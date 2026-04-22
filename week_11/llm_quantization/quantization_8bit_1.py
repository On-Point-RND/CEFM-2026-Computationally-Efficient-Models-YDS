import os
import time
from typing import Optional, Union, Tuple

import torch
import gc

import datasets
from datasets import load_dataset

from pathlib import Path

import transformers

# from bitsandbytes.nn import Linear8bitLt

from bnbtriton.quantize_rowwise import quantize_rowwise
from bnbtriton.int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast
)
# from huggingface_hub import snapshot_download

# cache директория для хранения файлов, загруженных с hf
# os.environ["HF_HOME"] = "/content/hf_cache"
# os.environ["TRANSFORMERS_CACHE"]= "/content/hf_cache"

def print_memory():
    # Функция измерения затраченной GPU памяти
    device='cuda'
    mem_allocated = torch.cuda.memory_allocated(device=device) / 1024**3
    mem_reserved = torch.cuda.memory_allocated(device=device) / 1024**3
    print(f"allocated: {mem_allocated:,.2f} gb")
    print(f" reserved: {mem_reserved:,.2f} gb")


def time_pytorch_function(func, input):
    # Функция для имерения скорости расчета `func` для входа `input`

    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(*input)

    start.record()
    func(*input)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end)

def torch_quantize_rowwise(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    W_int8 = W.clone()
    W_scale = W_int8.abs().max(dim=1)[0]
    W_int8 = W_int8 / (W_scale.view(W.shape[0], 1)) * 127.0 #2^(b-1) - 1
    W_scale = W_scale.half()
    W_int8 = torch.round(W_int8).to(torch.int8)
    return W_int8, W_scale

@torch.compile()
def torch_compile_quantize_rowwise(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    W_int8 = W.clone()
    W_scale = W_int8.abs().max(dim=1)[0]
    W_int8 = W_int8 / (W_scale.view(W.shape[0], 1)) * 127.0
    W_scale = W_scale.half()
    W_int8 = torch.round(W_int8).to(torch.int8)
    return W_int8, W_scale

def torch_int8_matmul_rowwise_dequantize(
    X_int8_torch, 
    W_int8_torch_transpose, 
    X_scale_torch,
    W_scale_torch,
    bias = None
):
    divfactor = 1.0 / (127.0 * 127.0)
    acc = torch._int_mm(X_int8_torch, W_int8_torch_transpose)
    acc = acc * divfactor
    acc = X_scale_torch.view(-1, 1).float() * acc
    acc = W_scale_torch.view(1, -1).float() * acc
    acc = acc.half()

    if bias is not None:
        acc += bias.reshape(1, -1).half()

    return acc

@torch.compile()
def torch_compile_int8_matmul_rowwise_dequantize(
    X_int8_torch, 
    W_int8_torch_transpose, 
    X_scale_torch,
    W_scale_torch,
    bias = None
):
    divfactor = 1.0 / (127.0 * 127.0)
    acc = torch._int_mm(X_int8_torch, W_int8_torch_transpose)
    acc = acc * divfactor
    acc = X_scale_torch.view(-1, 1).float() * acc
    acc = W_scale_torch.view(1, -1).float() * acc
    acc = acc.half()

    if bias is not None:
        acc += bias.reshape(1, -1).half()

    return acc

def main():
    W = torch.randn((11008, 4096)).to(dtype=torch.float16, device=torch.device('cuda:0'))

    W_int8_torch, W_scale_torch = torch_quantize_rowwise(W)
    W_int8_bnb, W_scale_bnb = quantize_rowwise(W)

    assert torch.allclose(W_int8_torch, W_int8_bnb, atol=1.0), 'Quantized matrices do not match'
    assert torch.allclose(W_scale_torch, W_scale_bnb), 'Scales do not match'

    t1 = time_pytorch_function(torch_quantize_rowwise, [W])
    t2 = time_pytorch_function(quantize_rowwise, [W])
    t3 = time_pytorch_function(torch_compile_quantize_rowwise, [W])

    print(f"\n quantization time pytorch: {t1}, triton: {t2}, torch_compile: {t3} \n")
    breakpoint()
    X = torch.randn((2048, 4096)).to(dtype=torch.float16, device=torch.device('cuda:0'))
    bias = torch.randn(11008).to(dtype=torch.float16, device=torch.device('cuda:0'))
    X_int8_torch, X_scale_torch = torch_quantize_rowwise(X)
    X_int8_bnb, X_scale_bnb = quantize_rowwise(X)

    out_torch = torch_int8_matmul_rowwise_dequantize(
        X_int8_torch, 
        W_int8_torch.t(), 
        X_scale_torch,
        W_scale_torch,
        bias 
    )

    out_bnb = int8_matmul_rowwise_dequantize(
        X_int8_torch, 
        W_int8_torch.t(), 
        X_scale_torch,
        W_scale_torch,
        bias
    )

    assert torch.allclose(out_torch, out_bnb), 'Matmul outputs do not match'

    t = time_pytorch_function(torch_int8_matmul_rowwise_dequantize, [
        X_int8_torch, 
        W_int8_torch.t(), 
        X_scale_torch,
        W_scale_torch,
        bias
    ])
    print(f"pytorch int matmul: {t}")

    
    t = time_pytorch_function(int8_matmul_rowwise_dequantize, [
        X_int8_torch, 
        W_int8_torch.t(), 
        X_scale_torch,
        W_scale_torch,
        bias
    ])
    print(f"triton int matmul: {t}")

    
    t = time_pytorch_function(torch.nn.functional.linear, [X, W, bias])
    print(f"torch.nn.linear: {t}")

    t = time_pytorch_function(torch_compile_int8_matmul_rowwise_dequantize, [
        X_int8_torch, 
        W_int8_torch.t(), 
        X_scale_torch,
        W_scale_torch,
        bias
    ])
    print(f"torch compile int: {t}")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()