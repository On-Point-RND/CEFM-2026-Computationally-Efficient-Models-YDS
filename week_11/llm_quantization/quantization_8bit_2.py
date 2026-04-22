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
from huggingface_hub import snapshot_download

# cache директория для хранения файлов, загруженных с hf
os.environ["HF_HOME"] = "/data/m.zhelnin/llm_quantization_with_bnbtriton/hf_cache"
os.environ["TRANSFORMERS_CACHE"]= "/data/m.zhelnin/llm_quantization_with_bnbtriton/hf_cache"

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

@torch.compile()
def torch_quantize_rowwise(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    W_int8 = W.clone()
    W_scale = W_int8.abs().max(dim=1)[0]
    W_int8 = W_int8 / (W_scale.view(W.shape[0], 1)) * 127.0
    W_scale = W_scale.half()
    W_int8 = torch.round(W_int8).to(torch.int8)
    return W_int8, W_scale

@torch.compile()
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

import random

# Load and process wikitext2 dataset
def get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    # Load test datasets
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    trainloader = None
    return trainloader, testenc


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"
    model.seqlen = 2048

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_wikitext2(seqlen=model.seqlen, tokenizer=tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


class BnbLinearW8A8OF16(torch.nn.Module):
    '''
    Линейный слой с квантизованными в int8 весами.
    При расчете forward pass активации квантизуются в int8
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: Union[torch.tensor, float] = 1.0,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty(
                    self.out_features,
                    dtype=torch.float16,
                    requires_grad=False,
                ),                
            )
        else:
            self.register_parameter("bias", None)

        # Одномерный массив параметров масштабирования для каждой строки матрицы весов
        self.register_buffer("weight_scale", torch.ones(out_features))


    def forward(self, X_3D):
        X = X_3D.view(-1, X_3D.size(-1))

        # Квантизовать входные активации X, используя функцию `quantize_rowwise`
        # X_int8, X_scale = quantize_rowwise(X)
        X_int8, X_scale = torch_quantize_rowwise(X)

        # Вычислить произведение весов на активации с 
        # использованием `int8_matmul_rowwise_dequantize`
        # res = int8_matmul_rowwise_dequantize(
        #     X_int8, self.weight.t(), 
        #     X_scale, self.weight_scale, 
        #     self.bias
        # ).view(*X_3D.size()[:-1], -1)
        res = torch_int8_matmul_rowwise_dequantize(
            X_int8, self.weight.t(), 
            X_scale, self.weight_scale, 
            self.bias
        ).view(*X_3D.size()[:-1], -1)
        
        return res

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear
    ):
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half()

        linear_weight = linear.weight.data.clone()
        # Квантизовать веса linear_weight в int8, используя методы pytorch
        # w_bit = 8
        # weight_scale = linear_weight.abs().max(dim=1)[0].half() / 2**(w_bit - 1)
        # weight_scale = weight_scale.to(linear_weight.device)
        # linear_weight = linear_weight.div_(weight_scale.view(linear.out_features, 1))
        # linear_weight = linear_weight.round_().to(torch.int8)

        # linear_weight, weight_scale = quantize_rowwise(linear_weight)
        linear_weight, weight_scale = torch_quantize_rowwise(linear_weight)

        assert (
            linear_weight.min() >= -128 and 
            linear_weight.max() <= 127
        ), "Quantized weight out of range"

        q_linear.weight_scale = weight_scale.contiguous()
        q_linear.weight.data = linear_weight.contiguous()

        return q_linear

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})'

def replace_with_qlinear(root_module):
    '''
    Процедура для замены линейных слоев в блоках трансформеров модели 
    на квантизованные линейные слои BnbLinearW8A8OF16
    '''

    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, torch.nn.Linear):
        # if isinstance(module, torch.nn.Linear) and (name.find("down_proj") != -1):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            
            q_linear = BnbLinearW8A8OF16.from_linear(module)
            # q_linear = Linear8bitLt()
            setattr(father, name[ind + 1 :], q_linear)
            print(f"replace layer {name} with {q_linear}")
            del module

def main():
    model_path = Path("/data/m.zhelnin/llm_quantization_with_bnbtriton/Llama-3.2-1B")

    # with open("./hf_token.txt", "r") as f:
    #     hf_token = f.read()
    # os.environ["HF_TOKEN"] = hf_token

    # model_path.mkdir(parents=True, exist_ok=True)
    # snapshot_download(
    #     repo_id="meta-llama/Llama-3.2-1B",
    #     local_dir=model_path,
    #     allow_patterns=[
    #         "config.json",
    #         "generation_config.json",
    #         "model.safetensors",
    #         "special_tokens_map.json",
    #         "tokenizer.json",
    #         "tokenizer_config.json"
    #     ]
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code = True,
        device_map = 'cuda:0'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    print_memory()

    questions = [
        "What is result of 2^5?",
        "Как добраться до Сколтеха?"
    ]

    answers = []

    for question in questions:
        tokenized_input = tokenizer(
            f"QUESTION: {question}\n ANSWER:",
            return_tensors="pt"
        )
        with torch.no_grad():
            output = model.generate(
                tokenized_input["input_ids"].cuda(),
                max_length=50, num_beams=3, early_stopping=True
            )[0]
        answer = tokenizer.decode(output, skip_special_tokens=True)
        print(answer)
        answers.append(answer[:answer.find(".")] + ".")
        

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    ppl = eval_ppl(model, tokenizer)
    end.record()
    torch.cuda.synchronize()
    print(f"ppl: {ppl}")
    print(f"time: {start.elapsed_time(end) / 1000:.2f}")
    breakpoint()

    replace_with_qlinear(model.model)
    gc.collect()
    torch.cuda.empty_cache()
    print_memory()

    answers_quant = []

    for question in questions:
        tokenized_input = tokenizer(
            f"QUESTION: {question}\n ANSWER:",
            return_tensors="pt"
        )
        with torch.no_grad():
            output = model.generate(
                tokenized_input["input_ids"].cuda(),
                max_length=50, num_beams=3, early_stopping=True
            )[0]
        answer = tokenizer.decode(output, skip_special_tokens=True)
        print(answer)
        answers_quant.append(answer[:answer.find(".")] + ".")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    ppl = eval_ppl(model, tokenizer)
    end.record()
    torch.cuda.synchronize()
    start.elapsed_time(end)
    print(f"ppl: {ppl}")
    print(f"time: {start.elapsed_time(end) / 1000:.2f}")

if __name__ == "__main__":
    main()