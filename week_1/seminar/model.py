from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def get_loader(batch_size: int = 256, num_workers: int = 4) -> DataLoader:
    ds = datasets.FashionMNIST(
        root="mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, persistent_workers=True, pin_memory=True)


class SDPABlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        h = n_heads
        dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.h, self.dh = h, dh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        with record_function("FWD/SDPA"):  
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = self.ln1(x + self.o(y))
        x = self.ln2(x + self.ff(x))
        return x


class MLPBlock(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(F.linear(x, self.fc1.weight, self.fc1.bias))
        x = self.fc2(x)
        return x


class Head(nn.Module):
    def __init__(self, d_model: int, out_classes: int, head_shards: int = 1):
        super().__init__()
        assert out_classes % head_shards == 0
        self.shards = head_shards
        if head_shards == 1:
            self.single = nn.Linear(d_model, out_classes)
            self.parts = None
        else:
            k = out_classes // head_shards
            self.single = None
            self.parts = nn.ModuleList([nn.Linear(d_model, k) for _ in range(head_shards)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shards == 1:
            with record_function("FWD/ONE_BIG_LINEAR"):
                return self.single(x)
        outs = []
        for i, layer in enumerate(self.parts):
            with record_function(f"FWD/SMALL_LINEAR[{i}]"):
                outs.append(layer(x))
        with record_function("CAT"):
            return torch.cat(outs, dim=-1)


class Net(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 head_out_classes: int = 10,
                 head_shards: int = 1,
                 mlp_depth: int = 16):
        super().__init__()
        self.embed = nn.Linear(28, d_model)
        self.attn = SDPABlock(d_model=d_model, n_heads=n_heads)
        self.mlp_blocks = nn.ModuleList([
            MLPBlock(d_model=d_model)
            for _ in range(mlp_depth)
        ])
        self.head = Head(d_model=d_model, out_classes=head_out_classes, head_shards=head_shards)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x = self.embed(x)
        x = self.attn(x)
        for blk in self.mlp_blocks:
            x = blk(x)
        x = x.mean(dim=1)
        return self.head(x)


@dataclass
class RunConfig:
    device: str = "cuda:0"
    batch_size: int = 256
    num_workers: int = 4
    steps: int = 12
    d_model: int = 1024
    n_heads: int = 16
    use_autocast: bool = False
    use_fused_adam: bool = False
    use_flash: bool = False
    use_bool_mask: bool = False
    head_out_classes: int = 10
    head_shards: int = 1
    lr: float = 1e-3
    trace_name: str = "trace.json"


def _make_profiler(steps: int):
    sched = schedule(skip_first=1, wait=1, warmup=1, active=min(8, max(1, steps - 3)), repeat=1)
    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=sched, record_shapes=True, profile_memory=True)


def run(mode: str,
        *,
        device: str = "cuda:0",
        batch_size: int = 256,
        steps: int = 12,
        head_out_classes: int = 10,
        head_shards: int = 1,
        trace_dir: str = ".") -> None:
    """
      'baseline',           # fp32, no fused, no flash
      'autocast',           # только autocast
      'fused',              # fused adam
      'flash',              # только flash-attention (bool mask)
      'single_head',        # one big head
      'sharded_head',       # a lot of small heads
      'all'                 # autocast + fused + flash, one head
    """
    cfg = RunConfig(device=device, batch_size=batch_size, steps=steps,
                    head_out_classes=head_out_classes, head_shards=head_shards)

    if mode == "baseline":
        pass
    elif mode == "autocast":
        cfg.use_autocast = True
    elif mode == "fused":
        cfg.use_autocast = True
        cfg.use_fused_adam = True
    elif mode == "flash":
        cfg.use_fused_adam = True
        cfg.use_bool_mask = True
        cfg.use_flash = True
        cfg.use_autocast = True  # обычно flash эффективен в fp16/bf16
    elif mode == "single_head":
        cfg.head_shards = 1
        cfg.head_out_classes = max(1024, head_out_classes)
    elif mode == "sharded_head":
        cfg.head_shards = max(16, head_shards)
        k = max(2048, head_out_classes)
        while k % cfg.head_shards != 0:
            k += 1
        cfg.head_out_classes = k
    elif mode == "all":
        cfg.use_autocast = True
        cfg.use_fused_adam = True
        cfg.use_bool_mask = True
        cfg.use_flash = True
        cfg.head_shards = 1
        cfg.head_out_classes = max(1024, head_out_classes)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    torch.manual_seed(0)
    loader = get_loader(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    net = Net(d_model=cfg.d_model, n_heads=cfg.n_heads,
              head_out_classes=cfg.head_out_classes,
              head_shards=cfg.head_shards).to(cfg.device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, fused=cfg.use_fused_adam)
    loss_fn = nn.CrossEntropyLoss()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ac_ctx = (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)) if cfg.use_autocast else (lambda: torch.cuda.amp.autocast(enabled=False))
    sdp_ctx = (lambda: torch.backends.cuda.sdp_kernel(enable_flash=cfg.use_flash, enable_math=True, enable_mem_efficient=True))

    trace_path = f"{trace_dir}/trace_{mode}.json"

    with _make_profiler(cfg.steps) as prof:
        it = iter(loader)
        net.train()
        for i in tqdm(range(cfg.steps)):
            data, label = next(it)
            data = data.to(cfg.device, non_blocking=True)
            label = label.to(cfg.device, non_blocking=True)
            
            with sdp_ctx():
                with ac_ctx():
                    with record_function("FWD"):
                        logits = net(data)
                    with record_function("LOSS"):
                        if cfg.head_out_classes != 10:
                            label = torch.remainder(label, cfg.head_out_classes)
                        loss = loss_fn(logits, label)

            opt.zero_grad(set_to_none=True)
            with record_function("BWD"):
                loss.backward()
            with record_function("OPT_STEP"):
                opt.step()

            prof.step()

    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    print(f"Saved trace: {trace_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
