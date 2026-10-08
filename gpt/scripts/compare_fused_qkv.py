from __future__ import annotations
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import GPTConfig
from model.attention import MultiHeadAttention

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.head_size
        self.key = nn.Linear(config.n_embd, self.head_size)
        self.query = nn.Linear(config.n_embd, self.head_size)
        self.value = nn.Linear(config.n_embd, self.head_size)
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        _, seq_len, _ = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)

        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf"),
        ) / math.sqrt(self.head_size)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        return weight @ v

class OldMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        return self.dropout(output)

def copy_old_to_fused(old: OldMultiHeadAttention, fused: MultiHeadAttention) -> None:
    c = fused.n_embd
    hs = fused.head_size
    with torch.no_grad():
        for i, h in enumerate(old.heads):
            fused.c_attn.weight[i * hs: (i + 1) * hs].copy_(h.query.weight)
            fused.c_attn.bias[i * hs: (i + 1) * hs].copy_(h.query.bias)
            fused.c_attn.weight[c + i * hs: c + (i + 1) * hs].copy_(h.key.weight)
            fused.c_attn.bias[c + i * hs: c + (i + 1) * hs].copy_(h.key.bias)
            fused.c_attn.weight[2 * c + i * hs: 2 * c + (i + 1) * hs].copy_(h.value.weight)
            fused.c_attn.bias[2 * c + i * hs: 2 * c + (i + 1) * hs].copy_(h.value.bias)
            fused.proj.weight.copy_(old.proj.weight)
            fused.proj.bias.copy_(old.proj.bias)

def main() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        block_size=32,
        n_layer=1,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        vocab_size=100,
    )

    old = OldMultiHeadAttention(cfg)
    fused = MultiHeadAttention(cfg)
    copy_old_to_fused(old, fused)
    old.eval()
    fused.eval()

    x = torch.randn(2, 16, cfg.n_embd)
    with torch.no_grad():
        y_old = old(x)
        y_new = fused(x)

    max_abs = (y_old - y_new).abs().max().item()
    mean_abs = (y_old - y_new).abs().mean().item()
    print(f"Max absolute difference: {max_abs:.6f}")
    print(f"Mean absolute difference: {mean_abs:.6f}")

    tol = 1e-5
    if max_abs >= tol:
        raise SystemExit(f"FAIL: max_abs_diff {max_abs} >= {tol}")
    print(f"PASS: fused QKV matches ModuleList within {tol}")

if __name__ == "__main__":
    main()