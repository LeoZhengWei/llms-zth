import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# fused QKV 把每个头各自的三套Linear(Q/K/V)，合成一次大线性层(C->3C)，再切成Q、K、V，再分头
# 数学上一样，实现GEMM更整齐
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be disvisible by n_head")
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.head_size

        # (B, T, C) -> (B, T, 3C)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x) # (B, T, 3C)        # (B, T, 3C)
        # 按照qkv的3c 切分为3个(B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # (B, T, C) -> (B, T, C) * 3
        
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # scale 在softmax 前：mask填-inf 再softmax
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.attention_mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y
