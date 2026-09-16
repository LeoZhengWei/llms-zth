import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

import math
torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768       # n_embd 也叫hidden_dim, hidden_size,
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    # tiktoken使用和gpt2一样的词表
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size

        # attention_mask 通过 Register_buffer注册
        # 无需计算梯度，节约内存和显存
        self.register_buffer(
            "attention_mask", 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)  # @ == torch.matmul
        # 1 qk^T / sqrt(d_k) 计算注意力
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
            ) / math.sqrt(self.head_size) # 此处单头注意力，这里的hidden_size是head_size
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        # 2 * v
        out = weight @ v
        # 3 输出
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads], 
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x