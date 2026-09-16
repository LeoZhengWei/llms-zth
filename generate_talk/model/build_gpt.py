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
        self.head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, self.head_size)
        self.query = nn.Linear(config.n_embd, self.head_size)
        self.value = nn.Linear(config.n_embd, self.head_size)

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

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # linear(4 -> 8); weight shape 是 8 * 4
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 是输入的token ids
        batch, seq_len = idx.size()
        if seq_len > self.block_size:
            raise ValueError(f"seq_len {seq_len} > block_size {self.block_size}")
        token_embeddings = self.token_embedding_table(idx)

        # seq 长度是本次输入的最大长度
        pos_emb = self.position_embedding_table(
            torch.arange(seq_len, device=idx.device)
        )
        # 为什么embedding和position可以相加
        x = token_embeddings + pos_emb # shape (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)       # shape (batch, seq_len, vocab_size)

        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits_flat = logits.view(batch * seq_len, vocab_size)
            targets_flat = targets.reshape(batch * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后block_size个token
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 获得预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :] # becomes(B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__ == "__main__":
    from model.data import MyDataset

    train_dataset = MyDataset(path='data/train.jsonl')
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    train_loader = DataLoader(train_subset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=12, shuffle=False)

    model = GPT(GPTConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # 打印模型一共有多少参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    for epoch in range(2):
        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch)
        val_loss = eval(model, val_loader, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

        # 保存模型
        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': GPTConfig(),
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
