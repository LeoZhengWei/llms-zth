import sys 
from pathlib import Path

import torch 
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GPTConfig
from model.gpt import GPT

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / "checkpoints" / "checkpoint.pt",
                      map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    enc = tiktoken.get_encoding(ckpt["tokenizer"])
    prompt = "<|用户|>今天玩什么游戏\n<|助手|>"
    ids = enc.encode(prompt)
    print(f"prompt: {prompt} (len: {len(prompt)})")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    n0 = idx.size(1)
    print(f"n0: {n0}, batch_size: {idx.size(0)}")

    max_new = 16
    with torch.no_grad():
        for _ in range(max_new):
            cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
            logits, _ = model(cond)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)

    new_ids = idx[0, n0:].tolist()
    print("new_tokens:", len(new_ids), new_ids)
    print("text:", enc.decode(new_ids))

if __name__ == "__main__":
    main()