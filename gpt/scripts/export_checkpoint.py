import sys 
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GPTConfig

def main() -> None:
    src = ROOT / "checkpoints" / "model_epoch_4.pt"
    dst = ROOT / "checkpoints" / "checkpoint.pt"

    if not src.exists():
        raise FileNotFoundError(src)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    # 保留原有字段，补充tokenizer
    out = {
        "epoch": ckpt.get("epoch"),
        "model_state_dict": ckpt["model_state_dict"],
        "config": ckpt["config"],
        "val_loss": ckpt.get("val_loss"),
        "tokenizer": "gpt2",
    }
    torch.save(out, dst)
    print(f"wrote {dst}")
    print(f"keys: {list(out.keys())}")
    print(f"tokenizer: {out['tokenizer']}")
    print(f"epoch:{out['epoch']} val_loss: {out.get('val_loss')}")

if __name__ == "__main__":
    main()