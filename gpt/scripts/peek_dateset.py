import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print(f"ROOT={ROOT}")
from data.dataset import TextDataset

# 小block
ds = TextDataset(
    path=str(ROOT / "data" / "train.jsonl"),
    block_size=32,
    max_lines=200,
)
loader = DataLoader(ds, batch_size=2, shuffle=False)
x, y = next(iter(loader))

print(f"x.shape: {x.shape}, y.shape: {y.shape}")
print(f"batch0 x: {x[0].tolist()}")
print(f"batch0 y: {y[0].tolist()}")
print(f"shift ok?", torch.equal(x[0, 1:], y[0, :-1]))

# 解码
print(f"x decode: {ds.decode(x[0].tolist())}")
print(f"y decode: {ds.decode(y[0].tolist())}")