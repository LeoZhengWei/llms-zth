import os

import torch
from torch.utils.data import DataLoader, random_split

from config import GPTConfig
from data.dataset import TextDataset
from model.gpt import GPT


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    config = GPTConfig()
    if torch.cuda.is_available():
        target_idx = 1 if torch.cuda.device_count() > 1 else 0
        device = torch.device(f"cuda:{target_idx}")
        torch.cuda.set_device(device)
        print(f"Using CUDA device: {device} -> {torch.cuda.get_device_name(target_idx)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    os.makedirs("checkpoints", exist_ok=True)

    dataset = TextDataset(path="data/train.jsonl", block_size=config.block_size, max_lines=None)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    for epoch in range(config.max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, f"checkpoints/model_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
