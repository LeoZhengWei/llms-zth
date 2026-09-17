import json
from typing import List

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple text corpus dataset for causal language modeling."""

    def __init__(self, path: str, block_size: int = 512, max_lines: int | None = None):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover
            raise ImportError("tiktoken is required for GPT tokenization.") from exc

        self.encoder = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.encoder.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        self.encoded_data: List[List[int]] = []

        raw_data = []
        with open(path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if max_lines is not None and i >= max_lines:
                    break
                try:
                    text = json.loads(line.strip())["text"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                raw_data.append(text)

        full_encoded: List[int] = []
        for text in raw_data:
            full_encoded.extend(self.encoder.encode(text) + [self.eos_token])

        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i : i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.encoder.decode(ids)
