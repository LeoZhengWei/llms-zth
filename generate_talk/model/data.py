import json

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, path, block_size=512, max_lines=1000):
        import tiktoken
        self.encoder = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.eos_token = self.encoder.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        self.encoded_data = []

        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.encoder.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        # 将长文本分割成训练样本，每条样本 block_size + 1 个 token
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        # x 与 y 相对错开一位：预测下一个 token
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.encoder.decode(ids)
