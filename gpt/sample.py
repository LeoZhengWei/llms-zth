import torch

from config import GPTConfig
from model.gpt import GPT


def main():
    config = GPTConfig()
    model = GPT(config)
    checkpoint = torch.load("checkpoints/model_epoch_0.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=8)
    print(generated)


if __name__ == "__main__":
    main()
