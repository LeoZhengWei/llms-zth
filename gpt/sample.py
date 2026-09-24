import argparse
import glob

import torch
import tiktoken

from config import GPTConfig
from model.gpt import GPT


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    candidates = sorted(glob.glob(f"{checkpoint_dir}/model_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
    parser.add_argument("--prompt", type=str, default="北京天气很好，", help="Text prompt to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file. Defaults to the latest checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. cuda:1 or cpu.")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"] if isinstance(checkpoint.get("config"), GPTConfig) else GPTConfig()
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoder = tiktoken.get_encoding("gpt2")
    prompt_ids = torch.tensor([encoder.encode(args.prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens)

    full_text = encoder.decode(generated[0].tolist())
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Generated text ===")
    print(full_text)


if __name__ == "__main__":
    main()
