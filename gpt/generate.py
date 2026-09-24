#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py —— 从训练好的 checkpoint 生成文本（修正版）

相比原 sample.py 修了四个问题：

 1. **`�` 乱码的真正成因**：tiktoken(gpt2) 对中文是逐字节切分的。
    例如「玩」= UTF-8 `E7 8E A9`，会被切成 3 个 token（id 163/236/102）。
    如果 prompt 恰好停在这 3 个 token 的**中间**，最后一个 token 就是一个
    残缺字节；若模型生成的下一个 token 不是它的补位字节，这个残缺字节
    无法构成合法 UTF-8，`decode()` 就吐出 U+FFFD（`�`）。
    本脚本用 `_safe_decode` 做「宽松解码」：解码时把无效字节丢弃，
    正文一字不差，也不影响后续 token 的生成。

 2. **分布外 prompt**：训练语料每一轮都以 `<|用户|>` / `<|助手|>` 开头。
    直接喂裸的「今天去哪里玩」，模型没见过这个分布，会乱接。
    默认用 `--chat` 自动包成 `<|用户|>{prompt}\\n<|助手|>`。

 3. **采样策略**：原脚本是无 top-k / 无温度的纯多项式采样，在 5 万词表上
    容易抽到生僻字和乱码组合。现在默认 top-k=40 + temperature=0.8，
    并提供 `--greedy`（确定性）。

 4. **解码范围**：只解码**新生成的 token**，而不是「prompt + 生成」整体。
    这样 prompt 末端的残缺字节根本不会进入解码范围，从源头消掉 `�`。
"""

import argparse
import glob
import sys

import torch
import tiktoken

from config import GPTConfig
from model.gpt import GPT

REPLACEMENT = "\ufffd"


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    candidates = sorted(
        glob.glob(f"{checkpoint_dir}/model_epoch_*.pt"),
        key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return candidates[-1]


def safe_decode(encoder, ids: list[int]) -> str:
    """
    宽松解码：把 token 序列还原成字节流后按 UTF-8 解码，
    遇到残缺字节则丢弃（而不是变成 U+FFFD）。

    这是本脚本解决 `�` 的核心。注意它只作用于**输出**，
    不改变喂给模型的 token —— 模型看到的上下文始终是完整的。
    """
    raw = b"".join(encoder.decode_single_token_bytes(i) for i in ids)
    return raw.decode("utf-8", errors="ignore")


def build_prompt(text: str, chat: bool, user_tag: str, assistant_tag: str,
                 terminate: bool = True) -> str:
    """
    把裸 prompt 包成训练语料里的角色格式。

    `terminate` 很重要（踩过的坑）：
    训练语料是「连续对话流」，一条样本长这样：
        <|用户|>A\n<|助手|>B\n<|用户|>C\n<|助手|>D\n\n
    注意 <|助手|> 后面**直接跟内容**，没有换行。而 <|用户|> 后面永远有换行。
    所以 `<|助手|>` 之后若再补一个 `\n`，就制造了「<|助手|> 紧跟空行」
    这个语料里不存在的状态 —— 模型会立刻认为「这一轮说完了」，
    于是吐出一个 `<|用户|>` 开启下一轮，回复内容反而没了。

    正确做法：**让 prompt 恰好停在 `<|助手|>`，不要加换行**。
    """
    if not chat:
        return text
    if text.startswith(user_tag):
        # 用户自己写了角色标记，只补缺失的结尾
        return text if text.endswith(assistant_tag) else text.rstrip("\n") + assistant_tag
    body = f"{user_tag}{text}\n{assistant_tag}"
    return body if terminate else body + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="从 checkpoint 生成文本（中文友好版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--prompt", type=str, default="今天去哪里玩")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--checkpoint", type=str, default=None, help="默认取最新的 epoch")
    ap.add_argument("--device", type=str, default=None, help="如 cuda:1 / cpu")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40, help="0 = 关闭 top-k")
    ap.add_argument("--greedy", action="store_true", help="贪心解码（确定性）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--chat", dest="chat", action="store_true", default=True,
                    help="把 prompt 包成 <|用户|>..\\n<|助手|>（默认开）")
    ap.add_argument("--no-chat", dest="chat", action="store_false",
                    help="原样使用 prompt，不做包装")
    ap.add_argument("--user-tag", default="<|用户|>")
    ap.add_argument("--assistant-tag", default="<|助手|>")
    ap.add_argument("--terminate", dest="terminate", action="store_true", default=True,
                    help="prompt 停在 <|助手|>（默认，推荐）")
    ap.add_argument("--no-terminate", dest="terminate", action="store_false",
                    help="在 <|助手|> 后再补一个换行，看看会怎样")
    ap.add_argument("--raw", action="store_true", help="额外打印完整原始输出（含角色标记）")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ---------------------------------------------------------- 设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        idx = 1 if torch.cuda.device_count() > 1 else 0
        device = torch.device(f"cuda:{idx}")
    else:
        device = torch.device("cpu")

    # ---------------------------------------------------------- 载入
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt["config"] if isinstance(ckpt.get("config"), GPTConfig) else GPTConfig()
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if "val_loss" in ckpt:
        print(f"  epoch      = {ckpt.get('epoch')}")
        print(f"  val_loss   = {ckpt['val_loss']:.4f}")
    print(f"  device     = {device}")
    if device.type == "cuda":
        print(f"  gpu        = {torch.cuda.get_device_name(device)}")

    enc = tiktoken.get_encoding("gpt2")

    # ---------------------------------------------------------- 编码 prompt
    full_prompt = build_prompt(args.prompt, args.chat, args.user_tag,
                               args.assistant_tag, args.terminate)
    prompt_ids = enc.encode(full_prompt)
    if len(prompt_ids) > cfg.block_size:
        print(f"  [warn] prompt 长度 {len(prompt_ids)} > block_size {cfg.block_size}，已右截断")
        prompt_ids = prompt_ids[-cfg.block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_prompt = idx.size(1)

    mode = "greedy" if args.greedy else f"top-k={args.top_k}, temp={args.temperature}"
    print(f"\n=== Prompt ({n_prompt} tokens, {mode}) ===")
    print(full_prompt.replace("\n", "\\n"))

    # ---------------------------------------------------------- 生成
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if args.greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(args.temperature, 1e-6)
                if args.top_k and args.top_k > 0:
                    k = min(args.top_k, logits.size(-1))
                    topv, _ = torch.topk(logits, k)
                    logits[logits < topv[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, nxt], dim=1)

    # ---------------------------------------------------------- 解码
    # 关键：只解码新生成的 token，prompt 末端的残缺字节不进解码范围
    new_ids = idx[0, n_prompt:].tolist()
    new_text = safe_decode(enc, new_ids)

    # 备用：解码全序列（若 prompt 末端残缺会看到 �，属正常）
    full_text = safe_decode(enc, idx[0].tolist())

    print(f"\n=== Generated ({len(new_ids)} tokens) ===")
    print(new_text)
    if args.raw:
        print(f"\n=== Full (含 prompt) ===")
        print(full_text)

    if REPLACEMENT in full_text:
        print(f"\n[note] 全序列解码含 U+FFFD —— 这是 prompt 末端「半个汉字」"
              f"所致，属已知现象；上面 Generated 段是干净的。")


if __name__ == "__main__":
    main()
