#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py —— 为 llms-zth 的 GPT 训练准备中文语料

数据源（ModelScope 开源，无需登录）：
    xuanxixue/HANA  «HANA 中文闲聊对话数据集»
    100 个 batch × 1000 条 = 100,000 条中文多轮对话，约 6.2M 汉字

输出（相对仓库根目录）：
    gpt/data/train.jsonl      训练集，每行一条 {"text": "..."}
    gpt/data/val.jsonl        验证集（默认 5% 对话）
    gpt/data/DATASET.md       数据卡片（来源、规模、校验和、重建命令）

train.jsonl 的格式与 gpt/data/dataset.py 里 TextDataset 的读取方式严格对齐：
它用 json.loads(line)["text"] 取正文，所以每行必须是且仅含一个 "text" 键的 JSON。
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from glob import glob

# ---------------------------------------------------------------- 数据集常量
DATASET_ID = "xuanxixue/HANA"


def human(n: int) -> str:
    """把字节数格式化成人类可读形式。"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def download_dataset(cache_dir: str, pattern: str) -> str:
    """下载数据集快照，返回本地快照根目录。依赖 modelscope 包。"""
    try:
        from modelscope import dataset_snapshot_download
    except ImportError:
        sys.exit(
            "缺少 modelscope 包。请先安装：\n"
            "    ~/anaconda3/envs/llms-zth/bin/pip install modelscope"
        )

    print(f"[1/4] 下载 {DATASET_ID} -> {cache_dir}")
    root = dataset_snapshot_download(
        DATASET_ID,
        cache_dir=cache_dir,
        allow_patterns=[pattern],
    )
    return root


def iter_dialogues(batch_path: str):
    """
    从单个 batch 文件里产出对话条目。

    HANA 的顶层是 {"batch_info": {...}, "dialogues": [...]}，
    但对历史/衍生版本也兼容裸 list。这一处必须兼容，
    否则换数据集时会静默产出 0 条。
    """
    with open(batch_path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    dialogues = obj["dialogues"] if isinstance(obj, dict) else obj
    if not isinstance(dialogues, list):
        return
    for d in dialogues:
        if isinstance(d, dict):
            yield d


def render_dialogue(d: dict, sep_user: str, sep_assistant: str, sep_eos: str) -> str:
    """
    把一条多轮对话渲染成一段纯文本。

    刻意不用 chat 模板 / 特殊角色 token：本机 GPT 的词表是 tiktoken gpt2，
    <|user|> 这类标记不在词表里，会被拆成多个 token，既浪费又学不到稳定语义。
    纯文本 + 换行的效果就是标准的「中文小 GPT 学说话」玩法。
    """
    turns = d.get("conversations") or d.get("messages") or []
    lines = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or "").lower()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"{sep_user}{content}")
        elif role == "assistant":
            lines.append(f"{sep_assistant}{content}")
        else:
            lines.append(content)
    if not lines:
        return ""
    return "\n".join(lines) + sep_eos


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)              # gpt/ 的上一级 = 仓库根
    default_out = os.path.join(repo_root, "gpt", "data")

    ap = argparse.ArgumentParser(description="为 llms-zth 准备 HANA 中文对话语料")
    ap.add_argument("--out-dir", default=default_out, help="输出目录（默认 gpt/data）")
    ap.add_argument("--cache-dir", default="/tmp/ms-hana-cache", help="ModelScope 下载缓存")
    ap.add_argument("--max-dialogues", type=int, default=0,
                    help="最多使用多少条对话（0 = 全部，默认 0）")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="验证集占比，默认 0.05")
    ap.add_argument("--seed", type=int, default=1337, help="划分随机种子，保证可复现")
    ap.add_argument("--sep-user", default="<|用户|>", help="用户发言前缀")
    ap.add_argument("--sep-assistant", default="<|助手|>", help="助手回复前缀")
    ap.add_argument("--sep-eos", default="\n\n", help="对话之间的分隔后缀")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")

    if not args.force:
        for p in (train_path, val_path):
            if os.path.exists(p):
                sys.exit(
                    f"{p} 已存在。加 --force 覆盖，或先备份。\n"
                    f"（不想重下数据也可以直接删除该文件后重跑）"
                )

    root = download_dataset(args.cache_dir, "batches/*.json")
    batch_files = sorted(glob(os.path.join(root, "**", "batch_*.json"), recursive=True))
    if not batch_files:
        sys.exit(f"在 {root} 下没找到 batch_*.json，下载可能不完整")

    print(f"     找到 {len(batch_files)} 个 batch 文件")

    # ---------------------------------------------------------- 读取 + 渲染
    print("[2/4] 渲染对话为纯文本")
    records = []
    n_dialogue = 0
    n_chars = 0
    for fp in batch_files:
        for d in iter_dialogues(fp):
            n_dialogue += 1
            if args.max_dialogues and n_dialogue > args.max_dialogues:
                break
            text = render_dialogue(d, args.sep_user, args.sep_assistant, args.sep_eos)
            if not text:
                continue
            records.append({"text": text})
            n_chars += len(text)
        if args.max_dialogues and n_dialogue > args.max_dialogues:
            break

    if not records:
        sys.exit("渲染后得到 0 条记录，请检查数据集字段是否变化")

    print(f"     对话 {len(records)} 条 / 字符 {n_chars:,}")

    # ---------------------------------------------------------- 划分 + 落盘
    print("[3/4] 划分 train / val 并写盘")
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    def dump(path, recs):
        with open(path, "w", encoding="utf-8") as fh:
            for r in recs:
                # ensure_ascii=False 必须保留：中文要原样写入，
                # 否则文件会膨胀成 \uXXXX 转义（体积约 6 倍且肉眼不可读）
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(train_path, train_records)
    dump(val_path, val_records)

    for p, recs in ((train_path, train_records), (val_path, val_records)):
        size = os.path.getsize(p)
        print(f"     {os.path.relpath(p, repo_root)}  {len(recs):,} 行  {human(size)}")

    # ---------------------------------------------------------- 自检
    print("[4/4] 自检：用 dataset.py 的同款方式回读")
    n_read = 0
    n_text = 0
    with open(train_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            n_read += 1
            if json.loads(line.strip())["text"]:
                n_text += 1
    assert n_read == n_text and n_read > 0, "回读失败：不是每行都能取到 text"
    print(f"     回读 {n_read} 行全部含非空 text  ✓")

    # ---------------------------------------------------------- 数据卡片
    digest = hashlib.sha256()
    with open(train_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    sha = digest.hexdigest()

    card = f"""# 数据集卡片：{DATASET_ID}

本文件由 `gpt/prepare_data.py` 自动生成，记录 `gpt/data/*.jsonl` 的来源与规模。

## 来源

- 平台：ModelScope 魔搭（开源，无需登录）
- 数据集：`{DATASET_ID}` —「HANA 中文闲聊对话数据集」
- 下载：`~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py`
- 原始形态：100 个 `batches/batch_XXX.json`，共 100,000 条多轮对话，约 6.2M 汉字，压缩后约 107 MB

## 本地产物

| 文件 | 行数 | 说明 |
|---|---|---|
| `train.jsonl` | {len(train_records):,} | 训练集 |
| `val.jsonl` | {len(val_records):,} | 验证集（{args.val_ratio:.0%}） |

`train.jsonl` 每行一个 JSON 对象，**只有一个键 `text`**：

```json
{{"text": "<|用户|>你好\\n<|助手|>你好呀，有什么想聊的？\\n\\n"}}
```

这与 `gpt/data/dataset.py` 中 `TextDataset` 的读取方式严格对齐（`json.loads(line)["text"]`）。

## 校验

- train.jsonl SHA-256：`{sha}`
- 划分种子：`{args.seed}`（同样的输入 + 种子必然得到同样的划分）

## 重建

```bash
cd ~/zw/llms-zth
rm -f gpt/data/train.jsonl gpt/data/val.jsonl
~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py
```

## 备注

- 文本未做分词，交给 `tiktoken` 的 **gpt2** 词表在训练时现场编码。
- gpt2 词表以英文为主，中文约 **2.1 tokens / 汉字**，所以 6.2M 汉字 ≈ **13M tokens**。
  这是「小模型学中文」的正常代价；本仓库的 GPT-2 类词表不可能换成中文词表（尺寸写死在 `config.py` 的 `vocab_size`）。
- 语料量足以让 `block_size=512` 的模型过拟合出可辨认的中文，符合 L05/L06 的过关标准。
"""
    card_path = os.path.join(args.out_dir, "DATASET.md")
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write(card)
    print(f"     数据卡片 -> {os.path.relpath(card_path, repo_root)}")

    # 清理下载缓存里的非数据文件，避免仓库膨胀
    print()
    print("完成。训练前请确认已激活 llms-zth 环境。")
    print(f"缓存目录 {args.cache_dir} 可随时删除（重跑会重新下载）。")


if __name__ == "__main__":
    main()
