#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedup_data.py —— 针对 HANA 语料的「模板化」问题做去重/降采样

== 问题诊断（实测数据）==

HANA 的 100,000 条对话是从一个**固定回复池**里组合生成的：

  发言总数 427,628 / 唯一发言 61,705  ->  重复率 85.6%
  出现最多的单条发言出现约 9,098 次
  「助手」侧常见回复池里有「其实X因人而异」「我懂你这种感觉」
  「求助万能的朋友圈！」「哈哈，笑死我了！」等

这带来两个后果：
  1. 模型极易过拟合到模板句式 —— 表现为 val_loss 很低(0.16)但生成内容高度重复；
  2. 别指望它能「好好聊天」，它只能学会「这个语料的说话风格」。

**注意「整条对话去重」是无效的**：实测 95,000 条里只重复 24 条(0.0%)，
因为生成器把回复池随机排列组合，每条对话的序列都不同，但组成它的句子大量重复。

== 本脚本做什么 ==

不做「删数据」（会毁掉语料的组合多样性），而是：

  1. `--max-utterance-freq N`：把**出现超过 N 次**的发言在**采样时降权**，
     而不是删除。默认对高频句做下采样，让长尾句的相对比重上升。
  2. `--report`：只体检不改数据，输出重复率/模板句榜单。
  3. `--mode resample|report`：resample 写新文件，report 只打印。

产出 `train.dedup.jsonl` / `val.dedup.jsonl`（默认），不动原文件。
"""

import argparse
import json
import random
import sys
from collections import Counter

USER_TAG = "<|用户|>"
ASSISTANT_TAG = "<|助手|>"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def split_lines(text):
    return [l for l in text.strip().split("\n") if l.strip()]


def report(convs, title):
    utt = []
    for c in convs:
        utt.extend(split_lines(c))
    cnt = Counter(utt)
    rep = 100 * (1 - len(cnt) / len(utt)) if utt else 0
    print(f"--- {title} ---")
    print(f"  对话 {len(convs):,}  发言 {len(utt):,}  唯一发言 {len(cnt):,}  重复率 {rep:.1f}%")
    return cnt


def main():
    ap = argparse.ArgumentParser(description="对模板化中文对话语料做降采样去重")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out-train", default="data/train.dedup.jsonl")
    ap.add_argument("--out-val", default="data/val.dedup.jsonl")
    ap.add_argument("--max-utterance-freq", type=int, default=200,
                    help="出现次数超过该值的发言所在对话会被降采样（0=关闭）")
    ap.add_argument("--keep-ratio", type=float, default=0.35,
                    help="含高频发言的对话被保留的概率")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--report", action="store_true", help="只体检，不写文件")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    train = load(args.train)
    try:
        val = load(args.val)
    except FileNotFoundError:
        val = []

    cnt_all = Counter()
    for c in train:
        cnt_all.update(split_lines(c))

    cnt_before = report(train, "降采样前 train")
    if val:
        report(val, "降采样前 val")

    hot = {s for s, n in cnt_before.items() if n > args.max_utterance_freq}
    print(f"\n  出现 >{args.max_utterance_freq} 次的高频发言种类: {len(hot):,}"
          f"（占唯一发言 {100*len(hot)/max(len(cnt_before),1):.1f}%）")
    print(f"  它们贡献的发言量占比: "
          f"{100*sum(n for s,n in cnt_before.items() if s in hot)/max(sum(cnt_before.values()),1):.1f}%")

    if args.report:
        print("\n(report 模式，未写文件)")
        return

    # 降采样：含高频发言的对话按 keep_ratio 抽稀
    kept, dropped = [], 0
    for c in train:
        lines = split_lines(c)
        if any(l in hot for l in lines):
            if rng.random() < args.keep_ratio:
                kept.append(c)
            else:
                dropped += 1
        else:
            kept.append(c)

    print(f"\n  保留 {len(kept):,} 条，丢弃 {dropped:,} 条"
          f"（丢弃率 {100*dropped/max(len(train),1):.1f}%）")

    cnt_after = report(kept, "降采样后 train")

    with open(args.out_train, "w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    if val:
        with open(args.out_val, "w", encoding="utf-8") as f:
            for c in val:
                f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")

    n_chars = sum(len(c) for c in kept)
    print(f"\n  写出 {args.out_train}  ({len(kept):,} 行, {n_chars:,} 字符)")
    if val:
        print(f"  写出 {args.out_val}  ({len(val):,} 行)")
    print("\n  用法：把 train.py 里的 path 改成 data/train.dedup.jsonl 即可。")


if __name__ == "__main__":
    main()
