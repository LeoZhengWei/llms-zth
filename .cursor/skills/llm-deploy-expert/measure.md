# S4 测量与对拍

没有基线表，后面所有「加速」都无法验收。嵌在 S3 末尾做完即可，不必单独拖很久。

## 固定协议

```
prompt 若干条（含短/长）
greedy，temperature=0
max_new_tokens = 16 或 32
记录：backend, has_kv, batch, prompt_len, ttft_ms, tpot_ms, tokens_per_s, peak_mem
对拍：token 列表；或 last-logits max abs diff
```

PyTorch naive 与 PyTorch+KV 是后面所有后端的对照锚。新后端先对齐 token，再比速度。

数字写入 `learn/PROGRESS.md` 的基线表。换机器要重测，不要抄旧表。
