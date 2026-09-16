# S3 解码与 KV cache

这是部署主干里最容易被跳过、也最不该跳过的一课。学员常把「加速」理解成换 ONNX/TRT/vLLM；先在这里把单请求做对、做快。

## 要建立的画面

- **Prefill**：一次吃完整 prompt，得到每个位置的 logits，并写出全部 K/V。
- **Decode**：每步输入 1 个 token；q 只来自这一步；k/v **追加**进 cache；因果性靠 cache 长度而不是再乘一整张 T×T 矩阵。

现有 `SingleHeadAttention` 每步对当前序列全做 `q @ k^T`。没有 cache 时，generate 循环若把整个 `idx` 重新 `forward`，复杂度随步数近似二次。

## 本阶段产物

```
infer_pt/generate.py     # naive：每步 full forward（基线，必须留着）
infer_pt/generate_kv.py  # 带 cache
infer_pt/bench.py        # 同一 prompt 比 naive vs kv
```

模型侧：给 attention / block / GPT 增加可选 `kv_cache`，decode 时 `T=1`。不要为了 cache 先上 paged layout（那是 B4）。

## 实现要点

- cache 形状按层：`k,v: (B, n_head, T_past, head_size)`（若仍是单头 ModuleList，先重构 fused QKV 再 cache，避免 12 个头各拷一份）。
- 与 naive 对拍必须 **greedy**、同一 checkpoint、同一 prompt。
- 位置编码：decode 步用的是 `pos = T_past`，不是 0。这是第一常见 bug。
- 第一课不做 sampling 花活；temperature/top-k 放加课。

## 验收

1. 16 个 greedy token 与 naive 完全一致。
2. `bench.py`：prompt 长度 ≥64、gen 32 token 时，KV 的 TPOT 明显低于 naive。
3. 数字写入 PROGRESS。过关后再允许 B1–B4。
