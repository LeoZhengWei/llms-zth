# B4 mini-vLLM（服务，不是编译器）

前置：**S3 的 KV cache 已对拍**。官方 vLLM 只作对照。禁止 `pip install vllm` 当本阶段完成。

vLLM 快在 **高并发解码**：连续组批 + KV 分页，不是把单次 GEMM 变得比 TRT 更强。单请求、batch=1 时，它常常并不比 PyTorch+KV 快。

## 阶梯（A 已在 S3 完成，不要重做）

| 步 | 做啥 | 验收 |
|----|------|------|
| C | 静态 batch + 正确 mask | 多请求 greedy 正确 |
| D | `add_request` / `step` 连续批处理 | 长短混合吞吐 > 静态等齐 |
| E | block table + paged KV（逻辑页即可） | 显存/内存占用可解释地下降 |
| F | CUDA Graph / 真 kernel | 加课 |

复用 `infer_pt` 的 cache 实现，不要第三套 attention。

## 接口

```
add_request(id, prompt_ids, max_new)
step() -> 本轮每个 running request 的 1 个新 token
```

## 测速

8 请求，prompt {16,64} 混合，`max_new=32`，greedy。对比：串行 KV vs 静态 batch vs 连续批。只报实测。

## 验收

单请求与 S3 greedy 对齐；吞吐表进 PROGRESS；能口头区分「KV cache / 连续批 / paged KV」。
