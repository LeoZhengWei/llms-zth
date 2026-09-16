# B1 ONNX

前置：S3+S4。ONNX 是**可移植的图 + 通用运行时**，不是加速器。比速度可以，但课的验收是**对齐**，不是「必须更快」。

## 导出顺序

1. 无 cache 的 prefill：`input_ids (B,T) → logits (B,T,vocab)`，动态轴 batch/seq。用来对拍。
2. 再导出 decode：`input_ids (B,1)` + 每层 KV in → logits + KV out。没走通 S3 不要做这一步。

`eval()`、固定 GELU、opset 17。mask 用已有 buffer 的 `[:T,:T]` 切片。

## 文件

`export/export_onnx.py` `infer_onnx/check_align.py` `infer_onnx/generate.py`

## 坑

- `tril`/bool mask、死写 512、GELU 近似不一致、动态形状
- FP16 对拍阈值放宽；分叉则 FP32 图作为后续锚

## 验收

last-token logits FP32 max abs diff < 1e-4；greedy 16 token 与 PyTorch+KV 一致。路径写入 PROGRESS。
