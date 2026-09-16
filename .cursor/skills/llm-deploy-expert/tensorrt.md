# B2 TensorRT

前置：S3+S4，建议 B1 的 ONNX 已对齐。无 NVIDIA GPU 只讲原理+占位脚本。

TRT 是 **GPU 上的图编译器**（融合、选 kernel、按 profile 定形状）。它不管请求队列，也不自动给你 paged KV。单请求若还在全量重算 Attention，先回去 S3。

## 顺序

FP32 engine 对拍 → 再 FP16。INT8 校准当加课。动态维必须设 min/opt/max profile，例如 `(1,8)/(1,128)/(1,512)`。

decode 要加速，就把 S3 的 KV 接口编进网络；不要幻想 TRT 替你缓存。

## 文件

`trt/build_engine.py` `trt/infer.py` `trt/bench.py`

## 环境

`nvidia-smi`、`import tensorrt` 版本写进 PROGRESS。不要和一份来路不明的 `apt nvidia-cuda-toolkit` 混编译。

## 验收

FP32 greedy 对齐；表：TRT vs ORT vs PyTorch+KV 的 TTFT/TPOT。有加速比更好，没有也如实记（小模型上 TRT 开销可能吃掉收益）。
