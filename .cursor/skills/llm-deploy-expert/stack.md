# 部署栈心智模型（纠偏用）

先用这段改写学员的问题，再动手。

## 生成慢，通常不是「没装 TensorRT」

自回归每步只吃 1 个新 token，却要带着全部历史。没 KV cache 时，第 t 步会把 1…t 再算一遍 Attention。  
**单请求最大的免费加速是 KV cache**（S3），不是换运行时。

四种工具解决的是不同问题：

| 工具 | 层 | 真正解决 | 不解决 |
|------|----|----------|--------|
| PyTorch + KV cache | 算法 | 别重算历史 K/V | 高并发下的显存碎片、请求调度 |
| ONNX / ORT | 格式 + 通用运行时 | 可移植、跟后续编译器对接 | 不保证比 PyTorch 快 |
| TensorRT | GPU 图编译器 | 层融合、kernel 选型、给定形状的前向 | 连续批处理、paged KV、请求队列 |
| GGML / llama.cpp | 轻量运行时 + 量化 | CPU/端侧、小内存把模型塞进去 | 服务端高 QPS 调度 |
| vLLM 类系统 | 推理服务 | 连续批处理、paged KV、吞吐 | 单请求 matmul 的理论峰值 |

## 把口头问题改写成工程问题

| 学员说 | 先改写成 | 你实际带他做 |
|--------|----------|----------------|
| 转 ONNX 来推理 | 要一个可移植、可对拍的图，不是自动加速 | S3 之后才 B1；先 naive vs cache 基线 |
| 用 TRT 加速我的模型 | 瓶颈是单步 GEMM 还是 decode 重算/调度？ | 无 cache → 拒绝进 B2 |
| 用 ggml 把模型用 C++ 重构 | 换运行时+语言+量化，不是 vLLM 前置 | B3，与 B4 无关 |
| 自己写 vLLM 加速推理 | 要学的是服务：组批与 KV 分页 | 必须先有 S3 cache，再 B4 |
| 把上面按 3→4→5→6 做完就会最快 | 想同时学四条支线 | 讲清正交，让他选最近目标 |

改写格式（每轮开头）：

```
你问的是：…
真正的问题是：…
所以今天做：…（不做：…）
```

## 环境（WSL）

- 有卡：`nvidia-smi` 正常才谈 B2。
- PyTorch CUDA 跟 **驱动** 走；再叠一套 `apt nvidia-cuda-toolkit` 很容易编译器/库冲突。本仓库默认：能 `torch.cuda.is_available()` 就先用这套；TRT 另装匹配版本，不要「系统 toolkit + conda 两套并行硬编」。
- 没卡或 WSL 没露出 GPU：S0 写清楚，B2 降级为讲原理+脚本占位。

## 测量词（S4 起强制使用）

- **TTFT**：首 token 延迟（prefill）
- **TPOT**：每输出 token 时间（decode）
- **吞吐**：并发下 tokens/s 或 requests/s
- **对拍**：同一 prompt、greedy、token 序列或 last-logits max abs diff

「变快了」必须带：对比对象、batch、prompt 长度、new tokens、硬件。
