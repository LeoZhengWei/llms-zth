# 路线（专家版）

主干是一条因果链；支线是同一套权重上的不同运行时。

```
S0  栈 + 环境
        ↓
S1  补全 Transformer / GPT
        ↓
S2  训练 → checkpoint.pt
        ↓
S3  greedy generate + KV cache     ← 单请求加速的主杠杆
        ↓
S4  基线表（TTFT/TPOT/吞吐/对拍）
        ↓
   ┌────┼────────────┬────────────┐
   ▼    ▼            ▼            ▼
  B1   B2           B3           B4
 ONNX  TensorRT    GGML/C++    mini-vLLM
 可移植 GPU编译     端侧量化      高并发服务
```

SFT/DPO、FlashAttention、投机解码、INT8 校准：全部是加课，不挡主干。

## 目录

```
generate_talk/     S1–S2 模型与训练
infer_pt/          S3–S4 PyTorch 解码、cache、bench（也可放 generate_talk/）
export/ infer_onnx/  B1
trt/               B2
ggml/              B3
mini_vllm/         B4（复用 S3 的 cache 实现，不要重写一套注意力）
learn/PROGRESS.md
```

## 默认推进

学员没指定支线时：S0→S1→S2→S3→S4，然后问一句「下一步更想要可移植、GPU 极限、C++ 端侧，还是服务吞吐？」再开支线。

指定了错误顺序：纠偏后仍从最近的主干缺口开始。
