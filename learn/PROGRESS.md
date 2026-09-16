# 学习进度

最后更新：课程改为专家主干 + 正交支线（尚未开始上课）

## 当前阶段

L01、L02 已过关。下一课 **L03：fused QKV 重构并对拍**（详见 `learn/PLAN.md`）。毕业标准：`learn/GOALS.md`。

## 会话日志

- 09-16：L01 过关（KV cache 优先，口试答对）。L02 补全 GPT 并修 4 个 bug（Embeding 拼写、modules.bias、targets=None 未定义、单头 head_size 用了 config.head_size），形状断言 (2,16,50257) 通过，loss 10.86。下一课 L03。

## 毕业（L20 才勾）

- [ ] 口试 5 题能闭卷答（见 GOALS.md）
- [ ] 产物清单齐全（ckpt / KV bench / ONNX / GGML / mini-vLLM / 基线表）
- [ ] 无 GPU 时 TRT 已书面降级，而不是假装完成

## 主干检查点

- [x] S0 能区分：KV cache / ONNX / TRT / GGML / vLLM 各解决什么
- [x] S1 `GPT.forward` 形状正确
- [ ] S2 `checkpoint.pt`，greedy 可采样
- [ ] S3 naive vs KV：token 一致，TPOT 下降
- [ ] S4 基线表（硬件、prompt、TTFT、TPOT）

## 支线检查点（S3 后再勾）

- [ ] B1 ONNX 与 PyTorch greedy 对齐
- [ ] B2 TensorRT engine（无 GPU 则记「降级」）
- [ ] B3 GGML C++ 前向 + 生成
- [ ] B4 mini-vLLM 连续批处理吞吐表

## 主线模型

- 定义：`generate_talk/model/build_gpt.py`
- 配置：默认 6L/768d/512ctx/50257 vocab（S2 可缩小，改了记这里）
- checkpoint / ONNX / engine / ggml：尚无

## 基线表

| backend | kv | prompt | new | ttft_ms | tpot_ms | notes |
|---------|----|--------|-----|---------|---------|-------|
| （S4 填写） |  |  |  |  |  |  |

## 环境笔记

- Ubuntu 22.04 WSL2，conda `~/miniconda3`
- **学习用环境：`conda activate llms`**（Python 3.12.14）
- torch 2.14.0+cu130，CUDA 可用，GPU：RTX 3050 8GB
- 不要用 `base` 的 Python 3.14 装 torch（官方 CUDA 轮子对不上）
- 系统 `nvcc` 11.5 / apt toolkit 与 PyTorch 自带 CUDA 13.0 无关，不要混编

## 会话日志

（每次课：日期、改写了什么问题、做了什么、数字、下一课）
