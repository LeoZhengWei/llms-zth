# 学习进度

最后更新：2026-10-12 · L07 过关

## 当前阶段

L01–L07 已过关。下一课 **L08：S3 KV cache**（详见 `learn/PLAN.md`）。毕业标准：`learn/GOALS.md`。

## 会话日志

- 09-16：L01 过关（KV cache 优先，口试答对）。L02 补全 GPT 并修 4 个 bug（Embeding 拼写、modules.bias、targets=None 未定义、单头 head_size 用了 config.head_size），形状断言 (2,16,50257) 通过，loss 10.86。下一课 L03。
- 10-08：L03 过关。`MultiHeadAttention` 改为 fused QKV（`c_attn: C→3C`）；`scripts/compare_fused_qkv.py` 对拍旧 ModuleList，`max_abs_diff=0`，`mean_abs_diff=0`，PASS。旧实现备份在 `model/attention_unfused.py`。口试（mask / 1/sqrt(d)）未单独考，可下节补。下一课 L04。
- 10-09：L04 过关。`peek_dateset.py` 取出 batch，`x/y` 形状一致，`shift ok? True`；理解移位是 next-token 监督对齐，不是位置编码。语料已有 `train.jsonl`/`val.jsonl`。下一课 L05。
- 10-10：L05 过关。`cuda:1` RTX 5090；train_loss 1.0445→0.1584，val_loss 0.1807→0.1629（5 epoch）。ckpt：`gpt/checkpoints/model_epoch_4.pt`。配置仍为默认 6L/768d/512ctx/batch12。下一课 L06。
- 10-10：L06 过关。`generate.py --greedy` 可读中文 + 对话格式碎片（「我觉得这个问题要辩证地看」/ `<|用户|>`）。`checkpoint.pt` 含 `tokenizer=gpt2`。注意：导出脚本误用了 `model_epoch_0.pt`（val 0.1807），应改成 `model_epoch_4.pt` 再导出。下一课 L07。
- 10-12：L07 过关。`infer_pt/generate.py` 每步整段重算，`logits[:, -1, :].argmax` greedy 出 16 token（`new_tokens: 16`，text「关于房价物价，�」）。ckpt 已是 epoch 4，`val_loss=0.1629`。下一课 L08。

## 毕业（L20 才勾）

- [ ] 口试 5 题能闭卷答（见 GOALS.md）
- [ ] 产物清单齐全（ckpt / KV bench / ONNX / GGML / mini-vLLM / 基线表）
- [ ] 无 GPU 时 TRT 已书面降级，而不是假装完成

## 主干检查点

- [x] S0 能区分：KV cache / ONNX / TRT / GGML / vLLM 各解决什么
- [x] S1 `GPT.forward` 形状正确
- [x] S1 fused QKV 与 ModuleList 对拍通过（L03）
- [x] S2 数据：Dataset 滑窗 x/y 移位验证（L04）
- [x] S2 训练：loss 下降（L05）；greedy 采样待 L06
- [x] S2 `checkpoint.pt` + greedy 可采样（L06；主线应改用 epoch 4 权重）
- [ ] S3 naive vs KV：token 一致，TPOT 下降
- [ ] S4 基线表（硬件、prompt、TTFT、TPOT）

## 支线检查点（S3 后再勾）

- [ ] B1 ONNX 与 PyTorch greedy 对齐
- [ ] B2 TensorRT engine（无 GPU 则记「降级」）
- [ ] B3 GGML C++ 前向 + 生成
- [ ] B4 mini-vLLM 连续批处理吞吐表

## 主线模型

- 定义：`gpt/model/`（`attention.py` / `block.py` / `gpt.py`）
- 配置：6L / 12H / 768d / block 512 / vocab 50257 / batch 12（L05 实跑）
- checkpoint：`gpt/checkpoints/checkpoint.pt`（epoch 4，`val_loss=0.1629`，`tokenizer=gpt2`）
- ONNX / engine / ggml：尚无

## 基线表

| backend | kv | prompt | new | ttft_ms | tpot_ms | notes |
|---------|----|--------|-----|---------|---------|-------|
| （S4 填写） |  |  |  |  |  |  |

## 环境笔记

- Ubuntu，conda env：`llms-zth`
- 训练设备：`cuda:1` → NVIDIA GeForce RTX 5090（多卡时 `train.py` 选了 idx=1）
- 另有卡时注意别混用错误 device；系统 toolkit 与 PyTorch 自带 CUDA 不要混编
