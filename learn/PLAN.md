# 学习计划（按课推进）

原则：一次只上一课，过关再下一课。不要按「ONNX → TensorRT → GGML → vLLM」当加速阶梯。

节奏建议：每周 3 课，每课 60–90 分钟。主干 10 课（约 4 周），支线 10 课（约 4 周）。总共约 **8 周 / 20 课 / 25–30 小时**。

**学完要成为什么样的人：** 见 `learn/GOALS.md`。终点不是「五个工具都点过」，而是同一套自训小 GPT 能对拍地跑在多后端上，并能按瓶颈选工具。

进度勾选：`learn/PROGRESS.md`。导师技能会读本文件的「当前课」。

---

## 怎么上每一课

1. 打开仓库 `llms-zth`，开 Cursor Agent。
2. 把该课的 **开场白** 原样发给 Agent（下面每课都写好了）。
3. 只追求该课的 **过关标准**。做完就停，不要顺手开下一课。
4. 让 Agent 更新 `PROGRESS.md`。你自己在本文件课号前打勾。

卡超过 40 分钟仍无产物：缩小模型配置，或结束本课并记下报错，下一课先修这个坑。

---

## 当前课

**L03**（S1 fused QKV 重构并对拍）

---

## 主干（必须按序，L01–L10）

### L01 · S0 环境与心智模型 · 60 分钟

- 做：`nvidia-smi`、`python -c "import torch; print(torch.cuda.is_available())"`。能跑 PyTorch 即可，**不要**再叠一套来路不明的系统 CUDA 去「配齐」。
- 懂：KV cache / ONNX / TensorRT / GGML / vLLM 各解决什么（见 skill 的 `stack.md`）。
- 过关：能用自己的话回答「单请求慢优先做什么」。
- 开场白：`按计划上 L01。核验 GPU/PyTorch，讲清四条运行时的分工。不要装新工具，不要写模型代码。`

### L02 · S1 补全 GPT · 90 分钟

- 做：在 `generate_talk/model/build_gpt.py` 补 `class GPT`（emb、blocks、ln_f、lm_head、forward）。
- 过关：`logits.shape == (B, T, vocab)` 的脚本断言通过。
- 开场白：`按计划上 L02。只补全 GPT.forward，给最小测试。不训练、不导出。`

### L03 · S1 巩固 · 60 分钟

- 做：口试 attention（mask 与 1/sqrt(d) 的位置）。把多头从 `ModuleList` 单头改成 fused QKV（为 L08 cache 铺路）。改完 forward 数值与改前对拍（固定输入）。
- 过关：能默写 Block 数据流；fused QKV 对拍通过。若时间不够，对拍过了就算过，口试下一课补。
- 开场白：`按计划上 L03。fused QKV 重构并对拍。不训练。`

### L04 · S2 数据 · 60 分钟

- 做：`tiktoken` + 小语料（tiny shakespeare 或一小段中文）→ token 文件。Dataset 滑窗，`x=buf[:-1], y=buf[1:]`。
- 过关：能取出一个 batch，打印 `x,y` 的移位关系。
- 开场白：`按计划上 L04。只做数据与 Dataset。不写完整 train loop。`

### L05 · S2 训练 · 90 分钟

- 做：`train.py`。无 GPU 或显存小：**主动缩小**（例如 2L/128d 或 4L/256d）。目标是 loss 下降，不是复现 GPT-2。
- 过关：train loss 明显下降；记下最终配置到 PROGRESS。
- 开场白：`按计划上 L05。写最小 train.py，配置按机器缩小。跑到 loss 下降即可。`

### L06 · S2 采样 · 60 分钟

- 做：`sample.py`，存 `checkpoint.pt`（state_dict + config + tokenizer 名）。greedy 采样。
- 过关：能生成非乱码文本（过拟合语料上应能认出原文碎片）；ckpt 路径写入 PROGRESS。
- 开场白：`按计划上 L06。从 ckpt greedy 采样。不要 ONNX。`

### L07 · S3 naive 解码基线 · 60 分钟

- 做：`infer_pt/generate.py`：每步把整个序列重新 forward（故意低效，作为对照）。
- 过关：能从 ckpt greedy 生成 16 token。
- 开场白：`按计划上 L07。只写 naive generate 当基线。不要 KV cache。`

### L08 · S3 KV cache · 90 分钟

- 做：attention/GPT 支持 cache；`infer_pt/generate_kv.py`。注意 decode 的 **位置编码是 T_past，不是 0**。
- 过关：代码能跑完 16 token（先不对拍也行，对拍是 L09）。
- 开场白：`按计划上 L08。给模型加 KV cache 并实现 generate_kv。先求能跑。`

### L09 · S3 对拍 · 60 分钟

- 做：同一 ckpt、同一 prompt、greedy，naive vs KV 的 token 必须一致。不一致就修 cache/pos。
- 过关：16 token 完全一致。
- 开场白：`按计划上 L09。naive 与 KV greedy 对拍，不一致就修到一致。`

### L10 · S4 测量 · 60 分钟

- 做：`infer_pt/bench.py`。记录 TTFT、TPOT、prompt 长度、是否 KV、硬件。
- 过关：PROGRESS 基线表有至少两行（naive / KV）。KV 的 TPOT 应低于 naive（prompt 拉长才明显）。
- 开场白：`按计划上 L10。写 bench，填 PROGRESS 基线表。不开支线。`

主干结束。下一课起才允许 ONNX / TRT / GGML / vLLM。

---

## 支线（L11–L20，顺序按专家排，不是你最初的工具清单）

为何是 **服务 → ONNX → GGML → TRT**：

- cache 刚做完，立刻做 mini-vLLM，调度还热。
- ONNX 练导出与对拍纪律。
- GGML 用上你已经装好的 cmake/gcc/clang。
- TensorRT 放最后：WSL 上 CUDA/TRT 摩擦最大；没卡就跳过，把两课改成量化或复盘。

### L11 · B4 静态 batch · 90 分钟

- 做：多请求 padding + 正确 mask，复用 L08 cache。
- 过关：两条不同长度 prompt greedy 结果与单条一致。
- 开场白：`按计划上 L11。静态 batch + KV。不要 paged、不要装 vllm 包。`

### L12 · B4 连续批处理 · 90 分钟

- 做：`add_request` / `step` 调度器。
- 过关：长短混合请求能跑完；能说明何时插入新请求。
- 开场白：`按计划上 L12。实现 continuous batching 的 step 循环。`

### L13 · B4 吞吐对比 · 60 分钟

- 做：串行 KV vs 静态 batch vs 连续批，8 请求混合长度。
- 过关：一张吞吐表进 PROGRESS。单请求仍须与 L09 greedy 对齐。
- 开场白：`按计划上 L13。bench 三种调度。不要上 TensorRT。`

### L14 · B1 ONNX 导出 · 90 分钟

- 做：prefill 图 `input_ids → logits`，动态轴。
- 过关：图能被 ORT 加载。
- 开场白：`按计划上 L14。导出 ONNX prefill 图。先不对 decode cache 图。`

### L15 · B1 ONNX 对拍 · 60 分钟

- 做：ORT vs PyTorch last-logits / greedy 16 token。
- 过关：FP32 max abs diff < 1e-4，token 一致。
- 开场白：`按计划上 L15。ONNX 与 PyTorch greedy 对拍。`

### L16 · B3 GGML 权重导出 · 60 分钟

- 做：从 ckpt dump 自定义二进制 + header（层数/维/词表）+ checksum。
- 过关：Python 读回 checksum 一致。
- 开场白：`按计划上 L16。只做权重导出与校验。不写 C++ 前向。`

### L17 · B3 C++ 单层对拍 · 90 分钟

- 做：CMake + ggml，实现一层 attention，与 PyTorch 固定输入对拍。
- 过关：打印前 8 个数对得上。
- 开场白：`按计划上 L17。ggml 单层 attention 对拍。`

### L18 · B3 C++ 整网生成 · 90 分钟

- 做：整网 forward + greedy。Tokenizer 可留在 Python。
- 过关：16 token 与 PyTorch 一致，或写明分叉点。
- 开场白：`按计划上 L18。C++ greedy 与 PyTorch 对拍。`

### L19 · B2 TensorRT 构建 · 90 分钟

- 前置：`nvidia-smi` 正常。否则改上「Q8 量化 GGML」加课，不要硬上 TRT。
- 做：用已对齐的 ONNX 打 FP32 engine，设动态 shape profile。
- 过关：engine 能跑 prefill。
- 开场白：`按计划上 L19。有 GPU 则建 TRT FP32 engine；没 GPU 则改 GGML Q8。`

### L20 · B2 测速或复盘 · 60 分钟

- 有 TRT：FP32 对拍 + TTFT/TPOT 对比 PyTorch-KV / ORT。
- 无 TRT：把 20 课的基线表收齐，写半页「我的模型在各后端的数字」。
- 开场白：`按计划上 L20。收尾测速或复盘，更新 PROGRESS。`

---

## 明确不做（直到计划写到）

- 下载 7B、`pip install vllm` 当作业
- SFT/DPO（那是训练课，不是部署主干）
- FlashAttention / 投机解码 / INT8 校准（加课）
- 为「配 CUDA」反复重装系统 toolkit

---

## 加课（全部 L20 之后，可选）

- 本模型 SFT 小数据
- ONNX decode 图带 KV
- GGML Q4 + 速度
- mini-vLLM paged KV 物理块
- 真 vLLM 跑对照（仍用同一套小模型或公开小模型）
