# 训练启动流程（手动操作版）

本文档说明本机（`ubuntu` / `hjadmin@172.88.88.12`）上 `llms-zth` 这个小 GPT 的
**数据准备 → 训练 → 采样** 完整手动流程。所有命令都已在真机跑通并记录实测数字。

---

## 0. 当前状态（一页速览）

| 项目 | 状态 |
|---|---|
| 数据集 | **HANA 中文闲聊对话数据集**（ModelScope `xuanxixue/HANA`），100,000 条多轮对话 |
| 已就位数据 | `gpt/data/train.jsonl`（95,000 行 / 22.7 MB）、`gpt/data/val.jsonl`（5,000 行 / 1.2 MB） |
| 语料规模 | 约 948 万汉字 → tiktoken **gpt2** 编码后约 **16.6M tokens**（32,499 个 `block_size=512` 的 block） |
| Python 环境 | conda env **`llms-zth`**（Python 3.11.16 / torch 2.14.0+cu130 / tiktoken 0.14.0） |
| 硬件 | GPU 0 = RTX 4090 24G，GPU 1 = RTX 5090 32G |
| 默认模型 | 6 层 / 768 维 / 12 头 / `block_size=512` / 词表 50257 → **约 1.39 亿参数** |
| 实测耗时 | 默认配置下 `max_lines=2000` 一个 epoch 约 **10 秒** |

> **词表提醒**：`config.py` 里 `vocab_size = 50257` 是写死的，用的是 **gpt2 词表**
> （英文为主）。中文在这个词表下约 **2.1 tokens/汉字**，所以 948 万汉字放大到 1660 万 tokens。
> 这是本仓库的既定设计，不要在 L04–L06 阶段试图换成中文词表。

---

## 1. 环境准备

```bash
# 目标机：hjadmin@172.88.88.12
ssh hjadmin@172.88.88.12
cd ~/zw/llms-zth
```

本机 conda 没有初始化到 PATH，**每次都要用绝对路径调用**：

```bash
PY=~/anaconda3/envs/llms-zth/bin/python
PIP=~/anaconda3/envs/llms-zth/bin/pip

# 自检（应当输出 True 和 2）
$PY -c "import torch, tiktoken, numpy; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

如果哪天报 `ModuleNotFoundError`，补装：

```bash
$PIP install numpy modelscope tiktoken
```

> 环境上一度缺 `numpy`（torch 依赖它，缺了就一跑就炸），已补装 `numpy 2.4.6`。

---

## 2. 数据准备

**已经跑完了，正常不用重跑。** 数据已经在 `gpt/data/` 下。
下面两条命令用于「想换数据」或「重建」。

### 2.1 看看现在有什么

```bash
cd ~/zw/llms-zth
ls -lh gpt/data/
head -1 gpt/data/train.jsonl          # 抽查一行，确认是可读中文
wc -l gpt/data/train.jsonl gpt/data/val.jsonl
```

### 2.2 重新生成（需要时）

```bash
cd ~/zw/llms-zth
rm -f gpt/data/train.jsonl gpt/data/val.jsonl
~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py
```

脚本会自动从 ModelScope 下载（约 107 MB，缓存在 `/tmp/ms-hana-cache`）、渲染、切分、写盘，
并刷新数据卡片 `gpt/data/DATASET.md`。常用参数：

```bash
# 只要 10000 条对话，快速试跑
~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py --max-dialogues 10000

# 换掉对话前缀标记
~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py --sep-user "问：" --sep-assistant "答："

# 改验证集比例与随机种子
~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py --val-ratio 0.02 --seed 42

# 覆盖已有文件
~/anaconda3/envs/llms-zth/bin/python gpt/prepare_data.py --force
```

### 2.3 数据格式（关键）

`gpt/data/train.jsonl` **每行一个 JSON 对象，且只有一个键 `text`**：

```json
{"text": "<|用户|>突然对养生保健很感兴趣\n<|助手|>不知道你有没有考虑过...\n\n"}
```

这个格式是**被 `gpt/data/dataset.py` 反向约束的** —— 它用
`json.loads(line.strip())["text"]` 取正文。所以：

- 每行必须是**合法单行 JSON**，不能有多余键、不能跨行；
- 正文里的换行必须写成 `\n` 转义（`json.dumps` 默认就会这么做）；
- 文件必须是 **UTF-8**。写盘时用 `ensure_ascii=False`，否则中文会变成
  `\uXXXX` 转义，体积涨约 6 倍且人眼不可读。

> 为什么用 `<|用户|>` / `<|助手|>` 而不是 `<|user|>`？
> 因为 gpt2 词表里**没有**这些特殊标记，写什么都会被拆成多个 token。
> 用中文标记的好处是「至少每个标记的 token 数固定、且角色边界肉眼可辨」。
> 这不是必须的，改成「问：」「答：」效果等价（见 2.2 的参数示例）。

---

## 3. 训练

```bash
cd ~/zw/llms-zth/gpt         # 必须在 gpt/ 下运行：代码用的是相对路径 data/train.jsonl
~/anaconda3/envs/llms-zth/bin/python train.py
```

**实测输出**（`max_lines=2000`、`max_epochs=2`，约 10 秒 / epoch）：

```
epoch=0 train_loss=4.4292 val_loss=2.2515
epoch=1 train_loss=1.9xxx val_loss=2.0xxx
```

产物：`gpt/checkpoints/model_epoch_0.pt`、`gpt/checkpoints/model_epoch_1.pt`
（每个约 **556 MB**，因为含 `model_state_dict` + config）。

### 3.1 先想清楚 `max_lines` 这个坑

`train.py` 第 44 行写死了 **`max_lines=2000`**：

```python
dataset = TextDataset(path="data/train.jsonl", block_size=config.block_size, max_lines=2000)
```

含义是**只用前 2000 行对话**（约 690 个 block / 12 万 tokens）。
这是 L05 刻意的设计：**目标是「loss 明显下降」，不是复现 GPT-2**。
好处是 10 秒一轮，改代码→看结果的循环极快。

想用全量 95,000 行，就把 `max_lines=2000` 改成 `max_lines=None`
（此时约 32,499 个 block / 1660 万 tokens，一轮大概几分钟到十几分钟，取决于配置）。

### 3.2 显存不够或想加速时

`config.py` 是唯一的旋钮：

```python
block_size    = 512      # 上下文长度
batch_size    = 12       # 显存不够先降这个
n_layer       = 6        # 层数
n_head        = 12       # 头数（必须整除 n_embd）
n_embd        = 768      # 隐层宽度
learning_rate = 3e-4
max_epochs    = 2
```

按学习计划的最小配置改法（L05 的建议）：

```python
# 4L / 256d 的小改法 —— 显存友好、loss 照样降
block_size = 256
batch_size = 16
n_layer    = 4
n_head     = 8
n_embd     = 256
```

> `n_embd % n_head != 0` 会在 `__post_init__` 里直接抛 `ValueError`，这是有意为之。

### 3.3 选哪张卡

`train.py` 用 `torch.cuda.is_available()`，**默认落在 GPU 0（RTX 4090）**。
想指定用卡，在命令前加环境变量：

```bash
# 用第二张卡（RTX 5090 32G）
CUDA_VISIBLE_DEVICES=1 ~/anaconda3/envs/llms-zth/bin/python train.py

# 观察占用
nvidia-smi
watch -n 1 nvidia-smi      # 实时看
```

### 3.4 后台跑长任务

全量数据训练会跑很久，建议挂后台并把日志落盘：

```bash
cd ~/zw/llms-zth/gpt
nohup ~/anaconda3/envs/llms-zth/bin/python train.py > train.log 2>&1 &
tail -f train.log          # 看进度
```

---

## 4. 采样验证

`sample.py` 从 `checkpoints/model_epoch_0.pt` 加载并生成。**训练至少要跑完 1 个 epoch**，
否则这个文件不存在。

```bash
cd ~/zw/llms-zth/gpt
~/anaconda3/envs/llms-zth/bin/python sample.py
```

当前 `sample.py` 存在的问题（L06 的动手内容）：

1. **prompt 是写死的 token id** `[[1, 2, 3, 4]]`，不是中文；
2. 用的是 `torch.multinomial`（随机采样），而 L06 要的是 **greedy**；
3. 用 `map_location="cpu"` 加载，然后在 CPU 上推理。

L06 的目标是能生成**非乱码中文**（过拟合语料的话应该能认出原文碎片）。
建议先手搓一个最小验证，确认权重是活的：

```bash
cd ~/zw/llms-zth/gpt
~/anaconda3/envs/llms-zth/bin/python - <<'PY'
import torch
from config import GPTConfig
from model.gpt import GPT
from data.dataset import TextDataset

cfg = GPTConfig()
model = GPT(cfg)
ck = torch.load("checkpoints/model_epoch_1.pt", map_location="cpu")
model.load_state_dict(ck["model_state_dict"])
model.eval()

ds = TextDataset(path="data/train.jsonl", block_size=cfg.block_size, max_lines=2000)

# 用真实中文当 prompt
prompt = "<|用户|>你好"
ids = ds.encode(prompt)
idx = torch.tensor([ids], dtype=torch.long)

# greedy 生成
for _ in range(40):
    logits, _ = model(idx)
    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, nxt], dim=1)

print(ds.decode(idx[0].tolist()))
PY
```

---

## 5. 常见问题

| 现象 | 原因 / 解法 |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | 用了系统 `python3`。必须用 `~/anaconda3/envs/llms-zth/bin/python` |
| `ModuleNotFoundError: No module named 'numpy'` | 环境缺 numpy：`~/anaconda3/envs/llms-zth/bin/pip install numpy` |
| `FileNotFoundError: data/train.jsonl` | 工作目录不对，必须在 `gpt/` 下运行 |
| `ImportError: tiktoken is required` | `pip install tiktoken` |
| `ValueError: n_embd must be divisible by n_head` | 改配置时没注意整除关系 |
| `json.decoder.JSONDecodeError` 刷屏 | `train.jsonl` 某行不是合法单行 JSON；`dataset.py` 会静默 `continue` 跳过，不会报错，表现为 dataset 变短 |
| 训练 OOM | 降 `batch_size` 或 `block_size` |
| 中文全变成 `\u4f60\u597d` | 写数据时漏了 `ensure_ascii=False` |
| `checkpoints/model_epoch_0.pt` 找不到 | 还没训练过，先跑 `train.py` |

### 数据自检一行命令

```bash
cd ~/zw/llms-zth/gpt
~/anaconda3/envs/llms-zth/bin/python - <<'PY'
import json
p = "data/train.jsonl"
n = 0
bad = 0
with open(p, encoding="utf-8") as f:
    for line in f:
        n += 1
        try:
            assert json.loads(line.strip())["text"]
        except Exception:
            bad += 1
print(f"总行数={n}  异常行={bad}  {'OK' if bad == 0 else '需要修数据'}")
PY
```

---

## 6. 文件清单

| 路径 | 说明 |
|---|---|
| `gpt/prepare_data.py` | 数据准备脚本（下载 → 渲染 → 切分 → 写盘 → 自检 → 出数据卡片） |
| `gpt/data/train.jsonl` | 训练集，95,000 行 |
| `gpt/data/val.jsonl` | 验证集，5,000 行 |
| `gpt/data/DATASET.md` | 数据卡片：来源、规模、SHA-256、重建命令 |
| `gpt/data/dataset.py` | `TextDataset`：滑窗切 block、`x=chunk[:-1]` / `y=chunk[1:]` |
| `gpt/config.py` | 全部超参 |
| `gpt/train.py` | 训练入口 |
| `gpt/sample.py` | 采样入口（L06 要改） |
| `gpt/checkpoints/` | 权重产物（约 556 MB / 个） |
| `.gitignore` | 已忽略 `*.jsonl` / `checkpoints/` / 权重文件 |

---

## 7. 按学习计划往下走

数据这一课对应 **L04**，训练对应 **L05**，采样对应 **L06**。
当前进度是 **L03**（fused QKV 重构并对拍）。

数据已就位，可以直接上 L04。把 `learn/PLAN.md` 里 L04 的开场白原样发给 Cursor Agent：

```
按计划上 L04。只做数据与 Dataset。不写完整 train loop。
```

过关标准：**能取出一个 batch，打印 `x, y` 的移位关系**。下面这行就已经验证过了：

```bash
cd ~/zw/llms-zth/gpt
~/anaconda3/envs/llms-zth/bin/python - <<'PY'
from data.dataset import TextDataset
import torch
ds = TextDataset(path="data/train.jsonl", block_size=512, max_lines=2000)
x, y = ds[0]
print("x.shape =", x.shape, " y.shape =", y.shape)
print("x[1:] == y[:-1] :", torch.equal(x[1:], y[:-1]))   # 必须是 True
print("解码前 120 token：", repr(ds.decode(x[:120].tolist())))
PY
```

预期输出：

```
x.shape = torch.Size([512])  y.shape = torch.Size([512])
x[1:] == y[:-1] : True
解码前 120 token： '<|用户|>突然对养生保健很感兴趣\n<|助手|>不知道你有没有考虑过...'
```
