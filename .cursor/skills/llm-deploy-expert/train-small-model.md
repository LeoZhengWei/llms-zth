# S2 训练小模型

产物：本仓库自己的 `checkpoint.pt`。不要用 HF GPT-2 换掉主线。

## 文件

```
generate_talk/model/build_gpt.py
generate_talk/data/prepare.py
generate_talk/train.py
generate_talk/sample.py
```

## 体量（主动缩小）

| 资源 | 配置 |
|------|------|
| GPU 16GB+ | 可接近 6L/768d，block_size 先 256 |
| 8GB / 弱 GPU | 4L/256d/8h/block 128 |
| 仅 CPU / WSL 无卡 | 2L/128d/4h/block 64，几 MB 文本过拟合 |

`n_embd % n_head == 0`。词表继续 50257，除非改字符级（改了必须记 PROGRESS，所有后端跟同一词表）。

## 必须写对

- 滑窗 `block_size+1` → `x=buf[:-1], y=buf[1:]`
- `eval` 时无 dropout；保存 `state_dict` + 完整 config + tokenizer 名
- 采样验收用 greedy，方便 S3 对拍

数据：tiny shakespeare 或一小段中文即可。KPI 是过拟合可见，不是开域流畅。

## 验收

train loss 下降；`sample.py` 非乱码；ckpt 路径入 PROGRESS。然后立刻进 S3，不要在这里插入 ONNX。
