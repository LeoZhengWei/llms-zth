# RoPE 暂存说明

这个目录是一个“临时实验夹”，用于保存把当前 GPT 绝对位置编码改成 RoPE（Rotary Positional Embedding）的思路和示例代码。

说明：
- 不修改现有代码
- 这里只是临时学习备份
- 适合在有空的时候单独研究和对照理解

---

## 目标

当前项目的输入逻辑是：

```python
x = token_embeddings + position_embeddings
```

这里使用的是绝对位置编码（learned positional embedding）。

RoPE 的核心思想不是“额外学一个位置表”，而是：

- 在 attention 的 q 和 k 上施加旋转
- 让相对位置关系编码进 q/k 的内积里

---

## 关键公式

对向量 `x = [x_0, x_1, ..., x_{d-1}]`，按偶数/奇数维做旋转：

```python
x' = [x_0 cos θ - x_1 sin θ, x_0 sin θ + x_1 cos θ, ...]
```

其中：

```python
θ = pos * inv_freq
inv_freq = 1 / (10000^(2i / d))
```

这样一来，两个 token 的相对距离会自然体现在 `q @ k` 中。

---

## 目录中的示例

- `rope_demo.py`：单独实现 RoPE 的最小示例
- 这个代码不是替换原项目代码
- 它仅用于理解 RoPE 的数学形式与张量计算方式

---

## 参考对象

当前项目的原始逻辑在：

- [llms-zth/gpt/model/gpt.py](../gpt/model/gpt.py)
- [llms-zth/gpt/model/attention.py](../gpt/model/attention.py)

你可以对照这两处代码理解：

- 现在：`position_embedding_table` 额外加位置表
- RoPE：在 q/k 上做相对位置旋转

---

## 学习建议

建议按下面顺序理解：

1. 先看当前项目的 `token embedding + position embedding`
2. 再看这个目录里的 `rope_demo.py`
3. 最后对照原始 attention 里 q/k 的计算流程
4. 重点理解：RoPE 不是替换 token embedding，而是替换位置表

