# GPT 输入句子到 embedding 的张量流

本文用一句中文示例来说明一个句子在 GPT 里是如何从原始文本，变成 token id，再变成 embedding 向量，再叠加位置编码，最终送进 Transformer 的。

示例句子：

> 今晚夜色真好，其实是在说我爱你！

---

## 1. 文字先经过 tokenizer

在这个项目里，编码使用的是 GPT-2 的 tokenizer（见 [llms-zth/gpt/data/dataset.py](../gpt/data/dataset.py)），也就是 `tiktoken.get_encoding("gpt2")`。

文本不会直接作为一个大向量输入，而是先被拆成若干个 token。每个 token 会对应到词表里的一个整数 id。

例如假设这句话被拆成 12 个 token，那么输入会是：

```python
idx = tensor([[t1, t2, t3, ..., t12]])
# shape: [batch_size, seq_len] = [1, 12]
```

也就是说：

- batch_size = 1
- seq_len = 12
- 句子长度为 12 个 token

---

## 2. token id 查词表，变成 embedding

在 [llms-zth/gpt/model/gpt.py](../gpt/model/gpt.py) 中：

```python
self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
```

这里的参数含义是：

- `vocab_size = 50257`
- `n_embd = 768`

它实际会创建一张词表：

```python
[50257, 768]
```

每个 token id 会查表，得到一个 768 维向量。

所以：

```python
token_embeddings = self.token_embedding_table(idx)
# shape: [1, 12, 768]
```

这表示：

- 12 个 token
- 每个 token 对应一个 768 维向量
- 整个句子形成一个矩阵

---

## 3. 位置编码：给每个 token 记住“在第几位”

在同一个文件里，还定义了：

```python
self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
```

这里：

- `block_size = 512`
- `n_embd = 768`
- 位置表大小是 `[512, 768]`

如果句子长度是 12，那么位置索引是：

```python
pos = [0, 1, 2, ..., 11]
```

对应位置向量：

```python
pos_embeddings = self.position_embedding_table(pos)
# shape: [12, 768]
```

再做广播：

```python
x = token_embeddings + pos_embeddings.unsqueeze(0).expand(batch, -1, -1)
```

最终：

```python
x.shape = [1, 12, 768]
```

也就是说：

- 每个 token 的词向量 + 这个 token 的位置信息向量
- 这样模型知道“哪个词出现在第几个位置”
- 否则模型无法区分“我爱你”和“你爱我”在不同位置上的语义差异

---

## 4. 整体张量变化一览

```mermaid
flowchart LR
    A[原始文本\n今晚夜色真好，其实是在说我爱你！] --> B[Tokenizer\nGPT-2 token ids\nshape: [1, 12]]
    B --> C[Embedding Lookup\nW_vocab: [50257, 768]\nshape: [1, 12, 768]]
    C --> D[Position IDs\n0..11\nshape: [12]]
    D --> E[Position Embedding\nW_pos: [512, 768]\nshape: [12, 768]]
    C --> F[Token Embedding + Position Embedding\nshape: [1, 12, 768]]
    F --> G[多层 Transformer Block\n保持 shape 不变\n[1, 12, 768]]
    G --> H[LM Head\nshape: [1, 12, 50257]]
    H --> I[输出每个位置的下一个 token 概率]
```

---

## 5. 为什么最终是 [batch, seq_len, 768]

在 [llms-zth/gpt/config.py](../gpt/config.py) 中：

```python
n_embd = 768
n_head = 12
head_size = n_embd // n_head = 64
```

在 attention 中：

- q, k, v 都是从每个 token 的 768 维特征投影出来
- 每个 head 处理 64 维
- 12 个 head 拼接后仍然是 768 维

也就是：

```python
q/k/v: [batch, seq_len, 64]  # per head
concat: [batch, seq_len, 768]
```

所以整个模型在每层中，序列长度和特征维度都保持稳定：

```python
[batch_size, seq_len, n_embd]
```

---

## 6. 一句话总结

一句话进入 GPT 时，流程可以概括为：

```text
原始文本
    -> tokenizer => token ids
    -> embedding lookup => [batch, seq_len, 768]
    -> 加 position embedding => [batch, seq_len, 768]
    -> 多层 Transformer => [batch, seq_len, 768]
    -> LM Head => [batch, seq_len, vocab_size]
```

也就是说，真正“模型看到”的不是字符串，而是一串带有词义和位置信息的 768 维向量序列。

这就是 GPT 从文本到预测下一个 token 的核心机制。

---

## 7. 代码对应关系

- 配置参数： [llms-zth/gpt/config.py](../gpt/config.py)
- 数据集编码： [llms-zth/gpt/data/dataset.py](../gpt/data/dataset.py)
- 模型前向计算： [llms-zth/gpt/model/gpt.py](../gpt/model/gpt.py)
- 注意力计算： [llms-zth/gpt/model/attention.py](../gpt/model/attention.py)
- block 结构： [llms-zth/gpt/model/block.py](../gpt/model/block.py)

如果你准备发到 GitHub，可以直接把这篇说明贴进博客、issue 或 README。 
