# S1 Transformer

对着 `generate_talk/model/build_gpt.py` 讲。目标是补全可 `forward` 的 `GPT`，为 S2/S3 服务，不是讲完论文。

## 现有事实

- Pre-LN：`x += attn(ln1(x)); x += ffn(ln2(x))`
- 单头：`softmax(mask(qk^T)/sqrt(hs)) v`；下三角 buffer，按 `seq_len` 切片
- 多头：`ModuleList` 单头 concat + `proj`（正确但慢；S3 前应改 fused QKV）
- FFN：`4h-GELU-h`
- 缺：token/pos emb、N 层 Block、`ln_f`、`lm_head`、`forward`/`generate`

## 只把这些钉死

1. 形状 `(B,T,n_embd)`，`hs = n_embd/n_head`
2. 缩放在 softmax 前
3. mask 填 `-inf` 再 softmax
4. 训练目标相对输入右移一位（Dataset 里做也行）
5. 预告 S3：decode 不该每步重算历史 K/V；现在先全量算，保证数值基线

## 产物与验收

补 `class GPT`。`logits.shape == (B,T,vocab)`。口试：复述 attention 三行。配置过大等到 S2 再缩。
