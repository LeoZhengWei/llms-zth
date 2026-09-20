import torch


def rotate_half(x):
    """对最后一维按 [x0, x1, x2, x3] -> [-x1, x0, -x3, x2] 的方式旋转."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return rotated


def apply_rope(x, seq_len):
    """
    x: [batch, seq_len, head_dim]
    作用：对每个位置施加旋转，编码相对位置。
    """
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).view(1, seq_len, 1)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )

    freqs = positions * inv_freq.view(1, 1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)

    x_rot = x * cos + rotate_half(x) * sin
    return x_rot


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    head_dim = 8

    x = torch.randn(batch_size, seq_len, head_dim)
    y = apply_rope(x, seq_len)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("first sample:\n", y[0])

    # RoPE 约束：旋转后仍保留相同的 shape
    assert x.shape == y.shape
    print("RoPE demo ok")
