import torch


_ROPE_CACHE = {}


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k


def build_rope_cache(
    seq_len,
    dim,
    device,
    theta=10000,
    dtype=torch.float32,
):
    device = torch.device(device)

    key = (
        device.type,
        device.index,
        int(seq_len),
        int(dim),
        float(theta),
        dtype,
    )

    cached = _ROPE_CACHE.get(key)

    if cached is not None:
        return cached

    with torch.no_grad():
        inv_freq = 1.0 / (
            theta
            **
            (
                torch.arange(
                    0,
                    dim,
                    2,
                    device=device,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        emb = torch.cat(
            (freqs, freqs),
            dim=-1,
        )

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)

    _ROPE_CACHE[key] = (cos, sin)

    return cos, sin
