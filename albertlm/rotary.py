import torch


def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]

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
    theta=10000
):

    inv_freq = 1.0 / (
        theta **
        (
            torch.arange(
                0,
                dim,
                2,
                device=device
            ).float()
            /
            dim
        )
    )


    t = torch.arange(
        seq_len,
        device=device
    ).float()


    freqs = torch.outer(t, inv_freq)

    emb = torch.cat(
        (freqs, freqs),
        dim=-1
    )


    return emb.cos(), emb.sin()
