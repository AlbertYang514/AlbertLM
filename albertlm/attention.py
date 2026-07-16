import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import LinearMode, make_linear
from .rotary import apply_rope, build_rope_cache


class GQAAttention(nn.Module):

    def __init__(self, config, linear_mode: LinearMode):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.head_dim = (
            config.hidden_size
            //
            config.num_attention_heads
        )

        assert (
            self.num_heads % self.num_kv_heads == 0
        )

        self.kv_repeat = (
            self.num_heads
            //
            self.num_kv_heads
        )


        self.q_proj = make_linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
            mode=linear_mode,
        )

        self.k_proj = make_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            mode=linear_mode,
        )

        self.v_proj = make_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            mode=linear_mode,
        )


        self.o_proj = make_linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
            mode=linear_mode,
        )


        self.rope_theta = config.rope_theta


    def repeat_kv(self, x):

        # x:
        # batch, kv_heads, seq, dim

        if self.kv_repeat == 1:
            return x


        return (
            x[:, :, None, :, :]
            .expand(
                x.shape[0],
                x.shape[1],
                self.kv_repeat,
                x.shape[2],
                x.shape[3],
            )
            .reshape(
                x.shape[0],
                self.num_heads,
                x.shape[2],
                x.shape[3],
            )
        )


    def forward(
        self,
        x,
        attention_mask=None,
    ):

        bsz, seq_len, _ = x.shape


        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        q = q.view(
            bsz,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )

        v = v.view(
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )


        # seq -> heads
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


        cos, sin = build_rope_cache(
            seq_len,
            self.head_dim,
            x.device,
            theta=self.rope_theta,
            dtype=q.dtype,
        )

        cos = cos[None,None,:,:]
        sin = sin[None,None,:,:]


        q, k = apply_rope(
            q,
            k,
            cos,
            sin
        )




        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=(
                self.num_heads
                != self.num_kv_heads
            ),
        )


        out = out.transpose(1,2)

        out = out.contiguous().view(
            bsz,
            seq_len,
            self.hidden_size
        )


        return self.o_proj(out)
