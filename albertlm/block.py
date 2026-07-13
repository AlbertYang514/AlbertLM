import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import GQAAttention
from .mlp import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.attention = GQAAttention(
            config
        )

        self.post_attention_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.mlp = SwiGLU(
            config.hidden_size,
            config.intermediate_size
        )


    def forward(
        self,
        x,
        attention_mask=None
    ):

        # Pre-Norm attention
        residual = x

        x = self.input_norm(x)

        x = self.attention(
            x,
            attention_mask
        )

        x = x + residual


        # Pre-Norm FFN
        residual = x

        x = self.post_attention_norm(x)

        x = self.mlp(x)

        x = x + residual


        return x

