import torch.nn as nn
import torch.nn.functional as F

from .linear import LinearMode, make_linear


class SwiGLU(nn.Module):

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        linear_mode: LinearMode,
    ):
        super().__init__()

        self.gate_proj = make_linear(
            hidden_size,
            intermediate_size,
            bias=False,
            mode=linear_mode,
        )

        self.up_proj = make_linear(
            hidden_size,
            intermediate_size,
            bias=False,
            mode=linear_mode,
        )

        self.down_proj = make_linear(
            intermediate_size,
            hidden_size,
            bias=False,
            mode=linear_mode,
        )


    def forward(self, x):

        return self.down_proj(
            F.silu(self.gate_proj(x))
            *
            self.up_proj(x)
        )
