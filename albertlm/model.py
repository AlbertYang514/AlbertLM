import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm
from .block import TransformerBlock


class AlbertLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )


        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.embed_tokens.weight
            )


    def forward(
        self,
        input_ids,
        labels=None
    ):

        x = self.embed_tokens(
            input_ids
        )


        for layer in self.layers:
            x = layer(x)


        x = self.norm(x)


        logits = self.lm_head(x)


        loss = None

        if labels is not None:

            shift_logits = (
                logits[:, :-1]
                .contiguous()
            )

            shift_labels = (
                labels[:, 1:]
                .contiguous()
            )


            loss = F.cross_entropy(
                shift_logits.view(
                    -1,
                    self.config.vocab_size
                ),
                shift_labels.view(-1)
            )


        return {
            "logits": logits,
            "loss": loss
        }
