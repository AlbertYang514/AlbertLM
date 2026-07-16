import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .linear import (
    LinearMode,
    export_native_state_dict,
    fp8_forward_context,
    prepare_state_dict_for_strict_load,
)
from .rmsnorm import RMSNorm


class AlbertLM(nn.Module):

    def __init__(
        self,
        config,
        *,
        linear_backend="native",
        fp8_enabled=False,
    ):
        super().__init__()

        self.config = config
        self.linear_mode = LinearMode(
            linear_backend,
            bool(fp8_enabled),
        )
        self.last_state_dict_load_report = None

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    self.linear_mode,
                )
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


    @property
    def uses_transformer_engine(self):
        return self.linear_mode.uses_transformer_engine


    @property
    def fp8_enabled(self):
        return self.linear_mode.fp8_enabled


    def load_state_dict(
        self,
        state_dict,
        strict=True,
        assign=False,
    ):
        if strict is not True:
            raise ValueError(
                "AlbertLM checkpoint loading requires strict=True"
            )

        converted, report = (
            prepare_state_dict_for_strict_load(
                self,
                state_dict,
            )
        )
        result = super().load_state_dict(
            converted,
            strict=True,
            assign=assign,
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "strict state_dict load returned incompatible keys: "
                f"missing={result.missing_keys} "
                f"unexpected={result.unexpected_keys}"
            )
        report["missing_keys"] = list(
            result.missing_keys
        )
        report["unexpected_keys"] = list(
            result.unexpected_keys
        )
        self.last_state_dict_load_report = report
        return result


    def native_state_dict(self):
        return export_native_state_dict(self)


    def forward(
        self,
        input_ids,
        labels=None,
        *,
        fp8_enabled=None,
    ):

        forward_mode = self.linear_mode
        if fp8_enabled is not None:
            forward_mode = LinearMode(
                self.linear_mode.backend,
                bool(fp8_enabled),
            )

        with fp8_forward_context(
            forward_mode
        ):
            return self._forward_impl(
                input_ids,
                labels=labels,
            )


    def _forward_impl(
        self,
        input_ids,
        labels=None,
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
