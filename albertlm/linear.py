"""Auditable native/Transformer Engine linear selection for AlbertLM."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping

import torch
import torch.nn as nn


LINEAR_BACKENDS = ("native", "te")
PROJECTION_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
TE_EXTRA_STATE_SUFFIX = "._extra_state"


@dataclass(frozen=True)
class LinearMode:
    backend: str = "native"
    fp8_enabled: bool = False

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower()
        object.__setattr__(self, "backend", backend)
        if backend not in LINEAR_BACKENDS:
            raise ValueError(
                f"linear backend must be one of {LINEAR_BACKENDS}, got {self.backend!r}"
            )
        if self.fp8_enabled and backend != "te":
            raise ValueError("FP8 requires linear_backend='te'")

    @property
    def uses_transformer_engine(self) -> bool:
        return self.backend == "te"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


GROUP_MODES = {
    "A": LinearMode("native", False),
    "B": LinearMode("te", False),
    "C": LinearMode("te", True),
    "A2": LinearMode("native", False),
}


def mode_for_group(group: str) -> LinearMode:
    normalized = group.strip().upper()
    try:
        return GROUP_MODES[normalized]
    except KeyError as error:
        raise ValueError(f"unknown A/B group {group!r}; expected A, B, C, or A2") from error


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    mode: LinearMode,
) -> nn.Module:
    if not mode.uses_transformer_engine:
        return nn.Linear(in_features, out_features, bias=bias)

    import transformer_engine.pytorch as te

    # The seed and native model use FP32 master parameters. Torch BF16 autocast
    # controls compute dtype; TE FP8 autocast quantizes only GEMM operands.
    return te.Linear(
        in_features,
        out_features,
        bias=bias,
        params_dtype=torch.float32,
    )


_FP8_RECIPE_KIND = "delayed"


def configure_fp8_recipe(kind: str) -> None:
    """Select the process-wide FP8 recipe before the first FP8 forward."""

    if kind not in {"delayed", "current"}:
        raise ValueError(f"unsupported FP8 recipe: {kind}")
    global _FP8_RECIPE_KIND
    _FP8_RECIPE_KIND = kind
    _fp8_recipe.cache_clear()


@lru_cache(maxsize=1)
def _fp8_recipe():
    from transformer_engine.common import recipe

    if _FP8_RECIPE_KIND == "delayed":
        return recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
    if _FP8_RECIPE_KIND == "current":
        return recipe.Float8CurrentScaling(fp8_format=recipe.Format.HYBRID)
    raise AssertionError(_FP8_RECIPE_KIND)


def fp8_forward_context(mode: LinearMode):
    if not mode.fp8_enabled:
        return nullcontext()

    import transformer_engine.pytorch as te

    return te.autocast(enabled=True, recipe=_fp8_recipe())


def activation_checkpoint(function, *args, mode: LinearMode, **kwargs):
    """Use TE's recompute helper whenever a block contains TE modules."""

    if mode.uses_transformer_engine:
        import transformer_engine.pytorch as te

        # Keep reentrant checkpointing mandatory for FP8 because the
        # non-reentrant recompute path corrupts DelayedScaling backward state in
        # the full multi-layer accumulation=64 workload.
        use_reentrant = kwargs.get("use_reentrant", True)
        if mode.fp8_enabled and use_reentrant is not True:
            raise ValueError(
                "FP8 activation checkpointing requires use_reentrant=True"
            )
        kwargs["use_reentrant"] = use_reentrant

        # TE's helper owns RNG/autocast restoration and does not accept the
        # native checkpoint helper's preserve_rng_state argument.
        kwargs.pop("preserve_rng_state", None)
        return te.checkpoint(function, *args, **kwargs)

    from torch.utils.checkpoint import checkpoint

    return checkpoint(function, *args, **kwargs)


def _is_projection_extra_state(key: str) -> bool:
    if not key.endswith(TE_EXTRA_STATE_SUFFIX):
        return False
    module_name = key[: -len(TE_EXTRA_STATE_SUFFIX)].rsplit(".", 1)[-1]
    return module_name in PROJECTION_NAMES


def _copy_state_dict(state_dict: Mapping[str, object]) -> OrderedDict[str, object]:
    copied = OrderedDict(state_dict.items())
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = metadata.copy()  # type: ignore[attr-defined]
    return copied


def prepare_state_dict_for_strict_load(
    model: nn.Module,
    state_dict: Mapping[str, object],
) -> tuple[OrderedDict[str, object], dict[str, object]]:
    """Convert only TE runtime extra-state while strictly preserving weights."""

    target_state = model.state_dict()
    converted = _copy_state_dict(state_dict)

    target_extra = {key for key in target_state if _is_projection_extra_state(key)}
    source_extra = {key for key in converted if _is_projection_extra_state(key)}
    target_native = set(target_state) - target_extra
    source_native = set(converted) - source_extra

    missing_native = sorted(target_native - source_native)
    unexpected_native = sorted(source_native - target_native)
    if missing_native or unexpected_native:
        raise RuntimeError(
            "native state_dict key mismatch: "
            f"missing={missing_native[:50]} unexpected={unexpected_native[:50]}"
        )

    shape_mismatches = []
    for key in sorted(target_native):
        source_value = converted[key]
        target_value = target_state[key]
        source_shape = getattr(source_value, "shape", None)
        target_shape = getattr(target_value, "shape", None)
        if source_shape != target_shape:
            shape_mismatches.append((key, source_shape, target_shape))
    if shape_mismatches:
        raise RuntimeError(f"state_dict shape mismatch: {shape_mismatches[:50]}")

    unexpected_extra = sorted(source_extra - target_extra)
    removed_extra = []
    if unexpected_extra:
        # Native targets intentionally discard only recognized TE runtime
        # metadata. All parameter keys have already been checked above.
        if getattr(model, "linear_mode", LinearMode()).uses_transformer_engine:
            raise RuntimeError(f"unexpected TE extra-state keys: {unexpected_extra[:50]}")
        for key in unexpected_extra:
            converted.pop(key)
            removed_extra.append(key)

    added_extra = []
    for key in sorted(target_extra - source_extra):
        converted[key] = target_state[key]
        added_extra.append(key)

    report = {
        "strict": True,
        "native_key_count": len(target_native),
        "source_te_extra_state_count": len(source_extra),
        "target_te_extra_state_count": len(target_extra),
        "added_te_extra_state_keys": added_extra,
        "removed_te_extra_state_keys": removed_extra,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
    }
    return converted, report


def export_native_state_dict(model: nn.Module) -> OrderedDict[str, object]:
    """Return a state dict consumable by the native model with identical weights."""

    state = _copy_state_dict(model.state_dict())
    for key in list(state):
        if _is_projection_extra_state(key):
            state.pop(key)
    return state


def projection_module_counts(model: nn.Module) -> dict[str, int]:
    import transformer_engine.pytorch as te

    counts = {"native": 0, "te": 0, "other_linear": 0}
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in PROJECTION_NAMES:
            continue
        if isinstance(module, te.Linear):
            counts["te"] += 1
        elif isinstance(module, nn.Linear):
            counts["native"] += 1
        else:
            counts["other_linear"] += 1
    return counts
