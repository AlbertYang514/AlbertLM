#!/usr/bin/env python3
"""Run BF16 runtime sampling from the trusted local latest checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT_ROOT = ROOT / "checkpoints"
LATEST_PATH = CHECKPOINT_ROOT / "latest"
QUARANTINED_TAG = "step_000000002519"


def trusted_latest_checkpoint():
    tag = LATEST_PATH.read_text(encoding="utf-8").strip()
    if not tag or tag == QUARANTINED_TAG:
        raise RuntimeError(f"refusing checkpoint tag: {tag!r}")
    checkpoint = (CHECKPOINT_ROOT / tag).resolve()
    root = CHECKPOINT_ROOT.resolve()
    if checkpoint.parent != root or not checkpoint.is_dir():
        raise RuntimeError(f"latest is not a trusted project checkpoint: {checkpoint}")
    model_state = checkpoint / "mp_rank_00_model_states.pt"
    if not model_state.is_file():
        raise RuntimeError(f"model state is missing: {model_state}")
    return tag, checkpoint, model_state


def file_snapshot(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def main():
    tag, checkpoint_path, model_state_path = trusted_latest_checkpoint()

    # This process loads only the trusted project-owned path validated above.
    os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")

    import torch

    from albertlm.config import load_config
    from albertlm.linear import configure_fp8_recipe
    from albertlm.model import AlbertLM
    from train.runtime_eval import generate_samples

    latest_before = (LATEST_PATH.read_bytes(), file_snapshot(LATEST_PATH))
    model_state_before = file_snapshot(model_state_path)

    config = load_config(str(ROOT / "configs/model/albertlm-1.7b.yaml"))
    configure_fp8_recipe("delayed")
    model = AlbertLM(config, linear_backend="te", fp8_enabled=True)

    checkpoint = torch.load(
        model_state_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["module"], strict=True, assign=True)
    optimizer_step = int(checkpoint.get("optimizer_step", int(tag.rsplit("_", 1)[1])))
    tokens_seen = int(checkpoint.get("tokens_seen", optimizer_step * 131072))
    del checkpoint

    model.cuda()
    model.train()
    result = generate_samples(
        model,
        optimizer_step=optimizer_step,
        tokens_seen=tokens_seen,
    )

    latest_after = (LATEST_PATH.read_bytes(), file_snapshot(LATEST_PATH))
    if latest_after != latest_before:
        raise RuntimeError("checkpoints/latest changed during sample")
    if file_snapshot(model_state_path) != model_state_before:
        raise RuntimeError("checkpoint model state changed during sample")
    if not model.training:
        raise RuntimeError("model training state was not restored")
    if not result["all_logits_finite"]:
        raise RuntimeError("sample returned non-finite logits")

    summary = {
        "checkpoint": str(checkpoint_path),
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
        "precision": result["precision"],
        "transformer_engine_fp8_enabled": result[
            "transformer_engine_fp8_enabled"
        ],
        "temperature": result["temperature"],
        "top_p": result["top_p"],
        "max_new_tokens": result["max_new_tokens"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "all_logits_finite": result["all_logits_finite"],
        "model_training_state_restored": True,
        "latest_unchanged": True,
        "checkpoint_unchanged": True,
        "samples": result["samples"],
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
