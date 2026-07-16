import json
import math
import os

os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["TORCH_EXTENSIONS_DIR"] = "/data/AlbertLM/.cache/torch_extensions"

import random
import re
import shutil
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import deepspeed
import bitsandbytes as bnb
import numpy as np
import torch
from train.runtime_eval import maybe_run_runtime_evaluation
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from albertlm.config import load_config
from albertlm.linear import (
    activation_checkpoint,
    configure_fp8_recipe,
    parse_bool,
    projection_module_counts,
)
from albertlm.model import AlbertLM


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = Path(
    os.environ.get(
        "ALBERTLM_CONFIG",
        ROOT / "configs/model/albertlm-1.7b.yaml",
    )
)

DEEPSPEED_CONFIG_PATH = Path(
    os.environ.get(
        "ALBERTLM_DEEPSPEED_CONFIG",
        ROOT / "configs/deepspeed/albertlm-1.7b-zero2.json",
    )
)

DATA_PATH = Path(
    os.environ.get(
        "ALBERTLM_DATA",
        ROOT / "data/processed/train.bin",
    )
)

CHECKPOINT_DIR = Path(
    os.environ.get(
        "ALBERTLM_CHECKPOINT_DIR",
        ROOT / "checkpoints",
    )
)

LOG_DIR = Path(
    os.environ.get(
        "ALBERTLM_LOG_DIR",
        ROOT / "logs",
    )
)

SEQ_LEN = int(
    os.environ.get("SEQ_LEN", "2048")
)

MICRO_BATCH_SIZE = int(
    os.environ.get("MICRO_BATCH_SIZE", "1")
)

GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get(
        "GRADIENT_ACCUMULATION_STEPS",
        "64",
    )
)

PEAK_LR = float(
    os.environ.get("PEAK_LR", "2e-4")
)

MIN_LR = float(
    os.environ.get("MIN_LR", "2e-5")
)

WEIGHT_DECAY = float(
    os.environ.get("WEIGHT_DECAY", "0.1")
)

WARMUP_TOKENS = int(
    os.environ.get(
        "WARMUP_TOKENS",
        "100000000",
    )
)

TARGET_TOKENS = int(
    os.environ.get(
        "TARGET_TOKENS",
        "10000000000",
    )
)

DECAY_TOKENS = int(
    os.environ.get(
        "DECAY_TOKENS",
        "500000000",
    )
)

CHECKPOINT_EVERY_TOKENS = int(
    os.environ.get(
        "CHECKPOINT_EVERY_TOKENS",
        "50000000",
    )
)

KEEP_LAST_CHECKPOINTS = int(
    os.environ.get(
        "KEEP_LAST_CHECKPOINTS",
        "3",
    )
)

METRICS_EVERY_UPDATES = int(
    os.environ.get(
        "METRICS_EVERY_UPDATES",
        "10",
    )
)

SEED = int(
    os.environ.get("SEED", "20260714")
)

ENABLE_ACTIVATION_CHECKPOINTING = (
    os.environ.get(
        "ACTIVATION_CHECKPOINTING",
        "1",
    )
    != "0"
)

TOKENS_PER_MICRO_BATCH = (
    MICRO_BATCH_SIZE * SEQ_LEN
)

TOKENS_PER_UPDATE = (
    TOKENS_PER_MICRO_BATCH
    * GRADIENT_ACCUMULATION_STEPS
)

WARMUP_UPDATES = max(
    1,
    math.ceil(
        WARMUP_TOKENS
        / TOKENS_PER_UPDATE
    ),
)

TOTAL_UPDATES = max(
    WARMUP_UPDATES + 1,
    math.ceil(
        TARGET_TOKENS
        / TOKENS_PER_UPDATE
    ),
)

DECAY_UPDATES = max(
    1,
    math.ceil(
        DECAY_TOKENS
        / TOKENS_PER_UPDATE
    ),
)

DECAY_START_UPDATE = max(
    WARMUP_UPDATES,
    TOTAL_UPDATES - DECAY_UPDATES,
)

STOP_REQUESTED = False


class TokenDataset(Dataset):
    def __init__(
        self,
        path,
        seq_len,
        start_block=0,
    ):
        self.data = np.memmap(
            path,
            dtype=np.uint32,
            mode="r",
        )

        self.seq_len = int(seq_len)

        self.total_blocks = (
            len(self.data) - 1
        ) // self.seq_len

        self.start_block = int(
            start_block
        )

        if self.start_block < 0:
            raise ValueError(
                "start_block cannot be negative"
            )

        if self.start_block > self.total_blocks:
            raise ValueError(
                f"start_block={self.start_block} "
                f"exceeds total_blocks="
                f"{self.total_blocks}"
            )

    def __len__(self):
        return (
            self.total_blocks
            - self.start_block
        )

    def __getitem__(self, index):
        block = (
            self.start_block
            + int(index)
        )

        start = (
            block
            * self.seq_len
        )

        end = (
            start
            + self.seq_len
        )

        tokens = np.array(
            self.data[start:end],
            dtype=np.int64,
            copy=True,
        )

        return torch.from_numpy(tokens)


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


def append_jsonl(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )



def compute_actual_grad_norm(parameters):
    """Compute L2 norm from the real parameter gradients."""
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not grads:
        return 0.0

    # foreach_norm avoids launching one Python-level reduction per tensor.
    norms = torch._foreach_norm(grads, 2.0)
    total_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                norm.float()
                for norm in norms
            ]
        ),
        2.0,
    )

    return float(total_norm.item())


def append_metrics(
    step,
    tokens_seen,
    loss,
    learning_rate,
    grad_norm,
    tokens_per_second,
):
    append_jsonl(
        LOG_DIR / "metrics.jsonl",
        {
            "schema_version": 1,
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "train_loss": float(loss),
            "learning_rate": float(
                learning_rate
            ),
            "grad_norm": float(
                grad_norm
            ),
            "tokens_per_second": float(
                tokens_per_second
            ),
            "timestamp": utc_now(),
        },
    )


def write_status(
    status,
    step,
    loss,
    checkpoint=None,
    message=None,
):
    payload = {
        "time": utc_now(),
        "status": status,
        "step": int(step),
        "loss": (
            None
            if loss is None
            else float(loss)
        ),
        "checkpoint": (
            None
            if checkpoint is None
            else str(checkpoint)
        ),
        "message": message,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

    path = LOG_DIR / "status.json"
    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def request_stop(
    signum,
    frame,
):
    del frame

    global STOP_REQUESTED
    STOP_REQUESTED = True

    print(
        f"received signal {signum}; "
        "will stop at the next optimizer boundary",
        flush=True,
    )


def install_stop_handlers():
    signal.signal(
        signal.SIGINT,
        request_stop,
    )

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    signal.signal(
        signal.SIGHUP,
        request_stop,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_transformer_layers(model):
    expected_layers = 28

    preferred_paths = (
        "layers",
        "blocks",
        "transformer.layers",
        "transformer.h",
        "model.layers",
        "model.blocks",
    )

    for path in preferred_paths:
        value = model

        for name in path.split("."):
            value = getattr(
                value,
                name,
                None,
            )

            if value is None:
                break

        if (
            isinstance(value, nn.ModuleList)
            and len(value)
            == expected_layers
        ):
            return path, value

    for name, module in model.named_modules():
        if (
            isinstance(module, nn.ModuleList)
            and len(module)
            == expected_layers
        ):
            return name, module

    raise RuntimeError(
        "cannot locate the 28 transformer "
        "layers for activation checkpointing"
    )


def enable_activation_checkpointing(
    model,
):
    if hasattr(
        model,
        "gradient_checkpointing_enable",
    ):
        model.gradient_checkpointing_enable()

        print(
            "activation checkpointing enabled "
            "through model API",
            flush=True,
        )

        return

    path, layers = find_transformer_layers(
        model
    )

    for layer in layers:
        original_forward = layer.forward

        def checkpointed_forward(
            *args,
            _original_forward=original_forward,
            **kwargs,
        ):
            if not torch.is_grad_enabled():
                return _original_forward(
                    *args,
                    **kwargs,
                )

            def run_forward(*inputs):
                return _original_forward(
                    *inputs,
                    **kwargs,
                )

            return activation_checkpoint(
                run_forward,
                *args,
                mode=model.linear_mode,
                use_reentrant=model.fp8_enabled,
                preserve_rng_state=False,
            )

        layer.forward = checkpointed_forward

    print(
        "activation checkpointing enabled "
        f"for {len(layers)} layers at {path}",
        flush=True,
    )



def enable_regional_compilation(model):
    mode = os.environ.get(
        "ALBERTLM_COMPILE_MODE",
        "default",
    ).strip()

    if mode.lower() in {
        "",
        "0",
        "false",
        "off",
        "none",
    }:
        print(
            "regional torch.compile disabled",
            flush=True,
        )

        return

    path, layers = find_transformer_layers(
        model
    )

    for layer in layers:
        layer.compile(
            mode=mode,
            fullgraph=False,
            dynamic=False,
        )

    print(
        "regional torch.compile enabled "
        f"mode={mode} "
        f"layers={len(layers)} "
        f"path={path}",
        flush=True,
    )

def wsd_multiplier(update_index):
    update = max(
        0,
        int(update_index),
    )

    if update < WARMUP_UPDATES:
        return max(
            1.0 / WARMUP_UPDATES,
            (update + 1)
            / WARMUP_UPDATES,
        )

    if update < DECAY_START_UPDATE:
        return 1.0

    progress = min(
        1.0,
        (
            update
            - DECAY_START_UPDATE
        )
        / max(
            1,
            TOTAL_UPDATES
            - DECAY_START_UPDATE,
        ),
    )

    minimum_ratio = (
        MIN_LR / PEAK_LR
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        minimum_ratio
        + (
            1.0
            - minimum_ratio
        )
        * cosine
    )


def extract_loss(output):
    if isinstance(output, dict):
        return output["loss"]

    if hasattr(output, "loss"):
        return output.loss

    if (
        isinstance(output, (tuple, list))
        and len(output) > 0
    ):
        return output[0]

    raise TypeError(
        "model output does not contain loss"
    )


def latest_checkpoint_tag():
    path = CHECKPOINT_DIR / "latest"

    if not path.is_file():
        return None

    tag = path.read_text(
        encoding="utf-8"
    ).strip()

    if not tag:
        return None

    if not (
        CHECKPOINT_DIR / tag
    ).is_dir():
        raise RuntimeError(
            f"latest checkpoint is missing: "
            f"{tag}"
        )

    return tag


def prune_checkpoints():
    pattern = re.compile(
        r"^step_(\d+)$"
    )

    checkpoints = []

    for path in CHECKPOINT_DIR.iterdir():
        if not path.is_dir():
            continue

        match = pattern.match(
            path.name
        )

        if match is None:
            continue

        checkpoints.append(
            (
                int(match.group(1)),
                path,
            )
        )

    checkpoints.sort(
        key=lambda item: item[0]
    )

    while (
        len(checkpoints)
        > KEEP_LAST_CHECKPOINTS
    ):
        _, path = checkpoints.pop(0)

        shutil.rmtree(
            path,
            ignore_errors=True,
        )


def save_training_state(
    engine,
    micro_batches_seen,
    tokens_seen,
    last_loss,
):
    step = int(
        engine.global_steps
    )

    tag = (
        f"step_{step:012d}"
    )

    temporary_root = (
        CHECKPOINT_DIR
        / (
            f".tmp-{tag}-"
            f"{os.getpid()}"
        )
    )

    shutil.rmtree(
        temporary_root,
        ignore_errors=True,
    )

    client_state = {
        "optimizer_step": step,
        "micro_batches_seen": int(
            micro_batches_seen
        ),
        "tokens_seen": int(
            tokens_seen
        ),
        "last_loss": (
            None
            if last_loss is None
            else float(last_loss)
        ),
        "config_path": str(
            CONFIG_PATH
        ),
        "data_path": str(
            DATA_PATH
        ),
        "sequence_length": SEQ_LEN,
        "micro_batch_size": (
            MICRO_BATCH_SIZE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "target_tokens": TARGET_TOKENS,
    }

    engine.save_checkpoint(
        str(temporary_root),
        tag=tag,
        client_state=client_state,
        save_latest=False,
    )

    source = (
        temporary_root / tag
    )

    destination = (
        CHECKPOINT_DIR / tag
    )

    if not source.is_dir():
        raise RuntimeError(
            f"DeepSpeed did not create "
            f"checkpoint directory: {source}"
        )

    if destination.exists():
        shutil.rmtree(
            destination
        )

    os.replace(
        source,
        destination,
    )

    shutil.rmtree(
        temporary_root,
        ignore_errors=True,
    )

    latest_tmp = (
        CHECKPOINT_DIR
        / "latest.tmp"
    )

    latest_tmp.write_text(
        tag + "\n",
        encoding="utf-8",
    )

    os.replace(
        latest_tmp,
        CHECKPOINT_DIR / "latest",
    )

    prune_checkpoints()

    print(
        "saved checkpoint: "
        f"{destination} "
        f"tokens={tokens_seen:,}",
        flush=True,
    )

    return destination


def load_training_state(engine):
    tag = latest_checkpoint_tag()

    if tag is None:
        return {
            "checkpoint": None,
            "micro_batches_seen": 0,
            "tokens_seen": 0,
            "last_loss": None,
        }

    load_path, client_state = (
        engine.load_checkpoint(
            str(CHECKPOINT_DIR),
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
    )

    if load_path is None:
        raise RuntimeError(
            f"failed to load checkpoint: "
            f"{tag}"
        )

    client_state = (
        client_state or {}
    )

    result = {
        "checkpoint": Path(load_path),
        "micro_batches_seen": int(
            client_state.get(
                "micro_batches_seen",
                0,
            )
        ),
        "tokens_seen": int(
            client_state.get(
                "tokens_seen",
                0,
            )
        ),
        "last_loss": (
            client_state.get(
                "last_loss"
            )
        ),
    }

    print(
        "resumed checkpoint: "
        f"{load_path} "
        f"optimizer_step="
        f"{engine.global_steps} "
        f"tokens="
        f"{result['tokens_seen']:,}",
        flush=True,
    )

    return result


def read_deepspeed_config():
    config = json.loads(
        DEEPSPEED_CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    config[
        "train_micro_batch_size_per_gpu"
    ] = MICRO_BATCH_SIZE

    config[
        "gradient_accumulation_steps"
    ] = GRADIENT_ACCUMULATION_STEPS

    config["train_batch_size"] = (
        MICRO_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    return config


def main():
    install_stop_handlers()

    for path in (
        CONFIG_PATH,
        DEEPSPEED_CONFIG_PATH,
        DATA_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    torch.cuda.set_device(
        local_rank
    )

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")

    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=False,
        )

    set_seed(SEED)

    torch.set_float32_matmul_precision(
        "high"
    )

    config = load_config(
        str(CONFIG_PATH)
    )

    linear_backend = os.environ.get(
        "ALBERTLM_LINEAR_BACKEND",
        "native",
    ).strip().lower()
    fp8_enabled = parse_bool(
        os.environ.get(
            "ALBERTLM_FP8_ENABLED",
            "0",
        )
    )
    fp8_recipe = os.environ.get(
        "ALBERTLM_FP8_RECIPE",
        "delayed",
    ).strip().lower()

    if fp8_enabled and not ENABLE_ACTIVATION_CHECKPOINTING:
        raise RuntimeError(
            "FP8 production training requires activation checkpointing"
        )
    if fp8_enabled and fp8_recipe != "delayed":
        raise RuntimeError(
            "FP8 production training requires DelayedScaling"
        )

    configure_fp8_recipe(fp8_recipe)

    model = AlbertLM(
        config,
        linear_backend=linear_backend,
        fp8_enabled=fp8_enabled,
    )

    projection_counts = projection_module_counts(
        model
    )
    expected_projection_count = (
        config.num_hidden_layers * 7
    )
    if model.uses_transformer_engine:
        if projection_counts != {
            "native": 0,
            "te": expected_projection_count,
            "other_linear": 0,
        }:
            raise RuntimeError(
                "unexpected Transformer Engine projection layout: "
                f"{projection_counts}"
            )
    elif projection_counts["native"] != expected_projection_count:
        raise RuntimeError(
            "unexpected native projection layout: "
            f"{projection_counts}"
        )

    print(
        "linear mode: "
        f"backend={linear_backend} "
        f"fp8_enabled={fp8_enabled} "
        f"fp8_recipe={fp8_recipe} "
        f"projections={projection_counts}",
        flush=True,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"model parameters: "
        f"{parameter_count:,}",
        flush=True,
    )

    if ENABLE_ACTIVATION_CHECKPOINTING:
        enable_activation_checkpointing(
            model
        )

    enable_regional_compilation(
        model
    )

    print("optimizer: bitsandbytes PagedAdamW8bit", flush=True)
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=wsd_multiplier,
    )

    engine, optimizer, _, scheduler = (
        deepspeed.initialize(
            model=model,
            model_parameters=(
                model.parameters()
            ),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=(
                read_deepspeed_config()
            ),
            dist_init_required=False,
        )
    )

    state = load_training_state(
        engine
    )

    micro_batches_seen = state[
        "micro_batches_seen"
    ]

    tokens_seen = state[
        "tokens_seen"
    ]

    last_loss = state[
        "last_loss"
    ]

    last_checkpoint = state[
        "checkpoint"
    ]

    start_block = (
        micro_batches_seen
        * MICRO_BATCH_SIZE
    )

    dataset = TokenDataset(
        DATA_PATH,
        SEQ_LEN,
        start_block=start_block,
    )

    available_micro_batches = (
        len(dataset)
        // MICRO_BATCH_SIZE
    )

    remaining_target_tokens = max(
        0,
        TARGET_TOKENS
        - tokens_seen,
    )

    target_micro_batches = (
        remaining_target_tokens
        // TOKENS_PER_MICRO_BATCH
    )

    run_micro_batches = min(
        available_micro_batches,
        target_micro_batches,
    )

    run_micro_batches -= (
        run_micro_batches
        % GRADIENT_ACCUMULATION_STEPS
    )

    if run_micro_batches <= 0:
        write_status(
            "completed",
            engine.global_steps,
            last_loss,
            checkpoint=last_checkpoint,
            message=(
                "No complete optimizer "
                "updates remain"
            ),
        )

        print(
            "training already completed",
            flush=True,
        )

        return 0

    loader = DataLoader(
        dataset,
        batch_size=(
            MICRO_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    engine.train()

    print(
        "training configuration:",
        flush=True,
    )

    print(
        f"  sequence length: {SEQ_LEN}",
        flush=True,
    )

    print(
        f"  micro batch: "
        f"{MICRO_BATCH_SIZE}",
        flush=True,
    )

    print(
        f"  grad accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}",
        flush=True,
    )

    print(
        f"  tokens/update: "
        f"{TOKENS_PER_UPDATE:,}",
        flush=True,
    )

    print(
        f"  warmup updates: "
        f"{WARMUP_UPDATES:,}",
        flush=True,
    )

    print(
        f"  decay starts: "
        f"{DECAY_START_UPDATE:,}",
        flush=True,
    )

    print(
        f"  total updates: "
        f"{TOTAL_UPDATES:,}",
        flush=True,
    )

    print(
        f"  available run tokens: "
        f"{run_micro_batches * TOKENS_PER_MICRO_BATCH:,}",
        flush=True,
    )

    write_status(
        "training",
        engine.global_steps,
        last_loss,
        checkpoint=last_checkpoint,
        message=(
            f"starting at tokens "
            f"{tokens_seen}"
        ),
    )

    next_checkpoint_tokens = (
        (
            tokens_seen
            // CHECKPOINT_EVERY_TOKENS
        )
        + 1
    ) * CHECKPOINT_EVERY_TOKENS

    window_tokens = 0

    window_started = (
        time.perf_counter()
    )

    processed_micro_batches = 0

    try:
        for batch in loader:
            if (
                processed_micro_batches
                >= run_micro_batches
            ):
                break

            batch = batch.to(
                engine.device,
                non_blocking=True,
            )

            output = engine(
                batch,
                labels=batch,
            )

            loss = extract_loss(
                output
            )

            # AlbertLM: token-weighted loss over the whole logging window.
            batch_tokens = int(
                batch.numel()
            )
            micro_loss = float(
                loss.detach().float().item()
            )

            engine._albert_window_loss_sum = (
                getattr(
                    engine,
                    "_albert_window_loss_sum",
                    0.0,
                )
                + micro_loss * batch_tokens
            )
            engine._albert_window_loss_tokens = (
                getattr(
                    engine,
                    "_albert_window_loss_tokens",
                    0,
                )
                + batch_tokens
            )

            # Keep a valid value for status/checkpoint paths between log events.
            last_loss = micro_loss

            engine.backward(
                loss
            )

            boundary = (
                engine
                .is_gradient_accumulation_boundary()
            )

            grad_norm = 0.0

            next_optimizer_step = (
                int(engine.global_steps) + 1
            )

            should_measure_grad_norm = (
                boundary
                and next_optimizer_step
                % METRICS_EVERY_UPDATES
                == 0
            )

            if should_measure_grad_norm:
                grad_norm = compute_actual_grad_norm(
                    engine.module.parameters()
                )

            engine.step()

            batch_tokens = int(
                batch.numel()
            )

            tokens_seen += (
                batch_tokens
            )

            micro_batches_seen += 1
            processed_micro_batches += 1
            window_tokens += batch_tokens

            if not boundary:
                continue

            optimizer_step = int(
                engine.global_steps
            )

            # AlbertLM runtime evaluation hook
            maybe_run_runtime_evaluation(
                engine=engine,
                optimizer_step=optimizer_step,
                tokens_seen=tokens_seen,
            )

            if (
                tokens_seen
                >= next_checkpoint_tokens
            ):
                last_checkpoint = (
                    save_training_state(
                        engine,
                        micro_batches_seen,
                        tokens_seen,
                        last_loss,
                    )
                )

                next_checkpoint_tokens = (
                    (
                        tokens_seen
                        // CHECKPOINT_EVERY_TOKENS
                    )
                    + 1
                ) * CHECKPOINT_EVERY_TOKENS

            if (
                optimizer_step
                % METRICS_EVERY_UPDATES
                == 0
            ):
                torch.cuda.synchronize()

                elapsed = max(
                    time.perf_counter()
                    - window_started,
                    1e-9,
                )

                tokens_per_second = (
                    window_tokens
                    / elapsed
                )

                learning_rate = float(
                    optimizer
                    .param_groups[0]["lr"]
                )

                loss_tokens = int(
                    getattr(
                        engine,
                        "_albert_window_loss_tokens",
                        0,
                    )
                )
                if loss_tokens <= 0:
                    raise RuntimeError(
                        "loss logging window contains no tokens"
                    )

                last_loss = float(
                    getattr(
                        engine,
                        "_albert_window_loss_sum",
                        0.0,
                    )
                    / loss_tokens
                )

                append_metrics(
                    optimizer_step,
                    tokens_seen,
                    last_loss,
                    learning_rate,
                    grad_norm,
                    tokens_per_second,
                )

                write_status(
                    "training",
                    optimizer_step,
                    last_loss,
                    checkpoint=last_checkpoint,
                )

                print(
                    f"step {optimizer_step} "
                    f"tokens {tokens_seen:,} "
                    f"loss {last_loss:.4f} "
                    f"lr {learning_rate:.8g} "
                    f"grad_norm {grad_norm:.4f} "
                    f"tokens/s "
                    f"{tokens_per_second:.1f}",
                    flush=True,
                )

                window_tokens = 0
                engine._albert_window_loss_sum = 0.0
                engine._albert_window_loss_tokens = 0
                window_started = (
                    time.perf_counter()
                )

            if STOP_REQUESTED:
                last_checkpoint = (
                    save_training_state(
                        engine,
                        micro_batches_seen,
                        tokens_seen,
                        last_loss,
                    )
                )

                write_status(
                    "stopped",
                    optimizer_step,
                    last_loss,
                    checkpoint=last_checkpoint,
                    message=(
                        "Stopped intentionally"
                    ),
                )

                print(
                    "training stopped cleanly",
                    flush=True,
                )

                return 130

        last_checkpoint = (
            save_training_state(
                engine,
                micro_batches_seen,
                tokens_seen,
                last_loss,
            )
        )

        reason = (
            "target tokens reached"
            if tokens_seen
            >= TARGET_TOKENS
            else "dataset exhausted"
        )

        write_status(
            "completed",
            engine.global_steps,
            last_loss,
            checkpoint=last_checkpoint,
            message=reason,
        )

        print(
            f"training completed: "
            f"{reason}; "
            f"tokens={tokens_seen:,}",
            flush=True,
        )

        return 0

    except BaseException as error:
        traceback.print_exc()

        write_status(
            "failed",
            engine.global_steps,
            last_loss,
            checkpoint=last_checkpoint,
            message=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

        raise


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        try:
            write_status(
                "failed",
                0,
                None,
                message=(
                    f"{type(error).__name__}: {error}"
                ),
            )
        except Exception:
            pass
        raise

    raise SystemExit(exit_code)
