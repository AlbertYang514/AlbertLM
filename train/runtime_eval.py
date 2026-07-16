from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SAMPLE_DIR = LOG_DIR / "samples"

EVAL_METRICS_PATH = LOG_DIR / "eval_metrics.jsonl"
EVAL_LATEST_PATH = LOG_DIR / "eval_latest.json"
EVAL_ERRORS_PATH = LOG_DIR / "eval_errors.log"
SAMPLES_JSONL_PATH = LOG_DIR / "samples.jsonl"
CHECKPOINT_INDEX_PATH = LOG_DIR / "checkpoints.json"

_FIRST_HOOK = True
_TOKENIZER = None
_TOKENIZER_PATH: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )


def log_error(event: str, error: BaseException) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EVAL_ERRORS_PATH.open("a", encoding="utf-8") as file:
        file.write(
            f"{utc_now()} event={event} "
            f"type={type(error).__name__} "
            f"error={error}\n"
        )


def distributed_rank() -> int:
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def distributed_world_size() -> int:
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def discover_valid_path() -> Path:
    explicit = os.environ.get("VALID_DATA_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"VALID_DATA_PATH does not exist: {path}"
            )
        return path

    processed = ROOT / "data" / "processed"

    preferred = [
        processed / "valid.bin",
        processed / "validation.bin",
        processed / "val.bin",
        processed / "valid-10b.bin",
        processed / "validation-10b.bin",
    ]

    for path in preferred:
        if path.is_file():
            return path

    candidates = sorted(
        path
        for path in processed.glob("*.bin")
        if (
            "valid" in path.name.lower()
            or path.name.lower().startswith("val")
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"no validation .bin file found under {processed}"
        )

    return candidates[0]


def discover_tokenizer_path() -> Path:
    global _TOKENIZER_PATH

    if _TOKENIZER_PATH is not None:
        return _TOKENIZER_PATH

    explicit = os.environ.get("TOKENIZER_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"TOKENIZER_PATH does not exist: {path}"
            )
        _TOKENIZER_PATH = path
        return path

    excluded = {
        ".venv",
        ".git",
        "backups",
        "checkpoints",
        "__pycache__",
    }

    candidates: list[Path] = []

    for path in ROOT.rglob("tokenizer.json"):
        if any(part in excluded for part in path.parts):
            continue
        if path.is_file():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "tokenizer.json was not found; set TOKENIZER_PATH"
        )

    candidates.sort(
        key=lambda path: (
            0 if "tokenizer" in str(path.parent).lower() else 1,
            len(path.parts),
            str(path),
        )
    )

    _TOKENIZER_PATH = candidates[0]
    return _TOKENIZER_PATH


def load_tokenizer():
    global _TOKENIZER

    if _TOKENIZER is not None:
        return _TOKENIZER

    from tokenizers import Tokenizer

    path = discover_tokenizer_path()
    _TOKENIZER = Tokenizer.from_file(str(path))
    return _TOKENIZER


def extract_loss(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "loss"):
        loss = output.loss
        if loss is not None:
            return loss

    if isinstance(output, dict):
        loss = output.get("loss")
        if loss is not None:
            return loss

    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first

    raise TypeError(
        f"cannot extract loss from output type {type(output)!r}"
    )


def extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        logits = output.logits
        if logits is not None:
            return logits

    if isinstance(output, dict):
        logits = output.get("logits")
        if logits is not None:
            return logits

    if isinstance(output, (tuple, list)):
        for value in output:
            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 3
            ):
                return value

    if isinstance(output, torch.Tensor) and output.ndim == 3:
        return output

    raise TypeError(
        f"cannot extract logits from output type {type(output)!r}"
    )


def unwrap_model(model):
    """Return the public model behind common wrappers, or the model itself."""
    return getattr(model, "module", model)


def model_device(model) -> torch.device:
    return next(unwrap_model(model).parameters()).device


@torch.inference_mode()
def evaluate_validation_loss(
    engine,
    *,
    max_batches: int,
    sequence_length: int,
) -> dict[str, Any]:
    valid_path = discover_valid_path()

    tokens = np.memmap(
        valid_path,
        mode="r",
        dtype=np.uint32,
    )

    if tokens.size < sequence_length:
        raise RuntimeError(
            f"validation file has only {tokens.size} tokens"
        )

    possible_starts = tokens.size - sequence_length
    batch_count = min(
        max_batches,
        max(1, tokens.size // sequence_length),
    )

    starts = np.linspace(
        0,
        possible_starts,
        num=batch_count,
        dtype=np.int64,
    )

    device = model_device(engine)
    was_training = bool(engine.training)

    total_loss = 0.0
    total_tokens = 0

    engine.eval()

    started = time.perf_counter()

    try:
        for start in starts:
            array = np.asarray(
                tokens[
                    int(start):
                    int(start) + sequence_length
                ],
                dtype=np.int64,
            ).copy()

            batch = torch.from_numpy(array)
            batch = batch.unsqueeze(0).to(
                device=device,
                non_blocking=True,
            )

            output = engine(
                batch,
                labels=batch,
            )

            loss = extract_loss(output)
            batch_tokens = int(batch.numel())

            total_loss += (
                float(loss.detach().float().item())
                * batch_tokens
            )
            total_tokens += batch_tokens

            del batch, output, loss

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    finally:
        if was_training:
            engine.train()
        else:
            engine.eval()

        del tokens
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started

    if total_tokens <= 0:
        raise RuntimeError("validation evaluated zero tokens")

    average_loss = total_loss / total_tokens

    return {
        "valid_loss": float(average_loss),
        "eval_tokens": int(total_tokens),
        "eval_batches": int(batch_count),
        "elapsed_seconds": float(elapsed),
        "valid_data_path": str(valid_path),
    }


def top_p_sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-5)
    top_p = min(max(float(top_p), 0.01), 1.0)

    logits = logits.float() / temperature

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove,
        float("-inf"),
    )

    probabilities = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    sampled_sorted_index = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return torch.gather(
        sorted_indices,
        dim=-1,
        index=sampled_sorted_index,
    )


def sample_prompts() -> list[str]:
    raw = os.environ.get("SAMPLE_PROMPTS_JSON")

    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise TypeError(
                "SAMPLE_PROMPTS_JSON must contain a JSON list"
            )
        return [str(value) for value in parsed]

    return [
        "人工智能的发展历史可以追溯到",
        "TCP 三次握手的主要过程是",
        "夜晚的上海，",
        "The history of artificial intelligence began",
        "def fibonacci(n):",
    ]


@torch.inference_mode()
def generate_samples(
    engine,
    *,
    optimizer_step: int,
    tokens_seen: int,
) -> dict[str, Any]:
    tokenizer = load_tokenizer()
    tokenizer_path = discover_tokenizer_path()

    max_new_tokens = env_int(
        "SAMPLE_MAX_NEW_TOKENS",
        96,
    )

    max_context = env_int(
        "SAMPLE_MAX_CONTEXT",
        512,
    )

    temperature = env_float(
        "SAMPLE_TEMPERATURE",
        0.8,
    )

    top_p = env_float(
        "SAMPLE_TOP_P",
        0.9,
    )

    prompts = sample_prompts()
    device = model_device(engine)
    was_training = bool(engine.training)

    if device.type != "cuda":
        raise RuntimeError("sample generation requires CUDA BF16")

    import transformer_engine.pytorch as te

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = None

    if torch.cuda.is_available():
        cuda_rng_states = torch.cuda.get_rng_state_all()

    torch.manual_seed(20260714 + optimizer_step)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            20260714 + optimizer_step
        )

    torch.cuda.reset_peak_memory_stats(device)
    engine.eval()

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    all_logits_finite = True

    print(
        "sample generation precision: "
        "BF16 autocast; Transformer Engine FP8 disabled",
        flush=True,
    )

    try:
        for prompt in prompts:
            encoded = tokenizer.encode(prompt)
            prompt_ids = list(encoded.ids)

            if not prompt_ids:
                raise RuntimeError(
                    f"tokenizer produced no tokens for {prompt!r}"
                )

            maximum_prompt_tokens = max(
                1,
                max_context - max_new_tokens,
            )
            prompt_ids = prompt_ids[-maximum_prompt_tokens:]

            input_ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )

            generated_ids = list(prompt_ids)

            for _ in range(max_new_tokens):
                model_input = input_ids[:, -max_context:]

                # The public precision override leaves the default FP8 training
                # path unchanged. These contexts own prefill and every decode
                # step, including non-aligned autoregressive sequence lengths.
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    with te.autocast(enabled=False):
                        output = engine(
                            model_input,
                            fp8_enabled=False,
                        )
                logits = extract_logits(output)

                if not bool(torch.isfinite(logits).all().item()):
                    all_logits_finite = False
                    raise FloatingPointError(
                        "sample logits contain non-finite values"
                    )
                next_logits = logits[:, -1, :]

                next_token = top_p_sample(
                    next_logits,
                    temperature=temperature,
                    top_p=top_p,
                )

                input_ids = torch.cat(
                    [input_ids, next_token],
                    dim=-1,
                )

                generated_ids.append(
                    int(next_token.item())
                )

                del output, logits, next_logits, next_token

            full_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            completion_text = tokenizer.decode(
                generated_ids[len(prompt_ids):],
                skip_special_tokens=True,
            )

            results.append(
                {
                    "prompt": prompt,
                    "completion": completion_text,
                    "full_text": full_text,
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": (
                        len(generated_ids) - len(prompt_ids)
                    ),
                }
            )

            del input_ids

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(cpu_rng_state)

        if (
            cuda_rng_states is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(cuda_rng_states)

        if was_training:
            engine.train()
        else:
            engine.eval()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started

    payload = {
        "schema_version": 1,
        "optimizer_step": int(optimizer_step),
        "tokens_seen": int(tokens_seen),
        "timestamp": utc_now(),
        "tokenizer_path": str(tokenizer_path),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "elapsed_seconds": float(elapsed),
        "precision": "bfloat16",
        "transformer_engine_fp8_enabled": False,
        "all_logits_finite": bool(all_logits_finite),
        "peak_cuda_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "samples": results,
    }

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    sample_path = (
        SAMPLE_DIR
        / f"samples_step_{optimizer_step:012d}.json"
    )

    atomic_write_json(sample_path, payload)
    append_jsonl(SAMPLES_JSONL_PATH, payload)

    payload["sample_path"] = str(sample_path)
    return payload


def read_latest_eval() -> dict[str, Any]:
    if not EVAL_LATEST_PATH.is_file():
        return {
            "schema_version": 1,
        }

    try:
        value = json.loads(
            EVAL_LATEST_PATH.read_text(
                encoding="utf-8"
            )
        )
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    return {
        "schema_version": 1,
    }


def update_latest_eval(
    *,
    validation: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
) -> None:
    latest = read_latest_eval()
    latest["schema_version"] = 1
    latest["updated_at"] = utc_now()

    if validation is not None:
        latest["latest_valid_loss"] = validation[
            "valid_loss"
        ]
        latest["latest_valid_loss_step"] = validation[
            "optimizer_step"
        ]
        latest["latest_valid_loss_tokens"] = validation[
            "tokens_seen"
        ]
        latest["latest_valid_eval_tokens"] = validation[
            "eval_tokens"
        ]

        if validation.get("valid_ppl") is not None:
            latest["latest_valid_ppl"] = validation[
                "valid_ppl"
            ]
            latest["latest_valid_ppl_step"] = validation[
                "optimizer_step"
            ]
            latest["latest_valid_ppl_tokens"] = validation[
                "tokens_seen"
            ]

    if sample is not None:
        latest["latest_sample_step"] = sample[
            "optimizer_step"
        ]
        latest["latest_sample_tokens"] = sample[
            "tokens_seen"
        ]
        latest["latest_sample_path"] = sample[
            "sample_path"
        ]

    atomic_write_json(EVAL_LATEST_PATH, latest)


def parse_checkpoint_tokens_from_log() -> dict[int, int]:
    result: dict[int, int] = {}
    train_log = LOG_DIR / "train.log"

    if not train_log.is_file():
        return result

    pattern = re.compile(
        r"saved checkpoint:\s+"
        r"(\S*step_(\d+)\S*)\s+"
        r"tokens=([\d,]+)"
    )

    try:
        lines = train_log.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except Exception:
        return result

    for line in lines[-10000:]:
        match = pattern.search(line)
        if not match:
            continue

        step = int(match.group(2))
        tokens = int(
            match.group(3).replace(",", "")
        )
        result[step] = tokens

    return result


def refresh_checkpoint_index() -> dict[str, Any]:
    checkpoint_root = ROOT / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    logged_tokens = parse_checkpoint_tokens_from_log()
    tokens_per_update = env_int(
        "TOKENS_PER_UPDATE",
        131072,
    )

    checkpoints: list[dict[str, Any]] = []

    for directory in checkpoint_root.glob("step_*"):
        if not directory.is_dir():
            continue

        match = re.fullmatch(
            r"step_(\d+)",
            directory.name,
        )
        if not match:
            continue

        step = int(match.group(1))

        model_state = (
            directory
            / "mp_rank_00_model_states.pt"
        )

        if not model_state.is_file():
            continue

        stat = model_state.stat()
        tokens = logged_tokens.get(
            step,
            step * tokens_per_update,
        )

        checkpoints.append(
            {
                "step": step,
                "tokens": int(tokens),
                "path": str(directory),
                "model_state_path": str(model_state),
                "model_state_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )

    checkpoints.sort(
        key=lambda value: value["step"],
        reverse=True,
    )

    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "latest": (
            checkpoints[0]
            if checkpoints
            else None
        ),
        "checkpoints": checkpoints,
    }

    atomic_write_json(
        CHECKPOINT_INDEX_PATH,
        payload,
    )

    return payload


def maybe_run_runtime_evaluation(
    *,
    engine,
    optimizer_step: int,
    tokens_seen: int,
) -> None:
    global _FIRST_HOOK

    if os.environ.get(
        "RUNTIME_EVAL_ENABLED",
        "1",
    ) != "1":
        return

    rank = distributed_rank()
    world_size = distributed_world_size()

    if rank != 0:
        return

    checkpoint_interval = env_int(
        "CHECKPOINT_INDEX_INTERVAL",
        10,
    )

    if (
        _FIRST_HOOK
        or (
            checkpoint_interval > 0
            and optimizer_step % checkpoint_interval == 0
        )
    ):
        try:
            refresh_checkpoint_index()
        except Exception as error:
            log_error("checkpoint_index", error)

    if world_size != 1:
        if _FIRST_HOOK:
            log_error(
                "runtime_eval",
                RuntimeError(
                    "runtime evaluation currently supports "
                    "WORLD_SIZE=1 only"
                ),
            )
        _FIRST_HOOK = False
        return

    valid_interval = env_int(
        "VALID_INTERVAL",
        500,
    )

    valid_batches = env_int(
        "VALID_BATCHES",
        128,
    )

    ppl_interval = env_int(
        "PPL_INTERVAL",
        2000,
    )

    ppl_batches = env_int(
        "PPL_BATCHES",
        512,
    )

    sample_interval = env_int(
        "SAMPLE_INTERVAL",
        5000,
    )

    sequence_length = env_int(
        "SEQUENCE_LENGTH",
        2048,
    )

    evaluate_now = (
        valid_interval > 0
        and optimizer_step % valid_interval == 0
    )

    ppl_now = (
        ppl_interval > 0
        and optimizer_step % ppl_interval == 0
    )

    if (
        _FIRST_HOOK
        and os.environ.get(
            "EVAL_ON_FIRST_STEP",
            "1",
        ) == "1"
    ):
        evaluate_now = True

    if ppl_now:
        evaluate_now = True

    if evaluate_now:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            evaluation = evaluate_validation_loss(
                engine,
                max_batches=(
                    ppl_batches
                    if ppl_now
                    else valid_batches
                ),
                sequence_length=sequence_length,
            )

            validation_payload = {
                "schema_version": 1,
                "event": (
                    "validation_with_ppl"
                    if ppl_now
                    else "validation"
                ),
                "optimizer_step": int(
                    optimizer_step
                ),
                "tokens_seen": int(tokens_seen),
                "valid_loss": float(
                    evaluation["valid_loss"]
                ),
                "valid_ppl": (
                    float(
                        math.exp(
                            min(
                                evaluation[
                                    "valid_loss"
                                ],
                                20.0,
                            )
                        )
                    )
                    if ppl_now
                    else None
                ),
                "eval_tokens": int(
                    evaluation["eval_tokens"]
                ),
                "eval_batches": int(
                    evaluation["eval_batches"]
                ),
                "elapsed_seconds": float(
                    evaluation["elapsed_seconds"]
                ),
                "valid_data_path": evaluation[
                    "valid_data_path"
                ],
                "timestamp": utc_now(),
            }

            append_jsonl(
                EVAL_METRICS_PATH,
                validation_payload,
            )

            update_latest_eval(
                validation=validation_payload,
            )

            message = (
                f"validation step {optimizer_step} "
                f"tokens {tokens_seen:,} "
                f"loss "
                f"{validation_payload['valid_loss']:.4f}"
            )

            if validation_payload["valid_ppl"] is not None:
                message += (
                    f" ppl "
                    f"{validation_payload['valid_ppl']:.2f}"
                )

            message += (
                f" eval_tokens "
                f"{validation_payload['eval_tokens']:,} "
                f"elapsed "
                f"{validation_payload['elapsed_seconds']:.1f}s"
            )

            print(message, flush=True)

        except torch.cuda.OutOfMemoryError as error:
            log_error("validation_cuda_oom", error)
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "validation skipped: CUDA OOM",
                flush=True,
            )

        except Exception as error:
            log_error("validation", error)
            print(
                f"validation skipped: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    sample_now = (
        sample_interval > 0
        and optimizer_step % sample_interval == 0
    )

    if sample_now:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sample_payload = generate_samples(
                engine,
                optimizer_step=optimizer_step,
                tokens_seen=tokens_seen,
            )

            update_latest_eval(
                sample=sample_payload,
            )

            print(
                f"generated samples step "
                f"{optimizer_step}: "
                f"{sample_payload['sample_path']}",
                flush=True,
            )

        except torch.cuda.OutOfMemoryError as error:
            log_error("sample_cuda_oom", error)
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "sample generation skipped: CUDA OOM",
                flush=True,
            )

        except Exception as error:
            log_error("sample_generation", error)
            print(
                f"sample generation skipped: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    _FIRST_HOOK = False


def main() -> int:
    print(f"root: {ROOT}")

    valid_path = discover_valid_path()
    print(f"validation: {valid_path}")

    try:
        tokenizer_path = discover_tokenizer_path()
        print(f"tokenizer: {tokenizer_path}")
    except Exception as error:
        print(f"tokenizer: unavailable ({error})")

    payload = refresh_checkpoint_index()
    print(
        "checkpoint count:",
        len(payload["checkpoints"]),
    )
    print(
        "checkpoint index:",
        CHECKPOINT_INDEX_PATH,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
