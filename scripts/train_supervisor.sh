#!/usr/bin/env bash

set -uo pipefail

ROOT="/data/AlbertLM"
LOG_DIR="$ROOT/logs"
CHECKPOINT_DIR="$ROOT/checkpoints"

PYTHON="$ROOT/.venv/bin/python"

export CUDA_HOME="/usr/local/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions"
export GRADIENT_ACCUMULATION_STEPS="64"
export MICRO_BATCH_SIZE="1"
export ALBERTLM_LINEAR_BACKEND="te"
export ALBERTLM_FP8_ENABLED="1"
export ALBERTLM_FP8_RECIPE="delayed"
export ALBERTLM_TE_EXPECTED_ROOT="/data/AlbertLM/.venv/te-deployments/transformer-engine-main-dc4a11dd-7122ebc8"
TARGET_BIN_PATH="${TARGET_BIN_PATH:-/data/AlbertLM/data/processed/train.bin}"

if [[ ! -e "$TARGET_BIN_PATH" ]]; then
    echo "ERROR: training bin not found: $TARGET_BIN_PATH" >&2
    exit 1
fi

TARGET_BIN_BYTES="$(stat -Lc '%s' "$TARGET_BIN_PATH")"

if (( TARGET_BIN_BYTES % 4 != 0 )); then
    echo "ERROR: training bin is not valid uint32: bytes=$TARGET_BIN_BYTES" >&2
    exit 1
fi

export TARGET_TOKENS="$((TARGET_BIN_BYTES / 4))"
echo "TARGET_TOKENS=$TARGET_TOKENS"
echo "TARGET_BIN_PATH=$(readlink -f "$TARGET_BIN_PATH")"
export ACTIVATION_CHECKPOINTING="1"
TRAIN_SCRIPT="$ROOT/train/pretrain.py"

SYSTEMCTL="/usr/bin/systemctl"
NVIDIA_SMI="$(command -v nvidia-smi || true)"

TRAIN_LOG="$LOG_DIR/train.log"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
MONITOR_LOG="$LOG_DIR/system-monitor.log"

mkdir -p \
    "$LOG_DIR" \
    "$CHECKPOINT_DIR"

touch \
    "$TRAIN_LOG" \
    "$SUPERVISOR_LOG" \
    "$MONITOR_LOG"

cd "$ROOT" || exit 1

stop_requested=0
child_pid=""
monitor_pid=""
display_was_active=0
restart_delay=15
normal_completion=0


log() {
    printf '%s %s\n' \
        "$(date --iso-8601=seconds)" \
        "$*" \
        | tee -a "$SUPERVISOR_LOG"
}


request_stop() {
    stop_requested=1

    log "stop requested"

    if [ -n "$child_pid" ] &&
       kill -0 "$child_pid" 2>/dev/null
    then
        log "forwarding SIGTERM to training pid=$child_pid"

        kill -TERM "$child_pid" \
            2>/dev/null || true
    fi
}


cleanup() {
    set +e

    log "supervisor cleanup started"

    if [ -n "$child_pid" ] &&
       kill -0 "$child_pid" 2>/dev/null
    then
        kill -TERM "$child_pid" \
            2>/dev/null || true

        wait "$child_pid" \
            2>/dev/null || true
    fi

    if [ -n "$monitor_pid" ] &&
       kill -0 "$monitor_pid" 2>/dev/null
    then
        kill "$monitor_pid" \
            2>/dev/null || true

        wait "$monitor_pid" \
            2>/dev/null || true
    fi

    if [ "$display_was_active" -eq 1 ]
    then
        log "restoring display manager"

        sudo -n "$SYSTEMCTL" \
            start \
            display-manager.service \
            || log "warning: failed to restore display manager"
    fi

    log "supervisor cleanup finished"
}


trap request_stop INT TERM HUP
trap cleanup EXIT


if [ ! -x "$PYTHON" ]
then
    log "fatal: python not found: $PYTHON"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]
then
    log "fatal: training script not found: $TRAIN_SCRIPT"
    exit 1
fi

log \
    "training mode: backend=$ALBERTLM_LINEAR_BACKEND fp8=$ALBERTLM_FP8_ENABLED recipe=$ALBERTLM_FP8_RECIPE"

if ! fp8_preflight_output="$("$PYTHON" - <<'PY' 2>&1
import os
from pathlib import Path

backend = os.environ["ALBERTLM_LINEAR_BACKEND"].strip().lower()
fp8_enabled = os.environ["ALBERTLM_FP8_ENABLED"].strip().lower()
recipe = os.environ["ALBERTLM_FP8_RECIPE"].strip().lower()

if backend not in {"native", "te"}:
    raise SystemExit(f"unsupported linear backend: {backend!r}")
if fp8_enabled not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
    raise SystemExit(f"invalid FP8 flag: {fp8_enabled!r}")

fp8_requested = fp8_enabled in {"1", "true", "yes", "on"}
if fp8_requested and backend != "te":
    raise SystemExit("FP8 requires ALBERTLM_LINEAR_BACKEND=te")
if fp8_requested and recipe != "delayed":
    raise SystemExit("production FP8 requires ALBERTLM_FP8_RECIPE=delayed")

if backend == "te":
    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch

    expected_root = Path(os.environ["ALBERTLM_TE_EXPECTED_ROOT"]).resolve()
    if transformer_engine.__version__ != "2.18.0.dev0+dc4a11dd":
        raise SystemExit(
            f"unexpected Transformer Engine version: {transformer_engine.__version__}"
        )

    module_paths = {
        "transformer_engine": Path(transformer_engine.__file__).resolve(),
        "transformer_engine_torch": Path(transformer_engine_torch.__file__).resolve(),
    }
    for name, path in module_paths.items():
        if expected_root not in path.parents:
            raise SystemExit(
                f"{name} loaded outside fixed deployment: {path}"
            )

    if torch.version.cuda != "13.0":
        raise SystemExit(f"unexpected PyTorch CUDA version: {torch.version.cuda}")
    if torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit(
            f"unexpected GPU compute capability: {torch.cuda.get_device_capability()}"
        )
    if torch.backends.cudnn.version() != 92000:
        raise SystemExit(
            f"unexpected cuDNN runtime version: {torch.backends.cudnn.version()}"
        )

    if fp8_requested:
        available, reason = te.is_fp8_available(return_reason=True)
        if not available:
            raise SystemExit(f"Transformer Engine FP8 unavailable: {reason}")

    print(f"Transformer Engine {transformer_engine.__version__}: {module_paths['transformer_engine']}")
    print(f"Transformer Engine torch extension: {module_paths['transformer_engine_torch']}")
    print(
        "CUDA/cuDNN/GPU: "
        f"{torch.version.cuda}/{torch.backends.cudnn.version()}/"
        f"{torch.cuda.get_device_name()} sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )
PY
)"
then
    log "fatal: FP8 preflight failed: $fp8_preflight_output"
    exit 1
fi

while IFS= read -r preflight_line
do
    log "FP8 preflight: $preflight_line"
done <<<"$fp8_preflight_output"


if "$SYSTEMCTL" \
    is-active \
    --quiet \
    display-manager.service
then
    display_was_active=1

    log "stopping display manager"

    if ! sudo -n "$SYSTEMCTL" \
        stop \
        display-manager.service
    then
        log "fatal: failed to stop display manager"
        exit 1
    fi

    sleep 3
else
    log "display manager already inactive"
fi


if [ -x "$ROOT/scripts/system_monitor.sh" ]
then
    "$ROOT/scripts/system_monitor.sh" \
        >>"$MONITOR_LOG" \
        2>&1 &

    monitor_pid=$!

    log "system monitor started pid=$monitor_pid"
fi


while [ "$stop_requested" -eq 0 ]
do
    run_started="$(date +%s)"

    log "starting training process"

    PYTHONPATH="$ROOT" \
    "$PYTHON" \
        -u \
        "$TRAIN_SCRIPT" \
        >>"$TRAIN_LOG" \
        2>&1 &

    child_pid=$!

    log "training pid=$child_pid"

    wait "$child_pid"
    exit_code=$?

    # wait 可能被 supervisor 收到的信号中断；
    # 人工停止时再次等待 Python 完成 checkpoint 保存。
    if [ "$stop_requested" -eq 1 ] &&
       kill -0 "$child_pid" 2>/dev/null
    then
        wait "$child_pid" \
            2>/dev/null

        exit_code=$?
    fi

    child_pid=""

    run_ended="$(date +%s)"
    runtime_seconds=$((run_ended - run_started))

    if [ "$stop_requested" -eq 1 ]
    then
        log "training stopped intentionally; no restart"
        break
    fi

    if [ "$exit_code" -eq 0 ]
    then
        normal_completion=1

        log "training completed normally; no restart"
        break
    fi

    log \
        "training exited abnormally: code=$exit_code runtime=${runtime_seconds}s"

    # 长时间运行后才崩溃，说明不是启动配置错误，
    # 重启等待恢复为 15 秒。
    if [ "$runtime_seconds" -ge 600 ]
    then
        restart_delay=15
    else
        restart_delay=$((restart_delay * 2))

        if [ "$restart_delay" -gt 300 ]
        then
            restart_delay=300
        fi
    fi

    # 等待驱动重新变得可用。
    if [ -n "$NVIDIA_SMI" ]
    then
        gpu_ready=0
        attempt=1

        while [ "$attempt" -le 30 ]
        do
            if "$NVIDIA_SMI" \
                -L \
                >/dev/null \
                2>&1
            then
                gpu_ready=1
                break
            fi

            log "GPU not ready: attempt=$attempt"

            sleep 10

            attempt=$((attempt + 1))
        done

        if [ "$gpu_ready" -eq 1 ]
        then
            log "GPU is available again"
        else
            log "warning: GPU still unavailable after checks"
        fi
    fi

    log \
        "restarting from latest checkpoint in ${restart_delay}s"

    sleep "$restart_delay"
done


if [ "$normal_completion" -eq 1 ]
then
    exit 0
fi

if [ "$stop_requested" -eq 1 ]
then
    exit 130
fi

exit 1
