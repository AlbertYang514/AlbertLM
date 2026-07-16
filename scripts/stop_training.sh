#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

SESSION="${ALBERTLM_TMUX_SESSION:-albertlm}"
TRAIN_PATTERN="$ROOT/train/pretrain.py"
STOP_TIMEOUT="${ALBERTLM_STOP_TIMEOUT:-180}"

log() {
    printf '%s %s\n' "$(date -Is)" "$*"
}

get_training_pids() {
    pgrep -f -- "$TRAIN_PATTERN" 2>/dev/null || true
}

tmux_running() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

mapfile -t INITIAL_PIDS < <(get_training_pids)

if ! tmux_running && ((${#INITIAL_PIDS[@]} == 0)); then
    log "AlbertLM training is not running"
    exit 0
fi

log "requesting graceful training stop"

#
# 先通知 Supervisor。它应停止重启循环，并负责退出时的系统清理。
#
if tmux_running; then
    PANE_PID="$(
        tmux list-panes \
            -t "$SESSION" \
            -F '#{pane_pid}' |
        head -n 1
    )"

    if [[ -n "${PANE_PID:-}" ]]; then
        log "sending SIGINT to supervisor pid=$PANE_PID"
        kill -INT "$PANE_PID" 2>/dev/null || true
    fi

    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
fi

#
# 再通知训练进程。pretrain.py 的信号处理器会设置停止标志，
# 在安全位置保存完整 DeepSpeed checkpoint 后退出。
#
sleep 1

mapfile -t TRAIN_PIDS < <(get_training_pids)

for pid in "${TRAIN_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    log "sending SIGINT to training pid=$pid"
    kill -INT "$pid" 2>/dev/null || true
done

#
# 不强杀。最多等待指定时间，让当前 micro-batch / optimizer step
# 和 checkpoint 保存正常完成。
#
START_TIME="$(date +%s)"

while true; do
    mapfile -t TRAIN_PIDS < <(get_training_pids)

    if ((${#TRAIN_PIDS[@]} == 0)); then
        break
    fi

    NOW="$(date +%s)"
    ELAPSED=$((NOW - START_TIME))

    if ((ELAPSED >= STOP_TIMEOUT)); then
        log "ERROR: training is still running after ${STOP_TIMEOUT}s"
        log "refusing to force-kill it because a checkpoint may be in progress"

        printf 'remaining training pid(s):'
        printf ' %s' "${TRAIN_PIDS[@]}"
        printf '\n'

        if [[ -f "$ROOT/logs/train.log" ]]; then
            echo "===== TRAIN LOG ====="
            tail -n 100 "$ROOT/logs/train.log"
        fi

        exit 1
    fi

    if ((ELAPSED % 10 == 0)); then
        log "waiting for checkpoint and exit: ${ELAPSED}s"
    fi

    sleep 1
done

log "training process exited cleanly"

#
# 给 Supervisor 一点时间执行退出清理，包括恢复 display-manager。
#
if tmux_running; then
    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true

    for _ in $(seq 1 20); do
        tmux_running || break
        sleep 1
    done
fi

#
# 训练进程已经退出，此时再清理残留 tmux，不会损坏 checkpoint。
#
if tmux_running; then
    log "removing residual tmux session: $SESSION"
    tmux kill-session -t "$SESSION"
fi

LATEST="$(
    find "$ROOT/checkpoints" \
        -type f \
        -name 'mp_rank_00_model_states.pt' \
        -printf '%T@ %p\n' 2>/dev/null |
    sort -n |
    tail -n 1 |
    cut -d' ' -f2-
)"

if [[ -z "$LATEST" ]]; then
    log "WARNING: training stopped, but no model checkpoint was found"
    exit 1
fi

log "latest checkpoint:"
ls -lh "$LATEST"

if [[ -f "$ROOT/logs/train.log" ]]; then
    SAVED_LINE="$(
        grep 'saved checkpoint:' "$ROOT/logs/train.log" |
        tail -n 1 ||
        true
    )"

    if [[ -n "$SAVED_LINE" ]]; then
        printf '%s\n' "$SAVED_LINE"
    fi
fi

log "graceful stop completed"
