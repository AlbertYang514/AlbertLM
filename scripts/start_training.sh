#!/usr/bin/env bash

set -euo pipefail

ROOT="/data/AlbertLM"
SESSION="albertlm"

cd "$ROOT"

mkdir -p logs checkpoints

# Transformer Engine serializes Python extra-state in checkpoints. This opt-in
# is restricted to trusted checkpoints produced by this project under
# /data/AlbertLM/checkpoints; never use it for arbitrary external checkpoints.
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1

if tmux has-session \
    -t "$SESSION" \
    2>/dev/null
then
    echo "training session already exists: $SESSION"
    exit 1
fi

tmux new-session \
    -d \
    -s "$SESSION" \
    -e NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 \
    "$ROOT/scripts/train_supervisor.sh"

echo "started training supervisor: $SESSION"
