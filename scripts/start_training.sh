#!/bin/bash

SESSION="albertlm"

cd ~/AlbertLM

if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Training session already exists"
    exit 1
fi

tmux new-session -d \
-s $SESSION \
"source .venv/bin/activate && PYTHONPATH=. python -u train/pretrain.py >> logs/train.log 2>&1"

echo "Started training in tmux session: $SESSION"
