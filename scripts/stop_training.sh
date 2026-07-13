#!/bin/bash

SESSION="albertlm"


if tmux has-session -t $SESSION 2>/dev/null; then
    tmux kill-session -t $SESSION
    echo "Stopped $SESSION"
else
    echo "No training session found"
fi
