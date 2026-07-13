#!/bin/bash

SESSION="albertlm"


if tmux has-session -t $SESSION 2>/dev/null; then
    echo "RUNNING"
else
    echo "STOPPED"
fi
