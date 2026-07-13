#!/bin/bash

case "$1" in

status)
    cat ~/AlbertLM/logs/status.json
    ;;

gpu)
    nvidia-smi \
    --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader
    ;;

start)
    ~/AlbertLM/scripts/start_training.sh
    ;;

stop)
    ~/AlbertLM/scripts/stop_training.sh
    ;;

tmux)
    tmux ls
    ;;

*)
    echo "Usage:"
    echo "  status"
    echo "  gpu"
    echo "  start"
    echo "  stop"
    echo "  tmux"
    ;;

esac
