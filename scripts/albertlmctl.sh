#!/bin/bash

ACTION=$1

case "$ACTION" in
datasets)

python3 - <<'PY'
import json
from pathlib import Path

roots = [
    Path("/data/AlbertLM/data/raw"),
    Path("/data/AlbertLM/data/processed"),
    Path("/data/AlbertLM/datasets/raw"),
    Path("/data/AlbertLM/datasets/instruction"),
]

items=[]

for root in roots:
    if not root.exists():
        continue

    for p in root.rglob("*"):
        if p.is_file():
            stat=p.stat()
            items.append({
                "name":str(p.relative_to("/data/AlbertLM")),
                "size":f"{stat.st_size/1024/1024:.2f} MB",
                "time":stat.st_mtime
            })
print(json.dumps(items))
PY

;;
experiment)
cat <<JSON
{
"name":"AlbertLM",
"model":"AlbertLM-1.7B",
"dataset":"$(ls /data/AlbertLM/data/*.jsonl 2>/dev/null | head -1 | xargs -r basename)",
"status":"$(tmux has-session -t albertlm 2>/dev/null && echo training || echo stopped)",
"step":"$(grep -o 'step [0-9]*' /data/AlbertLM/logs/train.log 2>/dev/null | tail -1 | awk '{print $2}')",
"loss":"$(grep 'step' /data/AlbertLM/logs/train.log 2>/dev/null | tail -1 | awk '{print $4}')",
"checkpoint":"$(ls /data/AlbertLM/checkpoints 2>/dev/null | tail -1)"
}
JSON
;;

status)

if [ -r /data/AlbertLM/logs/status.json ]; then
    cat /data/AlbertLM/logs/status.json
else
    cat <<JSON
{"time":"$(date -Iseconds)","status":"unknown","step":0,"loss":0,"checkpoint":null,"gpu":""}
JSON
fi

;;

gpu)

nvidia-smi \
--query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
--format=csv,noheader,nounits

;;

system)

/data/AlbertLM/scripts/system_status.sh

;;

checkpoints)

python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime

root = Path("/data/AlbertLM/checkpoints")

items=[]

if root.exists():
    for p in root.iterdir():
        if p.is_file():
            s=p.stat()
            items.append({
                "name":p.name,
                "size":f"{s.st_size/1024/1024:.1f} MB",
                "time":datetime.fromtimestamp(s.st_mtime).isoformat()
            })

items.sort(key=lambda item: item["time"], reverse=True)
print(json.dumps(items))
PY

;;

metrics)

LIMIT=${2:-500}
[[ "$LIMIT" =~ ^[0-9]+$ ]] || LIMIT=500
(( LIMIT > 5000 )) && LIMIT=5000
touch /data/AlbertLM/logs/metrics.jsonl
tail -n "$LIMIT" /data/AlbertLM/logs/metrics.jsonl

;;

system-log)

LIMIT=${2:-500}
[[ "$LIMIT" =~ ^[0-9]+$ ]] || LIMIT=500
(( LIMIT > 5000 )) && LIMIT=5000
touch /data/AlbertLM/logs/system.jsonl
tail -n "$LIMIT" /data/AlbertLM/logs/system.jsonl

;;

logs)

LIMIT=${2:-300}
[[ "$LIMIT" =~ ^[0-9]+$ ]] || LIMIT=300
(( LIMIT > 5000 )) && LIMIT=5000
touch /data/AlbertLM/logs/train.log
tail -n "$LIMIT" /data/AlbertLM/logs/train.log

;;

start)

MODE=${2:-pretrain}

if [ "$MODE" = "sft" ]; then

    TRAIN_SCRIPT="train/sft.py"

elif [ "$MODE" = "pretrain" ]; then

    TRAIN_SCRIPT="train/pretrain.py"

else

    echo "unknown training mode: $MODE"
    echo "available: pretrain | sft"
    exit 1

fi

if [ "$MODE" = "pretrain" ]; then
    /data/AlbertLM/scripts/start_training.sh
    exit $?
fi

tmux kill-session -t albertlm 2>/dev/null


tmux new-session -d -s albertlm \
"cd /data/AlbertLM && PYTHONPATH=. python -u $TRAIN_SCRIPT 2>&1 | tee -a logs/train.log"


echo "training started: $MODE"

;;

stop)

/data/AlbertLM/scripts/stop_training.sh

;;

tmux)

tmux ls

;;

*)

echo "usage:"
echo "$0 {status|gpu|system|checkpoints|datasets|experiment|metrics|system-log|logs|start|stop|tmux}"

;;

esac
