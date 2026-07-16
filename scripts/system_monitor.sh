#!/usr/bin/env bash

# Append one extensible workstation snapshot per line while training is active.

set -u

ROOT="/data/AlbertLM"
INTERVAL="${SYSTEM_METRICS_INTERVAL:-30}"
OUTPUT="$ROOT/logs/system.jsonl"
PID_FILE="$ROOT/logs/system_monitor.pid"

mkdir -p "$ROOT/logs"
touch "$OUTPUT"

if [[ -r "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        exit 0
    fi
fi

echo "$$" > "$PID_FILE"
cleanup() {
    if [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
        rm -f "$PID_FILE"
    fi
}
terminate() {
    exit 0
}
trap cleanup EXIT
trap terminate INT TERM

while true; do
    if snapshot="$($ROOT/scripts/system_status.sh 2>/dev/null)"; then
        timestamp="$(date --iso-8601=seconds)"
        jq -c --arg timestamp "$timestamp" '. + {timestamp:$timestamp, schema_version:1}' <<< "$snapshot" >> "$OUTPUT" 2>/dev/null || true
    fi
    sleep "$INTERVAL" &
    wait $!
done
