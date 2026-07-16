#!/usr/bin/env bash

# Unified, read-only workstation telemetry for AlbertLM Dashboard.
# Every probe is best-effort: unavailable hardware or commands become "N/A".

set -u

na="N/A"

command_exists() { command -v "$1" >/dev/null 2>&1; }

human_bytes() {
    if command_exists numfmt; then
        numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || printf '%s' "$na"
    else
        printf '%s' "$na"
    fi
}

trim() { xargs 2>/dev/null || true; }

cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}')"
cpu_model="${cpu_model:-$na}"
cpu_cores="$(lscpu -p=CORE 2>/dev/null | awk '!/^#/ {seen[$1]=1} END {print length(seen)}')"
cpu_cores="${cpu_cores:-$na}"
cpu_threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
cpu_threads="${cpu_threads:-$na}"
cpu_frequency="$(awk '/cpu MHz/ {sum+=$4; count++} END {if (count) printf "%.2f GHz", sum/count/1000}' /proc/cpuinfo 2>/dev/null)"
cpu_frequency="${cpu_frequency:-$na}"

declare -A previous_total previous_idle
while read -r name values; do
    [[ "$name" =~ ^cpu[0-9]*$ ]] || continue
    read -ra fields <<< "$values"
    total=0
    for index in 0 1 2 3 4 5 6 7; do total=$((total + ${fields[$index]:-0})); done
    previous_total["$name"]=$total
    previous_idle["$name"]=$((${fields[3]:-0} + ${fields[4]:-0}))
done < /proc/stat

sleep 0.2
cpu_usage="$na"
per_core_json='[]'
while read -r name values; do
    [[ "$name" =~ ^cpu[0-9]*$ ]] || continue
    read -ra fields <<< "$values"
    total=0
    for index in 0 1 2 3 4 5 6 7; do total=$((total + ${fields[$index]:-0})); done
    idle=$((${fields[3]:-0} + ${fields[4]:-0}))
    total_delta=$((total - ${previous_total[$name]:-$total}))
    idle_delta=$((idle - ${previous_idle[$name]:-$idle}))
    if (( total_delta > 0 )); then
        usage="$(awk -v total="$total_delta" -v idle="$idle_delta" 'BEGIN {printf "%.1f%%", 100 * (total-idle) / total}')"
        if [[ "$name" == "cpu" ]]; then
            cpu_usage="$usage"
        else
            core="${name#cpu}"
            per_core_json="$(jq -c --arg core "$core" --arg usage "$usage" '. + [{core:$core, usage:$usage}]' <<< "$per_core_json" 2>/dev/null || printf '[]')"
        fi
    fi
done < /proc/stat

cpu_temperature="$na"
if command_exists sensors; then
    cpu_temperature="$(sensors 2>/dev/null | awk '
        /^k10temp-/ {in_cpu=1; next}
        in_cpu && /^(Tctl|Tdie|Package id 0):/ {gsub(/[+°C]/, "", $2); printf "%.1f °C", $2; exit}
        in_cpu && /^[^[:space:]].*:/ && !/^Adapter:/ {in_cpu=0}
    ')"
    cpu_temperature="${cpu_temperature:-$na}"
fi

gpu_model="$na"; gpu_memory_used="$na"; gpu_memory_total="$na"; gpu_utilization="$na"
gpu_temperature="$na"; gpu_power="$na"; nvidia_driver="$na"; cuda_version="$na"
if command_exists nvidia-smi; then
    gpu_line="$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [[ -n "$gpu_line" ]]; then
        IFS=',' read -r gpu_model gpu_memory_used gpu_memory_total gpu_utilization gpu_temperature gpu_power nvidia_driver <<< "$gpu_line"
        gpu_model="$(printf '%s' "$gpu_model" | trim)"
        gpu_memory_used="$(printf '%s' "$gpu_memory_used" | trim) MiB"
        gpu_memory_total="$(printf '%s' "$gpu_memory_total" | trim) MiB"
        gpu_utilization="$(printf '%s' "$gpu_utilization" | trim)%"
        gpu_temperature="$(printf '%s' "$gpu_temperature" | trim) °C"
        gpu_power="$(printf '%s' "$gpu_power" | trim) W"
        nvidia_driver="$(printf '%s' "$nvidia_driver" | trim)"
    fi
    cuda_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([^ ]*\).*/\1/p' | head -1)"
    cuda_version="${cuda_version:-$na}"
fi

read -r memory_total_bytes memory_used_bytes memory_available_bytes < <(free -b 2>/dev/null | awk '/^Mem:/ {print $2, $3, $7}')
memory_total_bytes="${memory_total_bytes:-0}"; memory_used_bytes="${memory_used_bytes:-0}"; memory_available_bytes="${memory_available_bytes:-0}"
memory_total="$(human_bytes "$memory_total_bytes")"
memory_used="$(human_bytes "$memory_used_bytes")"
memory_available="$(human_bytes "$memory_available_bytes")"
memory_usage="$(awk -v used="$memory_used_bytes" -v total="$memory_total_bytes" 'BEGIN {if (total>0) printf "%.1f%%", used*100/total; else print "N/A"}')"

read -r swap_total_bytes swap_used_bytes swap_free_bytes < <(free -b 2>/dev/null | awk '/^Swap:/ {print $2, $3, $4}')
swap_total_bytes="${swap_total_bytes:-0}"; swap_used_bytes="${swap_used_bytes:-0}"; swap_free_bytes="${swap_free_bytes:-0}"
swap_total="$(human_bytes "$swap_total_bytes")"
swap_used="$(human_bytes "$swap_used_bytes")"
swap_free="$(human_bytes "$swap_free_bytes")"

disk_json() {
    local mount="$1" line filesystem total used available percent mounted
    line="$(df -B1 -P "$mount" 2>/dev/null | awk 'NR==2')"
    if [[ -z "$line" ]]; then
        jq -n --arg mount "$mount" --arg value "$na" '{mount:$mount,total:$value,used:$value,available:$value,usage:$value}'
        return
    fi
    read -r filesystem total used available percent mounted <<< "$line"
    jq -n \
        --arg mount "$mount" \
        --arg total "$(human_bytes "$total")" \
        --arg used "$(human_bytes "$used")" \
        --arg available "$(human_bytes "$available")" \
        --arg usage "$percent" \
        '{mount:$mount,total:$total,used:$used,available:$available,usage:$usage}'
}

system_disk="$(disk_json /)"
data_disk="$(disk_json /data)"
ssd_temperatures='[]'
if command_exists sensors; then
    current_sensor=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^(nvme-[^[:space:]]+) ]]; then
            current_sensor="${BASH_REMATCH[1]}"
        elif [[ -n "$current_sensor" && "$line" =~ ^Composite:[[:space:]]+\+?([0-9.]+) ]]; then
            temperature="${BASH_REMATCH[1]} °C"
            ssd_temperatures="$(jq -c --arg device "$current_sensor" --arg temperature "$temperature" '. + [{device:$device, temperature:$temperature}]' <<< "$ssd_temperatures" 2>/dev/null || printf '[]')"
            current_sensor=""
        elif [[ -z "$line" ]]; then
            current_sensor=""
        fi
    done < <(sensors 2>/dev/null)
fi

ubuntu_version="$(. /etc/os-release 2>/dev/null; printf '%s' "${PRETTY_NAME:-$na}")"
kernel_version="$(uname -r 2>/dev/null || printf '%s' "$na")"
uptime_text="$(uptime -p 2>/dev/null || printf '%s' "$na")"

jq -n \
    --arg cpu_model "$cpu_model" \
    --arg cpu_cores "$cpu_cores" \
    --arg cpu_threads "$cpu_threads" \
    --arg cpu_frequency "$cpu_frequency" \
    --arg cpu_usage "$cpu_usage" \
    --arg cpu_temperature "$cpu_temperature" \
    --argjson per_core "$per_core_json" \
    --arg gpu_model "$gpu_model" \
    --arg gpu_memory_used "$gpu_memory_used" \
    --arg gpu_memory_total "$gpu_memory_total" \
    --arg gpu_utilization "$gpu_utilization" \
    --arg gpu_temperature "$gpu_temperature" \
    --arg gpu_power "$gpu_power" \
    --arg memory_total "$memory_total" \
    --arg memory_used "$memory_used" \
    --arg memory_available "$memory_available" \
    --arg memory_usage "$memory_usage" \
    --arg swap_total "$swap_total" \
    --arg swap_used "$swap_used" \
    --arg swap_free "$swap_free" \
    --argjson system_disk "$system_disk" \
    --argjson data_disk "$data_disk" \
    --argjson ssd_temperatures "$ssd_temperatures" \
    --arg ubuntu "$ubuntu_version" \
    --arg kernel "$kernel_version" \
    --arg cuda "$cuda_version" \
    --arg driver "$nvidia_driver" \
    --arg uptime "$uptime_text" \
    '{
        cpu:{model:$cpu_model,cores:$cpu_cores,threads:$cpu_threads,frequency:$cpu_frequency,usage:$cpu_usage,per_core:$per_core,temperature:$cpu_temperature},
        gpu:{model:$gpu_model,memory_used:$gpu_memory_used,memory_total:$gpu_memory_total,utilization:$gpu_utilization,temperature:$gpu_temperature,power:$gpu_power},
        memory:{total:$memory_total,used:$memory_used,available:$memory_available,usage:$memory_usage,swap:{total:$swap_total,used:$swap_used,remaining:$swap_free}},
        disk:{system:$system_disk,data:$data_disk,ssd_temperatures:$ssd_temperatures},
        system:{ubuntu:$ubuntu,kernel:$kernel,cuda:$cuda,nvidia_driver:$driver,uptime:$uptime}
    }'
