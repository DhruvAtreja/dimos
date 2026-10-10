#!/usr/bin/env bash
# Sample eval containers' CPU and RAM, the GPU and the host every 10 s into FILE, until killed.
#   monitor.sh FILE
FILE="${1:?output file}"
while true; do
  ts=$(date -u +%H:%M:%S)
  names=$(docker ps --filter label=dimos-evals=1 --format '{{.Names}}')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  host=$(free -m | awk '/Mem:/{print $3"MB used,"$7"MB avail"}')
  load=$(cut -d" " -f1 /proc/loadavg)
  echo "$ts host=$host load=$load gpu_mem_MB,util%=$gpu" >> "$FILE"
  [ -n "$names" ] && docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' $names | sed "s/^/$ts   /" >> "$FILE"
  sleep 10
done
