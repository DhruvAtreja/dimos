#!/usr/bin/env bash
# Build dimos-evals/sim2-gx (the sim2 eval image plus GraspGenX) on the box.
#   docker/build_graspgenx.sh [CHECKOUT]     default checkout: src/osr
# The GraspGenX project files come from CHECKOUT/native/python/graspgenx. The build stops
# if free disk drops under 1.5 GB. Output goes to docker/build-gx.log.
set -u
cd "$(dirname "$0")/.."
SRC="${1:-src/osr}"
CTX=docker/gx-context
rm -rf "$CTX"; mkdir -p "$CTX/native/python"
cp -r "$SRC/native/python/graspgenx" "$CTX/native/python/"
cp docker/gx_link.py "$CTX/"
LOG=docker/build-gx.log
docker build -f docker/Dockerfile.graspgenx -t dimos-evals/sim2-gx:latest --progress=plain "$CTX" > "$LOG" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  free_mb=$(df --output=avail -BM / | tail -1 | tr -dc 0-9)
  if [ "$free_mb" -lt 1500 ]; then echo "ABORT: free disk ${free_mb}M" >> "$LOG"; kill $pid; break; fi
  sleep 3
done
wait $pid; rc=$?
echo "build exit $rc" >> "$LOG"
rm -rf "$CTX"
docker builder prune -f >/dev/null 2>&1
exit $rc
