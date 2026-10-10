#!/usr/bin/env bash
# Build the sim2 eval image from the osr checkout, stopping the build if free disk drops below 4 GB.
set -u
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile.sim2 -t dimos-evals/sim2:latest --progress=plain src/osr > docker/build.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt 4 ]; then echo "ABORT: free disk ${free_gb}G" >> docker/build.log; kill $pid; sudo systemctl restart docker; break; fi
  sleep 5
done
wait $pid; echo "build exit $?" >> docker/build.log
