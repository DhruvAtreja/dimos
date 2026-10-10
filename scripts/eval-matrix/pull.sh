#!/usr/bin/env bash
# Copy the matrix results from the EC2 evals box to this laptop, and free the box's disk by
# moving the trials' sensor recordings (the large files) off it.
#
#   ./pull.sh            results land in <this folder>/results/matrix/
#
# Recordings are deleted on the box only after rsync has copied each one here. Everything
# else (results, trajectories, logs) is copied and stays on the box too.
set -euo pipefail
HOST="${EVALS_HOST:-dimos-evals}"
BOX="${EVALS_DIR:-/home/ubuntu/dimos-evals}/results/${MATRIX_DIR:-matrix}/"
DEST="$(cd "$(dirname "$0")" && pwd)/results/${MATRIX_DIR:-matrix}/"
mkdir -p "$DEST"
before=$(ssh "$HOST" "df -h --output=avail / | tail -1" | tr -d ' ')
# Code 23: some folders of trials still running are unreadable until they finish; skip them.
rsync -a --exclude '.work/' --exclude '.claims/' --exclude '*.tmp' "$HOST:$BOX" "$DEST" 2>/dev/null || [ $? -eq 23 ]
rsync -a --remove-source-files --include '*/' --include 'recording.a*.tar.zst' --exclude '*' \
  "$HOST:$BOX" "$DEST" 2>/dev/null || [ $? -eq 23 ]
after=$(ssh "$HOST" "df -h --output=avail / | tail -1" | tr -d ' ')
echo "copied to $DEST ($(du -sh "$DEST" | cut -f1)); box disk free $before -> $after"
