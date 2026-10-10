#!/usr/bin/env bash
# Progress of the SceneReplica eval matrix on the EC2 evals box, refreshed every 30 s.
#   ./progress.sh            the main matrix (Ctrl-C to quit; runs keep going)
#   ./progress.sh cc         a named config set (auto.sh set:NAME)
#   ./progress.sh gc 20      ... with bars out of 20 instead of the set's case count
# One line per config: trials done, and of the current round's batch of 10.
HOST="${EVALS_HOST:-dimos-evals}"
BOX="${EVALS_DIR:-/home/ubuntu/dimos-evals}"
WIDTH=30
SET=${1:-}
TOTAL=${2:-100}
ENV=""
[ -n "$SET" ] && ENV="MATRIX_CONFIGS=$BOX/matrix/configs-$SET.tsv MATRIX_RESULTS=$BOX/results/matrix-$SET"
[ -n "$SET" ] && ENV="$ENV $(ssh -o ConnectTimeout=10 "$HOST" "[ -f $BOX/matrix/batches-$SET.tsv ] && echo MATRIX_BATCHES=$BOX/matrix/batches-$SET.tsv")"

bar() {
  local done=$1 total=$2 filled
  filled=$((done * WIDTH / total))
  printf '[%s%s]' "$(printf '%*s' "$filled" '' | tr ' ' '#')" "$(printf '%*s' $((WIDTH - filled)) '')"
}

while true; do
  status=$(ssh -o ConnectTimeout=10 "$HOST" "$ENV $BOX/matrix.sh status" 2>&1)
  clear
  echo "SceneReplica matrix${SET:+ ($SET)} on EC2 · $(date '+%H:%M')"
  echo
  if ! grep -q "^now:" <<<"$status"; then
    echo "Could not read the box: $status"
  else
    grep "^now:" <<<"$status"
    grep "^PAUSED" <<<"$status"
    echo
    while read -r line; do
      name=${line%%:*}
      n=$(grep -oP '^\S+: \K\d+(?=/\d+)' <<<"$line")
      t=$(grep -oP '^\S+: \d+/\K\d+' <<<"$line")
      [ -n "${2:-}" ] || TOTAL=$t
      b=$(grep -oP 'batch \d+: \K\d+(?=/10)' <<<"$line")
      bn=$(grep -oP 'batch \K\d+' <<<"$line")
      [ "$n" -gt "$TOTAL" ] && n=$TOTAL
      printf '%-14s %s %3d/%d' "$name" "$(bar "$n" "$TOTAL")" "$n" "$TOTAL"
      [ -n "$b" ] && printf '   batch %s: %2d/10' "$bn" "$b"
      echo
    done < <(grep -E '^\S+: [0-9]+/[0-9]+' <<<"$status")
    echo
    grep -E "^(running|disk free):" <<<"$status"
  fi
  sleep 30
done
