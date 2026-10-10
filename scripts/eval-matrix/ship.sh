#!/usr/bin/env bash
# Send the matrix configs (matrix/configs.tsv here) and their branches from this laptop's
# ~/dimos repo to the EC2 evals box, then move each config's checkout there to its
# branch's new commit.
#
#   ./ship.sh              the branches named in matrix/configs.tsv
#   ./ship.sh BRANCH ...   also these extra branches
# The sim overlay branch (matrix/sim-overlay.txt) is always sent too.
#
# Only commits the box lacks are sent (an incremental git bundle). Nothing is pushed to
# GitHub. Run it before a round when a branch has new commits; a round itself never
# changes code.
set -euo pipefail
HOST="${EVALS_HOST:-dimos-evals}"
REPO="${DIMOS_REPO:-$HOME/dimos}"
BOX="${EVALS_DIR:-/home/ubuntu/dimos-evals}"

HERE="$(cd "$(dirname "$0")" && pwd)"
if ssh "$HOST" "$BOX/matrix.sh status" | grep -q "(driver running)"; then
  echo "a round is running on the box; ship after it ends (or matrix.sh stop)" >&2; exit 1
fi
mapfile -t refs < <({ grep -v '^#' "$HERE/matrix/configs.tsv" | awk 'NF {print $2}'; printf '%s\n' "$@"
  head -1 "$HERE/matrix/sim-overlay.txt" 2>/dev/null; } | awk NF | sort -u)
for ref in "${refs[@]}"; do
  git -C "$REPO" rev-parse --verify -q "refs/heads/$ref" >/dev/null \
    || { echo "no local branch $ref in $REPO" >&2; exit 1; }
done

# Commits the box already has, so the bundle carries only the new ones.
have=()
while read -r sha; do
  git -C "$REPO" cat-file -e "$sha^{commit}" 2>/dev/null && have+=("^$sha")
done < <(ssh "$HOST" "git -C $BOX/src/store.git for-each-ref --format='%(objectname)'")

bundle=$(mktemp --suffix=.bundle)
trap 'rm -f "$bundle" "$bundle.err"' EXIT
if git -C "$REPO" bundle create "$bundle" "${refs[@]/#/refs/heads/}" "${have[@]}" 2>"$bundle.err"; then
  echo "bundle: $(du -h "$bundle" | cut -f1) for ${#refs[@]} branch(es)"
  ssh "$HOST" "mkdir -p $BOX/ship"
  scp -q "$bundle" "$HOST:$BOX/ship/incoming.bundle"
  ssh "$HOST" "git -C $BOX/src/store.git fetch -q $BOX/ship/incoming.bundle '+refs/heads/*:refs/heads/*' && rm -f $BOX/ship/incoming.bundle"
else
  # The box has every commit already; only the branch names need to move.
  grep -q "empty bundle" "$bundle.err" || { cat "$bundle.err" >&2; exit 1; }
  for ref in "${refs[@]}"; do
    ssh "$HOST" "git -C $BOX/src/store.git update-ref refs/heads/$ref $(git -C "$REPO" rev-parse "refs/heads/$ref")"
  done
fi
for ref in "${refs[@]}"; do
  printf '%-40s %s\n' "$ref" "$(git -C "$REPO" rev-parse --short "refs/heads/$ref")"
done
scp -q "$HERE/matrix/configs.tsv" "$HERE/matrix/require.txt" "$HOST:$BOX/matrix/"
if [ -f "$HERE/matrix/sim-overlay.txt" ]; then scp -q "$HERE/matrix/sim-overlay.txt" "$HOST:$BOX/matrix/"
else ssh "$HOST" "rm -f $BOX/matrix/sim-overlay.txt"; fi
ssh "$HOST" "$BOX/matrix.sh prepare && $BOX/matrix.sh check"
