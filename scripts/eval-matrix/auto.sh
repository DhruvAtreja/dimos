#!/usr/bin/env bash
# Run matrix steps one after another without anyone logged in.
#   auto.sh start [STEP ...]   run the steps in order, detached; survives ssh disconnect.
#                              A step is a round number (all configs), N:CONFIG[,CONFIG] (that
#                              round for those configs only), "prepare" (put new configs'
#                              checkouts in place), or "set:NAME" (from then on use
#                              matrix/configs-NAME.tsv and results/matrix-NAME, and
#                              matrix/batches-NAME.tsv when there is one). Default
#                              steps: rounds 2..10.
#   auto.sh stop               stop the chain and the round in progress
#   auto.sh detach             stop the chain only; a round in progress keeps going
#   auto.sh status             whether the chain is running, and its log
# The chain waits for any round already running before its first step. A round that
# pauses for low disk (below 2 GB free) just waits; run ./pull.sh on the laptop to free
# space and it carries on.
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$HERE/auto.pid"
LOG="$HERE/auto.log"

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
round_busy() { "$HERE/matrix.sh" status | grep -q "(driver running)"; }
stamp() { date -u '+%F %H:%M'; }

chain() {
  local step n cfgs
  while round_busy; do sleep 60; done
  for step in "$@"; do
    if [ "${step%%:*}" = set ]; then
      export MATRIX_CONFIGS="$HERE/matrix/configs-${step#set:}.tsv"
      export MATRIX_RESULTS="$HERE/results/matrix-${step#set:}"
      unset MATRIX_BATCHES
      [ -f "$HERE/matrix/batches-${step#set:}.tsv" ] && export MATRIX_BATCHES="$HERE/matrix/batches-${step#set:}.tsv"
      echo "$(stamp) now using $MATRIX_CONFIGS -> $MATRIX_RESULTS"
      # A round of this set may still be running (a chain restarted mid-set): wait for it.
      while round_busy; do sleep 60; done
      continue
    fi
    if [ "$step" = prepare ]; then
      echo "$(stamp) prepare"
      "$HERE/matrix.sh" prepare || { echo "prepare failed"; exit 1; }
      continue
    fi
    n=${step%%:*}
    cfgs=""
    [ "$step" != "$n" ] && cfgs=${step#*:}
    echo "$(stamp) starting round $n ${cfgs:-(all configs)}"
    # shellcheck disable=SC2086
    "$HERE/matrix.sh" round "$n" ${cfgs//,/ } || { echo "round $step failed to start"; exit 1; }
    sleep 60
    while round_busy; do sleep 60; done
    echo "$(stamp) round $step finished"
  done
  echo "$(stamp) all steps done"
  rm -f "$PIDFILE"
}

case "${1:-}" in
  start)
    running && { echo "already running (pid $(cat "$PIDFILE"))"; exit 1; }
    shift
    [ "$#" -gt 0 ] || set -- 2 3 4 5 6 7 8 9 10
    setsid nohup "$0" _chain "$@" >>"$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "steps $* started (pid $!); log: $LOG"
    ;;
  _chain) shift; chain "$@" ;;
  detach)
    running && kill "$(cat "$PIDFILE")" && echo "chain stopped; any round in progress keeps going"
    rm -f "$PIDFILE"
    ;;
  stop)
    running && kill "$(cat "$PIDFILE")" && echo "chain stopped"
    rm -f "$PIDFILE"
    "$HERE/matrix.sh" stop
    ;;
  status)
    running && echo "chain running (pid $(cat "$PIDFILE"))" || echo "chain not running"
    tail -5 "$LOG" 2>/dev/null
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
