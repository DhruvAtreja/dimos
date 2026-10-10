#!/usr/bin/env bash
# SceneReplica eval matrix on this box. Every config in matrix/configs.tsv runs the same
# 10 fixed batches of 10 trials (matrix/batches.tsv), one trial per container, so the
# configs' results pair up case by case.
#
#   matrix.sh round N [CONFIG ...]  run batch N (1-10) for every config (or the ones named),
#                                   SLOTS containers at once from one queue shared by the
#                                   configs, in the background. Trials already done are
#                                   skipped, so rerunning resumes.
#   matrix.sh status                what is running and how many trials each config has done
#   matrix.sh report [CONFIG]       grasped/placed per config and batch, side by side; with
#                                   CONFIG, that config's full SceneReplica summary
#   matrix.sh stop                  stop the round and its containers; finished trials stay
#   matrix.sh check [CONFIG ...]    check checkouts and suites; a round runs this first
#   matrix.sh prepare               put each config's checkout on its branch (ship.sh runs it)
#   matrix.sh smoke CONFIG CASE     one trial into results/matrix-smoke, not counted
#   matrix.sh batches CONFIG        write matrix/batches.tsv from CONFIG's suite (done once)
#
# Every config runs the same simulator: matrix/sim-overlay.txt names a branch (line 1) and
# the commit it starts from (line 2). prepare copies the files that branch changed since
# that commit over each config's checkout, so sim fixes reach every config without being
# merged into the branches under test. No file, or an empty one, leaves checkouts as they are.
#
# Settings (environment): IMAGE (default dimos-evals/sim2-gx:latest), MEMORY (RAM cap per
# container, 10g), SLOTS (containers at once in a round, 3), CASE_TIMEOUT_S (2700), STARTUP_TIMEOUT_S
# (stop a trial with no tool call this long after it starts, 120), SCRIPTED_TIMEOUT_S (whole-trial
# cap for configs run by matrix_scripted, 240), MIN_FREE_GB (no new trial starts with less free
# disk than this, 2), MATRIX_CONFIGS (another configs file, e.g. matrix/standin.tsv).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export MATRIX_HOME="$HERE"
export MATRIX_CONFIGS="${MATRIX_CONFIGS:-$HERE/matrix/configs.tsv}"
export MATRIX_RESULTS="${MATRIX_RESULTS:-$HERE/results/matrix}"
R="$MATRIX_RESULTS"
LIB="$HERE/matrix/lib"
IMAGE="${IMAGE:-dimos-evals/sim2-gx:latest}"
MEMORY="${MEMORY:-10g}"
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-2700}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-120}"   # no tool call by then: stuck (slowest seen: 96 s)
SCRIPTED_TIMEOUT_S="${SCRIPTED_TIMEOUT_S:-240}" # whole-trial cap with the scripted agent (slowest seen: 150 s)
SLOTS="${SLOTS:-3}"             # containers at once in a round, shared by every config
MIN_FREE_GB="${MIN_FREE_GB:-2}"
STORE="$HERE/src/store.git"
LFS_STORE="$HERE/src/lfs"
DATA_FROM="$HERE/src/baseline/data"   # extracted scene data, shared into every checkout
PIDFILE="$R/driver.pid"
STATE="$R/STATE"
VOXEL_BIN="$HERE/src/native/voxel_ray_tracing"
GX_KEY=08dc71384eb87574               # sha256("/app/native/python/graspgenx")[:16], as in the image
GX_LINK="$HERE/cache/dimos/isolated-python/$GX_KEY/.venv"
IMAGE_SRC="$HERE/src/osr"             # the checkout the image's venv and voxel binary were built from
SIM_OVERLAY="$HERE/matrix/sim-overlay.txt"

mx() { python3 "$LIB/matrix.py" "$@"; }
log() { echo "$(date -u +%FT%TZ) $*"; }
free_gb() { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
config_names() { grep -v '^[[:space:]]*#' "$MATRIX_CONFIGS" | awk 'NF {print $1}'; }

# Sets C_NAME C_REF C_SUITE C_PAR C_BP C_ENV (array) C_SRC for one config.
load_config() {
  local line rest e
  line=$(grep -v '^[[:space:]]*#' "$MATRIX_CONFIGS" | awk -v n="$1" '$1 == n' | head -1)
  [ -n "$line" ] || { echo "no config $1 in $MATRIX_CONFIGS" >&2; return 1; }
  read -r C_NAME C_REF C_SUITE C_PAR C_BP rest <<< "$line"
  [ "$C_BP" = "-" ] && C_BP=""
  C_ENV=()
  for e in $rest; do [ "$e" = "-" ] || C_ENV+=("$e"); done
  C_SRC="$HERE/src/matrix/$C_NAME"
}

# The pid of the running round driver, or nothing. Read from its pid file and checked
# against its command line, so no other process is ever taken for it.
driver_pid() {
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 0
  [ -n "$pid" ] && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "matrix.sh _round" \
    && echo "$pid"
  return 0
}

# ------------------------------------------------------------------ one trial

# Hold off starting a trial while the disk is nearly full; running trials carry on.
wait_for_disk() {
  local warned=0
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
    if [ "$warned" -eq 0 ]; then
      log "PAUSED: less than $MIN_FREE_GB GB free; ./pull.sh on the laptop moves recordings off"
      echo "disk low, waiting" > "$R/PAUSED"
    fi
    warned=1
    sleep 60
  done
  [ "$warned" -eq 1 ] && { log "resumed"; rm -f "$R/PAUSED"; }
  return 0
}

# Compress what a finished attempt left: the trial's sensor recording, the stack's logs
# and the model's raw request/response logs. Results and trajectories stay as they are.
pack() {
  local work=$1 out=$2 a=$3 raw
  if [ -n "$(ls -A "$work/recordings" 2>/dev/null)" ]; then
    tar -C "$work/recordings" -cf - . | zstd -q -T1 -3 -o "$out/recording.a$a.tar.zst.tmp" \
      && mv "$out/recording.a$a.tar.zst.tmp" "$out/recording.a$a.tar.zst"
  fi
  if [ -n "$(ls -A "$work/logs" 2>/dev/null)" ]; then
    tar -C "$work/logs" -cf - . | zstd -q -T1 -3 -o "$out/stack-logs.a$a.tar.zst"
  fi
  for raw in "$out"/evals/run-*/*/raw; do
    [ -d "$raw" ] || continue
    tar -C "$(dirname "$raw")" -cf - raw | zstd -q -T1 -3 -o "$(dirname "$raw")/raw.tar.zst" \
      && rm -rf "$raw"
  done
}

# One container running one case of the loaded config. Output in $R/CONFIG/CASE/:
# evals/run-*/ (results.jsonl, trajectory), aK.log (container output), aK.exit,
# recording.aK.tar.zst, stack-logs.aK.tar.zst.
one_attempt() {
  local case_id=$1 slot=$2
  local out="$R/$C_NAME/$case_id"
  mkdir -p "$out/evals"
  local a=$(( $(ls "$out"/a*.log 2>/dev/null | wc -l) + 1 ))
  local short=${case_id#scenereplica-}
  local c="mx-$C_NAME-$short-a$a"
  local work="$R/.work/$C_NAME-$short-a$a"
  mkdir -p "$work"/{state,recordings,logs,egg}
  # Compiled Python bytecode is kept between trials, one folder per commit (configs on the same
  # commit share it); the image's default (/tmp/pycache) makes every trial recompile ~2,200
  # modules. The sim overlay is the same for every config, so the commit names the code.
  local envargs=(-e "MATRIX_SUITE=$C_SUITE" -e "MATRIX_CASE=$case_id" -e "MATRIX_BLUEPRINT=$C_BP"
    -e "PYTHONPYCACHEPREFIX=/cache/pycache/$(git -C "$C_SRC" rev-parse --short=12 HEAD)")
  local e
  for e in "${C_ENV[@]}"; do envargs+=(-e "$e"); done
  docker rm -f "$c" >/dev/null 2>&1
  log "$C_NAME $case_id: attempt $a starts (slot $slot)"
  if ! docker run -d --name "$c" --label dimos-evals=1 --label dimos-evals.matrix=1 \
      --label "dimos-evals.config=$C_NAME" --label "dimos-evals.case=$case_id" \
      --gpus all --init --shm-size 4g --ulimit memlock=-1 --cap-add NET_ADMIN \
      ${MEMORY:+--memory "$MEMORY"} --env-file "$HERE/secrets.env" "${envargs[@]}" \
      -v "$C_SRC":/app:ro \
      -v "$work/state":/state \
      -v "$work/recordings":/app/recordings \
      -v "$work/logs":/app/logs \
      -v "$work/egg":/app/dimos.egg-info \
      -v "$out/evals":/state/dimos/evals \
      --tmpfs /state/dimos/runs \
      -v "$HERE/cache":/cache \
      -v "$LIB":/matrix:ro \
      "$IMAGE" bash -c 'PYTHONPATH=/matrix dimos evals run matrix_case --agent "${MATRIX_AGENT:-dimos.evals.agents.pi}" ${MATRIX_AGENT_SET:+--set "$MATRIX_AGENT_SET"}; rc=$?; chmod -R a+rwX /state/dimos/evals /app/recordings /app/logs 2>/dev/null; exit $rc' \
      >/dev/null; then
    log "$C_NAME $case_id: docker run failed"
    rm -rf "$work"
    sleep 30
    return
  fi
  docker logs -f "$c" > "$out/a$a.log" 2>&1 &
  local logs_pid=$! rc started=$SECONDS stopped="" cap=$CASE_TIMEOUT_S e
  for e in "${C_ENV[@]}"; do [ "$e" = MATRIX_AGENT=matrix_scripted ] && cap=$SCRIPTED_TIMEOUT_S; done
  ( docker wait "$c" > "$out/a$a.exit" 2>/dev/null ) &
  local wait_pid=$!
  # A trial whose stack never takes its first tool call is stuck starting up (the harness
  # can wait on the stack with no time limit); stop it so the attempt runs again.
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 10
    if [ $((SECONDS - started)) -ge "$cap" ]; then
      log "$C_NAME $case_id: still running ${cap}s after start, stopping"; stopped=timeout
    elif [ $((SECONDS - started)) -ge "$STARTUP_TIMEOUT_S" ] && ! grep -q "MCP tool call" "$out/a$a.log" 2>/dev/null; then
      log "$C_NAME $case_id: no tool call ${STARTUP_TIMEOUT_S}s after start, stuck starting up; stopping"; stopped=stuck
    fi
    [ -n "$stopped" ] && { docker stop -t 30 "$c" >/dev/null; break; }
  done
  wait "$wait_pid" 2>/dev/null
  rc=${stopped:-$(cat "$out/a$a.exit" 2>/dev/null)}
  wait "$logs_pid" 2>/dev/null
  echo "$rc" > "$out/a$a.exit"
  docker rm "$c" >/dev/null 2>&1
  # A killed container cannot open its own files up; the host does it instead.
  sudo -n chmod -R a+rwX "$out/evals" "$work" 2>/dev/null
  pack "$work" "$out" "$a"
  rm -rf "$work" 2>/dev/null || sudo -n rm -rf "$work"
  log "$C_NAME $case_id: attempt $a ended (exit $rc), case $(mx status-case "$C_NAME" "$case_id")"
}

# Run one case until it is done, at most two attempts in this round.
run_case() {
  local case_id=$1 slot=$2 tries=0 st
  while :; do
    st=$(mx status-case "$C_NAME" "$case_id")
    [ "$st" = done ] && return
    if [ "$tries" -ge 2 ]; then
      log "$C_NAME $case_id: still '$st' after $tries attempts; a rerun of the round tries again"
      return
    fi
    tries=$((tries + 1))
    wait_for_disk
    one_attempt "$case_id" "$slot"
  done
}

# One config's share of a batch: up to C_PAR containers at once, each taking the next case.
run_batch() {
  load_config "$1" || return
  local n=$2 cases pids=() k
  mapfile -t cases < <(mx cases "$n")
  local claims="$R/.claims/$C_NAME-b$n"
  rm -rf "$claims"; mkdir -p "$claims"
  log "$C_NAME batch $n: ${#cases[@]} cases, $C_PAR at a time, $C_REF @ $(git -C "$C_SRC" rev-parse --short HEAD)"
  for k in $(seq 1 "$C_PAR"); do
    (
      for case_id in "${cases[@]}"; do
        [ "$(mx status-case "$C_NAME" "$case_id")" = done ] && continue
        mkdir "$claims/$case_id" 2>/dev/null || continue
        run_case "$case_id" "$k"
      done
    ) &
    pids+=($!)
    sleep 20  # stagger boots so model loads do not all land at once
  done
  wait "${pids[@]}"
  log "$C_NAME batch $n finished: $(mx progress "$n" | grep "^$C_NAME:")"
}

# Every config's share of batch N in one queue: up to SLOTS containers at once, and a free
# slot takes the next unfinished (config, case) whichever config it belongs to, so no slot
# waits for another config's slowest trial. Jobs are ordered case by case, cycling the
# configs, so all configs move through the batch together.
run_round_queue() {
  local n=$1 cfg case_id job k pids=() jobs=() cases
  shift
  mapfile -t cases < <(mx cases "$n")
  for cfg in "$@"; do
    load_config "$cfg" || continue
    rm -rf "$R/.claims/$cfg-b$n"; mkdir -p "$R/.claims/$cfg-b$n"
    log "$cfg batch $n: ${#cases[@]} cases queued, $C_REF @ $(git -C "$C_SRC" rev-parse --short HEAD)"
  done
  for case_id in "${cases[@]}"; do
    for cfg in "$@"; do jobs+=("$cfg $case_id"); done
  done
  for k in $(seq 1 "$SLOTS"); do
    (
      for job in "${jobs[@]}"; do
        read -r cfg case_id <<< "$job"
        [ "$(mx status-case "$cfg" "$case_id")" = done ] && continue
        mkdir "$R/.claims/$cfg-b$n/$case_id" 2>/dev/null || continue
        load_config "$cfg" || continue
        run_case "$case_id" "$k"
      done
    ) &
    pids+=($!)
    sleep 20  # stagger boots so model loads do not all land at once
  done
  wait "${pids[@]}"
  for cfg in "$@"; do log "$cfg batch $n finished: $(mx progress "$n" | grep "^$cfg:")"; done
}

# ------------------------------------------------------------------ commands

cmd__round() {  # the detached driver behind `round`
  local n=$1 mon
  shift
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
  log "round $n starts: $* (shared queue, $SLOTS containers)"
  "$HERE/monitor.sh" "$R/resources.log" &
  mon=$!
  echo "round $n: $# configs, shared queue" > "$STATE"
  run_round_queue "$n" "$@"
  kill "$mon" 2>/dev/null
  echo "round $n finished $(date -u '+%F %H:%M') UTC" > "$STATE"
  log "round $n finished"
}

cmd_round() {
  local n=${1:-} pid names
  [[ "$n" =~ ^([1-9]|10)$ ]] || { echo "usage: matrix.sh round N [CONFIG ...]  (N = 1..10)" >&2; exit 2; }
  shift
  pid=$(driver_pid)
  [ -z "$pid" ] || { echo "a round is already running (pid $pid); see matrix.sh status, or matrix.sh stop"; exit 1; }
  if [ "$#" -gt 0 ]; then names=("$@"); else mapfile -t names < <(config_names); fi
  if ! cmd_check "${names[@]}"; then
    [ "${FORCE:-0}" = 1 ] || { echo "not started: fix the problems above (or FORCE=1 to run anyway)"; exit 1; }
  fi
  mkdir -p "$R"
  setsid bash "$HERE/matrix.sh" _round "$n" "${names[@]}" >> "$R/driver.log" 2>&1 < /dev/null &
  sleep 2
  echo "round $n started for: ${names[*]}"
  echo "progress: matrix.sh status   results: matrix.sh report   log: $R/driver.log"
}

cmd_stop() {
  local pid pgid
  pid=$(driver_pid)
  if [ -n "$pid" ]; then
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    # The driver was started with setsid, so its process group holds only the driver,
    # its workers and their helpers.
    if [ "$pgid" = "$pid" ]; then kill -TERM -- "-$pgid"; else kill -TERM "$pid"; fi
    rm -f "$PIDFILE"
    echo "stopped driver $pid"
  else
    echo "no round running"
  fi
  local ids
  ids=$(docker ps -aq --filter label=dimos-evals.matrix=1)
  if [ -n "$ids" ]; then
    echo "stopping containers: $(docker ps -a --filter label=dimos-evals.matrix=1 --format '{{.Names}}' | tr '\n' ' ')"
    docker stop -t 30 $ids >/dev/null
    docker rm $ids >/dev/null
  fi
  sudo -n rm -rf "$R/.work" 2>/dev/null || rm -rf "$R/.work"
  [ -n "$pid" ] && echo "round $(grep -oP '^round \K\d+' "$STATE" 2>/dev/null) stopped $(date -u '+%F %H:%M') UTC" > "$STATE"
  echo "finished trials are kept; a stopped trial runs again when the round is rerun"
}

cmd_status() {
  local pid round alive
  pid=$(driver_pid)
  [ -n "$pid" ] && alive="driver running" || alive="driver not running"
  echo "now: $(cat "$STATE" 2>/dev/null || echo 'no round yet') ($alive)"
  [ -e "$R/PAUSED" ] && echo "PAUSED: disk low; run ./pull.sh on the laptop"
  round=$(grep -oP '^round \K\d+' "$STATE" 2>/dev/null)
  mx progress $round
  echo "running: $(docker ps --filter label=dimos-evals.matrix=1 \
    --format '{{.Label "dimos-evals.config"}} {{.Label "dimos-evals.case"}} ({{.RunningFor}})' \
    | sed 's/scenereplica-//; s/ ago//' | paste -sd ';' | sed 's/;/; /g')"
  echo "disk free: $(free_gb) GB"
}

cmd_report() { mx report "$@"; }

# Case listing of the loaded config's suite, from its checkout in the image.
listing() {
  local e envargs=(-e "MATRIX_BLUEPRINT=$C_BP")
  for e in "${C_ENV[@]}"; do envargs+=(-e "$e"); done
  docker run --rm "${envargs[@]}" -v "$C_SRC":/app:ro -v "$LIB":/matrix:ro \
    -v "$HERE/cache":/cache --tmpfs /state --tmpfs /app/logs --tmpfs /app/recordings "$IMAGE" \
    python /matrix/matrix_case.py list "$C_SUITE"
}

cmd_check() {
  local names rc=0 name sha head key cached bp
  if [ "$#" -gt 0 ]; then names=("$@"); else mapfile -t names < <(config_names); fi
  [ -s "$HERE/matrix/batches.tsv" ] || { echo "no batch list yet: matrix.sh batches CONFIG"; return 1; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE missing"; return 1; }
  [ -L "$GX_LINK" ] || { echo "missing $GX_LINK (link to the image's GraspGenX venv)"; rc=1; }
  for name in "${names[@]}"; do
    load_config "$name" || { rc=1; continue; }
    if [ ! -d "$C_SRC/.git" ]; then echo "$name: no checkout; run ship.sh (laptop) or matrix.sh prepare"; rc=1; continue; fi
    sha=$(git -C "$STORE" rev-parse -q --verify "$C_REF^{commit}")
    head=$(git -C "$C_SRC" rev-parse HEAD)
    if [ "$sha" != "$head" ]; then
      echo "$name: checkout is at ${head:0:9} but $C_REF is ${sha:0:9}; run matrix.sh prepare"; rc=1; continue
    fi
    key=$(echo "$head $C_SUITE $C_BP ${C_ENV[*]} $(md5sum < "$LIB/matrix_case.py")" | md5sum | cut -c1-12)
    cached="$R/$name/listing-$key.tsv"
    mkdir -p "$R/$name"
    if [ ! -s "$cached" ]; then
      listing > "$cached.tmp" 2> "$R/$name/listing.err" && mv "$cached.tmp" "$cached" \
        || { echo "$name: could not list the suite's cases:"; tail -5 "$R/$name/listing.err"; rc=1; continue; }
    fi
    mx check "$name" "$cached" || rc=1
    bp=$(head -1 "$cached" | cut -f3 | cut -d' ' -f1)
    grep -q "\"$bp\":" "$C_SRC/dimos/robot/all_blueprints.py" \
      || { echo "  PROBLEM: blueprint $bp is not registered in this checkout"; rc=1; }
    git -C "$C_SRC" diff --quiet "$(git -C "$DATA_FROM/.." rev-parse HEAD)" HEAD -- data/.lfs 2>/dev/null \
      || { echo "  PROBLEM: this branch changes data/.lfs; the shared extracted data may not match"; rc=1; }
    git -C "$C_SRC" diff --quiet "$(git -C "$IMAGE_SRC" rev-parse HEAD)" HEAD -- pyproject.toml uv.lock \
      || { echo "  PROBLEM: pyproject.toml/uv.lock differ from the image's; rebuild the image (docker/build.sh)"; rc=1; }
    git -C "$C_SRC" diff --quiet "$(git -C "$IMAGE_SRC" rev-parse HEAD)" HEAD -- Cargo.toml Cargo.lock native ':!native/python' \
      || { echo "  PROBLEM: native code differs from the image's; the voxel binary may be stale"; rc=1; }
    gx_verify "$C_SRC" || rc=1
    while read -r need why; do
      git -C "$C_SRC" merge-base --is-ancestor "$need" HEAD 2>/dev/null \
        || { echo "  PROBLEM: lacks $need ($why)"; rc=1; }
    done < <(grep -v '^#' "$HERE/matrix/require.txt" 2>/dev/null | awk 'NF')
  done
  return $rc
}

# Copy the sim overlay's files into the current config's checkout ($C_SRC). Prints
# "BRANCH@SHA" when it applied one, nothing when there is none, or why it failed (exit 1).
apply_sim_overlay() {
  local ref base sha files
  [ -s "$SIM_OVERLAY" ] || return 0
  { read -r ref; read -r base; } < "$SIM_OVERLAY"
  [ -n "$ref" ] || return 0
  sha=$(git -C "$C_SRC" rev-parse -q --verify "origin/$ref^{commit}") \
    || { echo "sim overlay $ref is not on the box; run ship.sh on the laptop"; return 1; }
  mapfile -t files < <(git -C "$C_SRC" diff --name-only --diff-filter=AM "$base" "$sha")
  [ "${#files[@]}" -gt 0 ] || { echo "sim overlay $ref changes no files since $base"; return 1; }
  git -C "$C_SRC" checkout -q "$sha" -- "${files[@]}" || { echo "sim overlay $ref did not apply"; return 1; }
  echo "$ref@${sha:0:9}"
}

prepare_one() {
  local sha d n model thinking
  load_config "$1" || return 1
  sha=$(git -C "$STORE" rev-parse -q --verify "$C_REF^{commit}") \
    || { echo "$C_NAME: $C_REF is not on the box; run ship.sh on the laptop"; return 1; }
  if [ ! -d "$C_SRC/.git" ]; then
    mkdir -p "$(dirname "$C_SRC")"
    git clone -q --no-checkout --local "$STORE" "$C_SRC" || return 1
    git -C "$C_SRC" config lfs.storage "$LFS_STORE"
  else
    git -C "$C_SRC" fetch -q origin || return 1
  fi
  # Always reset, so a changed or removed sim overlay never leaves old files behind.
  git -C "$C_SRC" -c advice.detachedHead=false checkout -q --force --detach "$sha" || return 1
  local overlay
  overlay=$(apply_sim_overlay) || { echo "$C_NAME: $overlay"; return 1; }
  # Scene data the suites extract on first use, shared by hard links (no extra disk).
  for d in "$DATA_FROM"/*/; do
    n=$(basename "$d")
    [ -e "$C_SRC/data/$n" ] || cp -al "$d" "$C_SRC/data/$n"
  done
  # Mount points inside the read-only checkout.
  mkdir -p "$C_SRC/recordings" "$C_SRC/logs" "$C_SRC/dimos.egg-info"
  # The voxel mapper's native binary, built into the image from the same sources.
  mkdir -p "$C_SRC/target/release" && ln -f "$VOXEL_BIN" "$C_SRC/target/release/voxel_ray_tracing"
  model=$(grep -oP '^\s+model: str = "\K[^"]+' "$C_SRC/dimos/evals/agents/base.py" | head -1)
  thinking=$(grep -oP '^\s+thinking: Thinking = "\K[^"]+' "$C_SRC/dimos/evals/agents/pi.py" | head -1)
  mkdir -p "$R/$C_NAME"
  local agent="pi, model ${model:-?}, thinking ${thinking:-?} (code defaults, no --set)" e
  for e in "${C_ENV[@]}"; do
    case $e in MATRIX_AGENT=*) agent="${e#MATRIX_AGENT=}" ;; esac
  done
  for e in "${C_ENV[@]}"; do
    case $e in MATRIX_AGENT_SET=*) agent="$agent --set ${e#MATRIX_AGENT_SET=}" ;; esac
  done
  printf '{"ref": "%s", "sha": "%s", "prepared": "%s", "agent": "%s", "sim_overlay": "%s"}\n' \
    "$C_REF" "$sha" "$(date -u +%FT%TZ)" "$agent" "$overlay" > "$R/$C_NAME/config.json"
  echo "$C_NAME: $C_REF @ ${sha:0:9} (agent model ${model:-?}, thinking ${thinking:-?})${overlay:+, sim from $overlay}"
}

# GraspGenX's `uv run --with-editable /app` builds dimos from the checkout and adds dimos's
# packages on top of the GraspGenX venv. uv keys both on the checkout's files, so they are
# built here, once per checkout with the checkout writable, into the shared uv cache
# (/cache/uv); containers, where the checkout is read-only, then find them ready.
gx_prebuild() {
  local src=$1 out
  [ -d "$src/native/python/graspgenx" ] || return 0
  local run=(docker run --rm -v "$HERE/cache":/cache --tmpfs /state -w /app/native/python/graspgenx)
  local cmd=(env -u VIRTUAL_ENV "UV_PROJECT_ENVIRONMENT=/cache/dimos/isolated-python/$GX_KEY/.venv"
             uv run --frozen --with-editable /app python -c pass)
  # Offline first, from what the cache already holds; online only if that is not enough.
  if ! out=$("${run[@]}" -v "$src":/app "$IMAGE" "${cmd[@]}" 2>&1); then
    echo "  $(basename "$src"): cache lacks packages, fetching them"
    out=$("${run[@]}" -e UV_OFFLINE=0 -v "$src":/app "$IMAGE" "${cmd[@]}" 2>&1) \
      || { echo "  $(basename "$src"): GraspGenX environment failed:"; echo "$out" | tail -5; return 1; }
    "${run[@]}" "$IMAGE" /opt/venv/bin/python /opt/gx_link.py dedupe \
      /opt/venv/lib/python3.12/site-packages /cache/uv/archive-v0 | sed 's/^/  /'
  fi
  gx_verify "$src" && echo "  $(basename "$src"): GraspGenX environment ready"
}

# Start GraspGenX's environment the way a trial container does (checkout read-only); it
# must find everything built and install nothing.
gx_verify() {
  local src=$1 out
  [ -d "$src/native/python/graspgenx" ] || return 0
  out=$(docker run --rm -v "$HERE/cache":/cache --tmpfs /state -w /app/native/python/graspgenx \
    -v "$src":/app:ro "$IMAGE" env -u VIRTUAL_ENV \
    "UV_PROJECT_ENVIRONMENT=/cache/dimos/isolated-python/$GX_KEY/.venv" \
    uv run --frozen --with-editable /app python -c pass 2>&1) \
    || { echo "  PROBLEM: GraspGenX does not start with $(basename "$src") read-only (run matrix.sh prepare):"; echo "$out" | tail -3; return 1; }
  if grep -qE "Built|Installed|Downloaded" <<< "$out"; then
    echo "  PROBLEM: GraspGenX would build at every start for $(basename "$src"); run matrix.sh prepare"
    return 1
  fi
}

cmd_prepare() {
  local pid name rc=0 src
  pid=$(driver_pid)
  [ -z "$pid" ] || { echo "a round is running (pid $pid); checkouts move only between rounds"; exit 1; }
  if [ ! -d "$LFS_STORE/objects" ]; then
    mkdir -p "$LFS_STORE" && cp -al "$HERE/src/baseline/.git/lfs/objects" "$LFS_STORE/objects"
  fi
  if [ ! -f "$VOXEL_BIN" ]; then
    mkdir -p "$(dirname "$VOXEL_BIN")" && cp "$HERE/src/osr/target/release/voxel_ray_tracing" "$VOXEL_BIN"
  fi
  mkdir -p "$(dirname "$GX_LINK")" && ln -sfn /opt/graspgenx-venv "$GX_LINK"
  for name in $(config_names); do prepare_one "$name" || rc=1; done
  # Identical files across checkouts share one copy on disk. Not the files uv keys a
  # checkout's GraspGenX build on: linking them would make that build stale.
  hardlink -t -q -x '/(pyproject\.toml|setup\.py|setup\.cfg)$' "$HERE/src" 2>/dev/null
  # Linking changed file times; refresh git's index so `git status` stays clean.
  for src in "$HERE"/src/matrix/*/ "$HERE"/src/{osr,dan,baseline}/; do
    git -C "$src" update-index -q --refresh >/dev/null 2>&1
  done
  for name in $(config_names); do
    gx_prebuild "$HERE/src/matrix/$name" || rc=1
  done
  echo "disk free: $(free_gb) GB"
  return $rc
}

cmd_smoke() {
  local cfg=${1:?usage: matrix.sh smoke CONFIG CASE} case_id=${2:?usage: matrix.sh smoke CONFIG CASE}
  export MATRIX_RESULTS="$HERE/results/matrix-smoke"
  R="$MATRIX_RESULTS"
  load_config "$cfg" || exit 1
  mkdir -p "$R"
  one_attempt "$case_id" smoke
  tail -3 "$R/$C_NAME/$case_id"/a*.log | tail -4
}

cmd_batches() {
  local cfg=${1:?usage: matrix.sh batches CONFIG}
  load_config "$cfg" || exit 1
  listing > "$HERE/matrix/listing.tmp" || exit 1
  mx batches "$HERE/matrix/listing.tmp" "$HERE/matrix/batches.tsv" && rm -f "$HERE/matrix/listing.tmp"
}

cmd=${1:-}
shift || true
case "$cmd" in
  round|_round|status|report|stop|check|prepare|smoke|batches) "cmd_$cmd" "$@" ;;
  *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
