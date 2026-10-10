# Eval matrix: parallel SceneReplica sim runs on one GPU box

These scripts run many SceneReplica pick-and-place trials in parallel on one GPU machine,
one trial per Docker container, and compare configs (branches plus settings) case by case.
They were built for the closed-loop grasp evals; over two days they ran a few thousand
trials on an AWS `g5.2xlarge` (8 vCPU, 30 GB RAM, one A10G).

## What it does

- **A config** is one line in `matrix/configs*.tsv`: a name, a branch or commit, an eval
  suite, and environment settings (grasp mode, feature switches, which agent).
- **Every config runs the same cases** from a fixed list (`matrix/batches.tsv`: 100
  SceneReplica cases in 10 batches of 10), so results pair up case by case and can be
  compared with a paired test.
- **One trial = one container.** The container mounts the config's checkout read-only on
  `/app`, starts the sim2 stack, runs one case through `dimos evals run`, grades it, and
  exits. A crashed or stuck trial cannot affect the next one.
- **Slots:** a round runs `SLOTS` containers at once (default 3), all configs sharing one
  queue, so no slot waits for another config's slowest trial.
- **Rounds and chains:** `matrix.sh round N` runs batch N. `auto.sh start STEP ...` runs a
  list of rounds unattended, survives ssh disconnects, and can switch between config sets
  (`set:NAME`).
- **One simulator for every config:** `matrix/sim-overlay.txt` names a branch whose sim
  files are copied over every checkout at prepare time. Sim fixes then reach all configs
  without being merged into the branches under test.
- **Scripted agent:** `matrix/lib/matrix_scripted.py` is a model-free agent (scan, pick,
  place) loaded from this folder, so every branch runs the identical script. Without an LLM
  in the loop, differences between configs come from the robot code alone. The LLM agent
  (`dimos.evals.agents.pi`) is the default when a config doesn't set `MATRIX_AGENT`.

## Files

| File | What it is |
|---|---|
| `matrix.sh` | The runner on the box: `round`, `status`, `report`, `stop`, `check`, `prepare`, `smoke`. Read its header for every setting. |
| `auto.sh` | Runs a chain of rounds in the background (`start`, `stop`, `detach`, `status`). |
| `ship.sh` | Laptop side: sends the branches the configs name to the box as a git bundle (nothing is pushed to GitHub), then prepares checkouts. |
| `pull.sh` | Laptop side: copies results off the box and moves recordings off it to free disk. |
| `progress.sh` | Laptop side: progress bars for a config set, refreshed every 30 s. |
| `monitor.sh`, `monsum.py` | Sample CPU, RAM and GPU per container while trials run, then summarise. |
| `matrix/lib/matrix_case.py` | A one-case suite, so a container runs exactly one case. Also caps agent time per trial. |
| `matrix/lib/matrix_scripted.py` | The scripted pick agent (`--set attempts=3` retries the pick). |
| `matrix/lib/export.py`, `pick_report.py`, `classify.py`, `stats.py` | Read results: per-trial rows as JSON, pick outcome counts per config, failure stages, paired statistics. |
| `matrix/batches.tsv` | The 100 cases in 10 batches. A set can bring its own list as `matrix/batches-NAME.tsv`. |
| `matrix/configs-*.example.tsv` | Real config sets from the grasp work, as examples of the format. |
| `docker/` | The eval image: `Dockerfile.sim2` (sim2, MuJoCo on EGL, the voxel mapper, the pi CLI), then `Dockerfile.graspgenx` on top (a ready GraspGenX environment). |

## Setting up a box

The box was set up by hand. What it needs:

1. **Host:** an NVIDIA GPU, Docker, and the NVIDIA Container Toolkit. Pin the NVIDIA
   packages (`apt-mark hold`). An unattended driver upgrade once broke every container
   until the kernel module was reloaded.
2. **This folder** at `EVALS_DIR` on the box (default `/home/ubuntu/dimos-evals`), plus:
   - `src/store.git`: a bare git repo that `ship.sh` sends branches into;
   - `src/osr`: one dimos checkout to build the image from (and its voxel-mapper binary);
   - `src/baseline`: a checkout with `data/` extracted and `.git/lfs`, shared into every
     config's checkout by hard links;
   - `cache/`: mounted into every container (uv cache, the GraspGenX venv link, compiled
     Python bytecode);
   - `secrets.env`: environment for the containers (API keys for the LLM agent; the
     scripted agent needs none).
3. **The image:** `docker/build.sh`, then `docker/build_graspgenx.sh`.
4. **Laptop:** an ssh alias for the box (`EVALS_HOST`, default `dimos-evals`). Then
   `./ship.sh` to send branches and prepare checkouts.

Then:

```bash
./matrix.sh smoke CONFIG scenereplica-scene-02-mug     # one trial, not counted
SLOTS=4 ./auto.sh start set:lab prepare 1 2 3 4 5 6 7 8 9 10
./progress.sh lab                                        # on the laptop
sudo python3 matrix/lib/pick_report.py lab --pairs A:B   # on the box
```

## What we learned about running sims in parallel

- **CPU is the limit, not the GPU.** A trial uses about 1.8–1.9 cores while it runs, with
  or without camera perception. The `g5.2xlarge` has only 4 real cores (8 vCPU), so 4 slots
  is its ceiling (load 8–10). The GPU was never the limit: 0.8 GB per trial with perception
  from sim truth, 5.7 GB with the camera models. RAM is about 5–6.5 GB per trial, which caps
  that box at 5 slots.
- **Most CPU goes on things other than the robot moving.** About 70% of a trial's CPU is
  spent before the arm moves:
  - Python start-up;
  - loading GraspGenX;
  - removing the arm from the camera's point cloud (trimesh distance queries, about 29%).

  A 30–80 s trial spends 15–25 s actually picking.
- **Cheap wins:**
  - keep compiled bytecode between containers (`PYTHONPYCACHEPREFIX` on a shared cache,
    one folder per commit): about 15 s less start-up per trial;
  - turn off viewers nobody watches (`MANIPULATIONMODULE__VISUALIZATION__BACKEND=none`).
- **Speed-ups can change results.** A faster arm-removal filter in the same process as the
  100 Hz control loop tripled "Trajectory start differs" refusals: 6 in 100 trials vs 20 in
  70, and 1 vs 7 on the same 20 cases. Check any speed-up against the old path on the same
  cases before trusting it.
- **Trials can hang at start-up.** The harness waits on the stack with no time limit when
  `Dimos.connect()` gets no reply. `matrix.sh` stops a trial that makes no tool call within
  `STARTUP_TIMEOUT_S` (120 s) and retries it, and caps scripted trials at
  `SCRIPTED_TIMEOUT_S` (240 s).
- **Time limits interact with load.** A pick that takes 17 s on a laptop took 120+ s on the
  loaded box and hit the 120 s pick RPC limit. Measure slow skills on the box itself, alone,
  before queueing them.
- **Don't change a set's setup partway through.** Keep commit, filters and slots the same
  within a set, or comparisons mix setups. Put a change at a set boundary, or add a bridge
  config that reruns a baseline on the new setup.
- **Disk:** each trial leaves about 20 MB of recordings plus logs; `pull.sh` moves them off.
- **Physics stays on the CPU.** MuJoCo runs in real time per stack; GPU physics (MJX/Warp)
  would only pay off with batched simulation, which this harness doesn't do. Rendering
  already runs on the GPU (`MUJOCO_GL=egl`).
