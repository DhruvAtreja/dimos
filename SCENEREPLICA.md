# SceneReplica in sim2, with the localize memory as perception

This branch runs the SceneReplica tabletop benchmark (arXiv 2306.15620) in the sim2 MuJoCo
simulator on the xArm, with your localize memory (SigLIP retrieval, OWLv2, EdgeTAM, fused
sightings) as the perception stack. It is `main` from 8 October plus:

- sim2 and the SceneReplica suite: 20 scenes of five YCB objects, 100 trials.
- A sim fix that makes the xArm hold its commanded pose under gravity.
- `xarm-perception-sim2-dan`: the stock sim blueprint with `LocalizeSceneRegistrationModule`
  in place of the default scene registration. Policy `{"min_views": 1}`, because the arm
  holds still while it scans.
- `dimos.evals.suites.scenereplica_dan`: the same 100 trials on that blueprint, so its rows
  pair case by case with the default suite.
- `scripts/eval-matrix/`: the scripts we used to run thousands of trials on an AWS GPU box
  (see its README).

## Setup

1. Check out this branch. `data/.lfs/sim2.tar.gz` comes through git LFS as usual.
2. The SceneReplica scenes are not on LFS; they are on Dimensional cloud (20 MB). Pull them
   and unpack into `data/` (run `dimos login` first if asked):

       dimos data pull d29df7d1c587 --dest scenereplica.tar.gz
       tar -xzf scenereplica.tar.gz -C data/      # creates data/scenereplica/

3. Environment. The trials we report ran in the Docker image from
   `scripts/eval-matrix/docker/` (`build.sh`), which is the easiest way to match them. To run
   on your machine instead, you need Linux with an NVIDIA GPU, and:

       uv sync --python 3.12 --extra manipulation --extra cuda --extra misc
       cargo build --release -p dimos-voxel-ray-tracing --bin voxel_ray_tracing
       export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl VIEWER=none

   The LLM agent also needs the pi CLI (`npm install -g @earendil-works/pi-coding-agent@0.99.2`)
   and a model key (its default model is OpenAI's).

## Running

One scene (five trials) with your perception, then the summary:

    dimos evals run dimos.evals.suites.scenereplica_dan --agent dimos.evals.agents.pi --tags scene-01
    python -m dimos.evals.suites.lib.scenereplica_summary ~/.local/state/dimos/evals/run-*

The same trials with the default sim perception, for comparison:

    dimos evals run dimos.evals.suites.scenereplica --agent dimos.evals.agents.pi --tags scene-01

Without an LLM, use the scripted agent we used for all comparisons (scan, pick, place, the
same script for every config):

    PYTHONPATH=scripts/eval-matrix/matrix/lib dimos evals run dimos.evals.suites.scenereplica_dan \
        --agent matrix_scripted --tags scene-01

Override the localize policy with an environment variable. It replaces the whole policy, so
repeat `min_views`:

    LOCALIZESCENEREGISTRATIONMODULE__POLICY='{"min_views":1,"candidate_floor":0.07,"accept_score":0.1}'

Each trial is graded from the simulator's ground truth: half for lifting the object 5 cm,
half for leaving it at rest in the drop-off square. A trial whose target the scan never
reports counts as "target not found".

## What we found

We ran your perception once over all 100 trials, inside our newer grasping stack: GraspGenX
grasps, contact check, depth search, two extra views before grasping, objects fused at 3 mm.
That stack is not on this branch, so your grasp numbers here will differ. The perception
side is the same. Paired against the default sim perception in the same stack:

| | Target found | Grasped | Placed |
|---|---|---|---|
| Default sim perception (OWLv2 + YOLO-E) | 81 | 41 | 19 |
| Localize memory | 37 | 8 | 4 |

**Why targets were missed.** It is the thresholds, not the pipeline. We re-scored OWLv2 on
the wrist-camera frames recorded at each scan, with each trial's own prompt:

- All 47 trials where no 3D group formed had a best OWLv2 score under 0.25, your
  `candidate_floor`, so no box was ever made. 36 of them scored above 0.07.
- All 14 trials rejected at verification scored between 0.25 and 0.40 (`accept_score`).
- Every trial that scored above your thresholds was found, except two. Those passed
  verification but were dropped by `scan_scene`'s rule that a hit must be seen in a frame
  taken after the scan started.

Rendered YCB boxes and cans score far below real photos. Median best score: banana 0.83,
scissors 0.79, drill 0.66, marker 0.59, mug 0.45, clamp 0.42, bowl 0.35. Below those:
sugar box 0.19, mustard 0.19, cracker box 0.18, soup can 0.15, pudding box 0.13,
bleach 0.13, gelatin box 0.11, potted meat can 0.09, tuna can 0.05. The default sim
perception uses the same OWLv2 model with `detector_confidence=0.07` for this reason
(`dimos/robot/manipulators/xarm/blueprints/grasp.py`, `_scene_registration`). Your policy
keeps its real-camera defaults (0.25 and 0.40).

**Not explained yet.** On the 35 trials both stacks found, the default grasped 21 and yours
7. The object clouds differ: after the extra views the median cloud was about 1,200 points
for yours and 3,000 for the default. We have not looked further.

**Suggested first run:** the policy override above with `candidate_floor` 0.07 and
`accept_score` around 0.1, on a few scenes heavy in boxes and cans (scene-01, scene-19,
scene-20). Then check for false detections, which lower thresholds can bring.

## Notes

- The only camera is on the wrist. The arm scans from a fixed pose above the table.
- `potted meat can` defeats both perception stacks (0 of 7 found by either).
- The placed grader treats tiny simulator resting motion as "still moving" and marks about
  1 in 8 correctly placed objects as not placed. Grasp numbers are not affected.
