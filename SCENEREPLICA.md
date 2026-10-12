# SceneReplica in sim2 with the localize memory

## What is in `xarm-perception-sim2-dan`

`xarm-perception-sim2` with one module swapped:

- **Perception:** `LocalizeSceneRegistrationModule`, the localize memory (SigLIP frame
  retrieval, OWLv2 boxes, EdgeTAM masks), in place of scene registration (Moondream +
  EdgeTAM). Policy `{"min_views": 1}`, because the arm holds still while it scans.
- **Unchanged:** the xArm 7 and gripper in MuJoCo, the wrist RGB-D camera, the motion
  planner, pick and place with heuristic grasps, and the self filter and voxel map the
  planner uses for obstacles.

## Setup

    git checkout mustafa/feat/scenereplica-dan
    uv sync --python 3.12 --extra manipulation --extra cuda --extra misc
    cargo build --release -p dimos-voxel-ray-tracing --bin voxel_ray_tracing

    # SceneReplica scenes (20 MB, not on LFS)
    dimos data pull d29df7d1c587 --dest scenereplica.tar.gz
    tar -xzf scenereplica.tar.gz -C data/

## Run the blueprint on one scene

    dimos --simulation mujoco --scene-package $PWD/data/scenereplica/scene-01 \
        run xarm-perception-sim2-dan mcp-server

Then, in a second terminal, ask it to find an object:

    dimos mcp call scan_objects --json-args '{"prompts": ["mug"]}'

Scenes are `scene-01` to `scene-20`, five YCB objects each. Set
`SIMULATIONMODULE__VIEWER=true` to open the MuJoCo window.

To change the localize policy (it replaces the whole policy, so keep `min_views`):

    LOCALIZESCENEREGISTRATIONMODULE__POLICY='{"min_views":1,"candidate_floor":0.07,"accept_score":0.1}'

## Run the benchmark

    dimos evals run dimos.evals.suites.scenereplica_dan --agent dimos.evals.agents.pi --tags scene-01

`--tags scene-01` runs that scene's five trials; drop it for all 100. The pi agent needs the
pi CLI and an OpenAI key.
