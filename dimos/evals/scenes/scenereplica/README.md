# SceneReplica scenes

The 20 tabletop scenes of the SceneReplica benchmark
(<https://github.com/IRVLUTD/SceneReplica>, IRVL, UT Dallas) as sim2 scene
packages: five YCB objects on a cafe table, 100 object placements in total,
drawn from 16 objects (cracker box, sugar box, tomato soup can, mustard bottle,
tuna fish can, pudding box, gelatin box, potted meat can, banana, bleach
cleanser, bowl, mug, power drill, scissors, large marker, extra large clamp).

The converted scenes ship in the `scenereplica` LFS data package
(`data/.lfs/scenereplica.tar.gz`, 20 MB), which unpacks to:

```
data/scenereplica/
  _assets/<ycb id>/      visual.obj, texture.png (1024^2), collision_NN.obj, one folder per object
  _assets/manifest.json  mass, collision method and piece list per object
  scene-01 .. scene-20/  scene.xml (world only, no robot) + scene.json (dimos.scene.v1)
  reach_report.json      top-down reach of every object from the authored spawn
```

`scene-NN` follows SceneReplica's own order (`scene_ids.txt`); each
`scene.json` names the source scene id and metadata file in `provenance`.

## Run

```bash
MUJOCO_GL=egl dimos --simulation mujoco --transport zenoh \
  --scene-package data/scenereplica/scene-01 \
  run xarm-perception-sim2 mcp-server observe-skill --simulationmodule.viewer=false
```

`scene.json` carries the entities (`cracker_box`, `bowl`, ... with detection
labels such as "cracker box"), a `table/top` support region, a `dropoff/top`
support region on the near-left corner of the table, the settled `initial`
poses and a `workbench` spawn, so the xArm7 and its planner stand in the right
place without `--scene-spawn`.

## Frames and numbers

SceneReplica gives object poses in the Fetch `base_link` frame: x forward, y
left, z up, origin on the floor under the robot, quaternions as `[w, x, y, z]`.
Our world frame is that same frame, so positions copy across unchanged and only
the quaternion order changes to `[x, y, z, w]`.

| Item | Value |
|---|---|
| Table top | 0.913 m square (SceneReplica's `cafe_table_org`), top at z = 0.745 m, centred at x = 0.80 m |
| Objects | x 0.48..0.68 m, y -0.26..0.26 m (SceneReplica's 7x7 grid) |
| Arm base (`workbench` spawn) | (0, 0, 0.90) m on a pedestal, 0.155 m above the table top |
| Drop-off | `dropoff/top`, 0.14 m square centred (0.42, 0.38) on the table |

The spawn height came from a reach sweep (`convert.py reach --sweep`): a
top-down hover 0.10 m above each object's highest point, solved with a
damped-least-squares IK on the attached xArm7 and rejected on arm/table
penetration. Reachable hover poses out of 100: 0.65 m 85, 0.70 m 89, 0.745 m
(level with the table) 91, 0.80 m 94, 0.85 m 96, **0.90 m 99**, 1.00 m 97.
The one miss at 0.90 m is scene-11's upright cracker box in the far-right
corner (hover at 1.06 m, 1.6 cm short). SceneReplica's own drop-off point,
(0.78, 0.40), is out of the xArm7's reach from the origin, hence the nearer
corner.

Objects are placed as SceneReplica recorded them after settling in Gazebo,
lowered or raised by at most 2.4 mm so the lowest collision vertex touches the
table top, then settled for 2 s of MuJoCo physics (the marker rolls up to
1.3 cm; everything else moves under 2 mm). `convert.py settle` re-steps every
shipped scene for 2 s: the worst drift is 0.5 mm.

## Object conversion

`convert.py` takes each object's `textured_simple.obj` and texture from
SceneReplica's `models.zip`. Collision shapes: one convex hull for the eight
near-convex objects (boxes, cans, marker); CoACD convex decomposition for the
mustard bottle (4 pieces), banana (4), bleach cleanser (3), bowl (24), mug (24),
power drill (18), scissors (14) and extra large clamp (24), so the bowl, mug
and clamp are hollow where it matters for a grasp. Masses come from the
objects' Gazebo SDFs; inertia is the convex hull's, scaled to that mass.
Collision geoms sit in MJCF group 3, which `scene.json` hides from cameras.
Every scene uses elliptic friction cones with `impratio` 10, as the sim2
workbench scene does. Under MuJoCo's default pyramidal cones an object held by
the xArm7 gripper slides out of the fingers while the arm lifts it.

## Regenerating

```bash
# Downloads final_scenes.zip (16 MB) and models.zip (1.7 GB) into ~/.cache/dimos/scenereplica
python -m dimos.evals.scenes.scenereplica.convert convert
python -m dimos.evals.scenes.scenereplica.convert settle
python -m dimos.evals.scenes.scenereplica.convert reach
tar -C data -czf data/.lfs/scenereplica.tar.gz scenereplica
```

`--scenes` and `--models` accept the archives or their extracted folders;
`--reuse-objects` rewrites the scenes without redoing the decomposition.

## Licence

SceneReplica's code and scene data are MIT licensed (IRVL, UT Dallas, 2023).
The YCB object meshes are redistributed by SceneReplica from the YCB Object and
Model Set (<https://www.ycbbenchmarks.com>); their terms are the YCB
benchmark's, which have not been reviewed here. Verify before shipping the
meshes outside the team.
