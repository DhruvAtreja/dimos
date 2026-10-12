# sim2 Hardware Emulator

The existing G1 GR00T and xArm7 planner blueprints can use `sim2` in this branch.
Their controllers remain ordinary DimOS modules. MuJoCo owns physics; the
coordinator uses typed command/state streams to generic sim2 connections.
Shared memory stays internal to the simulated devices and sensor snapshots.
Cameras and lidar run independently of physics.
PimSim, task catalogs and population preparation are not involved.

## Connection Integration Branch

`pim/feat/sim2-connections` combines native sim2 core `65cd8b8e1` and Mustafa's
connection/coordinator stack through #4426, `8b91cbaba`. Merge `818027126`
contains those foundations; subsequent commits are this integration. Neither
foundation branch was rewritten. This is not a claim that the dependencies
have merged into main. SONIC and vendor DDS/SDK emulation are unchanged.

The new path is:

```text
Navigation / manipulation
        | Twist / trajectory goals
        v
ControlCoordinator: GR00T policy or joint trajectory task
        | typed MotorCommandArray or JointState
        v
SimMotorConnection or SimManipulatorConnection: ConnectionModule
        | internal shared-memory device
        v
SimulationModule: one MuJoCo world
        | shared-memory snapshots
        +--> camera worker --> images
        +--> lidar worker  --> point clouds
```

Feedback travels from physics through the connection's standard `joint_state`
and `imu` ports to the coordinator. `describe_control()`, `status()` and
`halt_robot()` are the same connection contract as upstream. The coordinator
no longer imports, registers or creates a sim2 hardware adapter.

## What Runs

The application remains a normal DimOS blueprint. sim2 replaces its devices,
not its navigation, perception, planner or policy. Three loops do the work:

1. **Control:** the existing ControlCoordinator reads joint/IMU feedback,
   runs its policy or trajectory controller and publishes commands to its
   connection. G1 runs this at 50 Hz; xArm at 100 Hz. The connection validates
   commands and owns the internal shared-memory device.
2. **Physics:** one world worker reads the latest motor targets from shared
   memory, applies them and steps MuJoCo. It publishes joint/IMU feedback and
   a world-state snapshot. The current blueprints request 200 Hz, paced against
   wall time; they do not wait for each control or sensor message.
3. **Sensing:** dedicated camera and lidar workers read the latest world
   snapshot and render or raycast using their own query model. G1 requests
   640x480 RGB-D and lidar at 10 Hz. Results use ordinary DimOS typed streams
   for mapping, perception and visualization, which feed the next commands.

Those rates are configured targets, not hard real-time guarantees. Scene
inspection, editing and reset are RPCs on the world owner, not a fourth task
engine. Robot definitions describe models, actuators and mounted devices;
they do not implement another control loop.

Physics starts at the existing `on_system_modules` graph-ready notification,
after devices and sensors finish starting. This prevents a humanoid falling
while a camera, lidar or planner is still initializing. It is not lockstep
control: subsequent physics ticks still use the latest accepted command.

This is why the code is a collection of DimOS modules: the simulator emulates
independently scheduled devices consumed by an existing robotics application.
Native MuJoCo provides physics, rendering and raycasting. Keeping policies and
application lifecycle in DimOS avoids imposing another framework's robot,
controller and task lifecycle on that application.

## Run

```bash
uv run dimos --simulation mujoco --transport zenoh --viewer rerun --scene-package kitchen run unitree-g1-groot-wbc
uv run dimos --simulation mujoco --transport zenoh --scene-package kitchen run xarm7-planner-coordinator
```

Run one stack at a time on the default transport bus. Both open the native
MuJoCo viewer. Disable it with the module override
`--simulationmodule.viewer=false` after the blueprint name. A scene name, an
absolute XML path, or a directory containing `scene.xml` uses the same loader.
Defaults without `--scene-package` remain the small logistics/workbench scenes.

The first download of existing robot meshes and GR00T policies is separate
from measured startup. Install the existing simulation and robot dependencies.
The simulation extra requires MuJoCo 3.10 or newer for batched raycasting.

This branch uses stock Zenoh 1.10.1; no locally patched wheel or router is
required for the verified local runs. Bounded checks use the real blueprints:

```bash
uv run python -m dimos.sim2.demo_smoke g1 --viewer --seconds 15 --move --reset
uv run python -m dimos.sim2.demo_smoke xarm --viewer --seconds 10 --move --reset
```

Add `--rerun` to include the Rerun bridge and viewer. These checks use the same
configuration parser as the CLI before deploying workers. Omit `--move` to
leave the robot holding its starting pose. `--reset` also exercises the normal
application reset over RPC and verifies resumed physics and the reset pose.

October 11 local headless connection-path checks, with assets downloaded,
stock Zenoh peers and the default logistics/workbench scenes:

| Blueprint | Startup | Real-time factor | Reset RPC sequence | Shutdown |
|---|---:|---:|---:|---:|
| G1 GR00T, 12 s motion | 3.66 s | 0.99996 | 8.18 ms | 0.096 s |
| xArm7 planner, 5 s motion | 14.07 s | 0.99952 | 5.93 ms | 0.076 s |

G1 advanced about 1.2 m and remained upright after reset; xArm reached a
0.25-radian joint change and returned to its reset position, not its old
target. Both published 640x480 RGB-D; G1 also published lidar. Reset timing is
the helper's synchronous RPC sequence, not time until the first new sensor
frame or settled pose. Each run separately checked two seconds of resumed
physics. These are short smoke measurements, not latency distributions,
sustained/cross-machine guarantees, or proof that streams beat direct SHM.
The old direct-adapter p99 measurements do not apply to this new path.

Focused verification: 167 control/connection/task/blueprint tests; 27 native
MuJoCo runtime/scene tests; 125 coordinator/registry/teleop regressions passed
(two hardware-marked cases deselected). Mypy passed for the 15 touched runtime
entrypoints. This was not a full repository test run. Initial xArm TF/state
warnings and macOS renderer depth warnings remain.

## Rolling Mid360

The G1 comparison preset changes only the lidar on the existing GR00T stack:

```bash
uv run dimos --simulation mujoco --transport zenoh --viewer rerun run unitree-g1-groot-mid360
```

Use `unitree-g1-groot-wbc` for the original ideal lidar. Both accept the same
`--scene-package` and viewer options. The Mid360 preset binds the existing
`mid360_link` asset site, including its inverted G1 mounting orientation; it
does not crop the scan's vertical field of view.

`Mid360` uses a continuous four-channel, two-rotor Fourier firing pattern.
Passive G1 hardware measurements exposed the opposite sweep direction in
the previous Livox reference table; that table and its artificial four-second
repetition have been replaced, not retained as another sensor mode.
The default is 20,000 emitted rays per 100 ms scan, with scene/robot motion
reconstructed in 200 Hz bins. The attributed data ships separately as the
small `mid360_pattern` data archive. Optional `model_kwargs`
such as `{"downsample": 4}` retain every fourth complete four-laser group.

The existing lidar worker exposes these streams, without an additional worker:

- `pointcloud`: truth-corrected returns for the existing mapper, in the
  configured world or scan-end sensor frame.
- `raw_pointcloud`: uncorrected acquisition-time sensor-frame returns, with
  `offset_time` in nanoseconds from the scan-start message timestamp and
  `line` identifying the laser channel. Misses are omitted, not fabricated.
- `imu_raw`: the device's optional IMU, with angular velocity in rad/s and
  specific force in m/s^2. No truth orientation is supplied. Both raw streams
  use the same acquisition clock, not callback arrival time. The G1 policy's
  existing control IMU remains separate.

Read raw data through the typed stream; the generic RPC `peek_stream` path
pickles point clouds and currently discards their extra per-point fields.
This port does not change that shared message implementation.

Only timed lidar enables bounded history on the existing world snapshot
channel (42 slots at G1's default settings). Motor channels remain double
buffered. Incomplete history delays a scan; reset boundaries never mix
episodes. `g1_lidar.sensor_status()` reports captured/dropped scans, ray and
return counts, last capture time and history availability. Restart the stack
after updating: the internal shared-memory layout changed.

One Mid360 combines Andrew's #4441 Fourier coefficients with his
range/incidence-dependent noise and grazing-angle dropout response. Set
`model_kwargs={"noise": False, "dropout": False}` for geometry diagnostics;
this does not select a different scanner. Noise is seeded per scan, so a
missed scan does not shift later samples. Two rotor rates were fitted on one
G1 capture and validated on a separate recording, with the shape coefficients
unchanged. See the [hardware comparison](/experiments/mid360/README.md)
for method, angular residuals and limitations. This is not a claim that every
Mid360 has identical calibration.

Self occlusion uses this robot's geometry, not a Go2 blindspot map. The G1
explicitly excludes its `head_link` mesh, which seals the optical window and
otherwise blocks every ray within 2-7 cm. Other robot links still occlude.
This also omits head-shell occlusion: separating the optical window in the
asset is required before claiming physically complete mounting fidelity.
Reflectivity, material response and IMU noise/bias are not calibrated here.

To run the **existing native Point-LIO estimator** on the same sensor:

```bash
uv run dimos --simulation mujoco --transport zenoh --viewer rerun run unitree-g1-groot-mid360-pointlio
```

This composition connects `raw_pointcloud -> lidar_raw` and `imu_raw` to
Point-LIO. Its corrected `lidar`, `odometry` and `tf` feed the existing
ray-tracing mapper and A* navigation. The device's truth-corrected cloud and
truth TF are disabled; robot truth odometry is separately named and not
connected to navigation. There is no truth fallback. The simulated lidar IMU
is colocated, so the blueprint explicitly configures zero translation and
identity rotation extrinsics rather than the hardware driver's offset.

The estimated map uses Point-LIO's local `odom` frame, not the authored scene
origin. The viewer uses the existing hardware odometry visualization. Reset
or teleport requires restarting this estimator run: upstream Point-LIO does
not expose filter reset yet. Native binaries build through the existing
NativeModule mechanism (Cargo/Rust is required on a cold installation).

`g1_lidar.sensor_status()` includes IMU sample/gap counts and a clearly named
`truth_pose` diagnostic. Only the measurement probe reads that truth; it is
not a sensor stream or an estimator input. Run
`python -m dimos.sim2.demo_pointlio --move --seconds 15` for a headless
end-to-end acquisition and estimator comparison with walking and turning.

Historical evidence from the pre-integration branches, verified October 6 on
the included logistics scene, headless with GR00T,
RGB-D, actual native Point-LIO and ray-tracing mapping active:

| 15-second moving run | Native sim2 | Robosuite sim2 |
|---|---:|---:|
| Real-time factor | 1.0002 | 1.0001 |
| Raw scans / IMU samples per second | 10.00 / 199.97 | 9.98 / 199.89 |
| Captured scan / IMU drops | 0 / 0 | 0 / 0 |
| Maximum observed IMU interval | 5 ms | 5 ms |
| Position error RMS / maximum | 5.7 / 8.8 mm | 4.2 / 11.0 mm |
| Maximum orientation error | 0.125 degrees | 0.102 degrees |
| Last full-rate scan capture | 17.5 ms | 16.6 ms |

The probe aligns the first estimated sensor pose once, then compares scan-end
poses within 25 ms. Both walked about 1.2 m and turned. These are short
integration checks, not long-run drift, hardware calibration, click-navigation
or all-scene performance guarantees. The initial lazy Open3D load is completed
before acquisition so it cannot create a gap midway through IMU streaming.

An October 6 native G1 check with walking, RGB-D and mapping active measured
0.9998 real-time factor, about 9.8 scans/s and 14 ms for the last full-rate
capture. Startup was 6.24 s and shutdown 0.23 s. This short headless run is
not an all-scenes performance guarantee; startup also skipped scan windows.
Reproduce with `python -m dimos.sim2.demo_smoke g1 --mid360 --move --seconds 6`.

## Included Scenes

The eight populated scenes ship together in the existing `data/.lfs/sim2.tar.gz`
data package. Shared mesh/texture files live once in `scenes/_assets`; no
PimSim install, bundle-path environment variable, or cooking step is needed
to run them. The existing DimOS LFS mechanism obtains/extracts the archive.

| Scene name | Named entities | Movable bodies | Fixture joints | Named spawn supports |
|---|---:|---:|---:|---|
| `kitchen` | 22 | 5 | 1 | `default`, `workbench` |
| `libero-kitchen-1` | 14 | 2 | 3 | `default` |
| `libero-kitchen-9` | 15 | 3 | 1 | `default`, `workbench` |
| `robocasa-kitchen-1` | 47 | 3 | 45 | `default` |
| `robocasa-kitchen-7` | 124 | 3 | 100 | `default` |
| `ithor-kitchen` | 84 | 28 | 25 | `default` |
| `procthor-house` | 90 | 51 | 15 | `default` |
| `hssd-home` | 232 | 0 | 0 | `default` |

HSSD is a furnished rigid navigation scene. Its furniture is not graspable.
The RoboCasa entries include three added movable mesh objects on an authored
counter. ProcTHOR is a multi-room house. These are scene imports, not claims
of passing the upstream benchmarks. Imported region labels are retained;
`kitchen` also contains explicitly authored regions for scene-control examples.

Scene spawns are support poses, independent of robot identity.
G1 selects `default` and adds its robot definition's `spawn_height` (0.793 m).
An arm selects `workbench` with zero offset. For example, a floor at Z=-1.5
places G1's root at -0.707.
Missing named supports fail clearly; an arm is never silently placed on a
floor. Raw scenes without spawn metadata use the blueprint's explicit support
default. Direct `RobotInstance` and live pose edits still use absolute root
poses. Scene metadata uses this single convention.

Old `office` is not an alias for one of these scenes: its
legacy collision wrapper still needs a separate visual/entity conversion.

```python
from dimos.sim2.scene import list_scenes

print(list_scenes())
```

The supplied scenes are finished native files. Edit their `scene.xml` and
`scene.json` directly; no preparation script or PimSim source library is
required. Source provenance is retained in each `scene.json`; source content
licensing remains subject to the original datasets' terms.

## Configure A Robot

Robot-local definitions live in:

- `dimos/robot/unitree/g1/sim2.py`: model, joint order, gains and named sensor bindings.
- `dimos/robot/manipulators/xarm/sim2.py`: native servos, gripper units and camera.

An existing blueprint selects simulated devices or real devices. It keeps its
controller, planner, perception and navigation modules:

```python
from pathlib import Path

from dimos.core.coordination.blueprints import autoconnect
from dimos.control.coordinator import ControlCoordinator
from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.xarm.sim2 import XARM7
from dimos.sim2.blueprint import simulation
from dimos.sim2.spec import RobotInstance

devices = simulation(
    scene=Path("/absolute/path/to/scene.xml"),
    sim_id="workbench",
    robots={"arm": RobotInstance(XARM7, xyz=(0, 0, 0.12))},
)

class ArmCoordinator(ControlCoordinator):
    arm_position_command: Out[JointState]
    arm_joint_state: In[JointState]


app = autoconnect(
    devices,
    ArmCoordinator.blueprint(
        tasks=[joint_trajectory_task([*(f"arm/joint{i}" for i in range(1, 8)), "arm/gripper"])],
    ),
)
```

Adding a robot with supported controls/sensors means adding its `sim2.py`
definition and changing its existing blueprint's device selection, plus a
robot contract test and assets. There is no central robot-name switch.

Stock sensors bind named cameras or sites in the native asset. Their positions
and orientations are not repeated in Python. A named camera's field of view
comes from its compiled asset. Select output size, rate and depth in Python:

```python
from dimos.sim2.sensors.spec import Camera

XARM7_SMALL_RGB = XARM7.with_sensor(
    Camera("wrist_camera", camera="wrist_camera", width=320, height=240, depth=False),
)
```

G1 uses `Imu("imu", site="control_imu")` and a lidar bound to `mid360_link`.
These sites are explicit in the MJCF;
G1's base-frame policy IMU is distinct from the torso hardware IMU.

An additional sensor can explicitly define a new mount on an existing body:

```python
from dimos.sim2.sensors.spec import Camera, Mount

XARM7_FRONT = XARM7.with_sensor(
    Camera("front", Mount("link_base", xyz=(0.1, 0, 0.3)), depth=False),
)
```

Names select sensor instances. A missing named camera/site/body fails during
composition, never triggering a replacement attachment. `Mount` means an
explicitly added device, not an alternate interpretation of a missing name.
Mount rotations use roll/pitch/yaw radians in the named body's local frame;
cameras use MuJoCo's -Z viewing direction and publish an optical-frame TF.
RGB-only and RGB-D modules have different declared ports. Repeated cameras
use `robot/sensor/port` names; multiple robots also namespace device ports.

`simulation()` returns an ordinary `Blueprint` declaring the world and
devices, not a wrapper with a parallel hardware configuration. `RobotConfig`
is data; its `describe_control()` supplies resource names, units and limits
to the connection. The coordinator discovers live connections after startup.
G1 uses four
emulator modules (physics, connection, camera, lidar); xArm uses three.
Physics, connections and camera/lidar have dedicated workers. The connection
worker follows upstream `ConnectionModule` placement. Each sensor worker holds a
local model/data copy, so isolation has a memory and state-reconstruction cost.

The existing GR00T, xArm7 planner and coordinator-xarm7 entrypoints
use this path. Old G1 vendor-action and other xArm perception/room/teleop,
xArm6 and Piper simulator paths are not yet all migrated. They are not a
fallback inside the migrated blueprints. The G1 vendor-action capability
requires an explicit retirement or preservation decision before replacing it.

Lidar configurations reference a concrete model factory, not an instance or
registered name. The worker constructs it once using its settings:

```python
from dimos.sim2.sensors.lidar.models.fibonacci import Fibonacci
from dimos.sim2.sensors.spec import Lidar

Lidar("lidar", "mid360_link", model=Fibonacci, model_kwargs={"ray_count": 15000})
```

New ideal ray patterns implement the `RayPattern` contract; they need no
model-name registry. World, robot and sensor settings are typed dataclasses.
The normal blueprint parser merges their settings as dictionaries, and module
configuration reconstructs the declared types. Python factory references
survive this process, just like custom IK solver classes elsewhere in DimOS.
The physics and connection module configs reconstruct the same `RobotConfig`.
There is no separate simulator configuration parser or adapter registration.

G1 uses `models.fibonacci.Fibonacci`: the previous PimSim 15,000-ray pattern
at 10 Hz, mounted at the calibrated upside-down MID360 pose. Its explicit
`maximum_world_elevation=0.0` discards upward rays after the mount transform.
This is an ideal mapping scan, without MID360 scan timing or noise. G1's
simulation costmap no longer forces a disk around world origin to be free.

## Runtime Ownership

`SimulationModule` owns one continuously stepping `SimulationRuntime` and
disposable model snapshot. Each camera/lidar worker loads its own query model
and receives stamped integration-state frames, not a new scene per frame.
All scene geom groups are visible to camera rendering; lidar excludes only
its own robot subtree. The native viewer reads the same state snapshots.

The whole-body channel contains complete position, velocity, gains and
feed-forward torque. The connection reads joints and IMU from one sample and
publishes them with a shared timestamp. The coordinator pairs these stamped
messages before making the PD robot ready for control. Native xArm servos retain their original actuator model;
the gripper retains the hardware API's 0-850 units. No second PD is applied.

The coordinator's `motor_gains={"g1": WholeBodyConfig(kp=..., kd=...)}` supplies
explicit gains in the description's motor-joint order. Position tasks produce
complete q/dq/kp/kd/tau frames; velocity and torque modes disable competing
servo terms. Missing or invalid PD gains fail during connection discovery.
This adds fixed-gain PD support, not a new task API for arbitrary changing gains.

The standard connection deadman handles lost coordinator commands. G1 halt
uses damping; xArm halt holds its measured pose with servos enabled. Damping
does not guarantee that a standing humanoid remains upright. A connection
process crash is also not covered by this deadman: the retained physics
reader has no independent command-age watchdog. Do not interpret this as
verified hardware emergency-stop behavior.

### Performance Boundaries

The previous MuJoCo engine performs camera rendering and lidar raycasting
inside its simulation loop; a separate publisher thread does not remove that
work from physics. sim2 moves those operations into dedicated workers. A slow
frame no longer directly holds up a physics step, although CPU/GPU contention
can still reduce throughput. Lidar uses native batched `mj_multiRay` queries.

The cost of this isolation is one model/data copy per sensor worker and state
snapshot transfer/reconstruction. Motor shared memory does not mean images
and point clouds are zero-copy end to end. Real-time pacing is not lockstep
determinism or a faster-than-real-time training scheduler. The ideal lidar
does not establish MID360 timing/noise or Point-LIO fidelity.

The pre-integration [matched G1 benchmark](/experiments/sim2_timing/README.md) measures
the old engine, current sim2 and a benchmark-only two-worker sim2 arrangement
on an M4 Max. Under sensor load both sim2 arrangements maintained about 200 Hz
physics while the old inline-sensor loop slowed down. Compact placement had
similar timing and used less memory; dedicated processes per sensor are not
established as optimal by those results. Kitchen lidar missed its requested
20 Hz even though physics stayed real-time. The report includes sensor age,
command consumption, incomplete private-memory accounting and an OpenMP
CPU/latency tradeoff rather than claiming unconditional superiority.
Its direct-adapter harness is retained as historical research, not migrated
to this branch's connection API; reproduce it on native core `65cd8b8e1`.

Multiple cameras/robots, viewer/mapping load, other hardware, the ten-minute
30 Hz camera gate, repeated-reset latency and a measured real/sim device swap
remain separate acceptance work. The short comparison does not close them.

## Scene Interface

Use the existing `Dimos.connect()` interface (or the same module proxies in
`dimos shell`). No separate simulator client or session object is required.
All poses use metres and world coordinates; quaternion order is XYZW.
Joint angles use radians, slide-joint positions use metres.

```python
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.porcelain.dimos import Dimos
from dimos.sim2.interaction import reset_scene
from dimos.sim2.scene_types import SceneUpdate

app = Dimos.connect()
sim = app.get_module("SimulationModule")
description = sim.describe_scene()
print(description.entities.keys(), description.joints.keys(), description.regions.keys())
state = sim.scene_state()
print(state.entities["block"].pose, state.regions["tray/interior"])

sim.set_scene_state(SceneUpdate(
    poses={"block": Pose(0.30, -0.16, 0.926)},
    joints={"cabinet-1/door-hinge": 0.8},
))
reset_scene(app)  # Captured scene defaults plus robot/controller homes.
app.stop()        # Disconnect; does not stop the running blueprint.
```

Complete operator RPC surface (in addition to normal Module lifecycle):

```python
status() -> dict[str, Any]
describe_scene() -> SceneDescription
scene_state() -> SceneState
set_scene_state(update: SceneUpdate) -> SceneState
reset(initial: SceneUpdate | None = None) -> SceneState
set_spawn(robot_id: str, xyz: tuple[float, float, float],
          rpy: tuple[float, float, float] = (0, 0, 0)) -> None
set_paused(paused: bool) -> None
set_truth_enabled(enabled: bool) -> None
```

`build()` and `describe()` are internal model/snapshot bootstrap RPCs used
by sensor workers, not another scene API.

The typed records are defined in `dimos/sim2/scene_types.py`:

| Record | Fields |
|---|---|
| `SceneUpdate` | `poses: dict[entity_or_robot_id, Pose]`, `joints: dict[fixture_joint_id, float]` |
| `SceneDescription` | `format`, `id`, `entities`, `joints`, `regions`, `initial`, `spawns`, `hidden_geom_groups`, `provenance` |
| `SceneEntity` | `body`, `label`, `kind`, `movable` |
| `SceneJoint` | `joint`, `entity`, `closed`, `opened` |
| `SceneRegion` | `body`, `kind` (support/containment/navigation), local `pose`, full `size` |
| `SceneState` | `world_id`, `scene_id`, `generation`, `tick`, `sim_time`, wall `ts`, `entities`, `robots`, `joints`, `regions`, `contacts` |
| `EntityState` | world `pose`, linear `velocity`, `angular_velocity`, world `bounds_min`, `bounds_max` |
| `RegionState` | world `pose`, full `size` |

`SceneState.robots` maps instance IDs to poses; contacts are pairs of entity
IDs or robot body names. Stable scene IDs are not MuJoCo array indices.
Robot joints remain on the ordinary control interface.

### Reset And Edit Rules

`set_scene_state` validates the whole update before mutation. It changes only
existing free/mocap bodies and declared scalar fixture joints, zeroes affected
velocities, advances the command generation and publishes immediately. Model,
viewer and sensor workers remain resident. Adding assets or changing structural
scene geometry requires a new run.

`sim.reset()` restores the captured authored physics baseline, then applies
optional overrides. Overrides do not redefine the baseline. It does not cancel
application goals. Use this application-side helper for a running stack:

```python
reset_scene(
    app: Dimos,
    initial: SceneUpdate | None = None,
    *, simulation: str = "SimulationModule",
    coordinators: Sequence[str] = ("ControlCoordinator",),
    before_reset: Sequence[Callable[[Dimos], None]] = (),
    after_reset: Sequence[Callable[[Dimos], None]] = (),
) -> SceneState
```

The helper waits for controller startup, pauses physics, latches coordinator
E-stop and confirms connection halt, cancels trajectories, deactivates tasks,
runs explicit cancellation hooks, resets physics and controller histories,
runs explicit post-reset hooks, then reactivates and clears E-stop. Held targets
are discarded. Stamped commands/feedback from before halt/reset are rejected;
this fence assumes a common wall clock (as in the measured single-host runs).
It does not add epoch fencing to unstamped Twist or guarantee cross-host clock
synchronization. Failure
leaves physics paused. The starter cases supply navigation/manipulation goal
cancellation. Mapping/perception histories require hooks from their actual
owners; they are not automatically inferred or cleared. Reset affects every
robot in the world. Moving an arm's physical base does not reconfigure its
planner, so retain its authored spawn for manipulation.

## Streams And Actions

| Module | Inputs | Outputs |
|---|---|---|
| Whole-body connection | `motor_command: MotorCommandArray` | `joint_state: JointState`, `imu: Imu`, `root_pose: PoseStamped`, `tf: TFMessage` |
| Manipulator connection | `position_command: JointState` | `joint_state: JointState` |
| RGB camera | none | `color_image: Image`, `camera_info: CameraInfo`, `tf: TFMessage` |
| RGB-D camera | none | RGB ports plus `depth_image: Image`, `depth_camera_info: CameraInfo` |
| Lidar | none | `pointcloud: PointCloud2` |
| SimulationModule | none | optional `sim_truth: SceneState` at 10 Hz |

These are the only command paths on the migrated blueprints. There is no
`command_source` switch or direct coordinator-to-sim2 adapter mode. Blueprint
wiring prefixes control topics by robot (`g1_motor_command`, `g1_joint_state`,
`g1_imu`; `arm_position_command`, `arm_joint_state`). Whole-body `root_pose`
is mapped to the existing navigation `odom` topic: it is full 6-DOF truth,
not the planar commanded-base capability in `describe_control()`. The Point-LIO
blueprint maps it to diagnostics-only `ground_truth_odom` instead.
Robot actions remain existing navigation/manipulation RPCs such as `set_goal`,
`plan_to_poses`, `execute`, and `set_gripper_position`.

## Evaluation Boundary

This branch provides physical scene controls and optional `sim_truth`, not a
second evaluation runner. Truth is disabled by default and is not exposed as
an agent skill or wired into perception. Main replaced `InteractiveEval` with
the `EvalCase`/environment API. The earlier three sim2 cases are preserved on
`feat/sim2-emulator-g1-xarm`; they are not runnable examples on this branch.
Adapting them to main's evaluation API is a separate follow-up.

## Scope Of This Checkpoint

Verified: G1 balance/walking, xArm joint control and gripper mapping, native
RGB-D, ideal instantaneous lidar, reset-frame invalidation, fixed-base
relocation, and independent two-robot channels/mounts.

The initial whole-body family requires one coherent control IMU. Standalone
IMU modules, splat rendering, automatic planner
scene obstacles, arbitrary live scene switching, full task generation/DR, and
the ten-minute latency/30-Hz-camera acceptance benchmark remain outside this
checkpoint. M20 and its coordinator extension remain a separate experiment.
Other robot blueprints remain on their existing backends until migrated.
