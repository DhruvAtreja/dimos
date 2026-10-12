# G1 SONIC controller

SONIC runs the planner, encoder and decoder at a 50 Hz policy rate. It accepts
standard Twist walking commands, selectable gaits, motion clips and optional
upper-body reference targets. No headset or teleop service is required.

The default gait is `SLOW_WALK`. Velocity commands set direction and speed
within that gait; they never select walking or running automatically. With a
walking gait selected, zero input idles while preserving that selection. Use
`set_locomotion_mode` on G1SonicConnection to change modes; `None` restores
`SLOW_WALK`.

Install the workstation dependencies and NVIDIA assets:

```bash
uv sync --extra control --extra cuda --extra sim \
  --no-install-package onnxruntime --reinstall-package onnxruntime-gpu
source .venv/bin/activate
dimos-sonic-models
dimos-sonic-models --check
dimos --transport zenoh --simulation mujoco --viewer none run unitree-g1-sonic-wbc
```

The CPU and GPU ONNX Runtime distributions share the same Python package;
excluding the CPU distribution prevents it overwriting the CUDA provider.
Keep these options when syncing this environment again.

Assets live in the DimOS cache (`~/.cache/dimos/sonic` by default), outside
the Git LFS data directory. `SONIC_MODEL_DIR` overrides this location for
both the installer and blueprint. The installer pins the Hugging Face revision and verifies SHA-256 hashes for
the policy, planner and observation configuration. The 13 example motion clips
come from a pinned NVIDIA deployment revision. `--check` performs no downloads.
SONIC v1.1 is the single supported policy bundle.

MuJoCo starts in the SONIC pose and waits for a complete control command before
advancing physics. Simulation arms automatically. Commands from a second shell:

```bash
dimos --transport zenoh shell
```

```python
c = app.G1SonicConnection
c.status()
c.set_locomotion_mode("SLOW_WALK")
c.list_motion_clips()
c.play_motion_clip("macarena_001__A545")
c.stop_motion_clip()
c.set_estop(True)
```

E-stop latches damping. Releasing a control, resetting runtime state, or arming
again cannot clear it; recovery requires restarting the stack. Hardware also
checks feedback and command freshness in its independent motor publisher.

For JetPack 5 or 6, use `bin/hardware/g1/setup-sonic`. It detects the release and
installs the matching ONNX Runtime and model assets; `--check` checks system
prerequisites only. JetPack 5 requires CUDA 11.8 with its compatibility driver
and cuDNN 8; JetPack 6 requires CUDA 12.6 and cuDNN 9.
Hardware starts unarmed with policy outputs in dry-run.
After checking startup and feedback, the hardware activation RPCs are:

```python
c = app.G1SonicConnection
c.set_dry_run(False)
c.arm()
```

This standalone wiring has not been validated on physical hardware.

Squat and kneeling request zero translation. Centered sticks stop crawling
while retaining its posture. Face-down mode 7 is unavailable, matching NVIDIA's
selectable motion menu. Floor transitions remain timed; crawl stability and
the earlier hardware instability remain open validation issues.

## Module interface

The blueprint runs `G1SonicConnection` directly, without ControlCoordinator,
coordinator tasks or a task-compatibility wrapper. Command flow is:

```text
navigation / teleop -> MovementManager -> G1SonicConnection -> SonicController -> motor IO
                      selects cmd_vel                        runs SONIC at 50 Hz
```

`MovementManager` combines `nav_cmd_vel` and `tele_cmd_vel`, with teleoperation
cancelling the active navigation goal. Its `cmd_vel` output feeds SONIC's
`base_command`. Applications should use those upstream inputs instead of
competing publishers on `cmd_vel`. There is no general task-priority scheduler
or timed arm-trajectory API in this stack. SONIC owns all 29 motor targets;
arm inputs condition the policy rather than independently controlling motors.

The connection exposes streams and RPCs and delegates to `SonicController` in
`sonic_controller.py`. That controller owns the 50 Hz loop, pose ramp and fault
latch; `SonicPipeline` handles model inference. `G1WholeBodyConnection` owns real
motor IO and its independent 500 Hz publisher. MuJoCo uses the existing
shared-memory whole-body adapter. SONIC sends no motor commands before its models load.

- `base_command: In[Twist]`: `linear.x`, `linear.y`, and `angular.z`, remapped
  to `cmd_vel`. After one second without fresh input, SONIC commands zero
  walking velocity while continuing to balance.
- `position_command: In[JointState]`: named arm references, on
  `/g1/position_command` in this blueprint. Use the arm names from measured
  `joint_state`, for example `g1/left_shoulder_pitch`. Partial arm updates hold
  the other reference values. Unknown names, leg targets, duplicate names,
  missing positions and non-finite values are rejected. These are encoder
  references, not independent joint servos.
- `joint_state: Out[JointState]`: measured state for all 29 joints, remapped to
  `g1_joints` (`/g1/joints`) for existing visualization.
- `imu: Out[Imu]`: measured IMU state. Raw hardware feedback uses the separate
  `low_level_imu` input. Odometry still comes from simulation or localization.

For example, send `JointState(name=["g1/left_shoulder_pitch"], position=[0.2])`
to `/g1/position_command`. `c.set_upper_body(positions)` sets a complete
14-element arm reference in DDS order. These update encoder references
directly; there is no coordinator trajectory interpolation or speed limiting.
`c.clear_upper_body()` restores the default arm references.
`set_vr_3point()` retains its existing wrist/head reference
contract; a generic Cartesian command stream is not implemented here.

Topic paths above are logical names. Typed Zenoh keys include the `dimos/`
prefix and message-type suffix, for example `dimos/cmd_vel/geometry_msgs.Twist`.

`arm()`, `disarm()`, `set_dry_run()`, `set_locomotion_mode()`, motion-clip
commands, `reset_runtime_state()` and `status()` are direct module RPCs.
`disarm()` returns to measured-pose hold; it does not keep the balancing policy
running. `set_estop(True)` and `halt_robot()` latch damping. Ordinary zero
walking commands and command expiry keep balancing and do not clear faults.

`set_velocity_command(vx, vy, yaw_rate)` is also available for direct shell
control. It bypasses `MovementManager`, so do not use it alongside active
navigation/teleoperation. Repeat commands to sustain motion; they expire after
the same one-second timeout. Zero velocity means stop walking and keep balancing,
not disarm. Disarming, resetting, and completing the activation ramp clear old
walking commands; send fresh commands after activation.

Stopping the complete stack stops SONIC and latches damping. This uses the
existing `Module` base and does not depend on the new `ConnectionModule` or
ControlCoordinator refactor. `task_invoke`, `set_activated`, `execute_trajectory`
and `/g1/joint_command` are not part of the SONIC interface.

## Focused validation

```bash
uv run pytest dimos/robot/unitree/g1/test_sonic_connection.py \
  dimos/robot/unitree/g1/test_wholebody_connection.py dimos/control/sonic \
  -m 'not mujoco'
```

These exercise standalone command routing, expiry, re-arm/reset, fault latching,
motor-publisher safeguards, inference semantics and blueprint composition.
Models and hardware IO are mocked. The blueprint tests import optional simulation
dependencies but do not load assets or start hardware. They do not establish
physical walking stability or hardware safety.
