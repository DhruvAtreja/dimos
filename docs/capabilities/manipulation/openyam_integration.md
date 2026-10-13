# OpenYAM integration

OpenYAM uses the generic Damiao whole-body adapter backed by
`can-motor-control`. The adapter owns one upstream robot containing the six-joint
arm and its gripper; motor topology is independent of the host operating system.

## CAN transport

The generic Damiao layer selects the native transport provided by
`can-motor-control>=0.0.8`:

| Host | Default | `--can-port` override |
|------|---------|-----------------------|
| Linux | `SocketCanBus("can0")` | SocketCAN interface, such as `can1` |
| macOS | first `1d50:606f` gs_usb adapter | USB serial number |

The macOS transport is implemented in Rust with native IOKit access. It does
not require PyUSB, libusb, `python-can`, or the Python `gs_usb` package.

Logical buses are ordered. Without explicit selectors, a future two-bus Damiao
robot maps to `can0`/`can1` on Linux and gs_usb indices 0/1 on macOS. Production
multi-arm deployments should configure USB serial numbers because enumeration
order is not a stable device identity.

List the selectors accepted on the current host:

```bash
dimos hardware can list
```

On Linux this prints SocketCAN network-interface names. Configure each physical
interface before use:

```bash
dimos hardware can setup can0
dimos hardware can setup can1
```

On macOS, connect a `gs_usb`/candleLight-compatible adapter with USB ID
`1d50:606f` for each arm. No SocketCAN interface or extra USB library is needed.
The motor buses still require CAN-H/CAN-L wiring, motor power, and correct bus
termination. To assign sides reliably, connect one adapter at a time, run the
list command, record and label its USB serial, then repeat for the other arm.
Adapters used together must expose unique serial numbers.

## Run

Use the default CAN device:

```bash
dimos run coordinator-openyam
```

Select another Linux SocketCAN interface:

```bash
dimos --can-port can1 run coordinator-openyam
```

Select a macOS adapter by USB serial number:

```bash
dimos --can-port <USB-SERIAL> run coordinator-openyam
```

The dual-arm WebXR blueprint is identical on both operating systems; only the
selector values differ:

```bash
# Linux
dimos run teleop-webxr-dual-openyam --left-can-port can0 --right-can-port can1

# macOS
dimos run teleop-webxr-dual-openyam \
  --left-can-port <LEFT-USB-SERIAL> \
  --right-can-port <RIGHT-USB-SERIAL>
```

Linux interfaces must already be configured for classical CAN at 1 Mbit/s.
The native macOS transport configures the adapter bitrate directly.

## Hardware topology

| Motor | Send ID | Reply ID | Type |
|-------|---------|----------|------|
| joint1–3 | `0x01`–`0x03` | send ID + `0x10` | DM4340 |
| joint4–6 | `0x04`–`0x06` | send ID + `0x10` | DM4310 |
| gripper | `0x08` | `0x18` | DM4310 |

The bus uses classical CAN at 1 Mbit/s. Opening the gripper decreases motor
position. The arm and gripper are exposed together as one whole-body hardware
component.

## Pick-and-place benchmark

The xArm grasp stack on the workcell, one arm per blueprint:

```bash
dimos run dual-openyam-grasp-right --left-can-port follower_l --right-can-port follower_r
dimos run dual-openyam-grasp-right-agent --left-can-port follower_l --right-can-port follower_r
dimos humancli      # second terminal: "scan for a soup can", "pick up the soup can", "put it in the bin"
```

`dual-openyam-grasp-left` and the `-graspgenx` variants select the other arm
and the learned grasp provider. The planning model carries both arms, but the
manipulation module binds one gripper per model, so each blueprint benchmarks
the arm whose gripper it names while the other arm rests on its support.
Every run records joint states, the three cameras and TF to sqlite.

What the stack does not have on the workcell, and therefore what the benchmark
measures as is: the model has no collision geometry, so the planner avoids
only the table (`floor_z`) and the detected objects; grasps are one top-down
heuristic candidate per object; a pick executes as soon as it is planned, so
the operator keeps a hand on the e-stop; grasp success is judged from the jaw
readback alone.

## Model assets

Two dual-arm model packages ship as LFS data:

- `dual_openyam_workcell_v1` is the default, `DUAL_OPENYAM_MODEL`. It describes
  the physical workcell: arm bases 0.43 m apart (tape, 2026-10-08) and joint 1
  limits of -225 to +90 deg, inside the stops measured when the motors were
  zeroed with Anvil's `damiao-set-zero-limit` on 2026-10-12. Its zero
  configuration is the folded rest pose on the supports.
- `dual_openyam_abc_box_v2`, exported as `DUAL_OPENYAM_ABC_MODEL`, keeps the
  0.62 m spacing of the ABC simulation bench for scenes that reproduce it.

Both packages carry the same i2rt arm geometry and meshes; only the base
spacing and the joint 1 limits differ. Each package's `SOURCE.md` records the
measurement or upstream commit every number comes from.

## Safety

- Keep the workspace clear and the emergency stop reachable during first
  activation.
- Verify motor IDs and types against the physical unit before commanding large
  motions; an incorrect type scales velocity and torque incorrectly.
- Deactivation removes motor torque, so support the arm before stopping it.
