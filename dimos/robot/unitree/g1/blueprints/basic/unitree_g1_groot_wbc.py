# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unitree G1 GR00T whole-body-control blueprint.

One blueprint, ``--simulation`` flag picks the backend:

Real hardware (default):
    G1WholeBodyConnection (DDS rt/lowstate <-> rt/lowcmd) + transport_lcm
    whole-body adapter. 500 Hz tick. Safety profile: unarmed + dry-run on
    start; activate explicitly through ControlCoordinator RPC after
    verifying commands. The policy ramps from the current pose to its
    bent-knee default over 10 s before taking torque control. The 14 arm
    joints accept bounded position commands through the lower-priority joint
    trajectory task.

Sim (``--simulation``):
    sim2 world + robot-mounted sensors + coherent SHM device adapter.
    50 Hz tick (matches the rate the policy was trained at). No arming
    ramp and no dry-run. The same bounded arm-command path is available in
    simulation and on hardware.

Usage:
    dimos run unitree-g1-groot-wbc                 # real hardware
    dimos --simulation mujoco run unitree-g1-groot-wbc    # sim
    dimos --simulation mujoco --scene-package /path/to/scene.xml run unitree-g1-groot-wbc

Overrides (replace the old env-var dance):
    dimos run unitree-g1-groot-wbc \\
        --network-interface=enp2s0
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from dimos.control.components import HardwareComponent, HardwareType
from dimos.control.coordinator import TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task
from dimos.control.teleop_coordinator import TeleopControlCoordinator
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.stream import Out
from dimos.core.transport import LCMTransport
from dimos.hardware.whole_body.spec import WholeBodyConfig
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import HeightCostConfig
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.nav_msgs.Path import Path as NavPath
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.navigation.go2.replanning_a_star.module import ReplanningAStarPlanner
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.g1.config import G1
from dimos.robot.unitree.g1.control_config import (
    G1_GROOT_KD,
    G1_GROOT_KP,
    g1_arms,
    g1_joints,
    g1_legs_waist,
)
from dimos.robot.unitree.g1.g1_rerun import (
    G1_RERUN_ROOT,
    g1_costmap,
    g1_urdf_joint_state,
    g1_urdf_static_robot,
)
from dimos.robot.unitree.g1.manip_config import G1_TELEOP_ARM_MODEL
from dimos.robot.unitree.g1.teleop_ik import G1PinkPoseTargetSolver
from dimos.utils.data import LfsPath
from dimos.visualization.rerun.scene_package import scene_package_static_entities
from dimos.visualization.vis_module import vis_module

# Lazy data handles. LfsPath only triggers the LFS pull on first
# str()/open(); using ``get_data(...)`` at import time would block the
# whole CLI on a multi-GB download every time the module is imported.
_GROOT_MODEL_DIR = LfsPath("groot")

_cmd_vel_topic = "/cmd_vel" if global_config.simulation else "/g1/cmd_vel"
_G1_NAV_VOXEL_RESOLUTION = 0.05
# go2 nav_3d resolution; 0.05 saturates the raytracer on the Orin.
_G1_REAL_NAV_VOXEL_RESOLUTION = 0.08
_G1_NAV_OVERHEAD_SAFETY_MARGIN = 0.2
_G1_NAV_MAX_STEP_HEIGHT = 0.10
_G1_NAV_ROTATION_DIAMETER = 0.8
_G1_NAV_SAFE_RADIUS_MARGIN = 0.6


class _G1GrootCoordinator(TeleopControlCoordinator):
    g1_joints: Out[JointState]


# Per-robot joint stream. Namespaced like the rest of the g1 wire topics, which
# also fixes its Rerun entity path (`world/` + topic).
_G1_JOINTS_TOPIC = "/g1/joints"
_G1_JOINTS_ENTITY = f"world{_G1_JOINTS_TOPIC}"


if global_config.simulation and global_config.simulation != "mujoco":
    raise ValueError("unitree-g1-groot-wbc only supports --simulation mujoco")

if global_config.simulation == "mujoco":
    from dimos.mapping.voxels.module import VoxelGridMapper
    from dimos.robot.unitree.g1.sim2 import G1_GROOT
    from dimos.sim2.blueprint import simulation
    from dimos.sim2.scene import scene_path, scene_robot

    _scene = scene_path(global_config.scene_package, "logistics.xml")
    _simulation = simulation(
        scene=_scene,
        robots={"g1": scene_robot(_scene, G1_GROOT, default=(0.0, 0.0, 0.0))},
        sim_id="g1-groot",
    )
    _backend = _simulation.blueprint
    _hardware = _simulation.hardware["g1"]
    _tick_rate = 50.0
    _auto_arm = True
    _auto_dry_run = False
    _default_ramp_seconds = 0.0
    _decimation: int | None = 1
    _n_workers = 2  # sim: keep the default worker count
    _mapper = VoxelGridMapper.blueprint(emit_every=1)
    _nav_stack = autoconnect(
        _mapper,
        CostMapper.blueprint(
            config=HeightCostConfig(
                resolution=_G1_NAV_VOXEL_RESOLUTION,
                can_pass_under=G1.height_clearance + _G1_NAV_OVERHEAD_SAFETY_MARGIN,
                can_climb=_G1_NAV_MAX_STEP_HEIGHT,
            ),
        ),
        ReplanningAStarPlanner.blueprint(
            robot_width=G1.width_clearance,
            robot_rotation_diameter=_G1_NAV_ROTATION_DIAMETER,
        ),
        MovementManager.blueprint(),
    )
    _nav_remappings = [(VoxelGridMapper, "lidar", "pointcloud")]
else:
    from dimos.hardware.sensors.lidar.pointlio.module import PointLio
    from dimos.hardware.sensors.lidar.pointlio.pointlio_blueprints import mid360_for_pointlio
    from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
    from dimos.robot.unitree.g1.wholebody_connection import G1WholeBodyConnection

    # Real-hw backend: DDS connection module + transport_lcm adapter.
    _backend = G1WholeBodyConnection.blueprint(release_sport_mode=True)
    _hardware = HardwareComponent(
        hardware_id="g1",
        hardware_type=HardwareType.WHOLE_BODY,
        joints=g1_joints,
        adapter_type="transport_lcm",
        address="",
        wb_config=WholeBodyConfig(kp=tuple(G1_GROOT_KP), kd=tuple(G1_GROOT_KD)),
    )
    # The onboard Jetson can't sustain a 500 Hz tick; it collapses to ~90 Hz
    # and starves the policy, so balance decays.
    _tick_rate = 100.0
    # Real hardware: come up unarmed + dry-run; operator must click
    # Activate (10 s ramp) after verifying commands.
    _auto_arm = False
    _auto_dry_run = True
    _default_ramp_seconds = 10.0
    _decimation = 2  # 100 Hz tick / 2 = 50 Hz policy (training + sim rate).
    # One process per heavy module; fewer workers starve the Rerun bridge.
    _n_workers = 11
    # Same nav middle as unitree-g1-nav-simple, fed by Point-LIO from the
    # MID-360, executed through the coordinator's twist_command.
    _nav_stack = autoconnect(
        mid360_for_pointlio(),
        PointLio.blueprint(),
        RayTracingVoxelMap.blueprint(
            voxel_size=_G1_REAL_NAV_VOXEL_RESOLUTION,
            emit_every=0,  # no local_map consumer here
            global_emit_every=4,  # ~1 Hz global map; also paces the costmap
            # Clearing matched to go2 nav_3d.
            max_health=10,
            graze_cos=0.85,
        ),
        CostMapper.blueprint(
            config=HeightCostConfig(
                resolution=_G1_REAL_NAV_VOXEL_RESOLUTION,
                can_pass_under=G1.height_clearance + _G1_NAV_OVERHEAD_SAFETY_MARGIN,
                can_climb=_G1_NAV_MAX_STEP_HEIGHT,
            ),
            initial_safe_radius_meters=G1.width_clearance + _G1_NAV_SAFE_RADIUS_MARGIN,
        ),
        ReplanningAStarPlanner.blueprint(
            robot_width=G1.width_clearance,
            robot_rotation_diameter=_G1_NAV_ROTATION_DIAMETER,
        ),
        MovementManager.blueprint(),
    )
    _nav_remappings = []


_arm_trajectory_task = joint_trajectory_task(
    g1_arms,
    priority=10,
    velocity_limits={name: 1.0 for name in g1_arms},
)


def _g1_groot_rerun_blueprint() -> Any:
    import rerun as rr
    import rerun.blueprint as rrb

    scene_view = rrb.Spatial3DView(
        origin="world",
        name="G1 GR00T WBC",
        background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
        line_grid=rrb.LineGrid3D(
            plane=rr.components.Plane3D.XY.with_distance(0.0),
        ),
    )
    layout = (
        rrb.Horizontal(
            scene_view,
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/color_image", name="RGB"),
                rrb.Spatial2DView(origin="world/depth_image", name="Depth"),
            ),
            column_shares=[3, 1],
        )
        if global_config.simulation
        else scene_view
    )
    return rrb.Blueprint(
        layout,
        rrb.TimePanel(state="collapsed"),
    )


def _g1_nav_path(path: NavPath) -> Any:
    return path.to_rerun(z_offset=0.3)


# Mesh root: sim roots under the /odom transform; real hw under the LIO's
# /odometry, whose world frame is the lidar boot pose (ground ~1.2 m below 0).
_G1_ROOT = G1_RERUN_ROOT if global_config.simulation == "mujoco" else "world/odometry/g1"

_G1_URDF_PATH = Path(__file__).resolve().parents[2] / "g1.urdf"
_G1_TELEOP_MODEL = RobotModelConfig(
    model=G1_TELEOP_ARM_MODEL,
    joint_names=list(g1_arms),
    base_link="pelvis",
)
_G1_TELEOP_PINK = PinkKinematicsConfig(
    dt=0.01,
    position_cost=8.0,
    orientation_cost=2.0,
    posture_cost=0.01,
    joint_limit_posture_margin=0.3,
    lm_damping=0.01,
    gain=0.25,
)
# Nominal standing pelvis height; matches G1GrootWBCTask's height_cmd.
_G1_NOMINAL_PELVIS_Z = 0.74
_g1_pelvis_mid360_cache: list[Any] = []


def _g1_pelvis_to_mid360() -> Any:
    """Rest-pose pelvis->mid360_link transform from the G1 URDF (cached)."""
    if not _g1_pelvis_mid360_cache:
        from importlib import import_module

        import numpy as np

        urdf = import_module("yourdfpy").URDF.load(str(_G1_URDF_PATH), load_meshes=False)
        urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
        _g1_pelvis_mid360_cache.append(urdf.get_transform("mid360_link", "pelvis"))
    return _g1_pelvis_mid360_cache[0]


def _g1_real_odometry_root(odom: Any) -> Any:
    """Robot-mesh root: pelvis pose from the LIO's mid360 odometry (rest offset)."""
    import numpy as np
    import rerun as rr

    from dimos.msgs.geometry_msgs.Quaternion import Quaternion

    t_world_mid360 = np.eye(4)
    # The MID-360 is mounted upside down (the URDF doesn't carry the flip):
    # un-roll by Rx(pi) == diag(1, -1, -1).
    t_world_mid360[:3, :3] = odom.orientation.to_rotation_matrix() @ np.diag([1.0, -1.0, -1.0])
    t_world_mid360[:3, 3] = (odom.x, odom.y, odom.z)
    t_world_pelvis = t_world_mid360 @ np.linalg.inv(_g1_pelvis_to_mid360())
    q = Quaternion.from_rotation_matrix(t_world_pelvis[:3, :3])
    return rr.Transform3D(
        translation=t_world_pelvis[:3, 3].tolist(),
        rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
    )


def _g1_real_ground_z() -> float:
    """Ground height in the LIO boot frame: -(mount z + nominal pelvis z)."""
    return -(float(_g1_pelvis_to_mid360()[2, 3]) + _G1_NOMINAL_PELVIS_Z)


def _g1_real_costmap(grid: Any) -> Any:
    """Costmap rendered on the actual ground plane of the boot frame."""
    return g1_costmap(grid, z_offset=_g1_real_ground_z() + 0.02)


_static_rerun_entities: dict[str, Any] = {
    _G1_ROOT: g1_urdf_static_robot(root_path=_G1_ROOT),
}
if not global_config.simulation:
    _static_rerun_entities.update(scene_package_static_entities(global_config.scene_package))

_rerun_config: dict[str, Any] = {
    "blueprint": _g1_groot_rerun_blueprint,
    "visual_override": {
        _G1_JOINTS_ENTITY: g1_urdf_joint_state(root_path=_G1_ROOT),
        "world/global_costmap": g1_costmap,
        "world/navigation_costmap": g1_costmap,
        "world/path": _g1_nav_path,
    },
    "max_hz": {
        _G1_JOINTS_ENTITY: 20.0,
        # Raw state streams arrive at ~440 Hz; useful only as debug plots.
        "world/g1/imu": 10.0,
        "world/g1/motor_states": 10.0,
        "world/g1/motor_command": 10.0,
        "world/odometry": 15.0,
        "world/global_map": 1.0,
        "world/global_costmap": 2.0,
        "world/navigation_costmap": 2.0,
        # The planner publishes an empty Path() immediately before the new
        # planned path. Throttling this entity drops the real path.
        "world/path": 0,
    },
    "static": _static_rerun_entities,
}

if global_config.simulation != "mujoco":
    _rerun_config["visual_override"]["world/odometry"] = _g1_real_odometry_root
    _rerun_config["visual_override"]["world/global_costmap"] = _g1_real_costmap
    _rerun_config["visual_override"]["world/navigation_costmap"] = _g1_real_costmap
    # Raw scan is sensor-frame (LIO contract); the voxel map is the live view.
    _rerun_config["visual_override"]["world/lidar"] = None
    _rerun_config["visual_override"]["world/lidar_raw"] = None


def _viewer() -> Any:
    return vis_module(viewer_backend=global_config.viewer, rerun_config=_rerun_config)


_coordinator = _G1GrootCoordinator.blueprint(
    instance_name="ControlCoordinator",
    publish_robot_joint_states=True,
    tick_rate=_tick_rate,
    hardware=[_hardware],
    tasks=[
        TaskConfig(
            name="groot_wbc",
            type="g1_groot_wbc",
            joint_names=g1_legs_waist,
            priority=50,
            auto_start=True,
            params={
                "model_path": _GROOT_MODEL_DIR,
                "hardware_id": "g1",
                "auto_arm": _auto_arm,
                "auto_dry_run": _auto_dry_run,
                "default_ramp_seconds": _default_ramp_seconds,
                "decimation": _decimation,
            },
        ),
        _arm_trajectory_task,
        # Shared bimanual WebXR task with G1-only model and objective tuning.
        TaskConfig(
            name="teleop_g1",
            type="teleop_ik",
            joint_names=g1_arms,
            priority=20,
            params={
                "robot_model": _G1_TELEOP_MODEL,
                "bindings": [
                    {"hand": "left", "target_frame": "left_rubber_hand"},
                    {"hand": "right", "target_frame": "right_rubber_hand"},
                ],
                "solver_type": G1PinkPoseTargetSolver,
                "pink": _G1_TELEOP_PINK,
                "timeout": 0.5,
                "max_command_tracking_error_deg": 10.0,
                "max_joint_velocity_rad_s": math.radians(120.0),
                "joint_command_filter_cutoff_hz": 5.0,
            },
        ),
    ],
)
if global_config.simulation:
    _coordinator = _coordinator.remappings([(_G1GrootCoordinator, "g1_joints", "g1/joints")])
else:
    _coordinator = _coordinator.transports(
        {
            ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
            ("g1_joints", JointState): LCMTransport(_G1_JOINTS_TOPIC, JointState),
            ("cmd_vel", Twist): LCMTransport(_cmd_vel_topic, Twist),
            # The real hardware adapter speaks to G1WholeBodyConnection over LCM.
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("imu", Imu): LCMTransport("/g1/imu", Imu),
            ("motor_command", MotorCommandArray): LCMTransport(
                "/g1/motor_command", MotorCommandArray
            ),
        }
    )

_unitree_g1_groot_wbc_core = (
    autoconnect(_backend, _coordinator)
    .remappings([(_G1GrootCoordinator, "twist_command", "cmd_vel")])
    .global_config(robot_model="unitree_g1", n_workers=_n_workers)
)

unitree_g1_groot_wbc = autoconnect(_unitree_g1_groot_wbc_core, _nav_stack, _viewer()).remappings(
    cast("Any", _nav_remappings)
)
