# Copyright 2026 Dimensional Inc.
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

"""Existing GR00T stack with PimSim's rolling Mid360 in place of the ideal scanner."""

from dataclasses import replace

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.pointlio.module import PointLioRust
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.robot.unitree.g1.blueprints.basic.unitree_g1_groot_wbc import (
    _G1_JOINTS_ENTITY,
    _g1_real_costmap,
    _g1_real_odometry_root,
    _rerun_config,
    unitree_g1_groot_wbc,
)
from dimos.robot.unitree.g1.g1_rerun import g1_urdf_joint_state, g1_urdf_static_robot
from dimos.robot.unitree.g1.sim2 import G1_GROOT_MID360
from dimos.sim2.blueprint import simulation
from dimos.sim2.scene import scene_path, scene_robot
from dimos.sim2.sensors.spec import Lidar
from dimos.visualization.vis_module import vis_module

if global_config.simulation != "mujoco":
    raise ValueError("unitree-g1-groot-mid360 requires --simulation mujoco")

_scene = scene_path(global_config.scene_package, "logistics.xml")
_devices = simulation(
    scene=_scene,
    robots={"g1": scene_robot(_scene, G1_GROOT_MID360, default=(0, 0, 0))},
    sim_id="g1-groot",
)
unitree_g1_groot_mid360 = autoconnect(unitree_g1_groot_wbc, _devices)

# Same device and policy, with the actual estimator instead of truth localization.
_raw_robot = G1_GROOT_MID360.with_sensor(
    replace(next(s for s in G1_GROOT_MID360.sensors if isinstance(s, Lidar)), truth_outputs=False)
)
_raw_devices = simulation(
    scene=_scene,
    robots={"g1": scene_robot(_scene, _raw_robot, default=(0, 0, 0))},
    sim_id="g1-groot",
)
unitree_g1_groot_mid360_pointlio = (
    autoconnect(
        unitree_g1_groot_wbc,
        _raw_devices,
        vis_module(
            global_config.viewer,
            rerun_config={
                **_rerun_config,
                "static": {
                    "world/odometry/g1": g1_urdf_static_robot(root_path="world/odometry/g1"),
                },
                "visual_override": {
                    **_rerun_config["visual_override"],
                    _G1_JOINTS_ENTITY: g1_urdf_joint_state(root_path="world/odometry/g1"),
                    "world/odometry": _g1_real_odometry_root,
                    "world/global_costmap": _g1_real_costmap,
                    "world/navigation_costmap": _g1_real_costmap,
                    "world/lidar": None,
                    "world/lidar_raw": None,
                    "world/ground_truth_odom": None,
                    "world/ground_truth_tf": None,
                },
            },
        ),
        PointLioRust.blueprint(
            instance_name="lidar_localization",
            sensor_frame_id="g1/lidar",
            frame_id="odom",
            # This simulated IMU is colocated with the laser, not Livox's real offset.
            extrinsic_t=[0.0, 0.0, 0.0],
            extrinsic_r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ),
        RayTracingVoxelMap.blueprint(
            world_frame="odom", voxel_size=0.05, emit_every=0, global_emit_every=4
        ),
    )
    .disabled_modules(VoxelGridMapper)
    .remappings(
        [
            ("g1_lidar", "raw_pointcloud", "lidar_raw"),
            (PointLioRust, "imu", "imu_raw"),
            ("g1_connection", "root_pose", "ground_truth_odom"),
            ("g1_connection", "tf", "ground_truth_tf"),
        ]
    )
)
