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

"""Go2 FREE connection + ControlCoordinator + sim2; optional real Point-LIO.

Install private weights explicitly: python -m dimos.control.go2_freewalk.models --install
Run: dimos --simulation mujoco --transport zenoh --viewer rerun run unitree-go2-freewalk
Use unitree-go2-freewalk-pointlio for Mid360-driven localization, not truth odometry.
These compositions do not connect to physical robot motors.
"""

from dataclasses import replace

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport_factory import make_transport
from dimos.hardware.sensors.lidar.pointlio.module import PointLioRust
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.pointclouds.occupancy import HeightCostConfig
from dimos.mapping.ray_tracing.module import RayTracingVoxelMap
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.navigation.go2.replanning_a_star.module import ReplanningAStarPlanner
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.unitree.go2.freewalk_connection import Go2FreewalkConnection
from dimos.robot.unitree.go2.sim2 import GO2_FREEWALK, GO2_FREEWALK_MID360
from dimos.sim2.blueprint import simulation
from dimos.sim2.connections.whole_body import WholeBodyConnection
from dimos.sim2.scene import scene_path, scene_robot
from dimos.sim2.sensors.spec import Lidar
from dimos.visualization.vis_module import vis_module


def _build(*, pointlio: bool) -> Blueprint:
    robot = GO2_FREEWALK
    if pointlio:
        robot = GO2_FREEWALK_MID360.with_sensor(
            replace(
                next(s for s in GO2_FREEWALK_MID360.sensors if isinstance(s, Lidar)),
                truth_outputs=False,
            )
        )
    scene = scene_path(global_config.scene_package, "logistics.xml")
    devices = simulation(
        scene=scene,
        robots={"go2": scene_robot(scene, robot, default=(0, 0, 0))},
        sim_id="go2-freewalk",
        timestep=0.0025,
    )
    joints = make_twist_base_joints("go2_base")
    control = (
        autoconnect(
            Go2FreewalkConnection.blueprint(auto_arm=True),
            ControlCoordinator.blueprint(
                tick_rate=50,
                publish_joint_state=False,
                hardware=[
                    HardwareComponent(
                        hardware_id="go2_base",
                        hardware_type=HardwareType.BASE,
                        joints=joints,
                        adapter_type="transport_lcm",
                    )
                ],
                tasks=[
                    TaskConfig(
                        name="walk",
                        type="velocity",
                        joint_names=joints,
                        priority=50,
                        auto_start=True,
                        params={"timeout": 0.5, "zero_on_timeout": True},
                    )
                ],
            ),
        )
        .remappings(
            [
                (ControlCoordinator, "twist_command", "cmd_vel"),
                (Go2FreewalkConnection, "base_command", "policy_velocity"),
            ]
        )
        .transports({("policy_velocity", Twist): make_transport("/go2_base/cmd_vel", Twist)})
    )
    localization = (
        autoconnect(
            PointLioRust.blueprint(
                instance_name="lidar_localization",
                sensor_frame_id="go2/lidar",
                frame_id="odom",
                extrinsic_t=[0.0, 0.0, 0.0],
                extrinsic_r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ),
            RayTracingVoxelMap.blueprint(
                world_frame="odom", voxel_size=0.05, emit_every=0, global_emit_every=4
            ),
        ).remappings([(PointLioRust, "imu", "imu_raw")])
        if pointlio
        else VoxelGridMapper.blueprint(voxel_size=0.05)
    )
    blueprint = autoconnect(
        devices.blueprint,
        # Replace the generated device instance's command source, not its implementation.
        WholeBodyConnection.blueprint(
            instance_name="go2_connection",
            definition=robot,
            address="go2-freewalk/go2",
            robot_id="go2",
            rate_hz=200,
            command_source="stream",
        ),
        control,
        localization,
        CostMapper.blueprint(
            config=HeightCostConfig(resolution=0.05, can_climb=0.05, can_pass_under=0.55),
            initial_safe_radius_meters=0.35,
        ),
        ReplanningAStarPlanner.blueprint(robot_width=0.35, robot_rotation_diameter=0.7),
        MovementManager.blueprint(),
        vis_module(global_config.viewer),
    )
    if not pointlio:
        # the sim lidar publishes `pointcloud`; the mapper reads `lidar` (as on the G1 sim2 blueprint)
        blueprint = blueprint.remappings([(VoxelGridMapper, "lidar", "pointcloud")])
    if pointlio:
        blueprint = blueprint.remappings(
            [
                ("go2_lidar", "raw_pointcloud", "lidar_raw"),
                ("go2_connection", "odom", "ground_truth_odom"),
                ("go2_connection", "tf", "ground_truth_tf"),
                ("go2_camera", "tf", "ground_truth_tf"),
            ]
        )
    return blueprint


if global_config.simulation != "mujoco":
    raise ValueError("Go2 FREE blueprints require --simulation mujoco; no hardware actuation here")

unitree_go2_freewalk = _build(pointlio=False).global_config(robot_model="unitree_go2", n_workers=8)
unitree_go2_freewalk_pointlio = _build(pointlio=True).global_config(
    robot_model="unitree_go2", n_workers=8
)
