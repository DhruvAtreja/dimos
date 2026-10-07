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

"""xArm7 with its native position servos, gripper units and wrist RGB-D camera."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.manipulators.xarm.config import XARM7_SIM_HOME, make_xarm7_sim_robot_config
from dimos.sim2.blueprint import simulation
from dimos.sim2.scene import scene_path, scene_robot
from dimos.sim2.sensors.spec import Camera
from dimos.sim2.spec import ControlInterface, Joint, RobotConfig, RobotInstance
from dimos.utils.data import LfsPath

if TYPE_CHECKING:
    from dimos.control.components import HardwareComponent
    from dimos.core.coordination.blueprints import Blueprint
    from dimos.manipulation.planning.spec.config import RobotModelConfig

XARM7 = RobotConfig(
    model=Path(__file__).parent / "assets" / "xarm7.xml",
    meshdir=LfsPath("xarm7/assets"),
    root_body="link_base",
    control=ControlInterface.MANIPULATOR,
    joints=(
        *(
            Joint(
                name=f"joint{i}",
                model_name=f"joint{i}",
                actuator=f"act{i}",
                home=home,
                mode="position",
                lower=lo,
                upper=hi,
            )
            for i, home, lo, hi in zip(
                range(1, 8),
                XARM7_SIM_HOME,
                (
                    -2 * math.pi,
                    -2.059,
                    -2 * math.pi,
                    -0.19198,
                    -2 * math.pi,
                    -1.69297,
                    -2 * math.pi,
                ),
                (2 * math.pi, 2.0944, 2 * math.pi, 3.927, 2 * math.pi, math.pi, 2 * math.pi),
                strict=True,
            )
        ),
        Joint(
            name="arm/gripper",
            model_name="left_driver_joint",
            actuator="gripper",
            home=850.0,
            mode="position",
            scale=-0.001,
            offset=0.85,
            ctrl_scale=-0.3,
            ctrl_offset=255.0,
            lower=0.0,
            upper=850.0,
            max_velocity=0.0,
        ),
    ),
    sensors=(Camera("wrist_camera", camera="wrist_camera"),),
)

# The same arm, with the wrist camera also publishing its depth as a point cloud
# for a blueprint that maps the workspace.
XARM7_MAPPING = XARM7.with_sensor(Camera("wrist_camera", camera="wrist_camera", pointcloud=True))


@dataclass(frozen=True)
class XArm7Simulation:
    """One simulated xArm7: its device modules, the adapter the coordinator drives it
    through, and a planning model standing on the same base pose."""

    devices: Blueprint
    hardware: HardwareComponent
    model: RobotModelConfig


def xarm7_simulation(
    scene_package: str | None,
    scene_spawn: tuple[float, float, float, float] | None = None,
    *,
    robot: RobotConfig = XARM7,
    tf_extra_links: list[str] | None = None,
    sim_id: str = "xarm7",
) -> XArm7Simulation:
    """Put one xArm7 into a sim2 scene and build its planner at the same base pose.

    Args:
        scene_package: A scene name from the sim2 data package, or a path to a scene.xml
            or the directory holding one. None loads the small workbench scene.
        scene_spawn: Base pose as (x, y, z, yaw), metres and radians in the world frame.
            None stands the arm on the scene's named "workbench" support. The arm and the
            planner share it, so a wrong value moves both and they still agree.
        robot: Which xArm7 definition to simulate; the definitions differ only in the
            devices mounted on the arm.
        tf_extra_links: Planning-model links whose world pose the planner publishes on
            tf, besides the tool tip. None publishes only ``link7``; a consumer that needs
            every collision link (a point-cloud self filter) must name them all here.
        sim_id: Name of the shared-memory channel between the coordinator and physics.
    """
    scene = scene_path(scene_package, "workbench.xml")
    if scene_spawn is None:
        arm = scene_robot(scene, robot, "workbench", default=(0.0, 0.0, 0.12))
    else:
        x, y, z, yaw = scene_spawn
        arm = RobotInstance(robot, xyz=(x, y, z), rpy=(0.0, 0.0, yaw))
    model = make_xarm7_sim_robot_config(tf_extra_links=tf_extra_links).model_copy(
        update={
            "base_pose": PoseStamped(
                position=Vector3(*arm.xyz),
                orientation=Quaternion.from_euler(Vector3(*arm.rpy)),
                frame_id="world",
            ),
        }
    )
    devices = simulation(scene=scene, robots={"arm": arm}, sim_id=sim_id)
    return XArm7Simulation(devices=devices.blueprint, hardware=devices.hardware["arm"], model=model)
