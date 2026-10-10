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

"""Basic xArm coordinator and planner blueprints."""

from __future__ import annotations

from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.common.blueprints import coordinator, planner, trajectory_task
from dimos.robot.manipulators.common.sim import mujoco_if_sim
from dimos.robot.manipulators.xarm.config import (
    XARM6_SIM_PATH,
    lite6_hardware,
    make_dual_xarm6_model_config,
    make_lite6_model_config,
    make_xarm7_model_config,
    make_xarm_hardware,
    xarm6_hardware,
    xarm7_hardware,
)

_dual_xarm6_model = make_dual_xarm6_model_config()
_mock_left_xarm6_hw = make_xarm_hardware(
    "left_arm", 6, canonical_joint_names=list(_dual_xarm6_model.planning_groups[0].joint_names)
)
_mock_right_xarm6_hw = make_xarm_hardware(
    "right_arm", 6, canonical_joint_names=list(_dual_xarm6_model.planning_groups[1].joint_names)
)

dual_xarm6_planner_coordinator = autoconnect(
    planner(
        model=_dual_xarm6_model,
        visualization={"backend": "viser"},
    ),
    coordinator(
        hardware=[_mock_left_xarm6_hw, _mock_right_xarm6_hw],
        tasks=[trajectory_task(_mock_left_xarm6_hw, _mock_right_xarm6_hw)],
    ),
)

_xarm7_devices = []


class _XArm7Coordinator(ControlCoordinator):
    arm_position_command: Out[JointState]
    arm_joint_state: In[JointState]


if global_config.simulation:
    from dimos.robot.manipulators.xarm.config import make_xarm7_sim_robot_config
    from dimos.robot.manipulators.xarm.sim2 import XARM7
    from dimos.sim2.blueprint import simulation
    from dimos.sim2.scene import scene_path, scene_robot

    if global_config.simulation != "mujoco":
        raise ValueError("xarm7-planner-coordinator supports --simulation mujoco")
    _scene = scene_path(global_config.scene_package, "workbench.xml")
    _arm = scene_robot(_scene, XARM7, "workbench", default=(0.0, 0.0, 0.12))
    _xarm7_model = make_xarm7_sim_robot_config(prefix="arm/").model_copy(
        update={
            "base_pose": PoseStamped(
                position=Vector3(*_arm.xyz),
                orientation=Quaternion.from_euler(Vector3(*_arm.rpy)),
                frame_id="world",
            ),
        }
    )
    _simulation = simulation(scene=_scene, robots={"arm": _arm}, sim_id="xarm7")
    _xarm7_devices = [_simulation]
    _xarm7_hardware = []
    _xarm7_trajectory = joint_trajectory_task([*_xarm7_model.joint_names, "arm/gripper"])
    _xarm7_coordinator_cls: type[ControlCoordinator] = _XArm7Coordinator
else:
    _xarm7_hw = xarm7_hardware("arm", gripper=True, mock_without_address=True)
    _xarm7_hardware = [_xarm7_hw]
    _xarm7_trajectory = trajectory_task(_xarm7_hw)
    _xarm7_coordinator_cls = ControlCoordinator
    _xarm7_model = make_xarm7_model_config(add_gripper=True, gripper_hardware_id="arm")


def _gripper_task(position_limits: dict[str, tuple[float, float]] | None = None) -> TaskConfig:
    return TaskConfig(
        name="arm_gripper",
        type="gripper",
        joint_names=["arm/gripper"],
        priority=20,
        params={"position_limits": position_limits},
    )


xarm7_planner_coordinator = autoconnect(
    *_xarm7_devices,
    planner(model=_xarm7_model),
    coordinator(
        cls=_xarm7_coordinator_cls,
        instance_name="ControlCoordinator",
        hardware=_xarm7_hardware,
        tasks=[
            _xarm7_trajectory,
            _gripper_task(
                {j.name: (j.lower, j.upper) for j in XARM7.joints if j.name == "arm/gripper"}
                if global_config.simulation
                else None
            ),
        ],
    ),
)

_coordinator_xarm7_hardware = [] if global_config.simulation else [xarm7_hardware("arm")]

coordinator_xarm7 = autoconnect(
    coordinator(
        cls=_xarm7_coordinator_cls,
        instance_name="ControlCoordinator",
        hardware=_coordinator_xarm7_hardware,
        tasks=[
            _xarm7_trajectory
            if global_config.simulation
            else trajectory_task(_coordinator_xarm7_hardware[0])
        ],
    ),
    *_xarm7_devices,
)

_coordinator_xarm6_hw = xarm6_hardware("arm", gripper=True)

coordinator_xarm6 = autoconnect(
    coordinator(
        hardware=[_coordinator_xarm6_hw],
        tasks=[trajectory_task(_coordinator_xarm6_hw), _gripper_task()],
    ),
    *mujoco_if_sim(XARM6_SIM_PATH, len(_coordinator_xarm6_hw.joints)),
)

_coordinator_lite6_hw = lite6_hardware("arm", gripper=True)

coordinator_lite6 = ControlCoordinator.blueprint(
    hardware=[_coordinator_lite6_hw],
    tasks=[trajectory_task(_coordinator_lite6_hw), _gripper_task()],
)

_lite6_hw = lite6_hardware("arm", gripper=True, mock_without_address=True)

lite6_planner_coordinator = autoconnect(
    planner(
        model=make_lite6_model_config(
            add_gripper=True,
            gripper_hardware_id="arm",
        ),
        visualization={"backend": "viser"},
    ),
    coordinator(
        hardware=[_lite6_hw],
        tasks=[trajectory_task(_lite6_hw), _gripper_task()],
    ),
)

_xarm7_left = xarm7_hardware(
    "left_arm", canonical_joint_names=[f"left_arm/joint{i}" for i in range(1, 8)]
)
_xarm6_right = xarm6_hardware(
    "right_arm", canonical_joint_names=[f"right_arm/joint{i}" for i in range(1, 7)]
)

coordinator_dual_xarm = ControlCoordinator.blueprint(
    hardware=[_xarm7_left, _xarm6_right],
    tasks=[trajectory_task(_xarm7_left, _xarm6_right)],
)
