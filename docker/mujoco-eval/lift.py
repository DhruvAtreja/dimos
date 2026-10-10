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

"""Fixture-specific RGB-D Lift positive control, not a general agent policy."""

from pathlib import Path
import threading
import time
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pinocchio as pin

from dimos.control.coordinator import ControlCoordinator
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JointTrajectoryTask,
    TrajectoryExecutionStatus,
)
from dimos.manipulation.planning.kinematics.pinocchio_ik import PinocchioIK, PinocchioIKConfig
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.msgs.trajectory_msgs.TrajectoryStatus import TrajectoryState
from dimos.robot.assets.model import RobotModel
from dimos.robot.manipulators.mujoco_robot_io import MujocoRobotIO

# Public robot mounting calibration, also used by the xarm-eval blueprint.
BASE = np.array([0.0, 0.0, 0.912])


def run_lift(coordinator: ControlCoordinator, camera: MujocoRobotIO) -> None:
    arm = [f"arm/joint{i}" for i in range(1, 8)]
    messages: dict[str, Any] = {}
    condition = threading.Condition()

    def receive(name: str, message: Any) -> None:
        with condition:
            messages[name] = message
            condition.notify_all()

    for name in ("color_image", "depth_image", "camera_info", "tf"):
        getattr(camera, name).subscribe(lambda message, channel=name: receive(channel, message))
    with condition:
        if not condition.wait_for(lambda: len(messages) == 4, timeout=10):
            raise RuntimeError("Missing public RGB-D/calibration frames")
        rgb = cast("Image", messages["color_image"]).to_rgb().data.copy()
        depth = cast("Image", messages["depth_image"]).data.copy()
        intrinsic = np.array(cast("CameraInfo", messages["camera_info"]).K).reshape(3, 3)
        optical = cast("TFMessage", messages["tf"]).transforms[0].to_matrix()

    description = Path("/opt/models/xarm_ros2/xarm_description")
    source = RobotModel.from_file(
        description / "urdf/xarm_device.urdf.xacro",
        package_paths={"xarm_description": description},
        xacro_args={
            "dof": "7",
            "robot_type": "xarm",
            "prefix": "",
            "limited": "true",
            "attach_xyz": "0 0 0",
            "attach_rpy": "0 0 0",
            "add_gripper": "true",
        },
    )
    fixed = [
        joint.name
        for joint in source.load().joints
        if joint.type != "fixed" and joint.name not in {f"joint{i}" for i in range(1, 8)}
    ]
    model = pin.buildModelFromXML(source.with_fixed_joints(*fixed).load().xml)
    data = model.createData()
    ik = PinocchioIK(
        model, data, model.getJointId("joint7"), PinocchioIKConfig(max_iter=500, eps=0.0002)
    )

    def positions() -> NDArray[np.float64]:
        state = coordinator.get_joint_positions()
        return np.array([state[name] for name in arm])

    def frame(name: str) -> Any:
        pin.forwardKinematics(model, data, positions())
        pin.updateFramePlacements(model, data)
        return data.oMf[model.getFrameId(name)].copy()

    # The red mask is a scripted fixture selector. No object TF, scene or seed is read.
    mask = (
        (rgb[:, :, 0] > 100)
        & (rgb[:, :, 1] < 0.6 * rgb[:, :, 0])
        & (rgb[:, :, 2] < 0.6 * rgb[:, :, 0])
        & np.isfinite(depth)
        & (depth > 0)
    )
    if np.count_nonzero(mask) < 20:
        raise RuntimeError("Red cube not visible in public RGB-D")
    v, u = np.nonzero(mask)
    z = depth[v, u]
    rays = np.vstack(
        (
            (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
            np.ones(len(u)),
        )
    )
    cloud = (frame("link7").homogeneous @ optical @ rays)[:3] + BASE[:, None]
    surface = np.median(cloud, axis=1)
    wrist_to_tcp = ik.forward_kinematics(positions()).actInv(frame("link_tcp"))
    task = coordinator.get_task("joint_trajectory")
    assert isinstance(task, JointTrajectoryTask)

    def move(target: NDArray[np.float64], opening: float, duration: float) -> None:
        assert coordinator.task_invoke("arm_gripper", "set_normalized", {"values": [opening]})
        current = positions()
        duration = max(duration, float(np.max(np.abs(target - current))) * 1.5)
        result = coordinator.execute_trajectory(
            JointTrajectory(
                joint_names=arm,
                points=[
                    TrajectoryPoint(time_from_start=0.0, positions=current.tolist()),
                    TrajectoryPoint(time_from_start=duration, positions=target.tolist()),
                    TrajectoryPoint(time_from_start=duration + 3.0, positions=target.tolist()),
                ],
            )
        )
        assert result.status == TrajectoryExecutionStatus.ACCEPTED, result
        deadline = time.monotonic() + duration + 15
        while time.monotonic() < deadline:
            status = task.get_status(time.monotonic())
            if status.state == TrajectoryState.COMPLETED:
                break
            assert status.state == TrajectoryState.EXECUTING, status.state
            threading.Event().wait(0.05)
        else:
            raise RuntimeError("Arm trajectory did not complete")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and np.max(np.abs(positions() - target)) > 0.02:
            threading.Event().wait(0.05)

    def cartesian(desired: NDArray[np.float64], opening: float, duration: float) -> None:
        target = desired.copy()
        for iteration in range(3):
            pose = pin.SE3(np.diag([1.0, -1.0, -1.0]), target - BASE)
            q, converged, error = ik.solve(pose * wrist_to_tcp.inverse(), positions())
            assert converged, error
            assert np.all(q >= model.lowerPositionLimit) and np.all(q <= model.upperPositionLimit)
            move(np.asarray(q, dtype=np.float64), opening, duration if iteration == 0 else 1.5)
            delta = desired - (frame("link_tcp").translation + BASE)
            if np.linalg.norm(delta) < 0.003:
                return
            target += delta
        assert np.linalg.norm(delta) < 0.008, delta

    move(positions(), 1.0, 1.0)
    approach = surface.copy()
    approach[2] += 0.12
    cartesian(approach, 1.0, 4.0)
    grasp = surface.copy()
    grasp[2] -= 0.04
    cartesian(grasp, 1.0, 3.0)
    move(positions(), 0.0, 2.0)
    raised = grasp.copy()
    raised[2] += 0.12
    cartesian(raised, 0.0, 3.0)
    # Keep the normal gripper task active until the sensor link ends. Only the host
    # evaluator reads whether the privately recorded object actually rose.
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        try:
            positions()
        except RuntimeError:
            return
        threading.Event().wait(0.1)
    raise RuntimeError("Trusted episode did not end")
