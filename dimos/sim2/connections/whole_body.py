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

"""PD motor connection. Shared memory is private to the simulated device."""

from __future__ import annotations

from typing import Any

from dimos.control.connection.connection_module import ConnectionModule
from dimos.control.contract.convert import IMU_INTERFACES, JOINT_STATE_FIELDS
from dimos.control.contract.description import ControlDescription
from dimos.core.core import rpc
from dimos.core.module import ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.whole_body.spec import MotorCommand
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.sim2.control.adapters import WholeBodyAdapter
from dimos.sim2.sensors.spec import Imu as ImuSensor
from dimos.sim2.spec import RobotConfig


class WholeBodyConnectionConfig(ModuleConfig):
    definition: RobotConfig
    address: str
    robot_id: str
    rate_hz: float = 50.0
    deadman_timeout_s: float = 0.2


class WholeBodyConnection(ConnectionModule):
    config: WholeBodyConnectionConfig
    motor_command: In[MotorCommandArray]
    joint_state: Out[JointState]
    imu: Out[Imu]
    # Full 6-DOF simulated pose, not the contract's planar base odometry.
    root_pose: Out[PoseStamped]
    tf: Out[TFMessage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._device = WholeBodyAdapter(
            address=self.config.address,
            dof=len(self.config.definition.joints),
            definition=self.config.definition,
        )
        self._sequence = -1
        self._description = self.describe()
        self._imu_name = next(
            s.name for s in self.config.definition.sensors if isinstance(s, ImuSensor)
        )

    def describe(self) -> ControlDescription:
        return self.config.definition.describe_control(
            self.config.robot_id,
            rate_hz=self.config.rate_hz,
            deadman_timeout_s=self.config.deadman_timeout_s,
        )

    def connect(self) -> None:
        self._device.connect()

    def read_state(self) -> dict[str, float] | None:
        frame = self._device.sample()
        if frame.metadata.sequence == self._sequence:
            return None
        self._sequence = frame.metadata.sequence
        v = frame.values
        ts = float(v["wall_time"][0])
        values = {
            f"{name}/{field}": float(v[field][i])
            for i, name in enumerate(self._description.joint_names())
            for field in JOINT_STATE_FIELDS
        }
        w, x, y, z = v["imu_quaternion"]
        imu_values = (x, y, z, w, *v["imu_gyroscope"], *v["imu_accelerometer"])
        values.update(
            {
                f"{self.config.robot_id}/{self._imu_name}/{field}": float(value)
                for field, value in zip(IMU_INTERFACES, imu_values, strict=True)
            }
        )
        pos = Vector3(*v["root_position"])
        q = v["root_quaternion"]
        rot = Quaternion(float(q[1]), float(q[2]), float(q[3]), float(q[0]))
        self.root_pose.publish(PoseStamped(position=pos, orientation=rot, frame_id="world", ts=ts))
        self.tf.publish(
            TFMessage(
                Transform(
                    translation=pos,
                    rotation=rot,
                    frame_id="world",
                    child_frame_id=f"{self.config.robot_id}/{self.config.definition.root_body}",
                    ts=ts,
                )
            )
        )
        return values

    def write(self, values: dict[str, float]) -> None:
        commands = [
            MotorCommand(
                *(
                    values[f"{name}/{field}"]
                    for field in ("position", "velocity", "kp", "kd", "effort")
                )
            )
            for name in self._description.joint_names()
        ]
        if not self._device.write_motor_commands(commands):
            raise RuntimeError("sim2 motor device is not ready")

    def halt(self) -> None:
        # Damping stops driving the policy; it does not promise to keep a humanoid upright.
        self._device.sample()
        if not self._device.write_motor_commands(
            [
                MotorCommand(q=0.0, dq=0.0, kp=0.0, kd=j.kd, tau=0.0)
                for j in self.config.definition.joints
            ]
        ):
            raise RuntimeError("sim2 motor device is not ready")

    @rpc
    def stop(self) -> None:
        try:
            super().stop()
        finally:
            self._device.disconnect()
