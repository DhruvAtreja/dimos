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

"""Private SHM devices used by sim2 connection modules, not coordinator adapters."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from dimos.hardware.manipulators.spec import ControlMode, ManipulatorInfo
from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.spec import POS_STOP, VEL_STOP, IMUState, MotorCommand, MotorState
from dimos.sim2.control.interface import connect, observation
from dimos.sim2.ipc.channel import ChannelFrame, FrameMetadata, RobotChannel
from dimos.sim2.spec import ControlInterface, RobotConfig


class DeviceAdapter:
    interface: ControlInterface

    def __init__(
        self, address: str, dof: int, definition: RobotConfig | dict[str, Any], **_: Any
    ) -> None:
        self.address = address
        self.dof = dof
        self.definition = TypeAdapter(RobotConfig).validate_python(definition)
        self.channel: RobotChannel | None = None
        self._sample: ChannelFrame | None = None
        self._lock = threading.RLock()

    def connect(self) -> bool:
        with self._lock:
            self.channel = connect(self.address, self.interface, self.dof)
            self._sample = observation(self.channel)
        return True

    def disconnect(self) -> None:
        with self._lock:
            if self.channel is not None:
                self.channel.close()
                self.channel = None
                self._sample = None

    def is_connected(self) -> bool:
        with self._lock:
            return self.channel is not None and self.channel.lifecycle == "ready"

    def sample(self) -> ChannelFrame:
        with self._lock:
            if self.channel is None:
                raise RuntimeError("sim2 device is disconnected")
            self._sample = observation(self.channel)
            return self._sample

    def _write(self, values: dict[str, Any]) -> bool:
        with self._lock:
            if self.channel is None or self.channel.lifecycle != "ready":
                return False
            if self._sample is None:
                return False
            self.channel.publish_action(
                values,
                FrameMetadata(
                    0,
                    self._sample.metadata.episode_id,
                    0,
                    0,
                    self._sample.metadata.sim_time,
                ),
            )
            return True


class WholeBodyAdapter(DeviceAdapter):
    interface = ControlInterface.WHOLE_BODY

    def has_motor_states(self) -> bool:
        return self.is_connected()

    def read_motor_states(self) -> list[MotorState]:
        values = self.sample().values
        return [
            MotorState(*row)
            for row in zip(
                values["position"],
                values["velocity"],
                values["effort"],
                strict=True,
            )
        ]

    def read_imu(self) -> IMUState:
        # Joints and IMU must use the same observation frame.
        with self._lock:
            if self._sample is None:
                raise RuntimeError("sim2 device is disconnected")
            v = self._sample.values
            return IMUState(
                quaternion=tuple(v["imu_quaternion"]),
                gyroscope=tuple(v["imu_gyroscope"]),
                accelerometer=tuple(v["imu_accelerometer"]),
                rpy=tuple(v["imu_rpy"]),
            )

    def get_limits(self) -> JointLimits | None:
        return None

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        if len(commands) != self.dof:
            raise ValueError(f"expected {self.dof} motor commands")
        values = np.array([[c.q, c.dq, c.kp, c.kd, c.tau] for c in commands])
        if not np.isfinite(values).all():
            raise ValueError("motor commands must be finite")
        q, dq, kp, kd, ff = values.T.copy()
        kp[q == POS_STOP] = 0
        q[q == POS_STOP] = 0
        kd[dq == VEL_STOP] = 0
        dq[dq == VEL_STOP] = 0
        return self._write(dict(enabled=[1], position=q, velocity=dq, kp=kp, kd=kd, effort=ff))


class ManipulatorAdapter(DeviceAdapter):
    interface = ControlInterface.MANIPULATOR

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._enabled = True
        self._mode = ControlMode.POSITION

    def activate(self) -> bool:
        return self.write_enable(True)

    def deactivate(self) -> bool:
        with self._lock:
            self._enabled = False
            if not self.is_connected():
                return True
            return self.write_stop()

    def get_info(self) -> ManipulatorInfo:
        return ManipulatorInfo(vendor="DimOS", model="sim2", dof=self.dof)

    def get_dof(self) -> int:
        return self.dof

    def get_limits(self) -> JointLimits:
        return JointLimits(
            position_lower=[j.lower for j in self.definition.joints],
            position_upper=[j.upper for j in self.definition.joints],
            velocity_max=[j.max_velocity for j in self.definition.joints],
        )

    def set_control_mode(self, mode: ControlMode) -> bool:
        if mode not in (ControlMode.POSITION, ControlMode.SERVO_POSITION):
            return False
        with self._lock:
            self._mode = mode
        return True

    def get_control_mode(self) -> ControlMode:
        with self._lock:
            return self._mode

    def read_joint_positions(self) -> list[float]:
        return [float(v) for v in self.sample().values["position"]]

    def read_joint_velocities(self) -> list[float]:
        with self._lock:
            if self._sample is None:
                raise RuntimeError("sim2 device is disconnected")
            return [float(v) for v in self._sample.values["velocity"]]

    def read_joint_efforts(self) -> list[float]:
        with self._lock:
            if self._sample is None:
                raise RuntimeError("sim2 device is disconnected")
            return [float(v) for v in self._sample.values["effort"]]

    def read_state(self) -> dict[str, int]:
        return {"state": int(any(abs(v) > 1e-4 for v in self.read_joint_velocities())), "mode": 0}

    def read_error(self) -> tuple[int, str]:
        return (0, "") if self.is_connected() else (1, "sim2 device is disconnected")

    def write_joint_positions(self, positions: list[float], velocity: float = 1.0) -> bool:
        if len(positions) != self.dof or not np.isfinite(positions).all():
            raise ValueError(f"expected {self.dof} finite joint positions")
        with self._lock:
            return self._write(
                dict(
                    enabled=[int(self._enabled)],
                    command_mode=[0],
                    position=positions,
                    velocity=np.zeros(self.dof),
                    effort=np.zeros(self.dof),
                    velocity_scale=[velocity],
                    gripper=[0.0],
                )
            )

    def write_joint_velocities(self, velocities: list[float]) -> bool:
        return False

    def write_joint_efforts(self, efforts: list[float]) -> bool:
        return False

    def write_stop(self) -> bool:
        return self.write_joint_positions(self.read_joint_positions())

    def write_enable(self, enable: bool) -> bool:
        with self._lock:
            self._enabled = enable
            return self.write_stop()

    def read_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def write_clear_errors(self) -> bool:
        return self.is_connected()

    def read_cartesian_position(self) -> dict[str, float] | None:
        return None

    def write_cartesian_position(self, pose: dict[str, float], velocity: float = 1.0) -> bool:
        return False

    def read_force_torque(self) -> list[float] | None:
        return None
