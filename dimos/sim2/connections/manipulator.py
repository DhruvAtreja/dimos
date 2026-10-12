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

"""Position-controlled arm/gripper connection in the robot's declared units."""

from __future__ import annotations

from typing import Any

from dimos.control.connection.connection_module import ConnectionModule
from dimos.control.contract.description import ControlDescription
from dimos.core.core import rpc
from dimos.core.module import ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.sim2.control.adapters import ManipulatorAdapter
from dimos.sim2.spec import RobotConfig


class SimManipulatorConnectionConfig(ModuleConfig):
    definition: RobotConfig
    address: str
    robot_id: str
    rate_hz: float = 100.0
    deadman_timeout_s: float = 0.2


class SimManipulatorConnection(ConnectionModule):
    """Emulated manipulator position commands and measured joint state."""

    config: SimManipulatorConnectionConfig
    position_command: In[JointState]
    joint_state: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._device = ManipulatorAdapter(
            address=self.config.address,
            dof=len(self.config.definition.joints),
            definition=self.config.definition,
        )
        self._sequence = -1
        self._description = self.describe()

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
        return {
            f"{self.config.robot_id}/{resource.name}/{field}": float(frame.values[field][i])
            for i, resource in enumerate(self._description.resources)
            for field in resource.state_interfaces
        }

    def write(self, values: dict[str, float]) -> None:
        # Partial position messages leave other joints at their measured position.
        positions = self._device.read_joint_positions()
        for i, name in enumerate(self._description.joint_names()):
            positions[i] = values.get(f"{name}/position", positions[i])
        if not self._device.write_joint_positions(positions):
            raise RuntimeError("sim2 manipulator device is not ready")

    def halt(self) -> None:
        if not self._device.write_stop():
            raise RuntimeError("sim2 manipulator device is not ready")

    @rpc
    def stop(self) -> None:
        try:
            super().stop()
        finally:
            self._device.disconnect()
