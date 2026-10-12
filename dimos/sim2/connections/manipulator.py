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

"""Position-controlled arm/gripper device streams in the robot's declared units."""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.sim2.control.adapters import ManipulatorAdapter
from dimos.sim2.spec import RobotConfig
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class SimManipulatorConnectionConfig(ModuleConfig):
    definition: RobotConfig
    address: str
    command_source: Literal["coordinator", "stream"] = "coordinator"
    rate_hz: float = 50.0


class SimManipulatorConnection(Module):
    """Emulated manipulator position commands and measured joint state."""

    config: SimManipulatorConnectionConfig
    joint_command: In[JointState]
    joint_states: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._device = ManipulatorAdapter(
            address=self.config.address,
            dof=len(self.config.definition.joints),
            definition=self.config.definition,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._device.connect()
        if self.config.command_source == "stream":
            self.register_disposable(Disposable(self.joint_command.subscribe(self._command)))
        self._stop.clear()
        self._thread = threading.Thread(target=self._publish, daemon=True)
        self._thread.start()

    def _command(self, command: JointState) -> None:
        expected = [j.name for j in self.config.definition.joints]
        if list(command.name) != expected:
            raise ValueError("joint_command names must match the complete device order")
        self._device.write_joint_positions(list(command.position))

    def _publish(self) -> None:
        try:
            while not self._stop.is_set():
                before = time.monotonic()
                values = self._device.sample().values
                self.joint_states.publish(
                    JointState(
                        name=[j.name for j in self.config.definition.joints],
                        position=values["position"].tolist(),
                        velocity=values["velocity"].tolist(),
                        effort=values["effort"].tolist(),
                        ts=float(values["wall_time"][0]),
                    )
                )
                self._stop.wait(max(0, 1.0 / self.config.rate_hz - (time.monotonic() - before)))
        except Exception:
            if not self._stop.is_set():
                logger.exception("sim2 manipulator connection stopped")

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._device.disconnect()
        super().stop()
