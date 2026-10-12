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

"""Sensor scheduling runs independently from physics and reports capture failures."""

from __future__ import annotations

from abc import ABC, abstractmethod
import queue
import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.sim2.module import SimulationSpec
from dimos.sim2.sensors.reader import WorldReader
from dimos.utils.logging_config import setup_logger
from dimos.utils.realtime import make_current_thread_periodic

logger = setup_logger()


class SensorModuleConfig(ModuleConfig):
    robot_id: str
    rate_hz: float = 10.0


class SensorModule(Module, ABC):
    config: SensorModuleConfig
    _simulation: SimulationSpec
    dedicated_worker = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready: queue.Queue[Exception | None] = queue.Queue(maxsize=1)
        self._error: str | None = None
        self._error_lock = threading.Lock()
        self.reader: WorldReader

    @rpc
    def start(self) -> None:
        super().start()
        description = self._simulation.describe()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(description,), daemon=True)
        self._thread.start()
        error = self._ready.get(timeout=60)
        if error is not None:
            raise error

    def _run(self, description: dict[str, Any]) -> None:
        make_current_thread_periodic(1.0 / self.config.rate_hz)
        opened = False
        reader_created = False
        try:
            self.reader = WorldReader(description)
            reader_created = True
            self.open()
            opened = True
            self._ready.put(None)
            while not self._stop.is_set():
                before = time.monotonic()
                if self.reader.update():
                    self.capture()
                self._stop.wait(max(0.0, 1.0 / self.config.rate_hz - (time.monotonic() - before)))
        except Exception as error:
            with self._error_lock:
                self._error = str(error)
            if not opened:
                self._ready.put(error)
            logger.exception("sim2 sensor failed", module=type(self).__name__)
        finally:
            self.close_sensor()
            if reader_created:
                self.reader.close()

    def open(self) -> None:
        pass

    @abstractmethod
    def capture(self) -> None:
        raise NotImplementedError

    def close_sensor(self) -> None:
        pass

    @rpc
    def sensor_status(self) -> dict[str, Any]:
        with self._error_lock:
            return {"error": self._error}

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("sensor worker did not stop")
        super().stop()
