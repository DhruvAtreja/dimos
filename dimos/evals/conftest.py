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

from collections.abc import Callable
from pathlib import Path

import pytest

from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import make_vector3


@pytest.fixture(autouse=True)
def shell(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Run as if the evals started from a shell with no viewer settings.

    Call ``shell(VIEWER="rerun")`` to set some for the rest of the test.
    """
    for name in (
        "VIEWER",
        "RERUN_OPEN",
        "RERUN_WEB",
        "SIMULATIONMODULE__VIEWER",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    def set_env(**env: str) -> None:
        for name, value in env.items():
            monkeypatch.setenv(name, value)

    return set_env


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    """A tiny on-disk memory dataset: 5 odom poses walking 4m in +x over 4s."""
    path = tmp_path / "tiny.db"
    with SqliteStore(path=str(path)) as store:
        stream = store.stream("odom", PoseStamped)
        for i in range(5):
            pose = PoseStamped(
                position=make_vector3(float(i), 0.0, 0.0),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame_id="world",
            )
            stream.append(pose, ts=1000.0 + i)
    return str(path)
