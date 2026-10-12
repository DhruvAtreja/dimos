# Copyright 2025-2026 Dimensional Inc.
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

import struct
from uuid import uuid4

import pytest

from dimos.sim2.control.interface import descriptor
from dimos.sim2.ipc.channel import _CREATOR_PID_OFFSET, _OWNER_PID_OFFSET, RobotChannel
from dimos.sim2.spec import ControlInterface


def _dead_pid() -> int:
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_create_replaces_a_channel_whose_creator_died():
    spec = descriptor(f"orphan-{uuid4().hex[:8]}/robot", ControlInterface.WHOLE_BODY, 3)
    leaked = RobotChannel.create(spec)
    dead = _dead_pid()
    struct.pack_into("<Q", leaked._shm.buf, _CREATOR_PID_OFFSET, dead)
    struct.pack_into("<Q", leaked._shm.buf, _OWNER_PID_OFFSET, dead)
    leaked._shm.close()  # a crashed run: the segment outlives its owner, never unlinked
    channel = RobotChannel.create(spec)
    try:
        assert channel.lifecycle == "starting"
    finally:
        channel.close()


def test_create_still_refuses_a_live_channel():
    spec = descriptor(f"live-{uuid4().hex[:8]}/robot", ControlInterface.WHOLE_BODY, 3)
    live = RobotChannel.create(spec)
    try:
        with pytest.raises(FileExistsError):
            RobotChannel.create(spec)
    finally:
        live.close()
