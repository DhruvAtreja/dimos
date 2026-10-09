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

"""Shared-memory segment ownership and frame coherence across processes; no MuJoCo needed."""

import json
import logging
from multiprocessing.shared_memory import SharedMemory
import os
import subprocess
import sys
import textwrap
import time
import uuid

import numpy as np
import pytest

from dimos.sim2.ipc.abi import ChannelDescriptor, make_channel_descriptor
from dimos.sim2.ipc.channel import ChannelFrame, RobotChannel, _untrack
from dimos.sim2.spec import ControlInterface

_CHANNEL_LOGGER = "dimos/sim2/ipc/channel.py"

# Creates the channel, then dies without unlinking it, like a killed physics
# worker. A killed worker leaves no resource tracker to tidy up after it, so the
# interpreter's own tracker is told to forget the segment before exit.
_ABANDON = textwrap.dedent(
    """
    import json, os, sys
    from multiprocessing import resource_tracker
    from dimos.sim2.ipc.abi import ChannelDescriptor
    from dimos.sim2.ipc.channel import RobotChannel

    channel = RobotChannel.create(ChannelDescriptor.from_dict(json.loads(sys.argv[1])))
    channel.set_lifecycle("ready")
    resource_tracker.unregister(channel._shm._name, "shared_memory")
    os._exit(0)
    """
)


def _descriptor() -> ChannelDescriptor:
    tag = uuid.uuid4().hex[:12]
    return make_channel_descriptor(
        sim_id=f"test-{tag}",
        robot_id="arm",
        generation="test",
        shm_name=f"dms2_test_{tag}",
        control_interface=ControlInterface.MANIPULATOR,
        dof=2,
        physics_dt=0.005,
        control_decimation=1,
    )


def _segment_exists(descriptor: ChannelDescriptor) -> bool:
    try:
        shm = SharedMemory(name=descriptor.shm_name, create=False)
    except FileNotFoundError:
        return False
    _untrack(shm)
    shm.close()
    return True


@pytest.fixture
def channel_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    lg = logging.getLogger(_CHANNEL_LOGGER)
    lg.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger=_CHANNEL_LOGGER)
    try:
        yield caplog
    finally:
        lg.removeHandler(caplog.handler)


def test_create_replaces_segment_whose_owner_died(
    channel_logs: pytest.LogCaptureFixture,
) -> None:
    descriptor = _descriptor()
    subprocess.run(
        [sys.executable, "-c", _ABANDON, json.dumps(descriptor.to_dict())],
        check=True,
        timeout=60,
    )
    assert _segment_exists(descriptor)

    channel = RobotChannel.create(descriptor)
    try:
        assert channel.owner_pid == os.getpid()
        assert channel.lifecycle == "starting"
        warnings = [r for r in channel_logs.records if "stale" in str(r.msg)]
        assert len(warnings) == 1
        assert descriptor.shm_name in str(warnings[0].msg)
    finally:
        channel.release()
    assert not _segment_exists(descriptor)


def test_create_refuses_segment_with_live_owner() -> None:
    descriptor = _descriptor()
    channel = RobotChannel.create(descriptor)
    try:
        channel.set_lifecycle("ready")
        with pytest.raises(RuntimeError, match=f"running process {os.getpid()}"):
            RobotChannel.create(descriptor)
        # The refusal must leave the live channel untouched.
        reader = RobotChannel.attach(descriptor)
        assert reader.lifecycle == "ready"
        assert reader.owner_pid == os.getpid()
        reader.close()
    finally:
        channel.release()
    assert not _segment_exists(descriptor)


def test_release_unlinks_even_when_an_earlier_step_fails() -> None:
    descriptor = _descriptor()
    channel = RobotChannel.create(descriptor)
    channel.close()  # the lifecycle write will now fail
    channel.release()
    assert not _segment_exists(descriptor)


# Attaches to the channel named in argv[1] and publishes action frames whose
# every number is the frame's index, so a mixed (torn) frame is easy to spot.
# The struct module seen by the channel is swapped for one that can pause the
# writer right after a chosen header or sequence store, as if the writer had
# been descheduled there.
#   mode "step":   publish frame 1, print "ready"; then for each stdin line k,
#                  publish the next frame, pausing after its k-th store (print
#                  "paused", wait for a line), and print "done <stores>".
#   mode "random": publish for argv[3] seconds, pausing 1 ms after a store with
#                  probability argv[4]; print the number of frames published.
_PAUSING_WRITER = textwrap.dedent(
    """
    import json, random, sys, time
    import numpy as np
    import dimos.sim2.ipc.channel as channel_module
    from dimos.sim2.ipc.abi import ChannelDescriptor
    from dimos.sim2.ipc.channel import FrameMetadata, RobotChannel

    real = channel_module.struct

    class PausingStruct:
        Struct = real.Struct
        unpack_from = staticmethod(real.unpack_from)
        stores = 0
        pause_after = 0
        pause_probability = 0.0
        rng = random.Random(0)

        @classmethod
        def pack_into(cls, *args):
            real.pack_into(*args)
            cls.stores += 1
            if cls.stores == cls.pause_after:
                print("paused", flush=True)
                sys.stdin.readline()
            elif cls.rng.random() < cls.pause_probability:
                time.sleep(0.001)

    channel_module.struct = PausingStruct
    channel = RobotChannel.attach(ChannelDescriptor.from_dict(json.loads(sys.argv[1])))
    dof = channel.descriptor.dof
    meta = FrameMetadata(0, 1, 0, 0, 0.0)

    def publish(n):
        channel.publish_action(
            {
                "command_mode": [n],
                "enabled": [1],
                "position": np.full(dof, float(n)),
                "velocity": np.full(dof, float(n)),
                "effort": np.full(dof, float(n)),
                "velocity_scale": [float(n)],
                "gripper": [float(n)],
            },
            meta,
        )

    if sys.argv[2] == "step":
        publish(1)
        print("ready", flush=True)
        n = 1
        while line := sys.stdin.readline():
            n += 1
            PausingStruct.stores = 0
            PausingStruct.pause_after = int(line)
            publish(n)
            print("done", PausingStruct.stores, flush=True)
    else:
        PausingStruct.pause_probability = float(sys.argv[4])
        stop = time.monotonic() + float(sys.argv[3])
        n = 0
        while time.monotonic() < stop:
            n += 1
            publish(n)
        print(n, flush=True)
    channel.close()
    """
)


def _frame_index(frame: ChannelFrame) -> int:
    """Index the test writer stored in every field; fails if the fields disagree."""
    numbers = {
        float(number)
        for name, value in frame.values.items()
        if name != "enabled"
        for number in np.ravel(value)
    }
    assert len(numbers) == 1, f"torn frame: {frame.values}"
    return int(numbers.pop())


def _start_writer(descriptor: ChannelDescriptor, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _PAUSING_WRITER, json.dumps(descriptor.to_dict()), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )


def test_writer_paused_mid_publish_never_blocks_the_reader() -> None:
    descriptor = _descriptor()
    channel = RobotChannel.create(descriptor)
    writer = _start_writer(descriptor, "step")
    assert writer.stdin is not None and writer.stdout is not None
    try:
        channel.set_lifecycle("ready")
        assert writer.stdout.readline().strip() == "ready"
        published = 1
        for store in (1, 2, 3):
            writer.stdin.write(f"{store}\n")
            writer.stdin.flush()
            assert writer.stdout.readline().strip() == "paused"
            frame = channel.read_action()
            assert frame is not None
            # Either the previous frame or the one being published, never a mix.
            assert _frame_index(frame) in (published, published + 1)
            writer.stdin.write("\n")
            writer.stdin.flush()
            # Every store of the publish was offered as a pause point.
            assert writer.stdout.readline().split() == ["done", "3"]
            published += 1
            frame = channel.read_action()
            assert frame is not None and _frame_index(frame) == published
        writer.stdin.close()
        assert writer.wait(timeout=30) == 0
    finally:
        writer.kill()
        channel.release()


def test_reads_stay_whole_while_a_stalling_writer_publishes() -> None:
    descriptor = _descriptor()
    channel = RobotChannel.create(descriptor)
    channel.set_lifecycle("ready")
    writer = _start_writer(descriptor, "random", "1.0", "0.02")
    assert writer.stdout is not None
    try:
        reads = 0
        last = 0
        while writer.poll() is None:
            frame = channel.read_action()
            if frame is None:
                time.sleep(0.001)
                continue
            index = _frame_index(frame)
            assert index >= last
            last = index
            reads += 1
        published = int(writer.stdout.read().strip())
        assert writer.returncode == 0
        assert reads > 100
        assert published > 100
    finally:
        writer.kill()
        channel.release()
