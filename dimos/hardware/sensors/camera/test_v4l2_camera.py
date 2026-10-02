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

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from dimos.core.module import Module
from dimos.hardware.sensors.camera.v4l2_camera import (
    CaptureClock,
    V4L2CameraModule,
    capture_source,
)
from dimos.msgs.sensor_msgs.Image import ImageFormat


def test_by_path_link_resolves_to_the_node_index(tmp_path: Any) -> None:
    """cv2 only reliably opens V4L2 devices by index, so links are resolved."""
    link = tmp_path / "platform-usb-0:2:1.0-video-index4"
    link.symlink_to("/dev/video18")

    assert capture_source(str(link)) == 18
    assert capture_source("/dev/video7") == 7
    assert capture_source("rtsp://camera/stream") == "rtsp://camera/stream"


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> V4L2CameraModule:
    def _fake_init(self: Any, **kwargs: Any) -> None:
        self.config = SimpleNamespace(
            device="/dev/v4l/by-path/fake-video-index4",
            width=848,
            height=480,
            fps=30.0,
            fourcc="YUYV",
            frame_id="wrist_left_optical",
            retry_s=0.0,
            max_missed_reads=3,
            stats_period_s=0.0,
            jpeg_quality=None,
        )

    monkeypatch.setattr(Module, "__init__", _fake_init)
    module = V4L2CameraModule()
    module.image_out = MagicMock()
    module.jpeg_out = MagicMock()
    return module


class _FakeCapture:
    """Enough of cv2.VideoCapture for the read loop: a scripted read sequence."""

    def __init__(self, reads: list[bool], opened: bool = True) -> None:
        self._reads = list(reads)
        self._opened = opened
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # cv2 spelling
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray[Any, Any] | None]:
        if not self._reads:
            return False, None
        ok = self._reads.pop(0)
        return (True, np.zeros((480, 848, 3), dtype=np.uint8)) if ok else (False, None)

    def release(self) -> None:
        self.released = True


def _published(module: V4L2CameraModule) -> MagicMock:
    return cast("MagicMock", module.image_out.publish)


def _fake_cv2(cap: _FakeCapture) -> SimpleNamespace:
    return SimpleNamespace(
        VideoCapture=lambda *_: cap,
        CAP_V4L2=200,
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        VideoWriter_fourcc=lambda *c: 0,
    )


def test_frames_are_published_as_bgr_images(module: V4L2CameraModule) -> None:
    cap = _FakeCapture(reads=[True, True, False, False, False])
    cap.set(3, 848)
    cap.set(4, 480)
    cap.set(5, 30.0)
    assert module._open(_fake_cv2(cap)) is cap

    module._pump(cap)  # returns once max_missed_reads failures pile up

    assert _published(module).call_count == 2
    image = _published(module).call_args[0][0]
    assert image.format == ImageFormat.BGR
    assert (image.width, image.height) == (848, 480)
    assert image.frame_id == "wrist_left_optical"


def test_frames_are_published_as_jpeg_at_the_configured_quality(module: V4L2CameraModule) -> None:
    module.config.jpeg_quality = 90
    cap = _FakeCapture(reads=[True, True, False, False, False])

    module._pump(cap)

    assert _published(module).call_count == 0
    jpegs = [call[0][0] for call in cast("MagicMock", module.jpeg_out.publish).call_args_list]
    assert len(jpegs) == 2
    assert jpegs[0].format == "jpeg" and jpegs[0].frame_id == "wrist_left_optical"
    decoded = jpegs[0].decode()
    assert (decoded.width, decoded.height) == (848, 480)


def test_a_missed_read_between_frames_is_tolerated(module: V4L2CameraModule) -> None:
    cap = _FakeCapture(reads=[True, False, True, False, False, False])

    module._pump(cap)

    assert _published(module).call_count == 2


def test_device_held_by_another_process_does_not_raise(module: V4L2CameraModule) -> None:
    """The vendor node still owning the camera is a wait, not a crash."""
    cap = _FakeCapture(reads=[], opened=False)

    assert module._open(_fake_cv2(cap)) is None
    assert cap.released
    assert _published(module).call_count == 0


def test_capture_clock_converts_the_driver_clock_exactly() -> None:
    """Stamps move onto the wall clock by the driver clock's offset, not by when frames arrived."""
    now = {"monotonic": 500.0}
    monotonic = lambda: now["monotonic"]  # noqa: E731
    soc_counter = lambda: now["monotonic"] + 14.0  # noqa: E731  # the Jetson's GMSL capture clock
    wall = lambda: now["monotonic"] + 1000.0  # noqa: E731
    clock = CaptureClock([monotonic, soc_counter], wall)

    stamps = []
    for capture, delay in ((514.0, 0.030), (514.0 + 1 / 30, 0.009), (514.0 + 2 / 30, 0.050)):
        now["monotonic"] = capture - 14.0 + delay
        stamps.append(clock.stamp(capture))

    # Frame k started at 500 + k/30 on CLOCK_MONOTONIC, however late it was read.
    assert stamps == pytest.approx([1500.0, 1500.0 + 1 / 30, 1500.0 + 2 / 30], abs=1e-9)


def test_capture_clock_gives_up_on_an_unknown_clock() -> None:
    assert CaptureClock([lambda: 500.0], time.time).stamp(9999.0) is None


def test_frames_carry_the_driver_capture_time(module: V4L2CameraModule) -> None:
    """Frames are stamped when the driver captured them, not when they were read."""
    captured_s: list[float] = []

    class _StampedCapture(_FakeCapture):
        def read(self) -> tuple[bool, np.ndarray[Any, Any] | None]:
            time.sleep(1 / 30)
            # uvcvideo stamps on CLOCK_MONOTONIC; this frame was captured 20 ms ago.
            captured_s.append(time.monotonic() - 0.020)
            self.props[0] = captured_s[-1] * 1e3  # CAP_PROP_POS_MSEC
            return super().read()

    module._pump(_StampedCapture(reads=[True, True, False, False, False]))

    stamps = [call[0][0].ts for call in _published(module).call_args_list]
    wall_minus_monotonic = time.time() - time.monotonic()
    # Only the wall/monotonic offset is re-read per frame; on a loaded CI machine that pair can be preempted ~0.1 ms.
    assert len(stamps) == 2
    assert stamps == pytest.approx([c + wall_minus_monotonic for c in captured_s[:2]], abs=1e-3)
