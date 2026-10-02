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

"""A UVC camera read straight off V4L2, with no vendor SDK in the loop.

Any UVC camera exposes its colour stream as a plain video node the kernel
already knows how to drive. When the vendor stack in front of that node is the
thing failing -- the R1 Pro's two wrist D405s stall inside the RealSense stack
as soon as both stream depth and colour, while the bare colour nodes run at
sensor rate side by side -- this module hands the frames to the rest of dimos
directly. Colour only; depth stays with the vendor driver.

The device is opened lazily and re-opened on any failure, so a camera that is
unplugged, or still held by the vendor driver, logs and waits instead of taking
the blueprint down with it.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import functools
import mmap
import os
import platform
import re
import struct
import sys
import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.CompressedImage import CompressedImage
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def capture_source(device: str) -> int | str:
    """What to hand ``cv2.VideoCapture`` for ``device``.

    OpenCV's V4L2 backend takes a node index on every build; opening by path is
    build-dependent and the wheels dimos runs with refuse it ("can't be used
    to capture by name"). So a by-path link, or a plain ``/dev/videoN``, is
    resolved to its index. Anything else is passed through untouched.
    """
    target = os.path.realpath(device)
    match = re.fullmatch(r"/dev/video(\d+)", target)
    return int(match.group(1)) if match else device


# cv2.CAP_PROP_POS_MSEC: on the V4L2 backend, the driver's capture timestamp of
# the frame last read, in milliseconds.
_CAP_PROP_POS_MSEC = 0


@functools.cache
def _soc_counter_reader() -> Callable[[], float] | None:
    """Seconds on the ARM generic timer (CNTVCT_EL0), or None off Linux on 64-bit ARM.

    Python cannot issue ``mrs``, so this maps two four-instruction routines into an
    executable page and calls them through ctypes; making the page executable has the
    kernel sync the instruction cache.
    """
    if sys.platform != "linux" or platform.machine() != "aarch64":
        return None
    isb, ret = 0xD5033FDF, 0xD65F03C0
    read_counter = (isb, 0xD53BE040, ret, 0)  # mrs x0, cntvct_el0
    read_frequency = (isb, 0xD53BE000, ret, 0)  # mrs x0, cntfrq_el0
    page = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    page.write(struct.pack("<8I", *read_counter, *read_frequency))
    address = ctypes.addressof(ctypes.c_char.from_buffer(page))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mprotect(ctypes.c_void_p(address), mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_EXEC):
        return None
    routine = ctypes.CFUNCTYPE(ctypes.c_uint64)
    counter, frequency = routine(address), routine(address + 16)
    hz = float(frequency())
    _soc_counter_reader.page = page  # type: ignore[attr-defined]  # keep the code mapped
    return lambda: counter() / hz


def _driver_clocks() -> list[Callable[[], float]]:
    """Clocks a V4L2 driver may stamp frames on, as readers of now in seconds."""
    soc_counter = _soc_counter_reader()
    return [time.monotonic, *([soc_counter] if soc_counter else [])]


class CaptureClock:
    """Turns a driver's capture timestamps into wall-clock time, exactly.

    V4L2 stamps each frame on a clock the driver picks: CLOCK_MONOTONIC for uvcvideo,
    the SoC counter (about 14 s ahead) for the Jetson's GMSL capture, though its buffers
    claim CLOCK_MONOTONIC. The first frame picks whichever clock reads within a second
    after its stamp; each stamp is then moved onto the wall clock by that clock's offset,
    re-read per frame between two reads of the clock and kept only when nothing ran in
    between. Use one per open device; a reopen may change clocks.
    """

    # Longest gap between the two clock reads around the wall read for that offset to be trusted.
    _MAX_BRACKET_S = 1e-4

    def __init__(
        self,
        clocks: list[Callable[[], float]] | None = None,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._clocks = _driver_clocks() if clocks is None else clocks
        self._wall = wall
        self._clock: Callable[[], float] | None = None
        self._wall_minus_clock_s: float | None = None

    def stamp(self, capture_s: float) -> float | None:
        """Wall-clock time (s) of a frame captured at ``capture_s``; None if no known clock matches."""
        if self._clock is None:
            self._clock = next((c for c in self._clocks if 0.0 <= c() - capture_s < 1.0), None)
            if self._clock is None:
                return None
        before, wall, after = self._clock(), self._wall(), self._clock()
        if self._wall_minus_clock_s is None or after - before < self._MAX_BRACKET_S:
            self._wall_minus_clock_s = wall - (before + after) / 2
        return capture_s + self._wall_minus_clock_s


class V4L2CameraConfig(ModuleConfig):
    # /dev/videoN, or better a /dev/v4l/by-path link, which pins the camera to
    # its USB port and survives the nodes being renumbered after a re-plug.
    device: str = "/dev/video0"
    width: int = 848
    height: int = 480
    fps: float = 30.0
    fourcc: str = "YUYV"
    frame_id: str = "camera_optical"
    # How long to wait before trying the device again after it could not be
    # opened or stopped delivering.
    retry_s: float = 3.0
    # Consecutive failed reads before the device is closed and reopened.
    max_missed_reads: int = 30
    # Log capture rate and publish cost this often; 0 disables.
    stats_period_s: float = 10.0
    # JPEG quality (1-100) to publish frames compressed on jpeg_out; None publishes raw on image_out.
    jpeg_quality: int | None = None


class V4L2CameraModule(Module):
    """Publish a UVC camera's colour stream as raw ``Image`` frames, or as JPEG when ``jpeg_quality`` is set."""

    config: V4L2CameraConfig

    image_out: Out[Image]
    jpeg_out: Out[CompressedImage]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._open_failed = False

    @rpc
    def start(self) -> None:
        super().start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"V4L2Camera:{self.config.device}"
        )
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # A blocked read returns on the driver's own timeout; do not hold
            # shutdown for it.
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()

    def _run(self) -> None:
        import cv2

        while not self._stop.is_set():
            cap = self._open(cv2)
            if cap is None:
                self._stop.wait(self.config.retry_s)
                continue
            try:
                self._pump(cap)
            finally:
                cap.release()
            self._stop.wait(self.config.retry_s)

    def _open(self, cv2: Any) -> Any | None:
        cfg = self.config
        cap = cv2.VideoCapture(capture_source(cfg.device), cv2.CAP_V4L2)
        if not cap.isOpened():
            if not self._open_failed:
                logger.warning(
                    "cannot open %s: absent, or held by another process such as the "
                    "vendor camera node; retrying every %.0fs",
                    cfg.device,
                    cfg.retry_s,
                )
            self._open_failed = True
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc.ljust(4)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        got = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )
        if got[:2] != (cfg.width, cfg.height):
            logger.warning(
                "%s gave %dx%d instead of %dx%d; the driver picked the nearest mode",
                cfg.device,
                *got[:2],
                cfg.width,
                cfg.height,
            )
        logger.info("opened %s at %dx%d @ %.0f fps", cfg.device, *got)
        self._open_failed = False
        return cap

    def _pump(self, cap: Any) -> None:
        """Read and publish until stopped or the device goes quiet."""
        missed = 0
        clock = CaptureClock()
        # Publish is synchronous through every in-process subscriber, so its
        # duration is the cost of the whole chain hanging off this camera.
        frames = 0
        publish_s = 0.0
        publish_max_s = 0.0
        window_t = time.monotonic()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                missed += 1
                if missed >= self.config.max_missed_reads:
                    logger.warning("%s stopped delivering frames; reopening", self.config.device)
                    return
                continue
            missed = 0
            capture_ms = cap.get(_CAP_PROP_POS_MSEC)
            # A backend without capture timestamps reports 0, or one on an unknown clock; stamp on arrival.
            ts = (clock.stamp(capture_ms / 1e3) if capture_ms > 0 else None) or time.time()
            t0 = time.perf_counter()
            image = Image.from_numpy(
                frame, format=ImageFormat.BGR, frame_id=self.config.frame_id, ts=ts
            )
            if self.config.jpeg_quality is None:
                self.image_out.publish(image)
            else:
                self.jpeg_out.publish(
                    CompressedImage.from_image(image, quality=self.config.jpeg_quality)
                )
            dt = time.perf_counter() - t0
            frames += 1
            publish_s += dt
            publish_max_s = max(publish_max_s, dt)
            elapsed = time.monotonic() - window_t
            if elapsed >= self.config.stats_period_s > 0:
                logger.info(
                    "%s: %.1f fps, publish %.1f ms avg / %.0f ms max",
                    self.config.frame_id,
                    frames / elapsed,
                    1e3 * publish_s / frames,
                    1e3 * publish_max_s,
                )
                frames, publish_s, publish_max_s, window_t = 0, 0.0, 0.0, time.monotonic()
