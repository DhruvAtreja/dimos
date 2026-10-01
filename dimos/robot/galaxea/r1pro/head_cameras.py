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

"""The R1 Pro's two head cameras, colour read straight off V4L2.

The head is a stereo pair of GMSL2 cameras (SENSING SG3S-ISX031C-GMSL2F, a Sony
ISX031 with its own ISP) behind a MAX96724 deserializer on the Jetson. It is
not a ZED. The ISP delivers finished UYVY frames to the Tegra VI, which exposes
them as plain V4L2 capture nodes, so ``V4L2CameraModule`` reads them as it
reads a UVC camera; nvarguscamerasrc does not apply, as it needs raw Bayer.

Only 1920x1536 at 30 fps is real. The driver lists smaller sizes and 60 fps,
but a smaller size is a corrupted crop of the full frame and 60 fps still
delivers 30. Galaxea's stereo calibration (``/opt/galaxea/body/stereo*.yaml``)
is for this size.

The vendor head pane (``hdas``, ``start_signal_camera_head.sh``) must not be
holding the cameras; while it is, these modules log and retry.
"""

from __future__ import annotations

import os
import subprocess
import sys

from dimos.core.core import rpc
from dimos.hardware.sensors.camera.v4l2_camera import V4L2CameraModule
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# The VI port numbers are fixed by the device tree, so by-path pins each eye to
# its GMSL link however the nodes are numbered.
_HEAD_V4L2 = "/dev/v4l/by-path/platform-tegra-capture-vi-video-index{index}"
HEAD_LEFT_V4L2 = _HEAD_V4L2.format(index=0)
HEAD_RIGHT_V4L2 = _HEAD_V4L2.format(index=10)

HEAD_WIDTH = 1920
HEAD_HEIGHT = 1536
HEAD_FPS = 30.0
HEAD_FOURCC = "UYVY"


# MIIVII's GMSL SDK, whose trigger MCU drives the cameras' FSYNC.
_MIIVII_GMSL_SDK = "/opt/miivii/lib/libmvgmslcamera_noopencv.so"
# Every link on the head's deserializer: a mask of only the two head links stops them streaming.
_TRIGGER_LINKS = 0x0F

# MvGmslCamera's config-only constructor hands the trigger config to GmslServer. Run in a
# throwaway interpreter: the SDK leaves a reader thread behind, and its destructor crashes on
# cameras it never opened.
_TRIGGER_SCRIPT = """
import ctypes, os, sys, time
class Config(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint8) for name in ("sync_camera_num", "sync_freq", "sync_camera_bit_draw", "async_camera_num", "async_freq", "async_camera_bit_draw")] + [("async_camera_pos", ctypes.c_uint8 * 8)]
sdk, links, hz = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
camera = ctypes.create_string_buffer(1 << 16)
ctypes.CDLL(sdk)._ZN6miivii12MvGmslCameraC1E23sync_out_a_cfg_client_t(camera, Config(bin(links).count("1"), hz, links))
time.sleep(1)
os._exit(0)
"""


def trigger_head_cameras(hz: int = int(HEAD_FPS)) -> None:
    """Fire both head cameras from one hardware trigger, so their frames start within ~30 us."""
    if not os.path.exists(_MIIVII_GMSL_SDK):
        logger.warning("no MIIVII GMSL SDK at %s; head cameras stay free-running", _MIIVII_GMSL_SDK)
        return
    subprocess.run(
        [sys.executable, "-c", _TRIGGER_SCRIPT, _MIIVII_GMSL_SDK, str(_TRIGGER_LINKS), str(hz)],
        check=True,
        timeout=10,
    )


# Distinct classes only because blueprints can't yet run two instances of one
# module (same reason as the wrist cameras).
class HeadLeftCamera(V4L2CameraModule):
    @rpc
    def start(self) -> None:
        trigger_head_cameras()
        super().start()


class HeadRightCamera(V4L2CameraModule):
    @rpc
    def start(self) -> None:
        trigger_head_cameras()
        super().start()
