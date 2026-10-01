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

from dimos.hardware.sensors.camera.v4l2_camera import V4L2CameraModule

# The VI port numbers are fixed by the device tree, so by-path pins each eye to
# its GMSL link however the nodes are numbered.
_HEAD_V4L2 = "/dev/v4l/by-path/platform-tegra-capture-vi-video-index{index}"
HEAD_LEFT_V4L2 = _HEAD_V4L2.format(index=0)
HEAD_RIGHT_V4L2 = _HEAD_V4L2.format(index=10)

HEAD_WIDTH = 1920
HEAD_HEIGHT = 1536
HEAD_FPS = 30.0
HEAD_FOURCC = "UYVY"


# Distinct classes only because blueprints can't yet run two instances of one
# module (same reason as the wrist cameras).
class HeadLeftCamera(V4L2CameraModule):
    pass


class HeadRightCamera(V4L2CameraModule):
    pass
