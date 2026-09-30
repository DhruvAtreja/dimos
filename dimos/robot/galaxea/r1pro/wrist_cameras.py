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

"""The R1 Pro's wrist D405 colour streams, read straight off V4L2.

Galaxea's RealSense nodes stall on this host once both wrists stream depth and
colour (four isochronous streams on one xHCI), while the bare colour nodes run
at sensor rate side by side. So the colour comes from ``V4L2CameraModule`` and
the vendor wrist pane (``hdas``, ``start_realsense_camera_r1pro.sh``) must not
be holding the cameras; while it is, these modules log and retry.
"""

from __future__ import annotations

from dimos.hardware.sensors.camera.v4l2_camera import V4L2CameraModule

# Each D405 exposes six video nodes; index4 is its colour stream. by-path pins
# the node to the USB port (port 1 right, port 2 left) so a re-plug cannot swap
# the wrists.
_WRIST_V4L2 = (
    "/dev/v4l/by-path/platform-14160000.pcie-pci-0004:01:00.0-usb-0:{port}:1.0-video-index4"
)
WRIST_LEFT_V4L2 = _WRIST_V4L2.format(port=2)
WRIST_RIGHT_V4L2 = _WRIST_V4L2.format(port=1)


# Distinct classes only because blueprints can't yet run two instances of one
# module (same reason the hosted xArm blueprints declare Front/WristCamera).
class WristLeftCamera(V4L2CameraModule):
    pass


class WristRightCamera(V4L2CameraModule):
    pass
