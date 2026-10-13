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

"""Handheld RealSense recording for tool_mesh: the IR pair, depth, color and IMU.

A camera window shows what the camera sees; its Record and Stop buttons start and end the take.

dimos run realsense-stereo-record
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from reactivex import operators as ops

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.memory.module import Recorder, default_recording_dir
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.perception.stereo.record_window import RecordWindow

if TYPE_CHECKING:
    from reactivex import Observable
    from reactivex.abc import DisposableBase

CAMERA_LINK = "camera_link"
DB_PATH = default_recording_dir() / "stereo.db"
# The default image codec is JPEG: it blurs the emitter dots and cannot hold 16-bit depth.
LOSSLESS_STREAMS = ("infrared_left", "infrared_right", "depth_image")


class StereoRecorder(Recorder):
    """Writes only between start_recording() and stop_recording()."""

    infrared_left: In[Image]
    infrared_right: In[Image]
    infrared_left_camera_info: In[CameraInfo]
    infrared_right_camera_info: In[CameraInfo]
    color_image: In[Image]
    camera_info: In[CameraInfo]
    depth_image: In[Image]
    depth_camera_info: In[CameraInfo]
    imu: In[Imu]

    _recording = False

    @rpc
    def start_recording(self) -> None:
        self._recording = True

    @rpc
    def stop_recording(self) -> None:
        self._recording = False

    def process_observable(
        self, observable: Observable[Any], async_cb: Callable[[Any], Any]
    ) -> DisposableBase:
        # The Recorder subscribes every recorded port through here.
        recorded = observable.pipe(ops.filter(lambda _: self._recording))
        return super().process_observable(recorded, async_cb)


realsense_stereo_record = autoconnect(
    # The emitter stays on: its dots texture the IR pair for stereo.
    RealSenseCamera.blueprint(frame_id=CAMERA_LINK, enable_infrared=True, enable_imu=True),
    StereoRecorder.blueprint(
        db_path=str(DB_PATH),
        stream_codecs={name: "lz4+lcm" for name in LOSSLESS_STREAMS},
        # Nothing publishes a world frame; poses come from the offline reconstruction.
        poseless_streams=list(StereoRecorder.__annotations__),
    ),
    RecordWindow.blueprint(db_path=str(DB_PATH)),
).global_config(n_workers=3)
