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

"""Recording blueprints.

`CollectionRecorder` (a memory Recorder) captures the obs/action/status
streams to a SQLite session DB during the run and flushes it durably on
shutdown. DataPrep reads that DB afterwards.
"""

from __future__ import annotations

from datetime import datetime

from dimos.constants import RECORDINGS_DIR
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.global_config import global_config
from dimos.core.stream import In
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.recorder import CollectionRecorder
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import teleop_webxr_dual_openyam
from dimos.teleop.webxr.blueprints import (
    teleop_webxr_piper,
    teleop_webxr_xarm7,
)


def _session_db(robot: str) -> str:
    """Timestamped session DB path under RECORDINGS_DIR, namespaced by robot."""
    return str(RECORDINGS_DIR / f"session_{robot}_{datetime.now():%Y%m%d_%H%M%S}.db")


def _camera_if_real() -> tuple[Blueprint, ...]:
    """Real RealSense only off-sim. In `--simulation` the teleop coordinator's
    MujocoSimModule already publishes color_image on /camera/color_image, so a
    real camera would be redundant (and fail with no device connected)."""
    if global_config.simulation:
        return ()
    return (RealSenseCamera.blueprint(enable_pointcloud=False),)


# teleop_buttons / color_image / coordinator_joint_state / status are left to
# autoconnect — each name is unique across the composed blueprint, so it
# resolves to a stable /<name> topic shared by producer and recorder. The
# recorder captures whatever joints are present, so the coordinator's aggregate
# stream is its intended input (see dimos/control/README.md).
# These generic teleoperation labels should be customized for a specific
# training task when composing a collection blueprint.
learning_collect_webxr_xarm7 = autoconnect(
    CollectionRecorder.blueprint(
        db_path=_session_db("xarm7"),
        poseless_streams=["color_image", "coordinator_joint_state", "status"],
        record_tf=False,
    ),
    EpisodeMonitorModule.blueprint(task="Teleoperate the xArm"),
    teleop_webxr_xarm7,
    *_camera_if_real(),
)


learning_collect_webxr_piper = autoconnect(
    CollectionRecorder.blueprint(
        db_path=_session_db("piper"),
        poseless_streams=["color_image", "coordinator_joint_state", "status"],
        record_tf=False,
    ),
    EpisodeMonitorModule.blueprint(task="Teleoperate the Piper"),
    teleop_webxr_piper,
    *_camera_if_real(),
)


# The benchmark rig's three D405s. Override per camera with
# --realsensecamera.serial-number, --left-wrist-camera.serial-number and
# --right-wrist-camera.serial-number.
DUAL_OPENYAM_WRIST_CAMERA_SERIALS = {"left": "260322272983", "right": "260322276650"}


class DualOpenYamCollectionRecorder(CollectionRecorder):
    """The collection streams plus both wrist views and the operator's commands.

    Wrist images are what a grasping policy looks at; the controller poses and
    gripper commands are the actions the operator asked for, so a dataset can
    use them instead of the measured state. DataPrep selects streams by name,
    so the extra streams cost nothing until a config asks for them.
    """

    left_wrist_color_image: In[Image]
    right_wrist_color_image: In[Image]
    left_cartesian_command: In[PoseStamped]
    right_cartesian_command: In[PoseStamped]
    left_gripper_command: In[Float32]
    right_gripper_command: In[Float32]


DUAL_OPENYAM_COLLECTION_STREAMS = [
    "color_image",
    "coordinator_joint_state",
    "status",
    "left_wrist_color_image",
    "right_wrist_color_image",
    "left_cartesian_command",
    "right_cartesian_command",
    "left_gripper_command",
    "right_gripper_command",
]


def _dual_openyam_wrist_camera(side: str) -> Blueprint:
    """A wrist D405 whose streams carry a ``{side}_wrist_`` prefix, so they never
    collide with the overhead camera. Colour only at 640x480 30 fps: the three
    D405s share USB 2 hubs, and the recorder needs stream names without a slash."""
    name = f"{side}_wrist_camera"
    camera = RealSenseCamera.blueprint(
        instance_name=name,
        frame_id_prefix=f"{side}_wrist",
        width=640,
        height=480,
        fps=30,
        enable_depth=False,
        align_depth_to_color=False,
        enable_pointcloud=False,
        serial_number=DUAL_OPENYAM_WRIST_CAMERA_SERIALS[side],
    )
    streams = [stream.name for stream in camera.blueprints[0].streams]
    return camera.remappings([(name, stream, f"{side}_wrist_{stream}") for stream in streams])


def _dual_openyam_cameras_if_real() -> tuple[Blueprint, ...]:
    if global_config.simulation:
        return ()
    return (
        RealSenseCamera.blueprint(enable_pointcloud=False),
        _dual_openyam_wrist_camera("left"),
        _dual_openyam_wrist_camera("right"),
    )


learning_collect_webxr_dual_openyam = autoconnect(
    DualOpenYamCollectionRecorder.blueprint(
        db_path=_session_db("dual_openyam"),
        poseless_streams=DUAL_OPENYAM_COLLECTION_STREAMS,
        record_tf=False,
    ),
    EpisodeMonitorModule.blueprint(task="Teleoperate the dual OpenYAM"),
    teleop_webxr_dual_openyam,
    *_dual_openyam_cameras_if_real(),
)
