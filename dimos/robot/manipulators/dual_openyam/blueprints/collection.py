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

"""What a dual OpenYAM collection runs and records: teleop tuning, cameras, recorder."""

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.core.stream import In
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.imitation.collection.recorder import CollectionRecorder
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.std_msgs.Float32 import Float32
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
    build_dual_openyam_webxr,
    dual_openyam_webxr_task,
)
from dimos.teleop.webxr.controller_types import Buttons

DUAL_OPENYAM_SIDES = ("left", "right")

# Joint-speed ceiling for collection teleop. The stock task allows 2.0 rad/s; on
# the benchmark rig the operator asked for a calmer arm (2026-10-13 session: 90th
# percentile joint speed 1.06 rad/s, 1.2 rad/s clips 6.5 percent of the samples).
DUAL_OPENYAM_COLLECTION_JOINT_SPEED_RAD_S = 1.2


def dual_openyam_collection_teleop_task() -> TaskConfig:
    """The stock dual-arm teleop task at the collection speed ceiling."""
    return dual_openyam_webxr_task(
        max_joint_velocity_rad_s=DUAL_OPENYAM_COLLECTION_JOINT_SPEED_RAD_S
    )


# The stock dual OpenYAM WebXR graph at the collection speed ceiling.
teleop_webxr_dual_openyam_collection = autoconnect(
    build_dual_openyam_webxr(teleop_task=dual_openyam_collection_teleop_task())
)

# The benchmark rig's wrist D405s. Override with --left-wrist-camera.serial-number
# and --right-wrist-camera.serial-number; the overhead camera takes
# --realsensecamera.serial-number.
DUAL_OPENYAM_WRIST_CAMERA_SERIALS = {"left": "260322272983", "right": "260322276650"}

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
    "teleop_buttons",
]


class DualOpenYamCollectionRecorder(CollectionRecorder):
    """The collection streams plus both wrist views and the operator's commands.

    Wrist images are what a grasping policy looks at; the controller poses and
    gripper commands are the actions the operator asked for, so a dataset can
    use them instead of the measured state; the button word says when the grips
    were held, which is when a controller pose was actually driving an arm.
    DataPrep selects streams by name, so the extra streams cost nothing until a
    config asks for them.
    """

    left_wrist_color_image: In[Image]
    right_wrist_color_image: In[Image]
    left_cartesian_command: In[PoseStamped]
    right_cartesian_command: In[PoseStamped]
    left_gripper_command: In[Float32]
    right_gripper_command: In[Float32]
    teleop_buttons: In[Buttons]


def dual_openyam_wrist_camera(side: str) -> Blueprint:
    """A wrist D405 whose streams carry a ``{side}_wrist_`` prefix, so they never
    collide with the overhead camera. Colour only at 640x480 30 fps: the three
    D405s share USB 2 hubs, and the recorder needs stream names without a slash."""
    if side not in DUAL_OPENYAM_SIDES:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
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
