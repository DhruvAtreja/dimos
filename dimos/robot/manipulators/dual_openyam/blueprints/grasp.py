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

"""The xArm grasp stack on the dual OpenYAM workcell, one arm per blueprint.

``dimos run dual-openyam-grasp-right --left-can-port follower_l --right-can-port follower_r``
``dimos run dual-openyam-grasp-left ...``             the other arm
``dimos run dual-openyam-grasp-right-graspgenx ...``  learned grasps

The planning model is the dual workcell model with both arms and both planning
groups. The manipulation module binds one gripper per model, so each blueprint
benchmarks one arm and names that arm's gripper; the other arm rests on its
support under gravity compensation. Everything else is the stack as it exists:
pick_object and place_at from the pick-and-place module, one top-down heuristic
grasp or GraspGenX, scene registration from the overhead D405, the planner with
the table, post and overhead camera as collision geometry in the workcell
model. No self filter or voxel map: the planner's obstacles are the detected
objects and the bin added per scene.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXModule
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.manipulation.planning.spec.config import RobotModelConfig
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.robot.manipulators.common.blueprints import planner
from dimos.robot.manipulators.dual_openyam.blueprints.basic import (
    DualOpenYamCoordinator,
    dual_openyam_trajectory_task,
)
from dimos.robot.manipulators.dual_openyam.config import (
    DUAL_OPENYAM_SIDES,
    dual_openyam_model_config,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_GRIPPER_JOINTS
from dimos.visualization.rerun.bridge import RerunBridgeModule

# Table top below the arm mounting plates (tape, 2026-10-08). The workcell model
# carries the slab 1 cm lower so objects resting on the table do not count as
# in collision with it; no separate planner floor, it would overlap the slab.
DUAL_OPENYAM_TABLE_TOP_Z = -0.042
# {side}_grasp_frame sits 10 cm below the gripper link on its axis; the finger
# pads of the URDF meet 3.7 cm further out. The printed fingers on the rig are
# longer than the URDF's by this much per arm (tool height check, 2026-10-09).
DUAL_OPENYAM_PAD_CENTRE_OFFSET = 0.037
DUAL_OPENYAM_FINGER_EXTENSION = {"left": 0.045, "right": 0.060}
# Pink's defaults do not converge on this model; the WebXR teleop uses these.
DUAL_OPENYAM_GRASP_PINK = PinkKinematicsConfig(
    dt=0.01,
    position_cost=8.0,
    orientation_cost=2.0,
    posture_cost=0.01,
    joint_limit_posture_margin=0.3,
    lm_damping=0.01,
    gain=1.0,
)
# Fixed camera on the centre post, in the frame midway between the arm bases
# (x forward, y toward the left arm, z up). Solved 2026-10-13 from four AprilTags
# laid by their corners from the J1 line and the X axis (97.5 mm black squares),
# 0.66 px reprojection RMS over 16 corners; tape to the camera agrees within 1 cm.
DUAL_OPENYAM_CAMERA_TRANSFORM = Transform(
    translation=Vector3(x=-0.0121, y=0.0147, z=0.4069),
    rotation=Quaternion(-0.01191, 0.48426, -0.01853, 0.87465),  # xyzw, pitched 58.0 deg down
    frame_id="world",
    child_frame_id="camera_link",
)
# RealSense D405 serials on the workcell; override with --realsensecamera.serial-number,
# --left-wrist-camera.serial-number and --right-wrist-camera.serial-number.
DUAL_OPENYAM_OVERHEAD_CAMERA_SERIAL = "230322272156"
DUAL_OPENYAM_WRIST_CAMERA_SERIALS = {"left": "260322272983", "right": "260322276650"}
# The gripper in GraspGenX's convention: origin on the gripper link, approach
# along +Z, jaws closing along X. The fingers slide 4.7 cm each, so the open and
# half-open sweep volumes share their centre.
DUAL_OPENYAM_GRIPPER_SWEEP_VOLUME = {
    "extents_open": (0.094, 0.028, 0.020),
    "offset_open": (0.0, 0.0, 0.137),
    "extents_half_open": (0.047, 0.028, 0.020),
    "offset_half_open": (0.0, 0.0, 0.137),
    "fingertip_depth": 0.1468,
}
# GraspGenX frame -> {side}_tcp. The tcp below is the grasp frame turned a half
# turn about Y, so GraspGenX's approach axis already runs along the tcp's +Z:
# swap X and Y, then move 13.7 cm back along the approach to the pad centre.
DUAL_OPENYAM_GRASP_FRAME_TO_TCP = (
    (0.0, -1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, -0.137),
    (0.0, 0.0, 0.0, 1.0),
)
# Streams a benchmark run keeps: joint states, every camera's colour, depth and
# intrinsics, and TF. Main publishes no commanded joint targets to record.
DUAL_OPENYAM_RECORD_TOPICS = ",".join(
    [
        "coordinator_joint_state",
        "color_image",
        "depth_image",
        "camera_info",
        "tf",
        *(
            f"{side}_wrist_{stream}"
            for side in DUAL_OPENYAM_SIDES
            for stream in ("color_image", "camera_info")
        ),
    ]
)


def _check_side(side: str) -> None:
    if side not in DUAL_OPENYAM_SIDES:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def dual_openyam_gripper_task(side: str, *, priority: int = 20) -> TaskConfig:
    """Gripper task named as the manipulation module expects: ``{side}_arm_gripper``."""
    _check_side(side)
    return TaskConfig(
        name=f"{side}_arm_gripper",
        type="gripper",
        joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[DUAL_OPENYAM_SIDES.index(side)]],
        priority=priority,
    )


def dual_openyam_grasp_model_config(side: str) -> RobotModelConfig:
    """The workcell model in the world frame with a fingertip tcp per arm and
    ``side``'s gripper bound to the model.

    The grasp frame's +Z points out of the back of the palm. The heuristic grasp
    and the pick-and-place module expect a tool whose +Z points at the object and
    back off along -Z, so the tcp is the grasp frame moved to the pad centre and
    turned a half turn about Y.
    """
    _check_side(side)
    config = dual_openyam_model_config()
    model = config.model
    for arm in DUAL_OPENYAM_SIDES:
        model = model.with_fixed_frame(
            f"{arm}_tcp",
            f"{arm}_grasp_frame",
            xyz=(0.0, 0.0, -(DUAL_OPENYAM_PAD_CENTRE_OFFSET + DUAL_OPENYAM_FINGER_EXTENSION[arm])),
            rpy=(0.0, math.pi, 0.0),
        )
    config.model = model
    config.base_pose = PoseStamped(frame_id="world")
    config.planning_groups = [
        replace(group, tip_link=f"{group.name.split('_')[0]}_tcp")
        for group in config.planning_groups
    ]
    config.gripper_hardware_id = f"{side}_arm"
    # The pads meet at the URDF's closed zero position, so the two fingertip
    # hulls always touch, and the wrist assembly nests inside the gripper body
    # so their hulls overlap at every pose. Adjacent links are filtered already.
    config.collision_exclusion_pairs = [
        *config.collision_exclusion_pairs,
        *((f"{arm}_tip_left", f"{arm}_tip_right") for arm in DUAL_OPENYAM_SIDES),
        *((f"{arm}_link4", f"{arm}_gripper") for arm in DUAL_OPENYAM_SIDES),
    ]
    return config


def dual_openyam_grasp_provider(*, graspgen: bool) -> Blueprint:
    if graspgen:
        return GraspGenXModule.blueprint(
            gripper=DUAL_OPENYAM_GRIPPER_SWEEP_VOLUME,
            grasp_frame_to_tcp=DUAL_OPENYAM_GRASP_FRAME_TO_TCP,
        )
    return HeuristicGraspModule.blueprint()


def dual_openyam_grasp_view() -> Any:
    """Rerun layout: the three cameras as tiles on the left, the scene on the right."""
    import rerun as rr
    import rerun.blueprint as rrb

    cameras = [
        ("world/color_image", "Overhead"),
        ("world/left_wrist_color_image", "Left wrist"),
        ("world/right_wrist_color_image", "Right wrist"),
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*[rrb.Spatial2DView(origin=path, name=name) for path, name in cameras]),
            rrb.Spatial3DView(
                origin="world",
                name="Scene",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(plane=rr.components.Plane3D.XY.with_distance(0.0)),
                # The images have their own tiles; in 3D they only clutter.
                overrides={path: rrb.EntityBehavior(visible=False) for path, _ in cameras},
            ),
            column_shares=[1, 2],
        ),
        rrb.TimePanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )


# Streams the Rerun bridge forwards; everything else stays off the viewer link.
DUAL_OPENYAM_VIEW_TOPICS = [
    "color_image",
    "left_wrist_color_image",
    "right_wrist_color_image",
    "pointcloud",
    "tf",
]


def dual_openyam_grasp_rerun() -> Blueprint:
    """The bridge for a headless rig: no window on the box, watch it from the laptop."""
    return RerunBridgeModule.blueprint(
        blueprint=dual_openyam_grasp_view,
        topics=DUAL_OPENYAM_VIEW_TOPICS,
        memory_limit="2GB",
        rerun_open="none",
    )


def dual_openyam_wrist_camera(side: str) -> Blueprint:
    """A wrist D405 whose streams and frames carry a ``{side}_wrist_`` prefix.

    A prefix with an underscore, not a namespace: the recorder uses stream names
    as SQL identifiers and rejects a slash. Colour only: the three D405s share
    USB 2 hubs and drop frames with depth on.
    """
    _check_side(side)
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


def dual_openyam_grasp_modules(side: str, *, graspgen: bool) -> tuple[Blueprint, ...]:
    _check_side(side)
    return (
        planner(
            model=dual_openyam_grasp_model_config(side),
            kinematics=DUAL_OPENYAM_GRASP_PINK,
            default_speed_scale=0.25,
            static_transforms=[DUAL_OPENYAM_CAMERA_TRANSFORM],
            visualization={"backend": "viser"},
            world_frame="world",
        ),
        ManipulationSkills.blueprint(),
        PickAndPlaceModule.blueprint(planning_frame="world"),
        dual_openyam_grasp_provider(graspgen=graspgen),
        # The D405 offers no 848x480 at 15 fps; 640x480 at 30 fps is its native mode.
        RealSenseCamera.blueprint(
            width=640,
            height=480,
            fps=30,
            enable_pointcloud=True,
            serial_number=DUAL_OPENYAM_OVERHEAD_CAMERA_SERIAL,
        ),
        dual_openyam_wrist_camera("left"),
        dual_openyam_wrist_camera("right"),
        ObjectSceneRegistrationModule.blueprint(
            target_frame="world",
            detector_backend="moondream",
            segmentation_backend="edgetam",
            detect_on_request=True,
            distance_threshold=0.08,
            # One explicit scan is the observation.
            min_detections_for_permanent=1,
            max_distance=1.5,
            use_aabb=True,
            max_obstacle_width=0.06,
        ),
        dual_openyam_grasp_rerun(),
        DualOpenYamCoordinator.blueprint(
            instance_name="ControlCoordinator",
            tasks=[
                dual_openyam_trajectory_task(),
                dual_openyam_gripper_task("left"),
                dual_openyam_gripper_task("right"),
            ],
        ),
    )


def dual_openyam_grasp_blueprint(side: str, *, graspgen: bool) -> Blueprint:
    return autoconnect(*dual_openyam_grasp_modules(side, graspgen=graspgen)).global_config(
        record="sqlite", record_topics=DUAL_OPENYAM_RECORD_TOPICS
    )


# Assigned through autoconnect so the registry generator sees them.
dual_openyam_grasp_right = autoconnect(
    *dual_openyam_grasp_modules("right", graspgen=False)
).global_config(record="sqlite", record_topics=DUAL_OPENYAM_RECORD_TOPICS)
dual_openyam_grasp_left = autoconnect(
    *dual_openyam_grasp_modules("left", graspgen=False)
).global_config(record="sqlite", record_topics=DUAL_OPENYAM_RECORD_TOPICS)
dual_openyam_grasp_right_graspgenx = autoconnect(
    *dual_openyam_grasp_modules("right", graspgen=True)
).global_config(record="sqlite", record_topics=DUAL_OPENYAM_RECORD_TOPICS)
dual_openyam_grasp_left_graspgenx = autoconnect(
    *dual_openyam_grasp_modules("left", graspgen=True)
).global_config(record="sqlite", record_topics=DUAL_OPENYAM_RECORD_TOPICS)
