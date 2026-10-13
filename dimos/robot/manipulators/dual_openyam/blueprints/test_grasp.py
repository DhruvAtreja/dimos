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

import math

import pytest

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, BlueprintAtom
from dimos.core.module import ModuleBase
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.manipulation.grasping.grasp_gen_x.module import GraspGenXModule
from dimos.manipulation.grasping.heuristic_grasp import HeuristicGraspModule
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.memory.utils.validation import validate_identifier
from dimos.robot.manipulators.dual_openyam.blueprints.agentic import (
    dual_openyam_agent_prompt,
    dual_openyam_grasp_right_agent,
)
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.blueprints.grasp import (
    DUAL_OPENYAM_RECORD_TOPICS,
    DUAL_OPENYAM_VIEW_TOPICS,
    dual_openyam_grasp_blueprint,
    dual_openyam_grasp_left,
    dual_openyam_grasp_model_config,
    dual_openyam_grasp_right,
)
from dimos.visualization.rerun.bridge import RerunBridgeModule


def _atom(blueprint: Blueprint, module: type[ModuleBase]) -> BlueprintAtom:
    return next(atom for atom in blueprint.active_blueprints if atom.module is module)


def _modules(blueprint: Blueprint) -> set[type[ModuleBase]]:
    return {atom.module for atom in blueprint.active_blueprints}


@pytest.mark.parametrize("side", ["left", "right"])
def test_model_keeps_both_arms_and_binds_one_gripper(side: str) -> None:
    config = dual_openyam_grasp_model_config(side)
    assert config.base_pose.frame_id == "world"
    assert config.gripper_hardware_id == f"{side}_arm"
    assert {group.name: group.tip_link for group in config.planning_groups} == {
        "left_manipulator": "left_tcp",
        "right_manipulator": "right_tcp",
    }
    assert len(config.joint_names) == 12


def test_tcp_is_the_grasp_frame_turned_to_point_down_the_fingers() -> None:
    config = dual_openyam_grasp_model_config("right")
    frames = {frame.name: frame for frame in config.model._fixed_frames}
    assert frames["right_tcp"].parent == "right_grasp_frame"
    assert frames["right_tcp"].rpy == (0.0, math.pi, 0.0)
    assert frames["right_tcp"].xyz[2] == pytest.approx(-0.097)
    assert frames["left_tcp"].xyz[2] == pytest.approx(-0.082)


def test_each_side_names_its_own_gripper_and_shares_the_tasks() -> None:
    for blueprint, side in ((dual_openyam_grasp_right, "right"), (dual_openyam_grasp_left, "left")):
        tasks: list[TaskConfig] = _atom(blueprint, DualOpenYamCoordinator).kwargs["tasks"]
        assert {task.name for task in tasks if task.type == "gripper"} == {
            "left_arm_gripper",
            "right_arm_gripper",
        }
        model = _atom(blueprint, ManipulationModule).kwargs["model"]
        assert model.gripper_hardware_id == f"{side}_arm"


def test_stack_is_the_xarm_grasp_stack_with_the_workcell_rig() -> None:
    manipulation = _atom(dual_openyam_grasp_right, ManipulationModule).kwargs
    assert manipulation["world_frame"] == "world"
    assert "floor_z" not in manipulation
    assert manipulation["static_transforms"][0].child_frame_id == "camera_link"
    assert _atom(dual_openyam_grasp_right, PickAndPlaceModule).kwargs == {"planning_frame": "world"}
    overhead = next(
        atom
        for atom in dual_openyam_grasp_right.active_blueprints
        if atom.module is RealSenseCamera and atom.kwargs.get("enable_pointcloud")
    )
    assert (overhead.kwargs["width"], overhead.kwargs["height"], overhead.kwargs["fps"]) == (
        640,
        480,
        30,
    )
    assert dual_openyam_grasp_right.global_config_overrides["record"] == "sqlite"
    bridge = _atom(dual_openyam_grasp_right, RerunBridgeModule).kwargs
    assert bridge["topics"] == DUAL_OPENYAM_VIEW_TOPICS
    assert bridge["rerun_open"] == "none"
    assert bridge["blueprint"]() is not None


def test_graspgen_flag_swaps_the_grasp_provider() -> None:
    heuristic = _modules(dual_openyam_grasp_blueprint("right", graspgen=False))
    learned = _modules(dual_openyam_grasp_blueprint("right", graspgen=True))
    assert HeuristicGraspModule in heuristic and GraspGenXModule not in heuristic
    assert GraspGenXModule in learned and HeuristicGraspModule not in learned


def test_recorded_stream_names_are_valid_identifiers() -> None:
    for name in DUAL_OPENYAM_RECORD_TOPICS.split(","):
        validate_identifier(name)


def test_agent_prompt_pins_the_arm_and_its_drop_pose() -> None:
    prompt = dual_openyam_agent_prompt("right")
    assert "right_manipulator" in prompt
    assert "place_at 0.46 -0.05 0.12" in prompt
    assert ManipulationModule in _modules(dual_openyam_grasp_right_agent)
