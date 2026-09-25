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

"""Offline tests for the TypeSafe arm policy: no API key, no simulator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul

from dimos.evals.agents.typesafe_policy import (
    DEFAULT_SCENE,
    GRIPPER,
    STEP_CRITERIA,
    STEPS,
    STOP,
    TypeSafePolicy,
    build_questions,
    derive_goal_label,
    load_scene,
)
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.tf2_msgs.TFMessage import TFMessage

# The gripper pointing straight down: roll 180 degrees, so the end-effector's
# +z is the world's -z and its +y is the world's -y; +x is shared.
DOWN = Quaternion.from_euler(Vector3(math.pi, 0.0, 0.0))


def _tf(child: str, x: float, y: float, z: float, rotation: Quaternion | None = None) -> Transform:
    return Transform(
        translation=Vector3(x, y, z),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0) if rotation is None else rotation,
        frame_id="world",
        child_frame_id=child,
        ts=1.0,
    )


def _choice(key: str, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(choice=key, confidence=confidence, probabilities={key: confidence})


class FakeTools:
    """Stands in for McpAdapter: records calls, answers like a SkillResult."""

    def __init__(self, success: bool = True) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.success = success

    def call_tool_text(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, dict(arguments or {})))
        return json.dumps({"success": self.success, "message": f"{name} done", "error_code": None})


@pytest.fixture
def policy() -> TypeSafePolicy:
    """A policy that has seen one tf frame: gripper down at (0.30, 0, 0.40), cup on the table."""
    agent = TypeSafePolicy()
    agent._wanted = frozenset({"link_tcp", "apple", "orange", "cup"})
    agent._on_tf(
        TFMessage(
            _tf("link_tcp", 0.30, 0.0, 0.40, DOWN),
            _tf("apple", 0.40, 0.08, 0.17),
            _tf("orange", 0.45, -0.08, 0.175),
            _tf("cup", 0.50, 0.0, 0.19),
        )
    )
    return agent


# --- contracts -------------------------------------------------------------------


def test_steps_gripper_and_stop_cover_the_criteria() -> None:
    """Every option the model can pick is handled, and vice versa."""
    assert set(STEP_CRITERIA) == set(STEPS) | set(GRIPPER) | {STOP}


def test_questions_are_one_choice_and_one_noul() -> None:
    questions = build_questions()
    assert isinstance(questions["step"], Choice)
    assert isinstance(questions["reached"], Noul)
    assert set(questions["step"].criteria) == set(STEP_CRITERIA)


# --- scene file --------------------------------------------------------------------


def test_shipped_scene_matches_the_mujoco_model() -> None:
    """Pins the hand transcription of data/xarm7/scene.xml."""
    scene = load_scene(DEFAULT_SCENE)
    assert scene.labels == ["apple", "orange", "cup"]
    cup = scene.objects[2]
    assert cup.shape == "cylinder"
    assert cup.size_m == {"radius_m": 0.035, "height_m": 0.12}
    assert cup.start_xyz == (0.50, 0.0, 0.19)
    assert scene.raw["table"]["top_z"] == pytest.approx(0.13)


def test_scene_without_objects_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"objects": []}))
    with pytest.raises(ValueError, match="no objects"):
        load_scene(path)


# --- goal from the instruction ---------------------------------------------------


def test_derive_goal_label_resolves_aliases_to_the_body() -> None:
    scene = load_scene(DEFAULT_SCENE)
    assert derive_goal_label("Pick up the cylinder and hold it in the air.", scene) == "cup"
    assert derive_goal_label("pick up the apple and hold it above the table", scene) == "apple"
    assert derive_goal_label("grab the cup", scene) == "cup"


def test_derive_goal_label_rejects_an_unnamed_goal() -> None:
    with pytest.raises(LookupError, match="names a scene object"):
        derive_goal_label("pick up the thing on the table", load_scene(DEFAULT_SCENE))


# --- state -------------------------------------------------------------------------


def test_objects_are_reported_in_the_end_effector_frame(policy: TypeSafePolicy) -> None:
    """Gripper down at (0.30, 0, 0.40): the cup at world (0.50, 0, 0.19) is 20 cm along +x
    and 21 cm ahead along the pointing axis (+z)."""
    scene = load_scene(DEFAULT_SCENE)
    ee, state = policy.observe(scene, "cup", tick=2)
    assert state.ee_xyz == (0.30, 0.0, 0.40)
    assert state.ee_rpy_deg == pytest.approx((180.0, 0.0, 0.0))
    by_label = {o.label: o for o in state.objects_in_ee_frame or []}
    assert by_label["cup"].xyz_ee == pytest.approx((0.20, 0.0, 0.21), abs=1e-3)
    assert by_label["apple"].xyz_ee == pytest.approx((0.10, -0.08, 0.23), abs=1e-3)
    assert state.goal_label == "cup"
    assert state.ticks_elapsed == 2
    assert state.gripper_open_fraction is None  # nothing commanded yet


def test_state_carries_the_scene_file_verbatim(policy: TypeSafePolicy) -> None:
    scene = load_scene(DEFAULT_SCENE)
    _, state = policy.observe(scene, "cup", tick=0)
    encoded = state.encode()
    assert encoded["scene"] == scene.raw
    assert encoded["objects_in_ee_frame"][2]["size_m"] == {"radius_m": 0.035, "height_m": 0.12}


def test_tf_is_ready_only_once_every_frame_arrived() -> None:
    agent = TypeSafePolicy()
    agent._wanted = frozenset({"link_tcp", "cup"})
    agent._on_tf(TFMessage(_tf("cup", 0.5, 0.0, 0.19), _tf("camera", 0.0, 0.0, 0.0)))
    assert not agent._tf_seen.is_set()
    assert "camera" not in agent._tf  # frames the state does not need are not kept
    agent._on_tf(TFMessage(_tf("link_tcp", 0.3, 0.0, 0.4)))
    assert agent._tf_seen.is_set()


# --- controller --------------------------------------------------------------------
# The pick is the direction, in the frame the objects were reported in.


def test_steps_are_rotated_from_the_end_effector_frame(policy: TypeSafePolicy) -> None:
    ee = policy._tf["link_tcp"]
    assert policy.target_for("+x", ee).to_tuple() == pytest.approx((0.33, 0.0, 0.40))
    assert policy.target_for("+y", ee).to_tuple() == pytest.approx((0.30, -0.03, 0.40))
    assert policy.target_for("+z", ee).to_tuple() == pytest.approx((0.30, 0.0, 0.37))  # down
    assert policy.target_for("-z", ee).to_tuple() == pytest.approx((0.30, 0.0, 0.43))  # up


def test_a_move_is_one_move_to_pose_call_holding_the_wrist(policy: TypeSafePolicy) -> None:
    """The default wrist orientation, straight down, rides along on every move."""
    tools = FakeTools()
    action = policy.act(tools, _choice("+z"), policy._tf["link_tcp"])  # type: ignore[arg-type]
    assert tools.calls == [
        (
            "move_to_pose",
            {"x": 0.3, "y": 0.0, "z": 0.37, "roll": math.pi, "pitch": 0.0, "yaw": 0.0},
        )
    ]
    assert action["ok"] and action["tool"] == "move_to_pose"


def test_rpy_none_keeps_the_arms_current_orientation() -> None:
    agent = TypeSafePolicy(rpy_rad=None)
    tools = FakeTools()
    agent.act(tools, _choice("+x"), _tf("link_tcp", 0.3, 0.0, 0.4, DOWN))  # type: ignore[arg-type]
    ((_, arguments),) = tools.calls
    assert set(arguments) == {"x", "y", "z"}


def test_gripper_picks_call_set_gripper_and_update_the_state(policy: TypeSafePolicy) -> None:
    tools = FakeTools()
    policy.act(tools, _choice("close"), policy._tf["link_tcp"])  # type: ignore[arg-type]
    assert tools.calls == [("set_gripper", {"position": 0.0})]
    assert policy._gripper == 0.0
    policy.act(tools, _choice("open"), policy._tf["link_tcp"])  # type: ignore[arg-type]
    assert policy._gripper == 1.0


def test_stop_calls_nothing(policy: TypeSafePolicy) -> None:
    tools = FakeTools()
    action = policy.act(tools, _choice(STOP), policy._tf["link_tcp"])  # type: ignore[arg-type]
    assert tools.calls == [] and action["tool"] is None


def test_a_failed_skill_is_recorded_not_raised(policy: TypeSafePolicy) -> None:
    tools = FakeTools(success=False)
    action = policy.act(tools, _choice("close"), policy._tf["link_tcp"])  # type: ignore[arg-type]
    assert action["ok"] is False
    assert policy._gripper is None  # the state does not claim what did not happen


def test_pick_is_applied_whatever_the_confidence(policy: TypeSafePolicy) -> None:
    """No threshold: the argmax is the command."""
    tools = FakeTools()
    policy.act(tools, _choice("+x", confidence=0.21), policy._tf["link_tcp"])  # type: ignore[arg-type]
    assert tools.calls[0][0] == "move_to_pose"


# --- preflight ---------------------------------------------------------------------


def test_preflight_accepts_the_arm_environment() -> None:
    from dimos.evals.suites.mujoco_xarm import arm_only_environment

    TypeSafePolicy().preflight(arm_only_environment())
    TypeSafePolicy(goal_label="cup").preflight(arm_only_environment())


def test_preflight_rejects_other_simulators() -> None:
    from dimos.evals.environments.dimsim import DimSimEnvironment

    with pytest.raises(ValueError, match="MuJoCo arm"):
        TypeSafePolicy().preflight(DimSimEnvironment(blueprint=["unitree-go2", "mcp-server"]))


def test_preflight_rejects_missing_scene(tmp_path: Path) -> None:
    from dimos.evals.suites.mujoco_xarm import arm_only_environment

    with pytest.raises(FileNotFoundError):
        TypeSafePolicy(scene_json=tmp_path / "nope.json").preflight(arm_only_environment())


def test_preflight_rejects_untracked_objects() -> None:
    from dimos.evals.environments.mujoco_sim import MujocoEnvironment

    env = MujocoEnvironment(
        blueprint=["xarm-perception-sim", "mcp-server"], tracked_bodies=("cup",)
    )
    with pytest.raises(ValueError, match="tracked_bodies"):
        TypeSafePolicy().preflight(env)


def test_preflight_rejects_unknown_goal() -> None:
    from dimos.evals.suites.mujoco_xarm import arm_only_environment

    with pytest.raises(LookupError, match="banana"):
        TypeSafePolicy(goal_label="banana").preflight(arm_only_environment())
