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


"""The SceneReplica tabletop benchmark (arXiv 2306.15620) on sim2: 20 scenes of five YCB
objects, 100 trials in all. Each trial asks the agent to pick one object and set it down in
the drop-off square on the table, with the perception stack (``scan_objects``,
``pick_object``, ``place_at``) available. A trial is graded from the recorded ground truth:
half for lifting the object 5 cm at any point (SceneReplica's grasp success), half for the
object ending at rest inside the drop-off (its pick-and-place success); a trial passes only
when both hold. A failed trial also records which stage lost it, perception, planning or
execution, read from the agent's tool calls.

    dimos evals run dimos.evals.suites.scenereplica --agent dimos.evals.agents.pi --tags scene-01
    python -m dimos.evals.suites.lib.scenereplica_summary ~/.local/state/dimos/evals/run-*

The paper picks the five objects of a scene in two orders; the ``order-<k>`` tag gives the
near-to-far order, by distance from the arm base, since the shipped scenes carry no other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue

from dimos.evals.environments.lib.scene_checks import (
    displacement,
    lifted_by,
    resting_in_region,
)
from dimos.evals.environments.sim2 import Sim2Environment
from dimos.evals.scorers import Check, weighted
from dimos.evals.types import EvalCase, Graded, Outcome, Suite, recording
from dimos.sim2.scene_types import SceneDescription
from dimos.utils.data import get_data

if TYPE_CHECKING:
    from dimos.evals.types import Trajectory
    from dimos.memory.store.base import Store

BLUEPRINT = ["xarm-perception-sim2", "mcp-server", "observe-skill"]
DROPOFF = "dropoff/top"
LIFT_M = 0.05  # SceneReplica counts a grasp once the object is clear of the table
DISTURBED_M = 0.02  # another object moved more than this was knocked
TIMEOUT_S = 900.0

SCENE = (
    "The table top is at z=0.745 m in the world frame, 0.155 m below the arm base, and spans "
    "roughly x=0.34 to 1.26 m ahead of the base and y=-0.46 to 0.46 m. Five household objects "
    "stand on its near half within reach of the arm; find where they are by looking through "
    "the wrist camera. The drop-off square lies on the table surface near its front-left "
    "corner. The gripper starts pointing straight down: for top-down moves omit "
    "roll/pitch/yaw in move_to_pose to keep the current orientation, or pass roll=3.1416, "
    "pitch=0. You also have perception skills: scan_objects(prompts=[...]) looks for the "
    "named objects in the current wrist-camera view and returns an object_id for each one it "
    "finds, pick_object(object_id) grasps that object and lifts it, and place_at(x, y, z) "
    "lowers the held object to that world-frame position and releases it. Neither reports "
    "where an object is; work that out from the camera and the table layout above."
)

Cause = Literal["perception", "planning", "execution"]

# How pick_object, place_at and the motion skills say the planner found no way to the pose:
# "The planner found no path to any of the 2 grasp candidate(s) ...", "... did not complete;
# planner returned FAILED: ...", "No plan; planner returned FAILED: ...".
_PLANNER_FAILED = re.compile(r"\bplanner (?:found no path|returned)\b")
# The skills whose planning failures count, by stage: picking (and explicit planning) or placing.
_PICK_CALL = re.compile(r"\b(pick_object|plan_\w*)\b")
_PLACE_CALL = re.compile(r"\bplace_at\b")
# An object the perception stack reported: scan_objects' JSON or detect's bullet list.
_DETECTION = re.compile(r'"name":\s*"([^"]*)"|^\s*-\s*(.+?)\s*\(object_id=', re.MULTILINE)


def failure_cause(trajectory: Trajectory, label: str, *, grasped: bool) -> Cause:
    """Which stage lost a failed trial, split the way SceneReplica reports failures.

    ``label`` is the target's detection label ("cracker box"); ``grasped`` says whether
    the truth check saw it lifted. "perception": no scan or detect result ever named the
    target (a name whose words all belong to the label counts, so "cracker" or "box" do,
    "red box" does not). "planning": the target was seen but the stage that lost it, a
    pick or plan call before any lift, or a place call after one, answered that the planner
    found no path; the agent's own exploratory moves do not count. "execution": anything
    else, such as a grasp that slipped, a knocked object or a drop outside the square.
    """
    wanted = set(label.lower().split())
    seen = planning_failed = False
    stage = _PLACE_CALL if grasped else _PICK_CALL
    for step in trajectory.steps:
        in_stage = any(
            stage.search(f"{call.function_name} {json.dumps(call.arguments)}")
            for call in step.tool_calls or ()
        )
        for result in step.observation.results if step.observation else ():
            for json_name, listed_name in _DETECTION.findall(result.content):
                words = set((json_name or listed_name).lower().split())
                seen |= bool(words) and words <= wanted
            planning_failed |= in_stage and bool(_PLANNER_FAILED.search(result.content))
    if not seen:
        return "perception"
    if planning_failed:
        return "planning"
    return "execution"


def truth(check: Callable[[Store], float]) -> Check:
    """A check on the opened recording, as a check on the trial's outcome."""

    def grade(outcome: Outcome) -> float:
        with recording(outcome) as store:
            return check(store)

    return grade


def pick_and_place(entity: str, label: str, others: tuple[str, ...]) -> Callable[[Outcome], Graded]:
    """Grader for one trial: lifted (0.5) and placed in the drop-off (0.5), with the
    disturbance of the other objects and the failure cause kept in the details."""
    score = weighted(
        {
            "lifted": (0.5, truth(lambda store: lifted_by(store, entity, LIFT_M))),
            "placed": (0.5, truth(lambda store: resting_in_region(store, entity, DROPOFF))),
        }
    )

    def grade(outcome: Outcome) -> Graded:
        graded = score(outcome)
        lifted = graded.details["lifted"]
        grasped = isinstance(lifted, float) and lifted >= 1.0
        with recording(outcome) as store:
            disturbed: list[JsonValue] = [e for e in others if displacement(store, e) > DISTURBED_M]
        return Graded(
            score=graded.score,
            details={
                **graded.details,
                "unmoved": 1.0 - len(disturbed) / len(others) if others else 1.0,
                "disturbed": disturbed,
                "cause": None
                if graded.score >= 1.0
                else failure_cause(outcome.trajectory, label, grasped=grasped),
            },
        )

    return grade


def scene_cases(
    scene_dir: Path,
    unreachable: frozenset[str] = frozenset(),
    blueprint: Sequence[str] = BLUEPRINT,
) -> list[EvalCase]:
    """The five trials of one scene package, in near-to-far order from the arm base.

    ``blueprint`` is the launch, by registry name, that every trial runs on.
    """
    scene = SceneDescription.model_validate_json((scene_dir / "scene.json").read_text())
    base = scene.spawns["workbench"].position
    objects = [name for name, entity in scene.entities.items() if entity.movable]
    objects.sort(
        key=lambda name: math.hypot(
            scene.initial.poses[name].position.x - base.x,
            scene.initial.poses[name].position.y - base.y,
        )
    )
    cases = []
    for order, name in enumerate(objects, start=1):
        label = scene.entities[name].label
        tags = {"scenereplica", scene_dir.name, name, "perception", f"order-{order}"}
        if name in unreachable:
            tags.add("unreachable")
        cases.append(
            EvalCase(
                id=f"scenereplica-{scene_dir.name}-{name}",
                inputs=(
                    f"Pick up the {label} and place it in the drop-off area on the table at "
                    f"x=0.42 m, y=0.38 m (a 14 cm square). {SCENE}"
                ),
                environment=Sim2Environment(
                    blueprint=list(blueprint), scene=str(scene_dir), truth_entities=tuple(objects)
                ),
                grade=pick_and_place(name, label, tuple(o for o in objects if o != name)),
                timeout_s=TIMEOUT_S,
                tags=frozenset(tags),
            )
        )
    return cases


def suite(package: Path, blueprint: Sequence[str] = BLUEPRINT) -> list[EvalCase]:
    """Every trial of every ``scene-NN`` in the data package, flagged by its reach report.

    ``blueprint`` is the launch, by registry name, that every trial runs on.
    """
    report = json.loads((package / "reach_report.json").read_text())
    return [
        case
        for scene_dir in sorted(package.glob("scene-*"))
        for case in scene_cases(
            scene_dir,
            frozenset(
                name
                for name, reach in report["scenes"].get(scene_dir.name, {}).items()
                if not reach["reachable"]
            ),
            blueprint,
        )
    ]


SUITE: Suite = suite(get_data("scenereplica"))
