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

"""The ``mujoco_xarm`` table tasks on sim2: the same prompts and scene text, the world-only
``xarm_table`` scene with the xArm7 at its ``workbench`` spawn, and ground truth read from
sim2's ``sim_truth`` stream instead of ``tf``. ``raw`` gives the agent the planner skills and
a wrist-camera image; ``perception`` launches ``xarm-perception-sim2`` instead, which adds
``scan_objects``, ``pick_object`` and ``place_at``.

    dimos evals run dimos.evals.suites.sim2_xarm --agent dimos.evals.agents.pi --tags raw
    dimos evals run dimos.evals.suites.sim2_xarm --agent dimos.evals.agents.pi --tags perception
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dimos.evals.environments.lib.sim_truth import first_entity_pose, last_entity_pose
from dimos.evals.environments.sim2 import Sim2Environment
from dimos.evals.scorers import Positions, lifted, stacked_on
from dimos.evals.suites.mujoco_xarm import PERCEPTION_SCENE, SCENE, TRACKED, Variant
from dimos.evals.types import EvalCase, Suite

if TYPE_CHECKING:
    from dimos.memory.store.base import Store
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

XARM_TABLE = Path(__file__).resolve().parents[1] / "scenes" / "xarm_table"

BLUEPRINTS: dict[Variant, list[str]] = {
    # The planner stack carries no skills of its own, so the manipulation skills are added.
    "raw": ["xarm7-planner-coordinator", "mcp-server", "observe-skill", "manipulation-skills"],
    # The perception stack already carries the manipulation skills, next to scene
    # registration, grasp generation and pick-and-place.
    "perception": ["xarm-perception-sim2", "mcp-server", "observe-skill"],
}


def _first(store: Store, entity: str) -> Vector3:
    return first_entity_pose(store, entity).position


def _last(store: Store, entity: str) -> Vector3:
    return last_entity_pose(store, entity).position


TRUTH_POSITIONS: Positions = (_first, _last)


def environment(variant: Variant) -> Sim2Environment:
    """The table scene on sim2, with truth required for the graded bodies."""
    return Sim2Environment(
        blueprint=BLUEPRINTS[variant], scene=str(XARM_TABLE), truth_entities=TRACKED
    )


def cases(variant: Variant) -> list[EvalCase]:
    """Both tasks for one variant. The ``raw`` ids are bare; ``perception`` ids carry a suffix."""
    suffix = "" if variant == "raw" else f"_{variant}"
    scene = SCENE if variant == "raw" else PERCEPTION_SCENE
    tags = frozenset({"sim2", "manipulation", "pick", variant})
    return [
        EvalCase(
            id=f"xarm_pick_cylinder{suffix}",
            inputs=f"Pick up the cylinder from the table and hold it in the air. {scene}",
            environment=environment(variant),
            grade=lifted("cup", by_m=0.05, positions=TRUTH_POSITIONS),
            timeout_s=600.0,
            tags=tags,
        ),
        EvalCase(
            id=f"xarm_ball_on_cylinder{suffix}",
            inputs=f"Pick up the red ball and place it on top of the cylinder. {scene}",
            environment=environment(variant),
            grade=stacked_on(
                "apple", "cup", rise_m=(0.08, 0.12), band_m=0.07, positions=TRUTH_POSITIONS
            ),
            timeout_s=900.0,
            threshold=0.5,
            tags=tags | {"place"},
        ),
    ]


SUITE: Suite = [*cases("raw"), *cases("perception")]
