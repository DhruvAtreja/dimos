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

"""The 2D navigation scene the TypeSafe nav suites grade against.

A DimSim detections export (or a Habitat scene file in the same layout) becomes
labelled footprints: the goal box, the room bounds and the obstacles a ground
robot has to steer around. On this branch the TypeSafe agent drives an arm; the
nav suites keep reading their scenes through these helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

Box2D = tuple[float, float, float, float]  # minx, miny, maxx, maxy

# DimSim's agent is a capsule (halfHeight 0.25 + radius 0.12) resting on the
# floor. Anything whose box bottom is above its top passes overhead: door
# headers, wall cabinets, the TV.
ROBOT_TOP_M = 0.74
# A footprint inside another kept footprint (books on a shelf, a plate on a
# cart) adds nothing for navigation, and Jev's accuracy falls with irrelevant
# state. The tolerance absorbs a throw blanket overhanging its bed, and a chair
# tucked under its table (the table's footprint already blocks that spot).
CONTAINMENT_TOL_M = 0.1
WALL_PREFIXES = ("wall", "yard")


@dataclass(frozen=True, kw_only=True)
class Obstacle:
    label: str
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class Scene:
    frame_id: str
    goal_label: str
    goal_xy: tuple[float, float]
    goal_box: Box2D
    room_bounds: Box2D
    obstacles: list[Obstacle]


def load_scene(path: Path, goal_label: str) -> Scene:
    """Detections JSON -> the 2D scene Jev sees. ``goal_label`` is a substring."""
    if not goal_label:
        raise ValueError("goal_label must name the goal; an empty string matches everything")
    raw = json.loads(Path(path).expanduser().read_text())
    goal = next(
        (d for d in raw["detections"] if goal_label.lower() in str(d["label"]).lower()), None
    )
    if goal is None:
        raise LookupError(f"no detection labelled like {goal_label!r} in {path}")
    grounded = [d for d in raw["detections"] if _box_bottom(d) < ROBOT_TOP_M]
    boxes = {str(d["id"]): _footprint(d) for d in grounded}
    kept = [d for d in grounded if not _contained(str(d["id"]), boxes)]
    walls = [boxes[str(d["id"])] for d in kept if str(d["label"]).startswith(WALL_PREFIXES)]
    bounds = walls or list(boxes.values())
    return Scene(
        frame_id=str(raw.get("frame_id", "world")),
        goal_label=str(goal["label"]),
        goal_xy=(float(goal["center_xyz"][0]), float(goal["center_xyz"][1])),
        goal_box=_footprint(goal),
        room_bounds=(
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        ),
        obstacles=[
            Obstacle(
                label=str(d["label"]),
                min_xy=(boxes[str(d["id"])][0], boxes[str(d["id"])][1]),
                max_xy=(boxes[str(d["id"])][2], boxes[str(d["id"])][3]),
            )
            for d in kept
        ],
    )


def scene_labels(path: Path) -> list[str]:
    raw = json.loads(Path(path).expanduser().read_text())
    return [str(d["label"]) for d in raw["detections"]]


# Instruction words that also occur inside object labels ("with chrome",
# "wall-kitchen-front") but never name the goal.
_STOP_WORDS = frozenset(
    {
        "back",
        "drive",
        "find",
        "front",
        "from",
        "goal",
        "head",
        "into",
        "left",
        "navigate",
        "near",
        "next",
        "over",
        "please",
        "reach",
        "right",
        "robot",
        "side",
        "then",
        "there",
        "toward",
        "towards",
        "walk",
        "with",
    }
)


def derive_goal_label(instruction: str, labels: Sequence[str]) -> str:
    """The longest word of the instruction that occurs in some scene label.

    Jev does no language grounding here: the instruction has to call the goal
    by a word from its label ("bathtub", "sectional"), and code does the lookup
    so the goal box in the state is always a real object. Letting the model
    pick the goal would be a separate Choice question and a separate eval.
    """
    lowered = [label.lower() for label in labels]
    words = re.findall(r"[a-z][a-z-]{3,}", instruction.lower())
    hits = [w for w in words if w not in _STOP_WORDS and any(w in label for label in lowered)]
    if not hits:
        raise LookupError(
            f"no word of {instruction!r} names a scene object; use a word from its label"
        )
    return str(max(hits, key=len))


def _footprint(d: dict[str, Any]) -> Box2D:
    (cx, cy, _), (sx, sy, _) = d["center_xyz"], d["size_xyz"]
    return (
        round(cx - sx / 2, 3),
        round(cy - sy / 2, 3),
        round(cx + sx / 2, 3),
        round(cy + sy / 2, 3),
    )


def _box_bottom(d: dict[str, Any]) -> float:
    return float(d["center_xyz"][2] - d["size_xyz"][2] / 2)


def _inside(a: Box2D, b: Box2D, tol: float) -> bool:
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


def _contained(own_id: str, boxes: dict[str, Box2D]) -> bool:
    """Whether the box lies inside another footprint that does not lie inside it."""
    box = boxes[own_id]
    return any(
        other_id != own_id
        and _inside(box, other, CONTAINMENT_TOL_M)
        and not _inside(other, box, CONTAINMENT_TOL_M)
        for other_id, other in boxes.items()
    )
