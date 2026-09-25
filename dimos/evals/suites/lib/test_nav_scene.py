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

"""The nav scene helpers, offline: no API key, no simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dimos.evals.suites.lib.nav_scene import derive_goal_label, load_scene, scene_labels

SHIPPED_SCENE = Path(__file__).parents[1] / "scenes" / "apartment_detections.json"


def _det(
    id_: str, label: str, cx: float, cy: float, cz: float, sx: float, sy: float, sz: float
) -> dict[str, Any]:
    """One entry in the DimSim detections schema (PR #4208)."""
    return {
        "id": id_,
        "label": label,
        "score": 1.0,
        "center_xyz": [cx, cy, cz],
        "size_xyz": [sx, sy, sz],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


SCENE = {
    "frame_id": "world",
    "timestamp": 0.0,
    "count": 6,
    "detections": [
        _det("couch", "Modern L-shaped sectional", 1.0, 4.0, 0.4, 2.0, 1.0, 0.8),  # goal
        _det("table", "Coffee table", 3.0, 0.0, 0.25, 1.0, 1.0, 0.5),
        _det("plate", "Dinner plate", 3.0, 0.0, 0.52, 0.2, 0.2, 0.02),  # on the table
        _det("header", "wall-south-header", 5.0, 0.0, 2.7, 0.2, 1.0, 0.9),  # overhead
        _det("wall-n", "wall-north", -5.0, 0.0, 1.5, 0.2, 12.0, 3.0),
        _det("wall-s", "wall-south", 5.0, 0.0, 1.5, 0.2, 12.0, 3.0),
    ],
}


@pytest.fixture
def scene_json(tmp_path: Path) -> Path:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(SCENE))
    return path


def test_scene_drops_overhead_and_contained_boxes(scene_json: Path) -> None:
    """The door header passes over the robot; the plate sits inside the table."""
    scene = load_scene(scene_json, "sectional")
    assert [o.label for o in scene.obstacles] == [
        "Modern L-shaped sectional",
        "Coffee table",
        "wall-north",
        "wall-south",
    ]


def test_scene_goal_box_and_room_bounds(scene_json: Path) -> None:
    scene = load_scene(scene_json, "sectional")
    assert scene.goal_label == "Modern L-shaped sectional"
    assert scene.goal_xy == (1.0, 4.0)
    assert scene.goal_box == (0.0, 3.5, 2.0, 4.5)
    assert scene.room_bounds == (-5.1, -6.0, 5.1, 6.0)  # from the walls, not the furniture


def test_scene_unknown_goal_raises(scene_json: Path) -> None:
    with pytest.raises(LookupError, match="hot tub"):
        load_scene(scene_json, "hot tub")


def test_shipped_apartment_scene_is_ground_truth() -> None:
    """Pins the copied PR #4208 snapshot and the filter: 15 walls + 33 objects.

    Of 107 detections, 43 are overhead (headers, wall cabinets, the TV) and 16
    are contained footprints (shelf and tabletop clutter, bedding, and one
    dining chair tucked fully under its table).
    """
    scene = load_scene(SHIPPED_SCENE, "sectional")
    labels = [o.label for o in scene.obstacles]
    assert sum(label.startswith(("wall", "yard")) for label in labels) == 15
    assert len(labels) == 48
    assert scene.goal_box == (0.156, 3.028, 1.956, 5.735)
    assert not any("header" in label for label in labels)


def test_derive_goal_label_picks_the_word_that_names_an_object(scene_json: Path) -> None:
    labels = scene_labels(scene_json)
    assert derive_goal_label("navigate to the sectional couch", labels) == "sectional"
    assert derive_goal_label("go to the bathtub", scene_labels(SHIPPED_SCENE)) == "bathtub"


def test_derive_goal_label_ignores_stop_and_short_words(scene_json: Path) -> None:
    """'with' occurs inside labels ("with chrome") but never names a goal."""
    with pytest.raises(LookupError, match="names a scene object"):
        derive_goal_label("go with the flow", scene_labels(SHIPPED_SCENE))


def test_load_scene_rejects_an_empty_goal(scene_json: Path) -> None:
    with pytest.raises(ValueError, match="matches everything"):
        load_scene(scene_json, "")
