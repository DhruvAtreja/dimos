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

from pydantic import ValidationError
import pytest

from dimos.sim2.scene_types import SceneDescription


def scene(*annotations: dict) -> SceneDescription:
    return SceneDescription.model_validate({"id": "s", "annotations": list(annotations)})


def test_points_paths_and_areas_parse_with_tags_and_props() -> None:
    parsed = scene(
        {
            "id": "goal_1",
            "kind": "point",
            "position": [1, 2, 0],
            "tags": ["nav:goal"],
            "props": {"tolerance": 0.5},
        },
        {"id": "walk", "kind": "path", "points": [[0, 0, 0], [3, 0, 0]], "tags": ["nav:mapping"]},
        {
            "id": "kitchen",
            "kind": "area",
            "points": [[0, 0, 0], [2, 0, 0], [2, 2, 0]],
            "height": 2.5,
        },
    )
    assert [a.kind for a in parsed.annotations] == ["point", "path", "area"]
    assert parsed.annotations[0].props["tolerance"] == 0.5


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "p", "kind": "point"},
        {"id": "w", "kind": "path", "points": [[0, 0, 0]]},
        {"id": "a", "kind": "area", "points": [[0, 0, 0], [1, 0, 0]]},
    ],
)
def test_each_kind_needs_its_geometry(bad: dict) -> None:
    with pytest.raises(ValidationError):
        scene(bad)
