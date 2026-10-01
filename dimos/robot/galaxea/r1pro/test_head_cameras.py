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

from __future__ import annotations

import pytest

from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.galaxea.r1pro.blueprints.basic.r1pro_coordinator import r1pro_control
from dimos.robot.galaxea.r1pro.connection import R1ProConnection
from dimos.robot.galaxea.r1pro.head_cameras import (
    HEAD_LEFT_V4L2,
    HEAD_RIGHT_V4L2,
    HeadLeftCamera,
    HeadRightCamera,
)

# The ISX031's only real mode: width, height (px), fps, pixel format.
_SENSOR_MODE = (1920, 1536, 30.0, "UYVY")


@pytest.mark.parametrize("wrist_depth", [False, True])
def test_each_eye_opens_its_own_node_at_the_sensor_mode(wrist_depth: bool) -> None:
    atoms = {
        atom.module: atom.kwargs
        for atom in r1pro_control(wrist_depth=wrist_depth).active_blueprints
    }

    left, right = atoms[HeadLeftCamera], atoms[HeadRightCamera]
    assert (left["device"], left["frame_id"]) == (HEAD_LEFT_V4L2, "head_left_optical")
    assert (right["device"], right["frame_id"]) == (HEAD_RIGHT_V4L2, "head_right_optical")
    for eye in (left, right):
        assert (eye["width"], eye["height"], eye["fps"], eye["fourcc"]) == _SENSOR_MODE


def test_head_colour_is_raw_image_under_the_old_names() -> None:
    blueprint = r1pro_control()

    assert {"head_left_color", "head_right_color"} <= set(blueprint.remapping_map.values())
    assert ("head_left_color", Image) in blueprint.transport_map
    assert ("head_right_color", Image) in blueprint.transport_map


def test_connection_no_longer_publishes_head_colour() -> None:
    assert not hasattr(R1ProConnection, "head_left_color")
    assert not hasattr(R1ProConnection, "head_right_color")
