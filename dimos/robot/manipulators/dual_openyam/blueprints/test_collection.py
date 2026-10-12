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

import pytest

from dimos.robot.manipulators.dual_openyam.blueprints.collection import (
    DUAL_OPENYAM_COLLECTION_STREAMS,
    DUAL_OPENYAM_WRIST_CAMERA_SERIALS,
    DualOpenYamCollectionRecorder,
    dual_openyam_wrist_camera,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_camera_streams_carry_the_side_prefix(side: str) -> None:
    camera = dual_openyam_wrist_camera(side)
    atom = camera.blueprints[0]

    assert atom.kwargs["serial_number"] == DUAL_OPENYAM_WRIST_CAMERA_SERIALS[side]
    assert atom.kwargs["enable_depth"] is False
    remapped = {camera.remapping_map[(atom.name, s.name)] for s in atom.streams}
    assert f"{side}_wrist_color_image" in remapped
    # a prefix with an underscore, not a namespace: the recorder stores stream
    # names as SQL identifiers
    assert all("/" not in name for name in remapped)


def test_wrist_camera_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError):
        dual_openyam_wrist_camera("centre")


def test_recorder_declares_every_collection_stream_as_an_input() -> None:
    atom = DualOpenYamCollectionRecorder.blueprint().blueprints[0]
    inputs = {s.name for s in atom.streams if s.direction == "in"}

    assert set(DUAL_OPENYAM_COLLECTION_STREAMS) <= inputs
