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

from typing import cast

import pytest

from dimos.control.coordinator import TaskConfig
from dimos.control.teleop_coordinator import TeleopControlCoordinator
from dimos.core.coordination.blueprints import Blueprint
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.blueprints.collection import (
    DUAL_OPENYAM_COLLECTION_JOINT_SPEED_RAD_S,
    DUAL_OPENYAM_COLLECTION_STREAMS,
    DUAL_OPENYAM_WRIST_CAMERA_SERIALS,
    DualOpenYamCollectionRecorder,
    dual_openyam_collection_teleop_task,
    dual_openyam_wrist_camera,
    teleop_webxr_dual_openyam_collection,
)
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import teleop_webxr_dual_openyam


def _tasks(blueprint: Blueprint) -> list[TaskConfig]:
    coordinator = next(
        atom for atom in blueprint.blueprints if issubclass(atom.module, TeleopControlCoordinator)
    )
    return cast("list[TaskConfig]", coordinator.kwargs["tasks"])


def _teleop_task(blueprint: Blueprint) -> TaskConfig:
    return next(task for task in _tasks(blueprint) if task.type == "teleop_ik")


def test_collection_teleop_is_the_stock_task_at_the_collection_speed() -> None:
    ours = dual_openyam_collection_teleop_task()
    stock = _teleop_task(teleop_webxr_dual_openyam)

    assert ours.params["max_joint_velocity_rad_s"] == DUAL_OPENYAM_COLLECTION_JOINT_SPEED_RAD_S
    assert ours.params["max_joint_velocity_rad_s"] < stock.params["max_joint_velocity_rad_s"]
    assert {k: v for k, v in ours.params.items() if k != "max_joint_velocity_rad_s"} == {
        k: v for k, v in stock.params.items() if k != "max_joint_velocity_rad_s"
    }
    assert (ours.name, ours.priority, ours.joint_names) == (
        stock.name,
        stock.priority,
        stock.joint_names,
    )


def test_collection_graph_only_swaps_the_teleop_task() -> None:
    stock = teleop_webxr_dual_openyam
    ours = teleop_webxr_dual_openyam_collection

    assert ours.remapping_map == stock.remapping_map
    assert [atom.module for atom in ours.blueprints] == [atom.module for atom in stock.blueprints]
    stock_task, our_task = _teleop_task(stock), _teleop_task(ours)
    assert our_task.name == stock_task.name
    assert our_task.joint_names == stock_task.joint_names
    assert our_task.params["bindings"] == stock_task.params["bindings"]
    assert our_task.params["solver_type"] is stock_task.params["solver_type"]
    assert our_task.params != stock_task.params
    for a, b in zip(ours.blueprints, stock.blueprints, strict=True):
        if a.module is not DualOpenYamCoordinator:
            assert a.kwargs == b.kwargs
    assert [t for t in _tasks(ours) if t.type != "teleop_ik"] == [
        t for t in _tasks(stock) if t.type != "teleop_ik"
    ]


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
