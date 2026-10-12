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

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.blueprints import Blueprint, BlueprintAtom
from dimos.core.coordination.module_coordinator import (
    _resolve_single_ref,
    _verify_no_name_conflicts,
)
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.perception.localize.scene_registration import (
    LocalizeSceneRegistrationConfig,
    LocalizeSceneRegistrationModule,
)
from dimos.robot.all_blueprints import all_blueprints
from dimos.robot.manipulators.xarm.blueprints.sim2 import xarm_perception_sim2
from dimos.robot.manipulators.xarm.blueprints.sim2_dan import xarm_perception_sim2_dan

BASELINE = xarm_perception_sim2
DAN = xarm_perception_sim2_dan


def _atoms(blueprint: Blueprint) -> dict[str, BlueprintAtom]:
    return {atom.name: atom for atom in blueprint.active_blueprints}


def _topic(blueprint: Blueprint, atom: BlueprintAtom, stream: str) -> str:
    """The topic a module's port ends up on, the way the coordinator resolves it."""
    topic = blueprint.remapping_map.get((atom.name, stream), stream)
    assert isinstance(topic, str)
    return topic


def test_only_the_perception_module_differs_from_the_baseline() -> None:
    baseline, dan = _atoms(BASELINE), _atoms(DAN)
    registration = ObjectSceneRegistrationModule.name
    memory = LocalizeSceneRegistrationModule.name

    assert set(baseline) - {registration} == set(dan) - {memory}
    assert registration in baseline and memory in dan and registration not in dan
    for name in set(baseline) - {registration}:
        assert dan[name] is baseline[name], name
    assert dict(DAN.remapping_map) == dict(BASELINE.remapping_map)
    assert dict(DAN.transport_map) == dict(BASELINE.transport_map)
    assert dict(DAN.global_config_overrides) == dict(BASELINE.global_config_overrides)
    _verify_no_name_conflicts(DAN)
    assert all_blueprints["xarm-perception-sim2-dan"].endswith(":xarm_perception_sim2_dan")


def test_the_memory_takes_scene_registrations_place_in_the_module_order() -> None:
    # Workers are filled in module order: the same order puts every other module
    # in a process with the same neighbours as in the baseline.
    registration = ObjectSceneRegistrationModule.name
    memory = LocalizeSceneRegistrationModule.name
    baseline = [atom.name for atom in BASELINE.active_blueprints]
    assert [atom.name for atom in DAN.active_blueprints] == [
        memory if name == registration else name for name in baseline
    ]
    assert DAN.disabled_modules_tuple == ()


def test_pick_and_place_reads_its_objects_from_the_localize_memory() -> None:
    pick = _atoms(DAN)[PickAndPlaceModule.name]
    (scene,) = [ref for ref in pick.module_refs if ref.name == "_scene"]
    provider = _resolve_single_ref(pick, scene, scene.spec, DAN, set(DAN.disabled_modules_tuple))
    assert provider == LocalizeSceneRegistrationModule.name


def test_the_memory_reads_the_wrist_camera_streams_registration_read() -> None:
    registration = _atoms(BASELINE)[ObjectSceneRegistrationModule.name]
    memory = _atoms(DAN)[LocalizeSceneRegistrationModule.name]
    camera = _atoms(DAN)["arm_wrist_camera"]
    inputs = {stream.name: stream.type for stream in memory.streams if stream.direction == "in"}
    camera_out = {stream.name: stream.type for stream in camera.streams}

    assert inputs.keys() == {"color_image", "depth_image", "camera_info", "tf"}
    for name, kind in inputs.items():
        assert _topic(DAN, memory, name) == _topic(BASELINE, registration, name)
        assert camera_out[name] is kind
        assert _topic(DAN, camera, name) == _topic(DAN, memory, name)


def test_the_memory_answers_in_the_world_frame_from_one_camera_position() -> None:
    parsed = BlueprintConfigParser(DAN).parse(environ={})
    config = LocalizeSceneRegistrationConfig.model_validate(
        parsed.module_kwargs(LocalizeSceneRegistrationModule.name)
    )
    assert config.world_frame == "world"
    assert config.policy == {"min_views": 1}
    assert not config.mobile
