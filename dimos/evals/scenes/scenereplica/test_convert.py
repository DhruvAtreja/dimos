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

"""Frame maths and scene-file shape for the SceneReplica converter. No downloads."""

from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.io import savemat

from dimos.evals.scenes.scenereplica.convert import (
    COLLISION_GROUP,
    OBJECTS,
    SCENE_IDS,
    SPAWN_HEIGHT,
    TABLE,
    ObjectAsset,
    Placement,
    base_link_to_world,
    quat_wxyz_to_xyzw,
    rest_on_table,
    scene_description,
    scene_xml,
    write_scene,
)
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.sim2.scene_types import SceneDescription

HALF = 0.05
CUBE = np.array(
    [[sx * HALF, sy * HALF, sz * HALF] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
)
YAW_90_WXYZ = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
YAW_90_XYZW = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))


def _asset(ycb: str) -> ObjectAsset:
    return ObjectAsset(
        spec=OBJECTS[ycb],
        mass=0.1,
        collision_method="test",
        pieces=("collision_00.obj",),
        inertial_pos=(0.0, 0.0, 0.0),
        fullinertia=(1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0),
        vertices=CUBE,
    )


FIVE = ("003_cracker_box", "024_bowl", "025_mug", "011_banana", "035_power_drill")


def _objects() -> list[tuple[ObjectAsset, Pose]]:
    return [
        (_asset(ycb), Pose(0.5 + 0.05 * i, 0.1 * i - 0.2, TABLE.top_z + HALF + 0.001))
        for i, ycb in enumerate(FIVE)
    ]


def test_quaternion_reorders_from_wxyz_to_xyzw() -> None:
    assert quat_wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0, 1.0)
    assert quat_wxyz_to_xyzw(YAW_90_WXYZ) == pytest.approx(YAW_90_XYZW)


def test_base_link_to_world_copies_positions_when_frames_coincide() -> None:
    pose = base_link_to_world(Placement("024_bowl", (0.48, -0.17, 0.77), YAW_90_WXYZ))
    assert pose.position.to_tuple() == pytest.approx((0.48, -0.17, 0.77))
    assert pose.orientation.to_tuple() == pytest.approx(YAW_90_XYZW)


def test_base_link_to_world_applies_a_base_offset() -> None:
    # Fetch base 1 m ahead and 2 m left of the arm, turned 90 degrees left: base_link
    # +x becomes world +y, so a point 0.5 m ahead of Fetch lands at (1, 2.5).
    base = Pose((1.0, 2.0, 0.0), YAW_90_XYZW)
    pose = base_link_to_world(Placement("025_mug", (0.5, 0.0, 0.7), (1.0, 0.0, 0.0, 0.0)), base)
    assert pose.position.to_tuple() == pytest.approx((1.0, 2.5, 0.7), abs=1e-9)
    assert pose.orientation.to_tuple() == pytest.approx(YAW_90_XYZW)


def test_rest_on_table_moves_only_z() -> None:
    floating = Pose(0.5, 0.2, 0.9)
    rested, shift = rest_on_table(CUBE, floating, TABLE.top_z)
    assert shift == pytest.approx(TABLE.top_z + 0.001 - (0.9 - HALF))
    assert rested.position.to_tuple() == pytest.approx((0.5, 0.2, TABLE.top_z + 0.001 + HALF))
    assert rested.orientation.to_tuple() == floating.orientation.to_tuple()
    # Tilted 45 degrees about x, the lowest corner is a half-diagonal below centre.
    tilted = Pose((0.5, 0.2, 0.9), (math.sin(math.pi / 8), 0.0, 0.0, math.cos(math.pi / 8)))
    rested, _ = rest_on_table(CUBE, tilted, TABLE.top_z)
    assert rested.position.z == pytest.approx(TABLE.top_z + 0.001 + HALF * math.sqrt(2))


def test_scene_description_has_the_sim2_shape() -> None:
    objects = _objects()
    description = scene_description(1, SCENE_IDS[0], objects, SPAWN_HEIGHT, {"mug": 0.0})
    assert description.format == "dimos.scene.v1"
    assert description.id == "scenereplica-01"
    assert set(description.entities) == {"cracker_box", "bowl", "mug", "banana", "power_drill"}
    assert all(e.movable and e.body == key for key, e in description.entities.items())
    assert description.entities["cracker_box"].label == "cracker box"
    assert set(description.initial.poses) == set(description.entities)
    assert description.initial.poses["mug"].position.to_tuple() == pytest.approx(
        objects[2][1].position.to_tuple()
    )
    table = description.regions["table/top"]
    assert table.kind == "support" and table.body == "world"
    assert table.pose.position.to_tuple() == pytest.approx((TABLE.centre_x, 0.0, TABLE.top_z))
    assert table.size == pytest.approx((TABLE.size, TABLE.size, 0.0))
    assert description.regions["dropoff/top"].kind == "support"
    assert description.spawns["workbench"].position.to_tuple() == (0.0, 0.0, SPAWN_HEIGHT)
    assert description.hidden_geom_groups == (COLLISION_GROUP,)
    assert description.provenance["source"] == "SceneReplica"
    assert description.provenance["source_scene_id"] == 36
    rebuilt = SceneDescription.model_validate_json(description.model_dump_json())
    assert rebuilt == description


def _child(parent: ET.Element, path: str) -> ET.Element:
    child = parent.find(path)
    assert child is not None, path
    return child


def _numbers(element: ET.Element, attribute: str) -> list[float]:
    value = element.get(attribute)
    assert value is not None, attribute
    return [float(v) for v in value.split()]


def test_scene_xml_places_bodies_with_wxyz_quaternions() -> None:
    objects = _objects()
    objects[2] = (objects[2][0], Pose(objects[2][1].position.to_tuple(), YAW_90_XYZW))
    root = ET.fromstring(scene_xml("scenereplica-01", objects, SPAWN_HEIGHT))
    assert root.tag == "mujoco" and root.get("model") == "scenereplica-01"
    assert _child(root, "compiler").get("meshdir") == "../_assets"
    option = _child(root, "option")
    assert (option.get("cone"), option.get("impratio")) == ("elliptic", "10")
    world = _child(root, "worldbody")
    bodies = {b.get("name"): b for b in world.findall("body")}
    assert set(bodies) == {a.spec.entity for a, _ in objects}
    mug = bodies["mug"]
    assert mug.find("freejoint") is not None
    assert _numbers(mug, "quat") == pytest.approx((YAW_90_XYZW[3], *YAW_90_XYZW[:3]), abs=1e-6)
    assert _numbers(mug, "pos") == pytest.approx(objects[2][1].position.to_tuple())
    assert [g.get("class") for g in mug.findall("geom")] == ["object_visual", "object_collision"]
    meshes = {m.get("name"): m.get("file") for m in _child(root, "asset").findall("mesh")}
    assert meshes["mug_visual"] == "025_mug/visual.obj"
    assert meshes["mug_collision_00"] == "025_mug/collision_00.obj"
    assert world.find("geom[@name='pedestal']") is not None


def test_write_scene_reads_mat_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    poses = np.array([[0.48, 0.26, 0.79, *YAW_90_WXYZ], [0.67, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]])
    savemat(
        metadata / "meta-000036.mat",
        {"object_names": np.array(["004_sugar_box      ", "011_banana         "]), "poses": poses},
    )
    from dimos.evals.scenes.scenereplica.convert import read_scene_metadata

    placements = read_scene_metadata(metadata / "meta-000036.mat")
    assert [p.ycb for p in placements] == ["004_sugar_box", "011_banana"]
    assert placements[0].quat_wxyz == pytest.approx(YAW_90_WXYZ)

    assets = {ycb: _asset(ycb) for ycb in ("004_sugar_box", "011_banana")}
    scene_dir = write_scene(tmp_path / "out", 1, placements, assets, 0.7, settle_seconds=0.0)
    description = SceneDescription.model_validate_json((scene_dir / "scene.json").read_text())
    assert set(description.entities) == {"sugar_box", "banana"}
    banana = description.initial.poses["banana"]
    assert banana.position.to_tuple() == pytest.approx((0.67, 0.0, TABLE.top_z + 0.001 + HALF))
    assert description.initial.poses["sugar_box"].orientation.to_tuple() == pytest.approx(
        YAW_90_XYZW
    )
    assert description.spawns["workbench"].position.z == 0.7
    assert json.loads((scene_dir / "scene.json").read_text())["provenance"]["rest_shift_m"][
        "banana"
    ] == pytest.approx(TABLE.top_z + 0.001 + HALF - 0.76, abs=1e-4)
    assert ET.parse(scene_dir / "scene.xml").getroot().tag == "mujoco"


@pytest.mark.mujoco
def test_generated_scene_compiles_in_mujoco(tmp_path: Path) -> None:
    import mujoco
    from PIL import Image
    import trimesh

    for ycb in FIVE:
        folder = tmp_path / "_assets" / ycb
        folder.mkdir(parents=True)
        trimesh.creation.box((2 * HALF,) * 3).export(folder / "visual.obj")
        trimesh.creation.box((2 * HALF,) * 3).export(folder / "collision_00.obj")
        Image.new("RGB", (4, 4), (200, 40, 60)).save(folder / "texture.png")
    scene_dir = tmp_path / "scene-01"
    scene_dir.mkdir()
    (scene_dir / "scene.xml").write_text(scene_xml("scenereplica-01", _objects(), SPAWN_HEIGHT))
    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    assert model.body("mug").id > 0
    assert model.joint("mug").type == mujoco.mjtJoint.mjJNT_FREE
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert data.xpos[model.body("mug").id][2] == pytest.approx(TABLE.top_z + HALF, abs=0.005)
