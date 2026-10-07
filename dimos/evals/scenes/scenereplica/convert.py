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

"""Turn the 20 SceneReplica tabletop scenes into sim2 scene packages.

SceneReplica (https://github.com/IRVLUTD/SceneReplica) places five YCB objects
on a cafe table in front of a Fetch robot. This script downloads its archives,
converts the 16 YCB meshes into MuJoCo assets, and writes one world-only
``scene.xml`` + ``scene.json`` per scene under ``data/scenereplica/``, so the
scenes run with ``dimos --simulation mujoco --scene-package <folder>``.

Frame convention. SceneReplica poses are in the Fetch ``base_link`` frame: x
forward, y left, z up, origin on the floor under the robot. Our world frame is
the same frame: the floor is z=0, the table top is at z=0.745 centred 0.8 m
ahead, and the xArm7 stands on a pedestal at x=y=0 (the ``workbench`` spawn).
Object positions therefore copy across unchanged; only the quaternion order
changes, from SceneReplica's ``[w, x, y, z]`` to our ``[x, y, z, w]``.

Run ``python -m dimos.evals.scenes.scenereplica.convert --help``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast
import zipfile

import numpy as np
from numpy.typing import NDArray
import requests
from scipy.spatial.transform import Rotation

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.sim2.scene_types import SceneDescription, SceneEntity, SceneRegion, SceneUpdate

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)

SOURCE_URL = "https://github.com/IRVLUTD/SceneReplica"
# Box "shared/static" form of the links in the SceneReplica README; curl-able.
FINAL_SCENES_URL = "https://utdallas.box.com/shared/static/47foq3nri7ob3gym853ynwrenfvv6oix.zip"
MODELS_URL = "https://utdallas.box.com/shared/static/zkt115qz5d5or0h1ehkzafnut8of4571.zip"

PACKAGE = "scenereplica"
ASSETS_DIR = "_assets"
# SceneReplica's official scene order (final_scenes/scene_ids.txt); scene-01 is id 36.
SCENE_IDS = (36, 84, 68, 10, 77, 148, 48, 25, 104, 38, 27, 122, 141, 65, 39, 83, 130, 161, 33, 56)

# Where the xArm7's base plate sits, in metres above the floor: 0.155 m above the
# table top. The `reach` sweep put 99 of the 100 top-down hover poses in reach
# from here (0.745 m, level with the table, reaches 91); README.md has the table.
SPAWN_HEIGHT = 0.90
# Collision geoms go in this MJCF group; scene.json hides it from the cameras.
COLLISION_GROUP = 3


@dataclass(frozen=True)
class ObjectSpec:
    """One YCB object: its SceneReplica name and how our scene names it."""

    ycb: str
    entity: str
    label: str
    kind: str


OBJECTS: dict[str, ObjectSpec] = {
    spec.ycb: spec
    for spec in (
        ObjectSpec("003_cracker_box", "cracker_box", "cracker box", "box"),
        ObjectSpec("004_sugar_box", "sugar_box", "sugar box", "box"),
        ObjectSpec("005_tomato_soup_can", "tomato_soup_can", "tomato soup can", "can"),
        ObjectSpec("006_mustard_bottle", "mustard_bottle", "mustard bottle", "bottle"),
        ObjectSpec("007_tuna_fish_can", "tuna_fish_can", "tuna fish can", "can"),
        ObjectSpec("008_pudding_box", "pudding_box", "pudding box", "box"),
        ObjectSpec("009_gelatin_box", "gelatin_box", "gelatin box", "box"),
        ObjectSpec("010_potted_meat_can", "potted_meat_can", "potted meat can", "can"),
        ObjectSpec("011_banana", "banana", "banana", "fruit"),
        ObjectSpec("021_bleach_cleanser", "bleach_cleanser", "bleach cleanser bottle", "bottle"),
        ObjectSpec("024_bowl", "bowl", "bowl", "bowl"),
        ObjectSpec("025_mug", "mug", "mug", "mug"),
        ObjectSpec("035_power_drill", "power_drill", "power drill", "tool"),
        ObjectSpec("037_scissors", "scissors", "scissors", "tool"),
        ObjectSpec("040_large_marker", "large_marker", "marker", "marker"),
        ObjectSpec("052_extra_large_clamp", "extra_large_clamp", "clamp", "tool"),
    )
}


@dataclass(frozen=True)
class Table:
    """The cafe table, in world metres. SceneReplica's ``cafe_table_org`` model has a
    0.913 m square top; placed at z=-0.03 its top lands at 0.745 m."""

    centre_x: float = 0.8
    top_z: float = 0.745
    size: float = 0.913
    thickness: float = 0.04


TABLE = Table()

# Where to put an object down. SceneReplica slides the Fetch gripper to
# (0.78, 0.40) in base_link, beyond the xArm7's reach from the origin, so our
# drop-off is the near-left corner of the same table.
DROPOFF_XY = (0.42, 0.38)
DROPOFF_SIZE = 0.14


@dataclass(frozen=True)
class Placement:
    """One object pose from a SceneReplica ``meta-*.mat`` file, in Fetch base_link."""

    ycb: str
    position: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Frames


def quat_wxyz_to_xyzw(q: Sequence[float]) -> tuple[float, float, float, float]:
    """Reorder a quaternion from SceneReplica's [w, x, y, z] to our [x, y, z, w]."""
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)


def base_link_to_world(placement: Placement, base_in_world: Pose | None = None) -> Pose:
    """Express a SceneReplica object pose in our world frame.

    Args:
        placement: Position (m) and [w, x, y, z] quaternion in the Fetch base_link frame.
        base_in_world: Where the Fetch base_link sits in our world. None means the two
            frames coincide, which is the convention the shipped scenes use.
    """
    local = Pose(placement.position, quat_wxyz_to_xyzw(placement.quat_wxyz))
    if base_in_world is None:
        return local
    return base_in_world + local


def rest_on_table(
    vertices: NDArray[np.float64], pose: Pose, table_top: float, clearance: float = 0.001
) -> tuple[Pose, float]:
    """Shift a pose straight up or down so the object's lowest point sits on the table.

    Args:
        vertices: Collision vertices in the object frame, metres, shape (N, 3).
        pose: The object's world pose.
        table_top: Height of the table surface in metres.
        clearance: Gap left between object and table in metres.

    Returns:
        The adjusted pose and the shift applied in metres (positive = raised).
    """
    rotation = Rotation.from_quat(pose.orientation.to_tuple())
    lowest = float((rotation.apply(vertices)[:, 2] + pose.position.z).min())
    shift = table_top + clearance - lowest
    position = pose.position.to_tuple()
    moved = Pose((position[0], position[1], position[2] + shift), pose.orientation.to_tuple())
    return moved, shift


# ---------------------------------------------------------------------------
# Source data


def download(url: str, destination: Path) -> Path:
    """Fetch a file once; a file that already exists is kept."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        partial = destination.with_suffix(destination.suffix + ".part")
        with partial.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
        partial.replace(destination)
    return destination


def unpack_scenes(source: Path, work: Path) -> Path:
    """Return the ``final_scenes`` folder, extracting ``final_scenes.zip`` if given one."""
    if source.is_dir():
        return source
    target = work / "final_scenes"
    if not (target / "metadata").is_dir():
        with zipfile.ZipFile(source) as archive:
            archive.extractall(work)
    return target


def unpack_models(source: Path, work: Path) -> Path:
    """Return the ``models`` folder holding the 16 YCB objects, extracting only the
    files the conversion needs from ``models.zip`` if given the archive."""
    if source.is_dir():
        return source
    target = work / "models"
    wanted = {
        f"models/{ycb}/{name}"
        for ycb in OBJECTS
        for name in ("textured_simple.obj", "texture_map.png", "model.sdf")
    }
    missing = [name for name in wanted if not (work / name).is_file()]
    if missing:
        with zipfile.ZipFile(source) as archive:
            archive.extractall(work, members=missing)
    return target


def read_scene_metadata(mat_path: Path) -> list[Placement]:
    """Read one ``meta-%06d.mat``: object names and [x y z w x y z] poses in base_link."""
    from scipy.io import loadmat

    data = loadmat(str(mat_path))
    names = [str(name).strip() for name in np.asarray(data["object_names"]).ravel()]
    poses = np.asarray(data["poses"], dtype=float).reshape(len(names), 7)
    placements = []
    for name, pose in zip(names, poses, strict=True):
        if name not in OBJECTS:
            raise ValueError(f"{mat_path.name}: unknown object {name!r}")
        placements.append(
            Placement(
                name,
                (float(pose[0]), float(pose[1]), float(pose[2])),
                (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
            )
        )
    return placements


# ---------------------------------------------------------------------------
# Object assets


@dataclass(frozen=True)
class ObjectAsset:
    """One converted YCB object under ``_assets/<ycb>/``."""

    spec: ObjectSpec
    mass: float
    collision_method: str
    pieces: tuple[str, ...]
    inertial_pos: tuple[float, float, float]
    # MuJoCo fullinertia order: Ixx Iyy Izz Ixy Ixz Iyz, kg m^2.
    fullinertia: tuple[float, float, float, float, float, float]
    vertices: NDArray[np.float64]

    def manifest(self) -> dict[str, object]:
        return {
            "entity": self.spec.entity,
            "label": self.spec.label,
            "kind": self.spec.kind,
            "mass_kg": self.mass,
            "collision_method": self.collision_method,
            "collision_pieces": list(self.pieces),
            "inertial_pos": list(self.inertial_pos),
            "fullinertia": list(self.fullinertia),
        }


def _sdf_mass(sdf: Path) -> float:
    import re

    match = re.search(r"<mass>\s*([0-9.eE+-]+)\s*</mass>", sdf.read_text())
    if match is None:
        raise ValueError(f"{sdf}: no <mass>")
    return float(match.group(1))


def _load_mesh(path: Path) -> trimesh.Trimesh:
    import trimesh

    return cast("trimesh.Trimesh", trimesh.load(path, force="mesh", process=False))


def _vec3(values: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


def _vec6(values: Iterable[float]) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, e, f = (float(v) for v in values)
    return (a, b, c, d, e, f)


def _strip_material_lines(obj: Path) -> str:
    """Drop mtllib/usemtl lines: MuJoCo takes the texture from the MJCF, not the OBJ."""
    return "".join(
        line
        for line in obj.read_text().splitlines(keepends=True)
        if not line.startswith(("mtllib", "usemtl"))
    )


# Objects whose shape a single convex hull already describes: boxes, cans, the
# marker. The rest (bowl, mug, drill, scissors, clamp, banana, bottles) have
# hollows or handles a hull would fill in, so they get a real decomposition.
CONVEX_OBJECTS = frozenset(
    {
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "040_large_marker",
    }
)


def _decompose(
    ycb: str, vertices: NDArray[np.float64], faces: NDArray[np.int64]
) -> tuple[str, list[tuple[NDArray[np.float64], NDArray[np.int64]]]]:
    """Return convex collision pieces: one hull for convex objects, CoACD pieces for
    the others (falling back to one hull if CoACD is not installed)."""
    import trimesh

    def hull() -> tuple[str, list[tuple[NDArray[np.float64], NDArray[np.int64]]]]:
        shape = trimesh.Trimesh(vertices, faces, process=False).convex_hull
        return "convex-hull", [(np.asarray(shape.vertices), np.asarray(shape.faces))]

    if ycb in CONVEX_OBJECTS:
        return hull()
    try:
        import coacd
    except ImportError:
        return hull()
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=0.05, max_convex_hull=24, seed=0)
    return "coacd", [(np.asarray(v), np.asarray(f)) for v, f in parts]


def convert_object(
    models_dir: Path, assets_dir: Path, spec: ObjectSpec, texture_size: int
) -> ObjectAsset:
    """Write ``_assets/<ycb>/``: visual.obj, texture.png and convex collision_NN.obj files."""
    from PIL import Image
    import trimesh

    source = models_dir / spec.ycb
    target = assets_dir / spec.ycb
    target.mkdir(parents=True, exist_ok=True)
    (target / "visual.obj").write_text(_strip_material_lines(source / "textured_simple.obj"))
    with Image.open(source / "texture_map.png") as texture:
        texture.convert("RGB").resize((texture_size, texture_size)).save(target / "texture.png")

    mesh = _load_mesh(source / "textured_simple.obj")
    method, parts = _decompose(
        spec.ycb, np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    )
    for old in target.glob("collision_*.obj"):
        old.unlink()
    pieces = []
    for index, (vertices, faces) in enumerate(parts):
        name = f"collision_{index:02d}.obj"
        trimesh.Trimesh(vertices, faces, process=False).export(target / name)
        pieces.append(name)

    # Mass from SceneReplica's Gazebo SDF; inertia of the convex hull scaled to it.
    mass = _sdf_mass(source / "model.sdf")
    hull = mesh.convex_hull
    hull.density = mass / hull.volume
    inertia = np.asarray(hull.moment_inertia)
    asset = ObjectAsset(
        spec=spec,
        mass=mass,
        collision_method=method,
        pieces=tuple(pieces),
        inertial_pos=_vec3(hull.center_mass),
        fullinertia=(
            float(inertia[0, 0]),
            float(inertia[1, 1]),
            float(inertia[2, 2]),
            float(inertia[0, 1]),
            float(inertia[0, 2]),
            float(inertia[1, 2]),
        ),
        vertices=np.vstack([v for v, _ in parts]),
    )
    logger.info("%s: %s, %d collision pieces, %.3f kg", spec.ycb, method, len(pieces), mass)
    return asset


def convert_objects(
    models_dir: Path, assets_dir: Path, texture_size: int = 1024
) -> dict[str, ObjectAsset]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        ycb: convert_object(models_dir, assets_dir, spec, texture_size)
        for ycb, spec in OBJECTS.items()
    }
    manifest = {
        "source": SOURCE_URL,
        "mesh": "YCB textured_simple.obj from SceneReplica models.zip",
        "texture_size": texture_size,
        "objects": {ycb: asset.manifest() for ycb, asset in assets.items()},
    }
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return assets


def load_assets(assets_dir: Path) -> dict[str, ObjectAsset]:
    """Read back converted objects from ``_assets/manifest.json`` and the piece meshes."""
    manifest = json.loads((assets_dir / "manifest.json").read_text())
    assets = {}
    for ycb, entry in manifest["objects"].items():
        pieces = tuple(str(piece) for piece in entry["collision_pieces"])
        vertices = np.vstack(
            [np.asarray(_load_mesh(assets_dir / ycb / piece).vertices) for piece in pieces]
        )
        assets[ycb] = ObjectAsset(
            spec=OBJECTS[ycb],
            mass=float(entry["mass_kg"]),
            collision_method=str(entry["collision_method"]),
            pieces=pieces,
            inertial_pos=_vec3(entry["inertial_pos"]),
            fullinertia=_vec6(entry["fullinertia"]),
            vertices=vertices,
        )
    return assets


# ---------------------------------------------------------------------------
# Scene files


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _mjcf_quat(pose: Pose) -> str:
    x, y, z, w = pose.orientation.to_tuple()
    return _fmt((w, x, y, z))


def scene_xml(
    scene_id: str,
    objects: Sequence[tuple[ObjectAsset, Pose]],
    spawn_height: float,
    table: Table = TABLE,
    assets_dir: str = f"../{ASSETS_DIR}",
) -> str:
    """Build the world-only MJCF: room, table, pedestal and the placed objects."""
    asset_lines = []
    body_lines = []
    for asset, pose in objects:
        e = asset.spec.entity
        asset_lines.append(
            f'    <texture type="2d" name="{e}_texture" file="{asset.spec.ycb}/texture.png"/>\n'
            f'    <material name="{e}_material" texture="{e}_texture" specular="0.2" shininess="0.1"/>\n'
            f'    <mesh name="{e}_visual" file="{asset.spec.ycb}/visual.obj"/>'
        )
        for index, piece in enumerate(asset.pieces):
            asset_lines.append(
                f'    <mesh name="{e}_collision_{index:02d}" file="{asset.spec.ycb}/{piece}"/>'
            )
        geoms = [f'      <geom class="object_visual" mesh="{e}_visual" material="{e}_material"/>']
        geoms += [
            f'      <geom class="object_collision" mesh="{e}_collision_{index:02d}"/>'
            for index in range(len(asset.pieces))
        ]
        body_lines.append(
            f'    <body name="{e}" pos="{_fmt(pose.position.to_tuple())}" quat="{_mjcf_quat(pose)}">\n'
            f'      <freejoint name="{e}"/>\n'
            f'      <inertial pos="{_fmt(asset.inertial_pos)}" mass="{asset.mass:.6g}"'
            f' fullinertia="{_fmt(asset.fullinertia)}"/>\n' + "\n".join(geoms) + "\n    </body>"
        )

    half = table.size / 2
    slab_z = table.top_z - table.thickness / 2
    column_top = table.top_z - table.thickness
    column_half = (column_top - 0.03) / 2
    assets_xml = "\n".join(asset_lines)
    bodies_xml = "\n".join(body_lines)
    return f"""<mujoco model="{scene_id}">
  <!-- World-only SceneReplica scene for sim2. The xArm7 is attached by load_scene at
       the "workbench" spawn in scene.json; there is no robot here.
       Generated by dimos/evals/scenes/scenereplica/convert.py. -->
  <compiler angle="radian" meshdir="{assets_dir}" texturedir="{assets_dir}"/>
  <!-- Elliptic friction cones with impratio 10 keep a gripped object from sliding
       out of the fingers under its own weight, as in the sim2 workbench scene. -->
  <option cone="elliptic" impratio="10"/>
  <statistic center="0.6 0 0.8" extent="1.4"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.26 0.27 0.29 1"/>
    <global azimuth="150" elevation="-20"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.33 0.31 0.29" rgb2="0.28 0.26 0.25" markrgb="0.38 0.36 0.34"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4"
      reflectance="0.05"/>
    <material name="wall" rgba="0.62 0.60 0.56 1"/>
    <material name="wall_back" rgba="0.42 0.46 0.50 1"/>
    <material name="ceiling" rgba="0.80 0.80 0.78 1"/>
    <material name="table_top" rgba="0.55 0.45 0.35 1"/>
    <material name="table_leg" rgba="0.4 0.35 0.3 1"/>
    <material name="pedestal" rgba="0.3 0.3 0.32 1"/>
{assets_xml}
  </asset>

  <default>
    <default class="object_visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="object_collision">
      <!-- xarm_table contact parameters, plus rolling friction (condim 6) so
           cylinders like the marker stop instead of creeping on their hull facets. -->
      <geom type="mesh" group="{COLLISION_GROUP}" rgba="0.8 0.3 0.3 0.4" condim="6"
        friction="1.0 0.02 0.001" solimp="0.95 0.99 0.001" solref="0.004 1"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 2.4" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="1.2 0.8 2.0" dir="-0.5 -0.4 -1" diffuse="0.35 0.35 0.35"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Enclosed room so cameras see walls past the table, not sky. -->
    <geom name="wall_back" type="box" size="0.05 2.0 1.4" pos="2.2 0 1.4" material="wall_back"/>
    <geom name="wall_front" type="box" size="0.05 2.0 1.4" pos="-1.2 0 1.4" material="wall"/>
    <geom name="wall_left" type="box" size="1.75 0.05 1.4" pos="0.5 2.0 1.4" material="wall"/>
    <geom name="wall_right" type="box" size="1.75 0.05 1.4" pos="0.5 -2.0 1.4" material="wall"/>
    <geom name="ceiling" type="box" size="1.75 2.0 0.05" pos="0.5 0 2.8" material="ceiling"/>

    <!-- Pedestal the arm stands on; its top is the workbench spawn height. -->
    <geom name="pedestal" type="cylinder" size="0.1 {spawn_height / 2:.6g}"
      pos="0 0 {spawn_height / 2:.6g}" material="pedestal"/>

    <!-- Cafe table: top at z={table.top_z:.6g}, {table.size:.6g} m square, centred {table.centre_x:.6g} m ahead. -->
    <geom name="table_top" type="box" size="{half:.6g} {half:.6g} {table.thickness / 2:.6g}"
      pos="{table.centre_x:.6g} 0 {slab_z:.6g}" material="table_top"/>
    <geom name="table_column" type="box" size="0.021 0.021 {column_half:.6g}"
      pos="{table.centre_x:.6g} 0 {0.03 + column_half:.6g}" material="table_leg"/>
    <geom name="table_base" type="box" size="0.28 0.28 0.015" pos="{table.centre_x:.6g} 0 0.015"
      material="table_leg"/>

{bodies_xml}
  </worldbody>
</mujoco>
"""


def scene_description(
    index: int,
    scene_id: int,
    objects: Sequence[tuple[ObjectAsset, Pose]],
    spawn_height: float,
    rest_shift: dict[str, float],
    table: Table = TABLE,
    settle_shift: dict[str, float] | None = None,
) -> SceneDescription:
    """Build the ``scene.json`` sidecar for scene number ``index`` (1-based).

    Args:
        rest_shift: Per entity, how far (m) the source pose was raised or lowered to
            sit on the table top.
        settle_shift: Per entity, how far (m) it then moved while the converter let
            physics settle the scene; None when that step was skipped.
    """
    entities = {
        asset.spec.entity: SceneEntity(
            body=asset.spec.entity, label=asset.spec.label, kind=asset.spec.kind, movable=True
        )
        for asset, _ in objects
    }
    regions = {
        "table/top": SceneRegion(
            kind="support",
            pose=Pose(table.centre_x, 0.0, table.top_z),
            size=(table.size, table.size, 0.0),
        ),
        "dropoff/top": SceneRegion(
            kind="support",
            pose=Pose(DROPOFF_XY[0], DROPOFF_XY[1], table.top_z),
            size=(DROPOFF_SIZE, DROPOFF_SIZE, 0.0),
        ),
    }
    return SceneDescription(
        id=f"{PACKAGE}-{index:02d}",
        entities=entities,
        regions=regions,
        initial=SceneUpdate(poses={asset.spec.entity: pose for asset, pose in objects}),
        spawns={"workbench": Pose(0.0, 0.0, spawn_height)},
        hidden_geom_groups=(COLLISION_GROUP,),
        provenance={
            "source": "SceneReplica",
            "source_url": SOURCE_URL,
            "source_scene_id": scene_id,
            "source_metadata": f"final_scenes/metadata/meta-{scene_id:06d}.mat",
            "license": "SceneReplica code MIT; YCB meshes under the YCB benchmark terms",
            "frame": "world = Fetch base_link (floor z=0, x forward); quaternions wxyz -> xyzw",
            "table": {"centre_x": table.centre_x, "top_z": table.top_z, "size": table.size},
            "source_dropoff_xy": [0.78, 0.40],
            "rest_shift_m": rest_shift,
            "settle_shift_m": settle_shift,
            "generator": "dimos/evals/scenes/scenereplica/convert.py",
        },
    )


def write_scene(
    out: Path,
    index: int,
    placements: Sequence[Placement],
    assets: dict[str, ObjectAsset],
    spawn_height: float,
    settle_seconds: float = 2.0,
) -> Path:
    """Write ``<out>/scene-NN/scene.xml`` and ``scene.json`` for one SceneReplica scene.

    Each object is first dropped onto the table top, then the scene is stepped for
    ``settle_seconds`` of physics and the settled poses are what the files keep, so
    a run starts with everything at rest. 0 skips the settling step.
    """
    scene_dir = out / f"scene-{index:02d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    objects: list[tuple[ObjectAsset, Pose]] = []
    rest_shift: dict[str, float] = {}
    for placement in placements:
        asset = assets[placement.ycb]
        pose, shift = rest_on_table(asset.vertices, base_link_to_world(placement), TABLE.top_z)
        objects.append((asset, pose))
        rest_shift[asset.spec.entity] = round(shift, 4)
    scene_id = SCENE_IDS[index - 1]
    description = scene_description(index, scene_id, objects, spawn_height, rest_shift)
    _write_scene_files(scene_dir, description, objects, spawn_height)
    if settle_seconds > 0:
        settled = settle_poses(scene_dir, settle_seconds)
        settle_shift = {
            asset.spec.entity: round(
                float(
                    np.linalg.norm(
                        np.subtract(
                            settled[asset.spec.entity].position.to_tuple(),
                            pose.position.to_tuple(),
                        )
                    )
                ),
                4,
            )
            for asset, pose in objects
        }
        objects = [(asset, settled[asset.spec.entity]) for asset, _ in objects]
        description = scene_description(
            index, scene_id, objects, spawn_height, rest_shift, settle_shift=settle_shift
        )
        _write_scene_files(scene_dir, description, objects, spawn_height)
    return scene_dir


def _write_scene_files(
    scene_dir: Path,
    description: SceneDescription,
    objects: Sequence[tuple[ObjectAsset, Pose]],
    spawn_height: float,
) -> None:
    (scene_dir / "scene.xml").write_text(scene_xml(description.id, objects, spawn_height))
    (scene_dir / "scene.json").write_text(description.model_dump_json(indent=2) + "\n")


def settle_poses(scene_dir: Path, seconds: float) -> dict[str, Pose]:
    """Step the world-only scene and return where each entity came to rest."""
    import mujoco

    from dimos.sim2.scene import describe_scene

    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    data = mujoco.MjData(model)
    description = describe_scene(scene_dir / "scene.xml")
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    poses = {}
    for key, entity in description.entities.items():
        body = model.body(entity.body).id
        w, x, y, z = (float(v) for v in data.xquat[body])
        poses[key] = Pose(_vec3(data.xpos[body]), (x, y, z, w))
    return poses


def write_scenes(
    out: Path, scenes_dir: Path, assets: dict[str, ObjectAsset], spawn_height: float
) -> list[Path]:
    written = []
    for index, scene_id in enumerate(SCENE_IDS, start=1):
        placements = read_scene_metadata(scenes_dir / "metadata" / f"meta-{scene_id:06d}.mat")
        written.append(write_scene(out, index, placements, assets, spawn_height))
    return written


# ---------------------------------------------------------------------------
# Checks


def scene_dirs(out: Path) -> list[Path]:
    return sorted(p for p in out.glob("scene-*") if (p / "scene.xml").is_file())


def settle_check(scene_dir: Path, seconds: float = 2.0) -> dict[str, float]:
    """Step the world-only scene and report how far each object drifted, in metres."""
    import mujoco

    from dimos.sim2.scene import describe_scene

    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    data = mujoco.MjData(model)
    description = describe_scene(scene_dir / "scene.xml")
    bodies = {key: model.body(entity.body).id for key, entity in description.entities.items()}
    mujoco.mj_forward(model, data)
    start = {key: data.xpos[body].copy() for key, body in bodies.items()}
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return {
        key: float(np.linalg.norm(data.xpos[body] - start[key])) for key, body in bodies.items()
    }


@dataclass(frozen=True)
class Reach:
    """Whether the xArm7 can hover straight down over one object."""

    target: tuple[float, float, float]
    ik_converged: bool
    collision_free: bool
    position_error_m: float
    contacts: tuple[str, ...]

    @property
    def reachable(self) -> bool:
        return self.ik_converged and self.collision_free


class ArmInScene:
    """The xArm7 attached to one scene, with a simple top-down IK for reach checks."""

    def __init__(self, scene_dir: Path, spawn_height: float) -> None:
        import mujoco

        from dimos.robot.manipulators.xarm.sim2 import XARM7
        from dimos.sim2.scene import describe_scene, load_scene
        from dimos.sim2.spec import RobotInstance, WorldConfig

        self.mujoco = mujoco
        self.model = load_scene(
            WorldConfig(
                scene=scene_dir / "scene.xml",
                robots={"arm": RobotInstance(XARM7, xyz=(0.0, 0.0, spawn_height))},
            )
        )
        self.data = mujoco.MjData(self.model)
        self.mocap = self.model.body_mocapid[self.model.body("arm/link_base").id]
        joints = [self.model.joint(f"arm/joint{i}") for i in range(1, 8)]
        self.qpos = np.array([j.qposadr[0] for j in joints])
        self.dof = np.array([j.dofadr[0] for j in joints])
        self.lower = np.array([j.range[0] for j in joints])
        self.upper = np.array([j.range[1] for j in joints])
        self.home = np.array([j.home for j in XARM7.joints[:7]])
        self.site = self.model.site("arm/link_tcp").id
        self.arm_geoms = {
            g
            for g in range(self.model.ngeom)
            if self.model.body(self.model.geom_bodyid[g]).name.startswith("arm/")
        }
        self.base_geoms = {
            g
            for g in self.arm_geoms
            if self.model.body(self.model.geom_bodyid[g]).name == "arm/link_base"
        }
        self.entities = {
            key: self.model.body(e.body).id
            for key, e in describe_scene(scene_dir / "scene.xml").entities.items()
        }

    def set_spawn_height(self, height: float) -> None:
        self.data.mocap_pos[self.mocap] = (0.0, 0.0, height)

    def object_top(self, key: str) -> tuple[float, float, float]:
        """Centre x, y and the highest collision vertex z of an entity, world metres."""
        mujoco = self.mujoco
        mujoco.mj_forward(self.model, self.data)
        body = self.entities[key]
        top = -math.inf
        for g in range(self.model.ngeom):
            if (
                self.model.geom_bodyid[g] != body
                or self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            ):
                continue
            mesh = self.model.geom_dataid[g]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            verts = self.model.mesh_vert[start : start + count]
            world = verts @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]
            top = max(top, float(world[:, 2].max()))
        centre = self.data.xpos[body]
        return (float(centre[0]), float(centre[1]), top)

    def solve(
        self, target: NDArray[np.float64], seed: NDArray[np.float64], iterations: int = 300
    ) -> tuple[bool, float]:
        """Damped least squares to put link_tcp at ``target`` pointing straight down."""
        mujoco = self.mujoco
        down = np.array([0.0, 0.0, -1.0])
        q = np.clip(seed, self.lower, self.upper)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        error = math.inf
        for _ in range(iterations):
            self.data.qpos[self.qpos] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            position = self.data.site_xpos[self.site]
            axis = self.data.site_xmat[self.site].reshape(3, 3)[:, 2]
            e_pos = target - position
            e_rot = np.cross(axis, down)
            error = float(np.linalg.norm(e_pos))
            if error < 1e-3 and float(np.dot(axis, down)) > math.cos(math.radians(1.0)):
                return True, error
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jac = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])
            residual = np.concatenate([e_pos, e_rot])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.01 * np.eye(6), residual)
            scale = min(1.0, 0.2 / max(float(np.abs(dq).max()), 1e-9))
            q = np.clip(q + scale * dq, self.lower, self.upper)
        return False, error

    def contacts(self) -> tuple[str, ...]:
        """Names of things the arm (not its base plate) is pushed into at the current pose."""
        mujoco = self.mujoco
        mujoco.mj_forward(self.model, self.data)
        found = set()
        for c in self.data.contact[: self.data.ncon]:
            if c.dist >= -1e-4:
                continue
            pair = (int(c.geom1), int(c.geom2))
            if not (pair[0] in self.arm_geoms or pair[1] in self.arm_geoms):
                continue
            if pair[0] in self.base_geoms or pair[1] in self.base_geoms:
                continue
            names = tuple(self.model.body(self.model.geom_bodyid[g]).name for g in pair)
            found.add(" <-> ".join(sorted(names)))
        return tuple(sorted(found))

    def reach(self, key: str, hover: float = 0.10, seeds: int = 6) -> Reach:
        x, y, top = self.object_top(key)
        target = np.array([x, y, top + hover])
        rng = np.random.default_rng(0)
        first = self.home.copy()
        first[0] = math.atan2(y, x)
        candidates = [first, self.home] + [
            rng.uniform(self.lower, self.upper) for _ in range(seeds - 2)
        ]
        best = math.inf
        for seed in candidates:
            converged, error = self.solve(target, seed)
            best = min(best, error)
            if converged:
                contacts = self.contacts()
                return Reach(tuple(target.tolist()), True, not contacts, error, contacts)
        return Reach(tuple(target.tolist()), False, False, best, ())


def reach_check(out: Path, spawn_height: float, hover: float = 0.10) -> dict[str, dict[str, Reach]]:
    results: dict[str, dict[str, Reach]] = {}
    for scene_dir in scene_dirs(out):
        arm = ArmInScene(scene_dir, spawn_height)
        results[scene_dir.name] = {key: arm.reach(key, hover) for key in arm.entities}
    return results


def reach_sweep(out: Path, heights: Sequence[float], hover: float = 0.10) -> dict[float, int]:
    """Count reachable hover poses per candidate spawn height, loading each scene once."""
    counts = dict.fromkeys(heights, 0)
    for scene_dir in scene_dirs(out):
        arm = ArmInScene(scene_dir, heights[0])
        for height in heights:
            arm.set_spawn_height(height)
            counts[height] += sum(arm.reach(key, hover).reachable for key in arm.entities)
    return counts


def reach_report(
    results: dict[str, dict[str, Reach]], spawn_height: float, hover: float, sweep: dict[float, int]
) -> dict[str, object]:
    total = sum(len(v) for v in results.values())
    reachable = sum(r.reachable for v in results.values() for r in v.values())
    return {
        "spawn_height_m": spawn_height,
        "hover_above_object_top_m": hover,
        "reachable": reachable,
        "total": total,
        "scenes_with_unreachable": sorted(
            name
            for name, objects in results.items()
            if not all(r.reachable for r in objects.values())
        ),
        "sweep": {f"{h:.3f}": n for h, n in sweep.items()},
        "scenes": {
            name: {
                key: {
                    "reachable": r.reachable,
                    "ik_converged": r.ik_converged,
                    "collision_free": r.collision_free,
                    "target": [round(v, 4) for v in r.target],
                    "position_error_m": round(r.position_error_m, 4),
                    "contacts": list(r.contacts),
                }
                for key, r in objects.items()
            }
            for name, objects in results.items()
        },
    }


# ---------------------------------------------------------------------------
# Command line


def _default_out() -> Path:
    from dimos.utils.data import get_data_dir

    return get_data_dir() / PACKAGE


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=None, help=f"scene package folder (default data/{PACKAGE})"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="download, convert objects, write the 20 scenes")
    convert.add_argument(
        "--scenes", type=Path, default=None, help="final_scenes.zip or its extracted folder"
    )
    convert.add_argument(
        "--models", type=Path, default=None, help="models.zip or its extracted folder"
    )
    convert.add_argument("--work", type=Path, default=Path.home() / ".cache" / "dimos" / PACKAGE)
    convert.add_argument("--texture-size", type=int, default=1024)
    convert.add_argument("--spawn-height", type=float, default=SPAWN_HEIGHT)
    convert.add_argument(
        "--reuse-objects", action="store_true", help="keep an existing _assets folder"
    )

    settle = commands.add_parser("settle", help="step each scene 2 s and report object drift")
    settle.add_argument("--seconds", type=float, default=2.0)
    settle.add_argument("--limit", type=float, default=0.01, help="allowed drift in metres")

    reach = commands.add_parser(
        "reach", help="IK a top-down hover over every object; write reach_report.json"
    )
    reach.add_argument("--spawn-height", type=float, default=SPAWN_HEIGHT)
    reach.add_argument("--hover", type=float, default=0.10)
    reach.add_argument(
        "--sweep", type=float, nargs="*", default=(), help="extra spawn heights to count"
    )

    args = parser.parse_args(argv)
    out = args.out or _default_out()

    if args.command == "convert":
        args.work.mkdir(parents=True, exist_ok=True)
        scenes = unpack_scenes(
            args.scenes or download(FINAL_SCENES_URL, args.work / "final_scenes.zip"), args.work
        )
        assets_dir = out / ASSETS_DIR
        if args.reuse_objects and (assets_dir / "manifest.json").is_file():
            assets = load_assets(assets_dir)
        else:
            models = unpack_models(
                args.models or download(MODELS_URL, args.work / "models.zip"), args.work
            )
            assets = convert_objects(models, assets_dir, args.texture_size)
        for scene_dir in write_scenes(out, scenes, assets, args.spawn_height):
            logger.info("wrote %s", scene_dir)
        return 0

    if args.command == "settle":
        worst = 0.0
        for scene_dir in scene_dirs(out):
            drift = settle_check(scene_dir, args.seconds)
            worst = max(worst, *drift.values())
            moved = {k: round(v, 4) for k, v in drift.items() if v >= args.limit}
            logger.info(
                "%s: max drift %.4f m%s",
                scene_dir.name,
                max(drift.values()),
                f" MOVED {moved}" if moved else "",
            )
        logger.info("worst drift %.4f m (limit %.3f)", worst, args.limit)
        return 0 if worst < args.limit else 1

    heights = [args.spawn_height, *[h for h in args.sweep if h != args.spawn_height]]
    sweep = reach_sweep(out, heights, args.hover) if len(heights) > 1 else {}
    results = reach_check(out, args.spawn_height, args.hover)
    report = reach_report(results, args.spawn_height, args.hover, sweep)
    (out / "reach_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for height, count in sorted(sweep.items()):
        logger.info("spawn height %.3f m: %d reachable", height, count)
    logger.info(
        "spawn height %.3f m: %d of %d hover poses reachable; scenes with unreachable: %s",
        args.spawn_height,
        report["reachable"],
        report["total"],
        ", ".join(report["scenes_with_unreachable"]) or "none",  # type: ignore[arg-type]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
