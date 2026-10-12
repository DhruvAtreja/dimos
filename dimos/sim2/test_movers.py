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

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from dimos.robot.unitree.go2.sim2 import GO2_FREEWALK
from dimos.sim2.movers import MoverSpec, _Track
from dimos.sim2.runtime import SimulationRuntime
from dimos.sim2.sensors.lidar.raycast import Raycaster
from dimos.sim2.spec import RobotInstance, WorldConfig

pytestmark = pytest.mark.mujoco

SCENE = """<mujoco model="movers">
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="mover" mocap="true" pos="2 -1 0.25">
      <geom name="mover_box" type="box" size="0.25 0.25 0.25" rgba="1 0.5 0 1"/>
    </body>
    <body name="crate" pos="2 0 0.15">
      <freejoint/>
      <geom name="crate_box" type="box" size="0.15 0.15 0.15" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_scene(root: Path, movers: list[dict]) -> Path:
    (root / "scene.xml").write_text(SCENE)
    (root / "scene.json").write_text(
        json.dumps(
            {
                "format": "dimos.scene.v1",
                "id": "movers",
                "entities": {
                    "mover": {"body": "mover", "label": "mover", "kind": "mover"},
                    "crate": {"body": "crate", "label": "crate", "kind": "box", "movable": True},
                },
                "movers": movers,
            }
        )
    )
    return root / "scene.xml"


@pytest.fixture
def world(tmp_path):
    worlds = []

    def create(movers: list[dict]) -> SimulationRuntime:
        scene = write_scene(tmp_path, movers)
        runtime = SimulationRuntime(
            WorldConfig(scene, {"go2": RobotInstance(GO2_FREEWALK, xyz=(-3, 0, 0.33))}),
            uuid4().hex,
        )
        worlds.append(runtime)
        return runtime

    yield create
    for runtime in worlds:
        runtime.close()


def test_loop_track_moves_at_constant_speed_and_wraps():
    spec = MoverSpec(body="m", waypoints=((0, 0, 0), (2, 0, 0), (2, 2, 0)), speed=1.0)
    track = _Track(spec, 0, np.array([1.0, 0, 0, 0]))
    assert np.allclose(track.pose(1.0)[0], (1, 0, 0))
    assert np.allclose(track.pose(3.0)[0], (2, 1, 0))
    # 2 + 2 + sqrt(8) per lap: back at the start after one lap
    assert np.allclose(track.pose(4 + np.sqrt(8))[0], (0, 0, 0))
    # facing the direction of travel: +y on the second leg is yaw 90 degrees
    assert np.allclose(track.pose(3.0)[1], (np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)))


def test_pingpong_track_retraces_its_path():
    spec = MoverSpec(
        body="m", waypoints=((0, 0, 0), (1, 0, 0)), speed=0.5, mode="pingpong", heading="fixed"
    )
    track = _Track(spec, 0, np.array([1.0, 0, 0, 0]))
    assert np.allclose(track.pose(1.0)[0], (0.5, 0, 0))
    assert np.allclose(track.pose(3.0)[0], (0.5, 0, 0))
    assert np.allclose(track.pose(4.0)[0], (0, 0, 0))
    assert np.allclose(track.pose(3.0)[1], (1, 0, 0, 0))


def test_mover_follows_path_and_reset_rewinds_it(world):
    runtime = world([{"body": "mover", "waypoints": [[2, -1, 0.25], [2, 1, 0.25]], "speed": 0.5}])
    for _ in range(round(1.0 / runtime.config.timestep)):
        runtime.step()
    pose = runtime.scene_state().entities["mover"].pose.position
    assert abs(pose.y - (-0.5)) < 0.02 and abs(pose.x - 2) < 1e-6
    runtime.reset()
    assert abs(runtime.scene_state().entities["mover"].pose.position.y - (-1)) < 1e-6


def test_mover_pushes_a_free_body_and_reports_the_contact(world):
    runtime = world([{"body": "mover", "waypoints": [[2, -1, 0.25], [2, 1.5, 0.25]], "speed": 0.5}])
    contacts = set()
    for _ in range(round(3.0 / runtime.config.timestep)):
        runtime.step()
        contacts |= set(runtime.scene_state().contacts) if runtime.tick % 20 == 0 else set()
    crate = runtime.scene_state().entities["crate"].pose.position
    assert crate.y > 0.3, "the mover should shove the crate along +y"
    assert any({"mover", "crate"} == set(pair) for pair in contacts)


def test_lidar_rays_hit_the_mover(world):
    runtime = world([{"body": "mover", "waypoints": [[2, -1, 0.25], [2, 1, 0.25]], "speed": 0.5}])
    runtime.step()
    raycaster = Raycaster(runtime.model, runtime.model.body("go2/base").id)
    origin = np.array([0.0, -1.0, 0.25])
    points = raycaster.cast(runtime.data, origin, np.array([[1.0, 0, 0]]), 0.1, 10)
    assert len(points) == 1 and abs(points[0][0] - 1.75) < 0.02
