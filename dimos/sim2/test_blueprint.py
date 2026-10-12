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

from dataclasses import dataclass, replace
import pickle

import numpy as np
import pytest

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.robot.manipulators.xarm.sim2 import XARM7
from dimos.robot.unitree.g1.sim2 import G1_GROOT, G1_GROOT_MID360
from dimos.sim2.blueprint import simulation
from dimos.sim2.connections.manipulator import SimManipulatorConnectionConfig
from dimos.sim2.connections.whole_body import SimMotorConnectionConfig
from dimos.sim2.module import SimulationModuleConfig
from dimos.sim2.scene import scene_path
from dimos.sim2.sensors.camera.module import CameraModuleConfig
from dimos.sim2.sensors.lidar.models.mid360 import Mid360
from dimos.sim2.sensors.lidar.module import LidarModuleConfig
from dimos.sim2.sensors.spec import Camera, Lidar, Mount
from dimos.sim2.spec import RobotInstance


def test_multiple_robots_and_rgb_cameras_have_separate_typed_ports():
    rgb = replace(XARM7, sensors=(Camera("front", Mount("link7"), depth=False),))
    devices = simulation(
        scene=scene_path(None, "workbench.xml"),
        robots={"left": RobotInstance(rgb), "right": RobotInstance(rgb)},
        viewer=False,
    )
    blueprint = devices
    parsed = BlueprintConfigParser(blueprint).parse(environ={})
    left = next(atom for atom in blueprint.active_blueprints if atom.name == "left_front")
    assert {stream.name for stream in left.streams} == {"color_image", "camera_info", "tf"}
    assert blueprint.remapping_map[("left_front", "color_image")] == "left/front/color_image"
    assert blueprint.remapping_map[("right_front", "color_image")] == "right/front/color_image"
    assert (
        blueprint.remapping_map[("left_connection", "position_command")] == "left_position_command"
    )
    assert (
        blueprint.remapping_map[("right_connection", "position_command")]
        == "right_position_command"
    )
    sensor = CameraModuleConfig(**parsed.module_kwargs("left_front")).sensor
    assert sensor == rgb.sensors[0]
    assert parsed.module_kwargs("left_connection")["address"] == "sim/left"
    assert parsed.module_kwargs("right_connection")["address"] == "sim/right"


def test_second_camera_does_not_require_shared_module_changes():
    robot = XARM7.with_sensor(Camera("front", Mount("link_base"), depth=False))
    blueprint = simulation(
        scene=scene_path(None, "workbench.xml"),
        robots={"arm": RobotInstance(robot)},
        viewer=False,
    )
    assert blueprint.remapping_map[("arm_front", "color_image")] == "arm/front/color_image"
    assert (
        blueprint.remapping_map[("arm_wrist_camera", "color_image")]
        == "arm/wrist_camera/color_image"
    )


def test_duplicate_sensor_names_fail_before_deployment():
    with pytest.raises(ValueError, match="sensor names must be nonempty and unique"):
        replace(XARM7, sensors=(*XARM7.sensors, *XARM7.sensors))


@dataclass(frozen=True)
class HorizontalRays:
    min_range: float = 0.1
    max_range: float = 12.0

    def directions(self):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    "robot,connection_config",
    [
        (G1_GROOT, SimMotorConnectionConfig),
        (G1_GROOT_MID360, SimMotorConnectionConfig),
        (XARM7, SimManipulatorConnectionConfig),
    ],
)
def test_physics_and_connection_reconstruct_the_same_robot(tmp_path, robot, connection_config):
    blueprint = simulation(
        scene=tmp_path / "scene.xml", robots={"robot": RobotInstance(robot)}, viewer=False
    )
    parsed = BlueprintConfigParser(blueprint).parse(
        environ={},
        overrides={"simulationmodule": {"world": {"robots": {"robot": {"xyz": [1, 2, 3]}}}}},
    )

    world_kwargs = pickle.loads(pickle.dumps(parsed.module_kwargs("simulationmodule")))
    world = SimulationModuleConfig(**world_kwargs).world
    device = connection_config(
        **pickle.loads(pickle.dumps(parsed.module_kwargs("robot_connection")))
    )

    assert world.robots["robot"].xyz == (1, 2, 3)
    assert world.robots["robot"].config == robot
    assert device.definition == robot
    assert [type(sensor) for sensor in device.definition.sensors] == [
        type(sensor) for sensor in robot.sensors
    ]


def test_custom_lidar_factory_and_overrides_survive_worker_serialization(tmp_path):
    robot = G1_GROOT.with_sensor(
        Lidar("lidar", "mid360_link", HorizontalRays, model_kwargs={"max_range": 12.0})
    )
    blueprint = simulation(
        scene=tmp_path / "scene.xml", robots={"robot": RobotInstance(robot)}, viewer=False
    )
    parsed = BlueprintConfigParser(blueprint).parse(
        environ={},
        overrides={"robot_lidar": {"sensor": {"model_kwargs": {"max_range": 20.0}}}},
    )

    kwargs = pickle.loads(pickle.dumps(parsed.module_kwargs("robot_lidar")))
    sensor = LidarModuleConfig(**kwargs).sensor
    pattern = sensor.model(**sensor.model_kwargs)

    assert type(pattern) is HorizontalRays
    assert pattern.max_range == 20.0
    np.testing.assert_array_equal(pattern.directions(), [[1, 0, 0], [0, 1, 0]])


def test_invalid_lidar_cannot_be_reconstructed_as_an_imu(tmp_path):
    blueprint = simulation(
        scene=tmp_path / "scene.xml", robots={"robot": RobotInstance(G1_GROOT)}, viewer=False
    )
    parsed = BlueprintConfigParser(blueprint).parse(environ={})
    kwargs = parsed.module_kwargs("simulationmodule")
    lidar = kwargs["world"]["robots"]["robot"]["config"]["sensors"][1]
    lidar["model"] = "unknown-model"

    with pytest.raises(ValueError, match="callable"):
        SimulationModuleConfig(**kwargs)


def test_timed_lidar_configuration_survives_worker_serialization(tmp_path):
    robot = G1_GROOT.with_sensor(
        Lidar("lidar", "mid360_link", Mid360, model_kwargs={"downsample": 8})
    )
    blueprint = simulation(
        scene=tmp_path / "scene.xml", robots={"g1": RobotInstance(robot)}, viewer=False
    )
    parsed = BlueprintConfigParser(blueprint).parse(environ={})
    world = SimulationModuleConfig(
        **pickle.loads(pickle.dumps(parsed.module_kwargs("simulationmodule")))
    ).world
    sensor = LidarModuleConfig(
        **pickle.loads(pickle.dumps(parsed.module_kwargs("g1_lidar")))
    ).sensor
    configured = next(s for s in world.robots["g1"].config.sensors if isinstance(s, Lidar))
    assert configured == sensor
    assert sensor.model(**sensor.model_kwargs) == Mid360(downsample=8)
