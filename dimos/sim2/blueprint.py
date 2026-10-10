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

"""Compose only the device modules declared by each robot configuration."""

from __future__ import annotations

from pathlib import Path

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.sim2.connections.manipulator import ManipulatorConnection
from dimos.sim2.connections.whole_body import WholeBodyConnection
from dimos.sim2.module import SimulationModule
from dimos.sim2.sensors.camera.module import SimCameraModule, SimRGBDCameraModule
from dimos.sim2.sensors.lidar.module import LidarModule
from dimos.sim2.sensors.spec import Camera, Imu, Lidar
from dimos.sim2.spec import ControlInterface, RobotInstance, WorldConfig


def simulation(
    *,
    scene: Path,
    robots: dict[str, RobotInstance],
    sim_id: str = "sim",
    viewer: bool = True,
    timestep: float = 0.005,
) -> Blueprint:
    """Build the world and devices. Controllers discover their connection contracts."""
    modules = [
        SimulationModule.blueprint(
            world=WorldConfig(scene=scene, robots=robots, timestep=timestep),
            sim_id=sim_id,
            viewer=viewer,
        )
    ]
    multiple = len(robots) > 1
    ports: tuple[str, ...]
    for robot_id, instance in robots.items():
        config = instance.config
        if config.control == ControlInterface.WHOLE_BODY:
            imu = next(s for s in config.sensors if isinstance(s, Imu))
            connection = WholeBodyConnection.blueprint(
                instance_name=f"{robot_id}_connection",
                definition=config,
                address=f"{sim_id}/{robot_id}",
                robot_id=robot_id,
                rate_hz=imu.rate_hz,
            )
            ports = ("motor_command", "joint_state", "imu")
            connection = connection.remappings(
                [
                    (
                        f"{robot_id}_connection",
                        "root_pose",
                        f"{robot_id}/odom" if multiple else "odom",
                    )
                ]
            )
        else:
            connection = ManipulatorConnection.blueprint(
                instance_name=f"{robot_id}_connection",
                definition=config,
                address=f"{sim_id}/{robot_id}",
                robot_id=robot_id,
            )
            ports = ("position_command", "joint_state")
        connection = connection.remappings(
            [(f"{robot_id}_connection", port, f"{robot_id}_{port}") for port in ports]
        )
        modules.append(connection)
        for sensor in config.sensors:
            kwargs = dict(
                instance_name=f"{robot_id}_{sensor.name}",
                robot_id=robot_id,
                sensor=sensor,
                rate_hz=sensor.rate_hz,
            )
            if isinstance(sensor, Camera):
                module = SimRGBDCameraModule if sensor.depth else SimCameraModule
                blueprint = module.blueprint(**kwargs)
                ports = ("color_image", "camera_info")
                if sensor.depth:
                    ports += ("depth_image", "depth_camera_info")
            elif isinstance(sensor, Lidar):
                if sensor.imu is not None:
                    kwargs["rate_hz"] = max(sensor.rate_hz, sensor.imu.rate_hz)
                blueprint = LidarModule.blueprint(**kwargs, root_body=config.root_body)
                ports = ("pointcloud", "raw_pointcloud", "imu_raw")
            elif isinstance(sensor, Imu):
                # IMU is sampled with the joint observation in the physics owner.
                continue
            else:
                raise TypeError(f"unsupported sensor configuration: {type(sensor).__name__}")
            if multiple or sum(type(s) is type(sensor) for s in config.sensors) > 1:
                blueprint = blueprint.remappings(
                    [
                        (f"{robot_id}_{sensor.name}", port, f"{robot_id}/{sensor.name}/{port}")
                        for port in ports
                    ]
                )
            modules.append(blueprint)
    return autoconnect(*modules)
