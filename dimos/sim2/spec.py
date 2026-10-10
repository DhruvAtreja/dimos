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

"""Robot definitions contain device bindings; instances contain world placement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Literal

from dimos.control.contract.convert import JOINT_STATE_FIELDS, MOTOR_INTERFACES
from dimos.control.contract.description import ControlDescription, Limits, Resource, ResourceKind
from dimos.control.contract.keys import Interface, Unit
from dimos.control.contract.presets import imu_resource
from dimos.sim2.sensors.spec import Imu, Lidar, Mount as Mount, Sensor


class ControlInterface(str, Enum):
    WHOLE_BODY = "whole_body"
    MANIPULATOR = "manipulator"


@dataclass(frozen=True)
class Joint:
    name: str
    model_name: str
    actuator: str
    home: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    # Native position servos retain their own gains. Motor actuators use PD once.
    mode: Literal["effort", "position"] = "effort"
    # Public joint coordinate -> MJCF joint coordinate, including gripper units.
    scale: float = 1.0
    offset: float = 0.0
    ctrl_scale: float = 1.0
    ctrl_offset: float = 0.0
    lower: float = -6.283185307179586
    upper: float = 6.283185307179586
    max_velocity: float = 3.141592653589793
    unit: Unit = Unit.RAD

    def __post_init__(self) -> None:
        if self.scale == 0 or not all(
            math.isfinite(v)
            for v in (
                self.scale,
                self.offset,
                self.ctrl_scale,
                self.ctrl_offset,
                self.home,
                self.kp,
                self.kd,
                self.lower,
                self.upper,
                self.max_velocity,
            )
        ):
            raise ValueError(f"joint {self.name!r} requires finite parameters and nonzero scale")
        if self.lower > self.upper or self.max_velocity < 0 or min(self.kp, self.kd) < 0:
            raise ValueError(f"joint {self.name!r} has invalid limits or gains")


@dataclass(frozen=True)
class RobotConfig:
    model: Path
    root_body: str
    control: ControlInterface
    joints: tuple[Joint, ...]
    sensors: tuple[Sensor, ...] = ()
    meshdir: Path | None = None
    floating: bool = False
    # Root height above a scene's support pose; explicit instances stay absolute.
    spawn_height: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.spawn_height) or self.spawn_height < 0:
            raise ValueError("robot spawn height must be finite and nonnegative")
        if not self.joints:
            raise ValueError("robot must declare its controlled joints")
        for label, names in (
            ("joint", [j.name for j in self.joints]),
            ("actuator", [j.actuator for j in self.joints]),
            ("sensor", [s.name for s in self.mounted_sensors]),
        ):
            if len(set(names)) != len(names) or any(not name for name in names):
                raise ValueError(f"robot {label} names must be nonempty and unique")
        imus = [s for s in self.sensors if isinstance(s, Imu)]
        if self.control == ControlInterface.WHOLE_BODY and len(imus) != 1:
            raise ValueError("whole-body control requires exactly one coherent control IMU")
        if self.control == ControlInterface.MANIPULATOR and imus:
            raise ValueError("standalone IMU streaming is not implemented in this migration")
        if any(not math.isfinite(s.rate_hz) or s.rate_hz <= 0 for s in self.mounted_sensors):
            raise ValueError("sensor rates must be finite and positive")

    @property
    def mounted_sensors(self) -> tuple[Sensor, ...]:
        """Physical mounts, including IMUs supplied as part of a lidar device."""
        return tuple(
            child
            for sensor in self.sensors
            for child in (
                (sensor, sensor.imu)
                if isinstance(sensor, Lidar) and sensor.imu is not None
                else (sensor,)
            )
        )

    def with_sensor(self, sensor: Sensor) -> RobotConfig:
        sensors = tuple(s for s in self.sensors if s.name != sensor.name)
        return replace(self, sensors=(*sensors, sensor))

    def describe_control(
        self, source: str, *, rate_hz: float, deadman_timeout_s: float
    ) -> ControlDescription:
        """Describe public coordinates, not MJCF actuator coordinates."""
        pd = self.control == ControlInterface.WHOLE_BODY
        command = MOTOR_INTERFACES if pd else (Interface.POSITION,)
        resources = []
        limits = {}
        for joint in self.joints:
            name = joint.name.rsplit("/", 1)[-1]
            if joint.unit == Unit.RAD:
                velocity, effort = Unit.RAD_PER_S, Unit.NM
            elif joint.unit == Unit.M:
                velocity, effort = Unit.M_PER_S, Unit.N
            elif not pd and joint.unit == Unit.UNITLESS:
                velocity = effort = Unit.UNITLESS
            else:
                raise ValueError(f"unsupported sim2 joint unit: {joint.unit}")
            state = (Interface.POSITION,) if joint.unit == Unit.UNITLESS else JOINT_STATE_FIELDS
            units = dict(zip(JOINT_STATE_FIELDS, (joint.unit, velocity, effort), strict=True))
            units.update({Interface.KP: Unit.UNITLESS, Interface.KD: Unit.UNITLESS})
            resources.append(
                Resource(
                    name=name,
                    kind=ResourceKind.JOINT,
                    state_interfaces=state,
                    command_interfaces=command,
                    units={key: units[key] for key in (*state, *command)},
                )
            )
            prefix = f"{source}/{name}"
            limits[f"{prefix}/position"] = Limits(joint.lower, joint.upper)
            if pd:
                limits[f"{prefix}/velocity"] = Limits(-joint.max_velocity, joint.max_velocity)
                limits[f"{prefix}/kp"] = Limits(lo=0.0)
                limits[f"{prefix}/kd"] = Limits(lo=0.0)
        resources.extend(imu_resource(s.name) for s in self.sensors if isinstance(s, Imu))
        return ControlDescription(
            source=source,
            resources=tuple(resources),
            limits=limits,
            state_rate_hz=rate_hz,
            deadman_timeout_s=deadman_timeout_s,
        )


@dataclass(frozen=True)
class RobotInstance:
    config: RobotConfig
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class WorldConfig:
    scene: Path
    robots: dict[str, RobotInstance]
    timestep: float = 0.005
    snapshot_hz: float = 60.0

    def __post_init__(self) -> None:
        if not self.robots or any(not name or "/" in name for name in self.robots):
            raise ValueError("world requires robots with nonempty, slash-free instance IDs")
        if any(not math.isfinite(v) or v <= 0 for v in (self.timestep, self.snapshot_hz)):
            raise ValueError("world timestep and snapshot rate must be finite and positive")
