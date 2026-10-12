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

"""Scripted obstacles: mocap bodies that loop along a waypoint path, set every physics step.

Declared in `movers.json` beside the scene's `scene.xml`. A mover is kinematic: it pushes the robot
and is never pushed back. Its pose is a pure function of sim time, so runs are deterministic and a
reset (which rewinds sim time) puts every mover back at its start.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveFloat

Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class MoverSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    body: str
    waypoints: tuple[Vec3, ...] = Field(min_length=2)
    speed: PositiveFloat
    # "loop" returns from the last waypoint to the first; "pingpong" retraces the path.
    mode: Literal["loop", "pingpong"] = "loop"
    # "path" faces the direction of travel (yaw only); "fixed" keeps the body's authored rotation.
    heading: Literal["path", "fixed"] = "path"
    phase: FiniteFloat = 0.0  # seconds of travel already done at sim time 0


class MoversFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["dimos.movers.v1"] = "dimos.movers.v1"
    movers: tuple[MoverSpec, ...] = ()


class _Track:
    def __init__(self, spec: MoverSpec, mocap_id: int, quat: NDArray[np.float64]) -> None:
        points = np.asarray(spec.waypoints, dtype=np.float64)
        if spec.mode == "loop":
            points = np.vstack([points, points[:1]])
        else:
            points = np.vstack([points, points[-2::-1]])
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if lengths.sum() <= 0:
            raise ValueError(f"mover {spec.body!r}: path has zero length")
        self.points = points
        self.segments = segments
        self.starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        self.lengths = lengths
        self.total = float(lengths.sum())
        self.spec = spec
        self.mocap_id = mocap_id
        self.quat = quat.copy()

    def pose(self, time: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        distance = ((time + self.spec.phase) * self.spec.speed) % self.total
        index = int(np.searchsorted(self.starts, distance, side="right") - 1)
        while self.lengths[index] == 0:
            index = (index + 1) % len(self.lengths)
        fraction = (distance - self.starts[index]) / self.lengths[index]
        position = self.points[index] + min(max(fraction, 0.0), 1.0) * self.segments[index]
        if self.spec.heading == "fixed":
            return position, self.quat
        dx, dy = self.segments[index][:2]
        yaw = math.atan2(dy, dx) if dx or dy else 0.0
        return position, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class Movers:
    """Every mover in one scene, applied in the physics thread before each step."""

    def __init__(self, tracks: list[_Track]) -> None:
        self.tracks = tracks

    @classmethod
    def load(cls, scene: Path, model: mujoco.MjModel) -> Movers:
        path = scene.with_name("movers.json")
        if not path.is_file():
            return cls([])
        spec = MoversFile.model_validate_json(path.read_text())
        tracks = []
        for mover in spec.movers:
            body = model.body(mover.body).id
            mocap_id = int(model.body_mocapid[body])
            if mocap_id < 0:
                raise ValueError(f'mover {mover.body!r} must be a mocap body (mocap="true")')
            tracks.append(_Track(mover, mocap_id, np.asarray(model.body_quat[body])))
        return cls(tracks)

    def apply(self, data: mujoco.MjData) -> None:
        for track in self.tracks:
            position, quat = track.pose(float(data.time))
            data.mocap_pos[track.mocap_id] = position
            data.mocap_quat[track.mocap_id] = quat
