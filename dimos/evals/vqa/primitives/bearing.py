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

"""Camera-relative horizontal bearing from LiDAR points in an object mask."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dimos.evals.vqa.contracts import InsufficientEvidenceError, NonEmptyString
from dimos.evals.vqa.primitives.range import _FrameProjection

if TYPE_CHECKING:
    from dimos.evals.vqa.contracts import ObjectMaskEstimator
    from dimos.evals.vqa.pointcloud_frame import PointCloudFrame

BEARING_CHOICES = ("left", "ahead", "right")
_AHEAD_HALF_ANGLE_DEG = 30.0
_MIN_SUPPORTING_POINTS = 5


class CameraBearingEvidence(BaseModel):
    """Horizontal bearing of an object relative to the camera forward axis."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    object_name: NonEmptyString
    bearing_deg: float
    bearing_quartiles_deg: tuple[float, float, float]
    supporting_point_count: int = Field(ge=1)
    synchronization_delta_s: float = Field(ge=0)
    calibration_source: str = Field(min_length=1)


def estimate_camera_bearing(
    frame: PointCloudFrame,
    mask_estimator: ObjectMaskEstimator,
    object_name: str,
) -> CameraBearingEvidence:
    """Estimate left/ahead/right from camera-frame LiDAR points inside one mask."""
    mask = mask_estimator.estimate(frame.image, object_name)
    projection = _FrameProjection.from_frame(frame)
    points = projection.camera_points_in_mask(mask.mask)
    if len(points) < _MIN_SUPPORTING_POINTS:
        raise InsufficientEvidenceError(
            f"camera bearing requires at least {_MIN_SUPPORTING_POINTS} supporting "
            f"points for {object_name!r}, got {len(points)}"
        )

    angles = np.degrees(np.arctan2(points[:, 0], points[:, 2]))
    lower, median, upper = map(float, np.quantile(angles, (0.25, 0.5, 0.75)))
    if _bearing_bucket(lower) != _bearing_bucket(upper):
        raise InsufficientEvidenceError(
            "camera bearing uncertainty crosses a bearing answer boundary"
        )
    return CameraBearingEvidence(
        object_name=object_name,
        bearing_deg=median,
        bearing_quartiles_deg=(lower, median, upper),
        supporting_point_count=len(points),
        synchronization_delta_s=float(frame.synchronization_delta_s),
        calibration_source=frame.calibration_source,
    )


def bearing_choice(evidence: CameraBearingEvidence) -> str:
    return BEARING_CHOICES[_bearing_bucket(evidence.bearing_deg)]


def _bearing_bucket(angle_deg: float) -> int:
    if angle_deg < -_AHEAD_HALF_ANGLE_DEG:
        return 0
    if angle_deg > _AHEAD_HALF_ANGLE_DEG:
        return 2
    return 1
