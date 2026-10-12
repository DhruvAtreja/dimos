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

from __future__ import annotations

import numpy as np
import pytest

from dimos.evals.vqa.contracts import InsufficientEvidenceError
from dimos.evals.vqa.pointcloud_frame import PointCloudFrame
from dimos.evals.vqa.primitives.bearing import (
    bearing_choice,
    estimate_camera_bearing,
)
from dimos.evals.vqa.primitives.edge_tam import ObjectMaskEvidence
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox


class _FullMaskEstimator:
    def estimate(self, image: Image, object_name: str) -> ObjectMaskEvidence:
        mask = np.ones((image.height, image.width), dtype=bool)
        detection = Detection2DBBox(
            bbox=(0.0, 0.0, float(image.width), float(image.height)),
            track_id=0,
            class_id=-1,
            confidence=1.0,
            name=object_name,
            ts=image.ts,
            image=image,
        )
        return ObjectMaskEvidence(
            object_name=object_name,
            prompt_bbox_xyxy=detection.bbox,
            detection=detection,
            mask=mask,
        )

    def estimate_many(self, image: Image, object_names: tuple[str, ...]) -> tuple[ObjectMaskEvidence, ...]:
        return tuple(self.estimate(image, name) for name in object_names)


def _spread_camera_points() -> np.ndarray:
    """Return many camera-frame points that project to distinct image pixels."""
    fx = fy = 10.0
    cx = cy = 10.0
    depth = 2.0
    coords: list[list[float]] = []
    for u in range(6, 14):
        for v in range(6, 14):
            coords.append([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])
    return np.asarray(coords, dtype=np.float64)


def _frame_with_points(points: np.ndarray) -> PointCloudFrame:
    image = Image.from_numpy(np.zeros((20, 20, 3), dtype=np.uint8), frame_id="camera_optical", ts=1.0)
    camera_info = CameraInfo.from_intrinsics(
        10.0, 10.0, 10.0, 10.0, image.width, image.height, frame_id="camera_optical"
    )
    cloud = PointCloud2.from_numpy(points, frame_id="camera_optical", timestamp=1.0)
    transform = Transform.identity()
    return PointCloudFrame(
        index=0,
        image=image,
        pointcloud=cloud,
        camera_info=camera_info,
        pointcloud_to_camera=transform,
        image_observation_timestamp=1.0,
        pointcloud_observation_timestamp=1.0,
        calibration_source="test",
    )


def test_camera_bearing_ahead() -> None:
    points = _spread_camera_points()
    frame = _frame_with_points(points)

    evidence = estimate_camera_bearing(frame, _FullMaskEstimator(), "chair")

    assert bearing_choice(evidence) == "ahead"


def test_camera_bearing_right() -> None:
    fx = fy = 10.0
    cx = cy = 10.0
    depth = 2.0
    x_offset = 2.0
    coords: list[list[float]] = []
    for u in range(6, 14):
        for v in range(6, 14):
            coords.append([(u - cx) * depth / fx + x_offset, (v - cy) * depth / fy, depth])
    frame = _frame_with_points(np.asarray(coords, dtype=np.float64))

    evidence = estimate_camera_bearing(frame, _FullMaskEstimator(), "chair")

    assert bearing_choice(evidence) == "right"


def test_camera_bearing_rejects_crossing_quartiles() -> None:
    left = _spread_camera_points() + np.array([-2.0, 0.0, 0.0], dtype=np.float64)
    right = _spread_camera_points() + np.array([2.0, 0.0, 0.0], dtype=np.float64)
    points = np.vstack((left, right))
    frame = _frame_with_points(points)

    with pytest.raises(InsufficientEvidenceError, match="bearing answer boundary"):
        estimate_camera_bearing(frame, _FullMaskEstimator(), "chair")
