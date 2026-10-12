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

"""Dominant-color evidence from a detection bounding box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dimos.evals.vqa.contracts import InsufficientEvidenceError, NonEmptyString

if TYPE_CHECKING:
    from dimos.msgs.sensor_msgs.Image import Image

COLOR_CHOICES = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "brown",
    "white",
    "black",
    "gray",
)

_MIN_VALID_PIXELS = 32
_MIN_BUCKET_FRACTION = 0.4


class ObjectColorEvidence(BaseModel):
    """Dominant color bucket inside one detection crop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_name: NonEmptyString
    dominant_bucket: NonEmptyString
    bucket_fraction: float = Field(ge=0, le=1)
    valid_pixel_count: int = Field(ge=1)
    bbox_xyxy: tuple[float, float, float, float]


def dominant_color_in_bbox(
    image: Image, object_name: str, bbox_xyxy: tuple[float, ...]
) -> ObjectColorEvidence:
    """Map the largest HSV bucket inside a bbox to a fixed color choice."""
    import cv2

    x0, y0, x1, y1 = (int(round(v)) for v in bbox_xyxy)
    x0 = max(0, min(x0, image.width - 1))
    x1 = max(x0 + 1, min(x1, image.width))
    y0 = max(0, min(y0, image.height - 1))
    y1 = max(y0 + 1, min(y1, image.height))

    crop = image.to_rgb().data[y0:y1, x0:x1]
    if crop.ndim != 3 or crop.shape[2] < 3:
        raise InsufficientEvidenceError("object color requires an RGB image crop")

    rgb = np.asarray(crop[:, :, :3], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.intp)
    sat = hsv[:, :, 1].astype(np.intp)
    val = hsv[:, :, 2].astype(np.intp)

    valid = (sat >= 40) & (val >= 40)
    if int(valid.sum()) < _MIN_VALID_PIXELS:
        raise InsufficientEvidenceError(
            "object color crop lacks enough saturated, well-lit pixels"
        )

    buckets = np.array([_hue_to_bucket(int(h), int(s), int(v)) for h, s, v in hsv[valid]])
    counts = np.bincount(buckets, minlength=len(COLOR_CHOICES))
    winner = int(counts.argmax())
    fraction = float(counts[winner] / counts.sum())
    if fraction < _MIN_BUCKET_FRACTION:
        raise InsufficientEvidenceError(
            "object color crop does not have one unambiguous dominant bucket"
        )

    return ObjectColorEvidence(
        object_name=object_name,
        dominant_bucket=COLOR_CHOICES[winner],
        bucket_fraction=fraction,
        valid_pixel_count=int(valid.sum()),
        bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
    )


def _hue_to_bucket(hue: int, saturation: int, value: int) -> int:
    if value < 40:
        return COLOR_CHOICES.index("black")
    if value > 220 and saturation < 35:
        return COLOR_CHOICES.index("white")
    if saturation < 40:
        return COLOR_CHOICES.index("gray")

    if hue < 10 or hue >= 170:
        return COLOR_CHOICES.index("red")
    if hue < 25:
        return COLOR_CHOICES.index("orange")
    if hue < 35:
        return COLOR_CHOICES.index("yellow")
    if hue < 85:
        return COLOR_CHOICES.index("green")
    if hue < 130:
        return COLOR_CHOICES.index("blue")
    if hue < 150:
        return COLOR_CHOICES.index("purple")
    return COLOR_CHOICES.index("brown")
