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

import functools

import numpy as np
from turbojpeg import TJPF_RGB, TurboJPEG


@functools.cache
def get_turbojpeg() -> TurboJPEG:
    """Return the shared TurboJPEG codec handle."""
    return TurboJPEG()


@functools.cache
def turbojpeg_available() -> bool:
    """Whether the libturbojpeg shared library can be loaded on this machine."""
    try:
        get_turbojpeg()
    except RuntimeError:
        return False
    return True


def encode_rgb_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """Compress an RGB image to JPEG bytes, with libturbojpeg or else OpenCV.

    Args:
        rgb: pixels, height x width x 3, uint8, in RGB channel order.
        quality: JPEG quality, 0-100; higher keeps more detail and makes bigger files.
    """
    if turbojpeg_available():
        return get_turbojpeg().encode(rgb, quality=quality, pixel_format=TJPF_RGB)  # type: ignore[no-any-return]
    import cv2

    ok, jpeg = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError(f"OpenCV could not JPEG-encode a {rgb.shape} {rgb.dtype} image")
    return jpeg.tobytes()


def decode_rgb_jpeg(data: bytes) -> np.ndarray:
    """Decompress JPEG bytes to RGB pixels (height x width x 3, uint8), with libturbojpeg or else OpenCV."""
    if turbojpeg_available():
        return get_turbojpeg().decode(data, pixel_format=TJPF_RGB)  # type: ignore[no-any-return]
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV could not decode the JPEG data")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
