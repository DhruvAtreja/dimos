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

"""The localize memory behind the scene-registration calls pick-and-place makes.

PickAndPlaceModule asks its scene for the prompted objects in view
(``scan_scene``) and then for one object's point cloud by id. This module
answers both from :class:`LiveLocalizeModule`'s memory, so a blueprint can put
it where ObjectSceneRegistrationModule stood without changing pick-and-place.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import uuid

import numpy as np
from pydantic import Field

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.vision_msgs.Detection3DArray import Detection3DArray
from dimos.perception.experimental.object_scene_registration_spec import (
    ObjectSceneRegistrationSpec,
)
from dimos.perception.localize.demo_blueprints.live_localize import (
    LiveLocalizeModule,
    LiveLocalizeModuleConfig,
    as_detection_array,
)
from dimos.perception.localize.types import Localization


class LocalizeSceneRegistrationConfig(LiveLocalizeModuleConfig):
    # How far back (seconds) each scan reads frames for new sightings.
    scan_window_s: float = Field(default=10.0, gt=0.0)
    # How long (seconds) a scan waits for the models to load and the feed to start.
    ready_timeout_s: float = Field(default=60.0, gt=0.0)
    # How long (seconds) a scan waits for a frame taken after the call to be
    # embedded; the memory embeds about one frame a second.
    fresh_frame_timeout_s: float = Field(default=15.0, gt=0.0)


@dataclass(frozen=True)
class FoundObject:
    """An object a scan reported: its label, box centre (world, metres) and point cloud."""

    name: str
    center: np.ndarray
    cloud: PointCloud2


class LocalizeSceneRegistrationModule(LiveLocalizeModule, ObjectSceneRegistrationSpec):
    """Find prompted objects with the localize memory, for pick-and-place.

    A scan reports an object only when a frame taken after the scan was called
    shows it, as a scan of the latest frame would. Its point cloud is fused
    over every camera position the memory saw it from. An object found again
    within the policy's ``cluster_radius_m`` of where it was keeps its id.

    The agent gets ``detect`` here and ``scan_objects`` from pick-and-place,
    the same perception tools ObjectSceneRegistrationModule gives it, so
    ``localize`` and ``state`` are RPCs only.
    """

    config: LocalizeSceneRegistrationConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._prompts: list[str] = []
        self._found: dict[str, FoundObject] = {}

    @rpc
    def set_prompts(self, text: list[str] | None = None) -> None:
        """Labels a later ``scan_scene()`` without its own looks for."""
        self._prompts = list(text or [])

    @rpc
    def scan_scene(self, text: list[str] | None = None) -> Detection3DArray:
        """The prompted objects the camera sees now, one box each in the world frame.

        Waits for the next embedded frame, so a scan right after the arm moves
        reflects where it moved to. Raises RuntimeError when the memory is not
        ready within ``ready_timeout_s`` or no new frame arrives in time.

        Args:
            text: Object labels, e.g. ["mug"]; None reuses the last set_prompts.
        """
        if text is not None:
            self._prompts = list(text)
        prompts = list(dict.fromkeys(p.strip() for p in self._prompts if p.strip()))
        world = self.config.world_frame
        if not prompts:
            return as_detection_array([], [], world)
        if not self._ready.wait(self.config.ready_timeout_s):
            raise RuntimeError(f"localize cannot answer yet: {self._stage}")
        since = self._color_feed.last().ts
        self._await_frame(since)
        window = self.config.scan_window_s
        with self._inference_lock:
            results = self.localize_objects(prompts, -window, window)
            fresh = [
                [h for h in hits if h.point_cloud is not None and h.last_seen_timestamp >= since]
                for hits in results
            ]
            detections = as_detection_array(prompts, fresh, world)
            self._assign_ids(prompts, fresh, detections)
        return detections

    @rpc
    def get_object_pointcloud_by_object_id(self, object_id: str) -> PointCloud2 | None:
        """The world-frame point cloud of an object a scan reported, or None."""
        found = self._found.get(object_id)
        return None if found is None else found.cloud

    @rpc
    def get_object_pointcloud_by_name(self, name: str) -> PointCloud2 | None:
        """The point cloud of the object with this label a scan reported last, or None."""
        latest = [found for found in self._found.values() if found.name == name]
        return latest[-1].cloud if latest else None

    @rpc
    def get_full_scene_pointcloud(
        self,
        exclude_object_id: str | None = None,
        depth_trunc: float = 2.0,
        voxel_size: float = 0.01,
    ) -> PointCloud2 | None:
        """Always None: this memory keeps objects, not a cloud of the whole scene."""
        return None

    @skill
    def detect(self, prompts: list[str]) -> str:
        """Detect objects matching the given text prompts.

        Do NOT call this tool multiple times for one query. Pass all objects in a single call.
        For example, to detect a cup and mouse, pass ["cup", "mouse"] in one call.

        Args:
            prompts: Text descriptions of concrete object categories to detect.

        Returns:
            str: Detected objects with their object_id and name.
        """
        if not prompts:
            return "No prompts provided."
        try:
            detections = self.scan_scene(prompts)
        except RuntimeError as exc:
            return str(exc)
        if not detections.detections:
            return "No objects detected."
        lines = [
            f"  - {d.results[0].hypothesis.class_id} (object_id='{d.id}')"
            for d in detections.detections
        ]
        return f"Detected {len(lines)} object(s):\n" + "\n".join(lines)

    @rpc
    def localize(
        self, objects: str, start: float = -10.0, duration: float = 10.0, policy: str = ""
    ) -> str:
        """LiveLocalizeModule.localize, for the shell rather than the agent."""
        return super().localize(objects, start, duration, policy)

    @rpc
    def state(self) -> str:
        """LiveLocalizeModule.state, for the shell rather than the agent."""
        return super().state()

    def _await_frame(self, since: float) -> None:
        """Block until a frame taken at or after ``since`` is embedded with its depth.

        ``since`` is on the camera's clock, in seconds.
        """
        deadline = time.monotonic() + self.config.fresh_frame_timeout_s
        while True:
            try:
                if self._depth_memory.last().ts >= since:
                    return
            except LookupError:
                pass
            if self._stop.is_set() or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"no new camera frame within {self.config.fresh_frame_timeout_s:.0f}s"
                )
            time.sleep(0.05)

    def _assign_ids(
        self, prompts: list[str], hits: list[list[Localization]], detections: Detection3DArray
    ) -> None:
        """Give each detection the id of the object it was before, or a new one.

        ``detections`` holds one box per hit, in the order of ``hits``.
        """
        radius = self._policy("").cluster_radius_m
        clouds = [
            (prompt, hit.point_cloud)
            for prompt, row in zip(prompts, hits, strict=True)
            for hit in row
        ]
        taken: set[str] = set()
        for detection, (name, cloud) in zip(detections.detections, clouds, strict=True):
            assert cloud is not None
            p = detection.bbox.center.position
            center = np.array([p.x, p.y, p.z])
            previous = [
                (float(np.linalg.norm(found.center - center)), object_id)
                for object_id, found in self._found.items()
                if found.name == name and object_id not in taken
            ]
            near = min(previous, default=(np.inf, ""))
            object_id = near[1] if near[0] <= radius else uuid.uuid4().hex
            taken.add(object_id)
            # Re-inserted, so the last entry for a label is the latest sighting.
            self._found.pop(object_id, None)
            self._found[object_id] = FoundObject(name, center, cloud)
            detection.id = object_id
