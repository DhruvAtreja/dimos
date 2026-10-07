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

"""Offline tests for the localize memory behind pick-and-place: what a scan reports, the
ids it hands out, the clouds read back by id, and the agent tools it exposes."""

from __future__ import annotations

from collections.abc import Iterator
import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.localize.demo_blueprints.live_localize import LiveLocalizeModule
from dimos.perception.localize.scene_registration import LocalizeSceneRegistrationModule
from dimos.perception.localize.types import Localization, LocalizePolicy

XYZ = tuple[float, float, float]
CALLED_AT = 10.0  # camera clock, seconds: the newest frame when a scan is called


def _hit(center: XYZ, last_seen: float) -> Localization:
    """A verified instance whose cloud is a 4 cm cube around ``center``, world frame."""
    x, y, z = center
    corners = np.array([[x - 0.02, y - 0.02, z - 0.02], [x + 0.02, y + 0.02, z + 0.02]])
    cloud = PointCloud2.from_numpy(
        corners.astype(np.float32), frame_id="world", timestamp=last_seen
    )
    return Localization(
        instance_id="query-0",
        semantic_score=0.9,
        identity_score=0.25,
        ambiguity_margin=1.0,
        position_world_xyz=center,
        orientation_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="world",
        support=None,
        pose_timestamp=last_seen,
        geometry_timestamp=last_seen,
        last_seen_timestamp=last_seen,
        point_cloud=cloud,
        coverage=0.125,
        n_views=1,
    )


def _ready(module: LiveLocalizeModule, embedded_at: float = CALLED_AT + 1.0) -> None:
    """Stand in for start(): models loaded, and a frame embedded at ``embedded_at``."""
    module._stop = threading.Event()
    module._ready = threading.Event()
    module._ready.set()
    module._stage = "ready"
    module._inference_lock = threading.RLock()
    module._groups = {}
    module._memory = MagicMock()
    module._color_feed = MagicMock()
    module._color_feed.last.return_value.ts = CALLED_AT
    module._depth_memory = MagicMock()
    module._depth_memory.last.return_value.ts = embedded_at
    module.rig = MagicMock(world_frame="world", default_localize_policy=LocalizePolicy)
    # What stop() tears down.
    module._thread = MagicMock()
    module._unsubs = []
    module._tf_buffer = MagicMock()


@pytest.fixture
def module() -> Iterator[LocalizeSceneRegistrationModule]:
    instance = LocalizeSceneRegistrationModule(
        world_frame="world", ready_timeout_s=0.1, fresh_frame_timeout_s=0.1
    )
    _ready(instance)
    instance.localize_objects = MagicMock()  # type: ignore[method-assign]
    yield instance
    instance.stop()


def _centers(module: LocalizeSceneRegistrationModule, prompts: list[str]) -> dict[str, XYZ]:
    """Scan, and map each reported id to its box centre."""
    found = module.scan_scene(prompts)
    return {
        d.id: (d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z)
        for d in found.detections
    }


def test_a_scan_reports_only_what_a_frame_after_the_call_saw(
    module: LocalizeSceneRegistrationModule,
) -> None:
    fresh, stale = _hit((0.5, 0.0, 0.8), CALLED_AT + 1.0), _hit((0.7, 0.1, 0.8), CALLED_AT - 5)
    module.localize_objects.return_value = [[fresh, stale]]  # type: ignore[attr-defined]

    found = module.scan_scene(["mug"])

    (detection,) = found.detections
    assert detection.results[0].hypothesis.class_id == "mug"
    assert detection.bbox.center.position.x == pytest.approx(0.5)
    assert found.header.frame_id == "world"
    assert module.get_object_pointcloud_by_object_id(detection.id) is fresh.point_cloud
    assert module.get_object_pointcloud_by_name("mug") is fresh.point_cloud
    assert module.get_object_pointcloud_by_object_id("unknown") is None
    # The query read the configured window back from the newest frame.
    module.localize_objects.assert_called_once_with(["mug"], -10.0, 10.0)  # type: ignore[attr-defined]


def test_an_object_found_again_nearby_keeps_its_id(
    module: LocalizeSceneRegistrationModule,
) -> None:
    query = module.localize_objects
    query.return_value = [[_hit((0.5, 0.0, 0.8), 11.0)]]  # type: ignore[attr-defined]
    (first,) = _centers(module, ["mug"])

    # 2 cm along, inside the 8 cm cluster radius: the same mug.
    query.return_value = [[_hit((0.52, 0.0, 0.8), 11.0)]]  # type: ignore[attr-defined]
    assert list(_centers(module, ["mug"])) == [first]

    # A second mug beside it gets its own id, and so does a bowl at the first mug's spot.
    query.return_value = [  # type: ignore[attr-defined]
        [_hit((0.52, 0.0, 0.8), 11.0), _hit((0.53, 0.01, 0.8), 11.0)],
        [_hit((0.52, 0.0, 0.8), 11.0)],
    ]
    ids = list(_centers(module, ["mug", "bowl"]))
    assert ids[0] == first and len(set(ids)) == 3

    # 20 cm away is another object.
    query.return_value = [[_hit((0.72, 0.0, 0.8), 11.0)]]  # type: ignore[attr-defined]
    assert first not in _centers(module, ["mug"])


def test_a_scan_fails_until_ready_and_without_a_new_frame(
    module: LocalizeSceneRegistrationModule,
) -> None:
    module._ready.clear()
    module._stage = "loading SigLIP, OWLv2 and EdgeTAM weights"
    with pytest.raises(RuntimeError, match="loading SigLIP"):
        module.scan_scene(["mug"])
    assert "loading SigLIP" in module.detect(["mug"])

    _ready(module, embedded_at=CALLED_AT - 1.0)
    with pytest.raises(RuntimeError, match="no new camera frame"):
        module.scan_scene(["mug"])
    module.localize_objects.assert_not_called()  # type: ignore[attr-defined]


def test_no_prompts_scans_nothing(module: LocalizeSceneRegistrationModule) -> None:
    assert module.scan_scene([" "]).detections == []
    module.set_prompts(["mug"])
    module.localize_objects.return_value = [[]]  # type: ignore[attr-defined]
    assert module.scan_scene().detections == []
    module.localize_objects.assert_called_once()  # type: ignore[attr-defined]


def test_detect_answers_in_scene_registrations_format(
    module: LocalizeSceneRegistrationModule,
) -> None:
    module.localize_objects.return_value = [[_hit((0.5, 0.0, 0.8), 11.0)]]  # type: ignore[attr-defined]
    answer = module.detect(["cracker box"])
    (object_id,) = module._found
    assert answer == f"Detected 1 object(s):\n  - cracker box (object_id='{object_id}')"

    module.localize_objects.return_value = [[]]  # type: ignore[attr-defined]
    assert module.detect(["mug"]) == "No objects detected."


def test_the_agent_gets_detect_and_not_localize_or_state() -> None:
    def skills(cls: type) -> set[str]:
        return {name for name in dir(cls) if hasattr(getattr(cls, name), "__skill__")}

    assert {"localize", "state"} <= skills(LiveLocalizeModule)
    assert skills(LocalizeSceneRegistrationModule) == {"detect"}


def test_localize_objects_folds_the_module_policy_under_the_calls() -> None:
    instance = LiveLocalizeModule(world_frame="world", policy={"min_views": 1, "tail_k": 3})
    _ready(instance)
    instance.index = MagicMock()
    instance.index.get_time_range.return_value = (0.0, 30.0)
    instance.index.time_range.side_effect = lambda lo, hi: ("frames", lo, hi)
    hits = [[_hit((0.5, 0.0, 0.8), 11.0)]]
    instance.detector = MagicMock()
    instance.detector.localize.return_value = hits
    try:
        assert instance.localize_objects(["mug"], -10.0, 10.0, json.dumps({"tail_k": 2})) == hits
        (store, prompts), kwargs = instance.detector.localize.call_args
        assert prompts == ["mug"] and kwargs["index"] == ("frames", 20.0, 30.0)
        assert kwargs["policy"] == LocalizePolicy(min_views=1, tail_k=2)
        assert kwargs["groups"] is instance._groups

        instance._ready.clear()
        with pytest.raises(RuntimeError, match="cannot answer yet"):
            instance.localize_objects(["mug"])
    finally:
        instance.stop()
