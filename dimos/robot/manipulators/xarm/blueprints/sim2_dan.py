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


"""xarm-perception-sim2 with the localize memory as its perception stack."""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.perception.localize.scene_registration import LocalizeSceneRegistrationModule
from dimos.robot.manipulators.xarm.blueprints.sim2 import xarm_perception_sim2

# xarm-perception-sim2 with one module swapped: the localize memory (SigLIP frame
# retrieval, OWLv2 boxes, EdgeTAM masks, sightings fused across camera positions)
# finds the objects pick-and-place grasps, where scene registration (Moondream
# boxes and EdgeTAM masks on the latest frame) did. The arm, planner, voxel map,
# self filter, grasp generator and viewers are the baseline's own.
xarm_perception_sim2_dan = autoconnect(
    xarm_perception_sim2.disabled_modules(ObjectSceneRegistrationModule),
    LocalizeSceneRegistrationModule.blueprint(
        world_frame="world",
        # The arm holds still while the agent scans, so every frame comes from
        # one camera position; the default policy wants two before it reports.
        policy={"min_views": 1},
    ),
)
