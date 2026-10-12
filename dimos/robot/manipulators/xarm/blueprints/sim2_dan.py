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

from dataclasses import replace

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.perception.experimental.object_scene_registration import ObjectSceneRegistrationModule
from dimos.perception.localize.scene_registration import LocalizeSceneRegistrationModule
from dimos.robot.manipulators.xarm.blueprints.sim2 import xarm_perception_sim2


def swap_scene_registration(blueprint: Blueprint, perception: Blueprint) -> Blueprint:
    """``blueprint`` with its scene-registration module replaced by ``perception``'s one module.

    The replacement takes the same place in the module order. Modules are spread over
    worker processes in that order, so every other module shares a process with the
    same neighbours as before.
    """
    (atom,) = perception.blueprints
    return replace(
        blueprint,
        blueprints=tuple(
            atom if each.module is ObjectSceneRegistrationModule else each
            for each in blueprint.blueprints
        ),
    )


# xarm-perception-sim2 with one module swapped: the localize memory (SigLIP frame
# retrieval, OWLv2 boxes, EdgeTAM masks, sightings fused across camera positions)
# finds the objects pick-and-place grasps, where scene registration (Moondream
# boxes and EdgeTAM masks on the latest frame) did. The arm, planner, voxel map,
# self filter, grasp generator and viewers are the baseline's own.
xarm_perception_sim2_dan = autoconnect(
    swap_scene_registration(
        xarm_perception_sim2,
        LocalizeSceneRegistrationModule.blueprint(
            world_frame="world",
            # The arm holds still while the agent scans, so every frame comes from
            # one camera position; the default policy wants two before it reports.
            policy={"min_views": 1},
        ),
    )
)
