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

"""The sim2 xArm7 holds a commanded pose under gravity, as the real arm does."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.robot.manipulators.xarm.config import XARM7_SIM_HOME
from dimos.robot.manipulators.xarm.sim2 import XARM7

pytestmark = pytest.mark.mujoco

# Arm joint targets, radians: the sim home pose, and the arm stretched forward and
# low as it is above a grasp on the table, where gravity pulls hardest.
POSES = {
    "home": list(XARM7_SIM_HOME),
    "stretched": [0.0, 0.9, 0.0, 1.0, 0.0, 0.1, 0.0],
    "pre-grasp": [-0.6, 0.7, 0.3, 1.4, -0.2, 0.7, 0.4],
}
# Metres. The real xArm tracks a position target within a few millimetres.
MAX_TCP_ERROR = 0.003
SETTLE_TIME = 3.0  # seconds


@pytest.mark.parametrize("pose", POSES)
def test_the_arm_holds_a_pose_under_gravity(pose: str) -> None:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(XARM7.model))
    spec.meshdir = str(XARM7.meshdir)
    model = spec.compile()
    data = mujoco.MjData(model)
    tcp = model.site("link_tcp").id
    target = POSES[pose]

    data.qpos[:7] = target
    mujoco.mj_kinematics(model, data)
    wanted = data.site_xpos[tcp].copy()
    data.ctrl[:7] = target
    for _ in range(round(SETTLE_TIME / model.opt.timestep)):
        mujoco.mj_step(model, data)

    error = float(np.linalg.norm(data.site_xpos[tcp] - wanted))
    assert error < MAX_TCP_ERROR, f"TCP sits {error * 1000:.1f} mm from its target at {pose}"
