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

"""The sim2 xArm7 keeps hold of a SceneReplica object while it lifts it."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
import pytest

from dimos.evals.scenes.scenereplica.convert import SPAWN_HEIGHT, ArmInScene
from dimos.robot.manipulators.xarm.sim2 import XARM7
from dimos.utils.data import LfsPath

pytestmark = pytest.mark.mujoco

# scene-16's tomato soup can (0.35 kg, 6.6 cm across) stands on its end with
# nothing within 13 cm, so a top-down grasp across it touches nothing else.
SCENE = "scene-16"
CAN = "tomato_soup_can"
# Metres. The pads' centre sits 1.75 cm above the TCP, so they close on the
# can 2.75 cm below its top.
TCP_BELOW_TOP = 0.045
LIFT = 0.10
# Seconds.
CLOSE_TIME = 1.0
LIFT_TIME = 1.0
HOLD_TIME = 2.0


def _gripper_ctrl(command: float) -> float:
    """Actuator value for a gripper command in xArm units (0 closed, 850 open)."""
    gripper = XARM7.joints[-1]
    return command * gripper.ctrl_scale + gripper.ctrl_offset


def test_the_gripper_lifts_a_can_without_letting_it_slip() -> None:
    import mujoco

    arm = ArmInScene(LfsPath("scenereplica") / SCENE, SPAWN_HEIGHT)
    model, data = arm.model, arm.data
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC, (
        f"data/scenereplica/{SCENE} uses pyramidal friction cones; delete data/scenereplica "
        "so it is unpacked again from data/.lfs/scenereplica.tar.gz"
    )

    x, y, top = arm.object_top(CAN)
    grasp = np.array([x, y, top - TCP_BELOW_TOP])
    seed = arm.home.copy()
    seed[0] = math.atan2(y, x)
    assert arm.solve(grasp + np.array([0.0, 0.0, LIFT]), seed)[0]
    q_lift = data.qpos[arm.qpos].copy()
    assert arm.solve(grasp, q_lift)[0]
    q_grasp = data.qpos[arm.qpos].copy()

    mujoco.mj_forward(model, data)
    arm_actuators = [model.actuator(f"arm/act{i}").id for i in range(1, 8)]
    gripper = model.actuator("arm/gripper").id
    jaw = model.joint("arm/left_driver_joint").qposadr[0]
    can = arm.entities[CAN]
    data.ctrl[arm_actuators] = q_grasp
    data.ctrl[gripper] = _gripper_ctrl(850.0)

    def step(seconds: float, lift_to: NDArray[np.float64] | None = None) -> None:
        """Run the sim; with ``lift_to`` (arm joint targets, rad), move the arm there."""
        steps = round(seconds / model.opt.timestep)
        for i in range(steps):
            if lift_to is not None:
                s = (i + 1) / steps
                # Minimum-jerk blend from the grasp pose to lift_to.
                blend = 10 * s**3 - 15 * s**4 + 6 * s**5
                data.ctrl[arm_actuators] = q_grasp + blend * (lift_to - q_grasp)
            mujoco.mj_step(model, data)

    def can_in_tcp() -> NDArray[np.float64]:
        """The can's position in the gripper's TCP frame, metres."""
        rotation = data.site_xmat[arm.site].reshape(3, 3)
        return np.asarray(rotation.T @ (data.xpos[can] - data.site_xpos[arm.site]))

    step(0.3)  # the arm settles at the grasp pose with the jaws open
    data.ctrl[gripper] = _gripper_ctrl(0.0)
    step(CLOSE_TIME)
    held_at, jaw_closed, can_z = can_in_tcp(), float(data.qpos[jaw]), float(data.xpos[can][2])
    step(LIFT_TIME, q_lift)
    step(HOLD_TIME)

    slip_mm = 1000 * float(np.linalg.norm(can_in_tcp() - held_at))
    assert slip_mm < 5.0, f"the can moved {slip_mm:.1f} mm in the gripper"
    assert data.xpos[can][2] - can_z > 0.8 * LIFT
    # The joint reads 0..0.85 rad over the full stroke; a reading that keeps
    # closing on a held object looks like an empty gripper to grasp checks.
    assert abs(float(data.qpos[jaw]) - jaw_closed) < 0.02
