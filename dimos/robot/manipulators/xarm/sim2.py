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

"""xArm7 with its native position servos, gripper units and wrist RGB-D camera."""

import math

from dimos.control.contract.keys import Unit
from dimos.robot.manipulators.xarm.config import XARM7_SIM_HOME
from dimos.sim2.sensors.spec import Camera
from dimos.sim2.spec import ControlInterface, Joint, RobotConfig
from dimos.utils.data import LfsPath

XARM7 = RobotConfig(
    model=LfsPath("xarm7/xarm7.xml"),
    root_body="link_base",
    control=ControlInterface.MANIPULATOR,
    joints=(
        *(
            Joint(
                name=f"joint{i}",
                model_name=f"joint{i}",
                actuator=f"act{i}",
                home=home,
                mode="position",
                lower=lo,
                upper=hi,
            )
            for i, home, lo, hi in zip(
                range(1, 8),
                XARM7_SIM_HOME,
                (
                    -2 * math.pi,
                    -2.059,
                    -2 * math.pi,
                    -0.19198,
                    -2 * math.pi,
                    -1.69297,
                    -2 * math.pi,
                ),
                (2 * math.pi, 2.0944, 2 * math.pi, 3.927, 2 * math.pi, math.pi, 2 * math.pi),
                strict=True,
            )
        ),
        Joint(
            name="arm/gripper",
            model_name="left_driver_joint",
            actuator="gripper",
            home=850.0,
            mode="position",
            scale=-0.001,
            offset=0.85,
            ctrl_scale=-0.3,
            ctrl_offset=255.0,
            lower=0.0,
            upper=850.0,
            max_velocity=0.0,
            unit=Unit.UNITLESS,
        ),
    ),
    sensors=(Camera("wrist_camera", camera="wrist_camera"),),
)
