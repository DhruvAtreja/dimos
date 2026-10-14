# Copyright 2025-2026 Dimensional Inc.
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


from dimos.robot.galaxea.r1pro.config import (
    R1PRO_CHASSIS_LIDAR_FRAME,
    R1PRO_CHASSIS_LIDAR_MOUNT_XYZ,
    R1PRO_MODEL,
)


def test_chassis_lidar_mount_matches_the_urdf() -> None:
    joint = R1PRO_MODEL.load().get_joint("lidar_chassis_left_joint")
    assert joint is not None
    assert (joint.parent_link, joint.child_link) == ("base_link", R1PRO_CHASSIS_LIDAR_FRAME)
    assert joint.origin_xyz == R1PRO_CHASSIS_LIDAR_MOUNT_XYZ
    assert joint.origin_rpy == (0.0, 0.0, 0.0)
