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

"""TARS (MuJoCo sim via tars_sdk) blueprints.

Usage:
    uv pip install -e experimental/tars_sdk
    dimos run tars-sim                                # TarsConnection + lidar global map in rerun
    dimos run tars-sim-keyboard-teleop                # same + WASD pygame teleop (Linux)
    dimos run tars-sim-nav                            # + costmap, A* to a clicked goal, exploration
    dimos run tars-sim-moon                           # tars-sim-nav on a cratered lunar site
    dimos run coordinator-tars-sim                    # TARS in the Go2 office scene + viewer, /cmd_vel
    dimos run coordinator-tars-sim-keyboard-teleop    # + WASD pygame teleop (Linux; crashes on macOS)
    python -m dimos.robot.tars.demo_keyboard_cmd_vel  # macOS: run next to coordinator-tars-sim
"""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import numpy as np
from tars_sdk.scenes import moon

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.navigation.experimental.frontier_exploration.wavefront_frontier_goal_selector import (
    WavefrontFrontierExplorer,
)
from dimos.navigation.experimental.patrolling.module import PatrollingModule
from dimos.navigation.go2.replanning_a_star.module import ReplanningAStarPlanner
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.robot.tars.connection import TarsConnection
from dimos.robot.tars.height_crop import HeightCrop
from dimos.robot.tars.lidar_registration import LidarRegistration
from dimos.robot.unitree.keyboard_teleop import KeyboardTeleop
from dimos.visualization.vis_module import vis_module

# Footprint for the planner (half-size TARS): 0.48 m across the slabs; walking spreads the
# feet ~0.3 m fore-aft, so turning in place sweeps a ~0.6 m circle.
TARS_WIDTH = 0.5
TARS_ROTATION_DIAMETER = 0.65

_tars_joints = make_twist_base_joints("tars")

_tars_hw = HardwareComponent(
    hardware_id="tars",
    hardware_type=HardwareType.BASE,
    joints=_tars_joints,
    adapter_type="tars",
    # same room as `dimos --simulation run unitree-go2`, in an open aisle facing +x
    adapter_kwargs={"scene": "office1", "spawn": (-4.5, 1.1, 0.0), "viewer": True},
)

_tars_vel_task = TaskConfig(
    name="vel_tars",
    type="velocity",
    joint_names=_tars_joints,
    priority=10,
)

coordinator_tars_sim = ControlCoordinator.blueprint(
    hardware=[_tars_hw],
    tasks=[_tars_vel_task],
).remappings([(ControlCoordinator, "twist_command", "cmd_vel")])

coordinator_tars_sim_keyboard_teleop = autoconnect(
    ControlCoordinator.blueprint(hardware=[_tars_hw], tasks=[_tars_vel_task]),
    KeyboardTeleop.blueprint(),
).remappings([(ControlCoordinator, "twist_command", "cmd_vel")])


def _convert_global_map(
    cloud: Any,
    max_z: float | None = None,
    cube: float | None = None,
    xy_box: tuple[float, float, float, float] | None = None,
) -> Any:
    # max_z drops a ceiling so the 3D view looks into a room; xy_box (x0, x1, y0, y1) keeps
    # the area of interest (the viewer frames the whole scene); cube draws each voxel as an
    # oversized sphere of that diameter so a sparse lidar map reads as a solid surface
    if max_z is not None or xy_box is not None:
        from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

        points, _ = cloud.as_numpy()
        keep = points[:, 2] < (max_z if max_z is not None else float("inf"))
        if xy_box is not None:
            x0, x1, y0, y1 = xy_box
            keep &= (points[:, 0] > x0) & (points[:, 0] < x1)
            keep &= (points[:, 1] > y0) & (points[:, 1] < y1)
        cloud = PointCloud2.from_numpy(points[keep], frame_id=cloud.frame_id)
    if cube is not None:
        # spheres, not mode="boxes": to_rerun's boxes mode passes one half_size, and
        # Boxes3D draws one box per half_size. Colored by height on a fixed scale (turbo,
        # -1.2..1.5 m): crater floors blue, the plain green, rims and tall things warm.
        from matplotlib import colormaps

        points, _ = cloud.as_numpy()
        z = np.clip((points[:, 2] + 1.2) / 2.7, 0.0, 1.0)
        colors = (colormaps["turbo"](z)[:, :3] * 255).astype(np.uint8)
        return cloud.to_rerun(voxel_size=cube, colors=colors)
    return cloud.to_rerun(bottom_cutoff=0)


def _convert_navigation_costmap(grid: Any) -> Any:
    return grid.to_rerun(colormap="Accent", z_offset=0.015, opacity=0.2, background="#484981")


def _static_tars_body(rr: Any) -> list[Any]:
    # the slab stack at half size, hanging from the hub (base_link)
    return [
        rr.Boxes3D(
            half_sizes=[0.07, 0.24, 0.38], centers=[[0.0, 0.0, -0.305]], colors=[(160, 160, 165)]
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _tars_rerun_blueprint(
    eye: tuple[float, float, float],
    target: tuple[float, float, float] | None = None,
    camera_share: float = 1.0,
) -> Any:
    """target None: the eye follows TARS from `eye` (relative); else a fixed world eye."""
    import rerun as rr
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="Camera"),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
                line_grid=rrb.LineGrid3D(plane=rr.components.Plane3D.XY.with_distance(0.0)),
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=list(eye),
                    look_target=list(target) if target else [0.0, 0.0, 0.3],
                    eye_up=[0.0, 0.0, 1.0],
                    tracking_entity=None if target else "world/robot_body",
                ),
            ),
            column_shares=[camera_share, 2],
        ),
        rrb.TimePanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )


def _tars_rerun_config(
    map_max_z: float | None,
    blueprint: Any,
    map_cube: float | None = None,
    map_xy_box: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "world/global_map": partial(
            _convert_global_map, max_z=map_max_z, cube=map_cube, xy_box=map_xy_box
        ),
        "world/navigation_costmap": _convert_navigation_costmap,
        # world/lidar is drawn in lidar_link (it rides the slab); the registered copy is a
        # duplicate
        "world/registered_lidar": None,
    }
    if map_xy_box is not None:  # the viewer frames everything: hide what reaches beyond it
        overrides |= {
            "world/lidar": None,
            "world/global_costmap": None,
            "world/navigation_costmap": None,
            "world/nav_map": None,
        }
    return {
        "blueprint": blueprint,
        "visual_override": overrides,
        "max_hz": {
            "world/global_map": 0,
            "world/global_costmap": 0,
            "world/color_image": 0,
            "world/lidar": 2,
        },
        "tf_axes": 0.3,
        "static": {"world/robot_body": _static_tars_body},
    }


def _tars_sim(
    scene: str,
    spawn: tuple[float, float, float],
    rerun_config: dict[str, Any],
    carve_columns: bool = True,
) -> Any:
    """TARS + TF-registered lidar map + rerun. Teleop (rerun viewer WASD -> tele_cmd_vel)
    and navigation (nav_cmd_vel) both go through MovementManager, which owns cmd_vel and
    gives teleop priority."""
    return autoconnect(
        vis_module(viewer_backend=global_config.viewer, rerun_config=rerun_config),
        # lidar arrives in lidar_link (on a swinging slab); TF at each scan instant registers it
        TarsConnection.blueprint(scene=scene, spawn=spawn),
        LidarRegistration.blueprint(),
        # carving clears each scanned (x, y) column before inserting: it removes things that
        # moved, but keeps only the latest scan's slice of anything tall
        VoxelGridMapper.blueprint(emit_every=5, carve_columns=carve_columns).remappings(
            [(VoxelGridMapper, "lidar", "registered_lidar")]
        ),
        MovementManager.blueprint(),
    )


def _tars_nav(sim: Any) -> Any:
    """Go2's 2D navigation stack (unitree-go2) with the TARS footprint. Its controller drives
    with (vx, wz) only, which suits TARS (no strafing). Click a point in rerun for a goal."""
    return autoconnect(
        sim,
        # the costmap sees the map below TARS's height only: the tilted lidar's ceiling hits
        # would otherwise read as walls (see HeightCrop)
        HeightCrop.blueprint(max_z=1.0),
        # TODO: can_pass_under should be ~0.9 m for TARS, but a HeightCostConfig in the
        # blueprint currently fails config validation; the default (0.6 m) is used for now.
        # The lidar can't see the floor right under TARS: treat the start area as free.
        CostMapper.blueprint(initial_safe_radius_meters=0.8).remappings(
            [(CostMapper, "global_map", "nav_map")]
        ),
        ReplanningAStarPlanner.blueprint(
            robot_width=TARS_WIDTH, robot_rotation_diameter=TARS_ROTATION_DIAMETER
        ),
        WavefrontFrontierExplorer.blueprint(),
        PatrollingModule.blueprint(),
    )


# the Go2 office scene, in an open aisle facing +x
tars_sim = _tars_sim(
    "office1",
    (-4.5, 1.1, 0.0),
    # behind and above TARS, following it
    _tars_rerun_config(2.0, partial(_tars_rerun_blueprint, (-3.5, -3.0, 3.5))),
).global_config(n_workers=4)

tars_sim_keyboard_teleop = autoconnect(
    tars_sim,
    KeyboardTeleop.blueprint().remappings([(KeyboardTeleop, "cmd_vel", "tele_cmd_vel")]),
)

tars_sim_nav = _tars_nav(tars_sim).global_config(n_workers=8)

# a lunar landing site (tars_sdk.scenes.moon), facing along its walking route
_moon_yaw = math.atan2(moon.ROUTE[1][1] - moon.ROUTE[0][1], moon.ROUTE[1][0] - moon.ROUTE[0][0])
# The map shown in rerun is cut to the landing site around the route, so the viewer frames
# it closely. Nothing moves on the moon, so the map keeps every scan (no column carving).
_moon_rerun = partial(_tars_rerun_blueprint, (-4.0, -1.5, 4.5), (9.0, 0.5, -0.8), 0.5)
tars_sim_moon = _tars_nav(
    _tars_sim(
        "moon",
        (0.0, 0.0, _moon_yaw),
        _tars_rerun_config(None, _moon_rerun, map_cube=0.12, map_xy_box=(-8.0, 30.0, -9.0, 9.0)),
        carve_columns=False,
    )
).global_config(n_workers=8)
