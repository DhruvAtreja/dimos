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

"""Focused emulator contracts with real MuJoCo models and shared-memory devices."""

import logging
from uuid import uuid4

import mujoco
import numpy as np
import pytest

from dimos.hardware.whole_body.spec import MotorCommand
from dimos.robot.manipulators.xarm.sim2 import XARM7
from dimos.robot.unitree.g1.sim2 import G1_GROOT
from dimos.sim2.control.adapters import ManipulatorAdapter, WholeBodyAdapter
from dimos.sim2.ipc.channel import IncoherentFrameError
from dimos.sim2.runtime import SimulationRuntime
from dimos.sim2.scene import scene_path
from dimos.sim2.sensors.camera.renderers.mujoco import MujocoCamera
from dimos.sim2.sensors.lidar.raycast import Raycaster
from dimos.sim2.sensors.reader import WorldReader
from dimos.sim2.spec import RobotInstance, WorldConfig

pytestmark = pytest.mark.mujoco


@pytest.fixture
def runtime():
    worlds = []

    def create(robot, robot_id, scene, xyz):
        world = SimulationRuntime(
            WorldConfig(
                scene_path(None, scene),
                {robot_id: RobotInstance(robot, xyz=xyz)},
            ),
            uuid4().hex,
        )
        worlds.append(world)
        return world

    yield create
    for world in reversed(worlds):
        world.close()


@pytest.fixture
def device():
    devices = []

    def create(world, robot_id, adapter):
        config = world.robots[robot_id].config
        client = adapter(
            address=world.snapshot_descriptor.sim_id + "/" + robot_id,
            dof=len(config.joints),
            definition=config,
        )
        client.connect()
        devices.append(client)
        return client

    yield create
    for client in reversed(devices):
        client.disconnect()


def test_g1_complete_commands_and_coherent_imu(runtime, device):
    world = runtime(G1_GROOT, "g1", "logistics.xml", (0, 0, 0.793))
    client = device(world, "g1", WholeBodyAdapter)
    motors = client.read_motor_states()
    imu = client.read_imu()
    assert [s.q for s in motors] == pytest.approx([j.home for j in G1_GROOT.joints])
    assert imu.quaternion == pytest.approx((1, 0, 0, 0))
    world.step()
    assert client.read_imu() == imu
    commands = [MotorCommand(q=s.q, dq=0.2, kp=0, kd=2, tau=0.3) for s in motors]
    assert client.write_motor_commands(commands)
    binding = world.robots["g1"]
    before = world.data.qvel[binding.dofs].copy()
    world.step()
    assert world.data.ctrl[binding.actuators] == pytest.approx(2 * (0.2 - before) + 0.3)


def test_xarm_moves_and_uses_real_gripper_units(runtime, device):
    world = runtime(XARM7, "arm", "workbench.xml", (0, 0, 0.12))
    client = device(world, "arm", ManipulatorAdapter)
    initial = client.read_joint_positions()
    assert initial[-1] == pytest.approx(850)
    target = initial.copy()
    target[0] = 0.25
    target[-1] = 0
    assert client.write_joint_positions(target)
    for _ in range(300):
        world.step()
    result = client.read_joint_positions()
    assert result[0] == pytest.approx(0.25, abs=0.025)
    assert result[-1] < 100
    world.reset()
    world.step()
    assert client.read_joint_positions()[0] == pytest.approx(0, abs=0.005)


def test_failed_action_read_holds_the_last_command(runtime, device, monkeypatch, caplog):
    world = runtime(XARM7, "arm", "workbench.xml", (0, 0, 0.12))
    client = device(world, "arm", ManipulatorAdapter)
    target = client.read_joint_positions()
    target[0] = 0.25
    assert client.write_joint_positions(target)
    world.step()
    binding = world.robots["arm"]
    commanded = world.data.ctrl[binding.actuators].copy()

    def unreadable(**_):
        raise IncoherentFrameError("action", 20, "the frame was being rewritten")

    monkeypatch.setattr(binding.channel, "read_action", unreadable)
    runtime_logger = logging.getLogger("dimos/sim2/runtime.py")
    runtime_logger.addHandler(caplog.handler)
    try:
        for _ in range(5):
            world.step()
    finally:
        runtime_logger.removeHandler(caplog.handler)
    # Position-controlled joints keep the commanded target, not the home pose.
    assert world.data.ctrl[binding.actuators] == pytest.approx(commanded)
    assert binding.action_read_failures == 5
    warnings = [r for r in caplog.records if "holding its last command" in str(r.msg)]
    assert len(warnings) == 1
    monkeypatch.undo()
    world.step()
    assert binding.action_read_failures == 5


def test_closed_action_channel_still_stops_physics(runtime):
    world = runtime(XARM7, "arm", "workbench.xml", (0, 0, 0.12))
    world.robots["arm"].channel.close()
    with pytest.raises(RuntimeError, match="shared memory buffer is closed"):
        world.step()


def test_arm_respawn_does_not_recompile(runtime):
    world = runtime(XARM7, "arm", "workbench.xml", (0, 0, 0.12))
    model = world.model
    world.set_spawn("arm", (1, 2, 0.8), (0, 0, 0.5))
    assert world.model is model
    assert world.data.xpos[world.robots["arm"].root] == pytest.approx((1, 2, 0.8))


def test_reset_immediately_publishes_new_epoch(runtime):
    world = runtime(XARM7, "arm", "workbench.xml", (0, 0, 0.12))
    world.step()
    before = world.snapshots.read_observation()
    world.reset()
    after = world.snapshots.read_observation()
    assert before is not None and after is not None
    assert after.metadata.episode_id == before.metadata.episode_id + 1
    assert after.metadata.sequence > before.metadata.sequence
    assert after.metadata.sim_time == 0


def test_arm_deactivate_is_idempotent(runtime, device):
    world = runtime(XARM7, "arm", "workbench.xml", (0, 0, 0.12))
    client = device(world, "arm", ManipulatorAdapter)
    assert client.deactivate()
    world.step()
    assert client.sample().values["enabled"].tolist() == [0]
    client.disconnect()
    assert client.deactivate()


def test_worker_snapshot_renders_scene_and_lidar(runtime, tmp_path):
    world = runtime(G1_GROOT, "g1", "logistics.xml", (0, 0, 0.793))
    model_path = tmp_path / "world.mjb"
    mujoco.mj_saveModel(world.model, str(model_path), None)
    reader = WorldReader(
        {"model": str(model_path), "snapshot": world.snapshot_descriptor.to_dict()}
    )
    renderer = None
    try:
        assert reader.update()
        camera = reader.model.camera("g1/front_camera").id
        renderer = MujocoCamera(reader.model, 320, 240)
        rgb, depth = renderer.capture(reader.data, camera, True)
        assert rgb.std() > 15
        assert depth is not None
        assert np.count_nonzero((depth > 1) & (depth < 10)) > 1000
        lidar = next(s for s in G1_GROOT.sensors if s.name == "lidar")
        site = reader.model.site("g1/mid360_link").id
        pattern = lidar.model(**lidar.model_kwargs)
        rays = pattern.directions() @ reader.data.site_xmat[site].reshape(3, 3).T
        reader.model.geom_group[:] = 5
        raycaster = Raycaster(reader.model, reader.model.body("g1/pelvis").id)
        points = raycaster.cast(reader.data, reader.data.site_xpos[site], rays, 0.15, 30)
        assert len(points) > 1000
        assert np.isfinite(points).all()
        assert np.count_nonzero(np.abs(points[:, 2]) < 0.01) > 100
    finally:
        if renderer is not None:
            renderer.close()
        reader.close()


def test_two_robots_have_independent_channels_and_mounts():
    world = SimulationRuntime(
        WorldConfig(
            scene_path(None, "logistics.xml"),
            {
                "left": RobotInstance(XARM7, xyz=(0, 0, 0.12)),
                "right": RobotInstance(XARM7, xyz=(2, 0, 0.12)),
            },
        ),
        uuid4().hex,
    )
    try:
        assert (
            world.robots["left"].channel.descriptor.shm_name
            != world.robots["right"].channel.descriptor.shm_name
        )
        left = world.model.camera("left/wrist_camera").id
        right = world.model.camera("right/wrist_camera").id
        assert world.data.cam_xpos[right] - world.data.cam_xpos[left] == pytest.approx((2, 0, 0))
    finally:
        world.close()
