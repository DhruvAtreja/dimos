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

import time
from types import SimpleNamespace

import numpy as np
import pytest

from dimos.core.global_config import global_config
from dimos.e2e_tests.dimos_cli_call import DimosCliCall
from dimos.evals.environments.lib.sim_truth import first_entity_pose, last_entity_pose
from dimos.evals.environments.sim2 import Sim2Environment
from dimos.memory.store.memory import MemoryStore
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.sim2.scene_types import EntityState, SceneState


def environment(**kwargs):
    kwargs.setdefault("scene", "/scenes/xarm_table")
    return Sim2Environment(
        blueprint=["xarm7-planner-coordinator", "mcp-server", "manipulation-skills"], **kwargs
    )


def truth(ts: float, **z_by_entity: float) -> SceneState:
    """A sim_truth row with each named entity at (0.4, 0.0, z)."""
    return SceneState(
        world_id="xarm7",
        scene_id="xarm_table",
        generation=0,
        tick=int(ts * 200),
        sim_time=ts,
        ts=ts,
        entities={
            name: EntityState(
                pose=Pose((0.4, 0.0, z), (0.0, 0.0, 0.0, 1.0)),
                velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                bounds_min=(0.35, -0.05, z - 0.05),
                bounds_max=(0.45, 0.05, z + 0.05),
            )
            for name, z in z_by_entity.items()
        },
        robots={},
        joints={},
        regions={},
        contacts=(),
    )


def test_launch_flags(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    proc = DimosCliCall()
    environment(scene_spawn="0.1, 0.2, 0.3, 0.4").configure_launch(proc)
    assert proc.simulator == "mujoco"
    assert proc.global_args == [
        "--transport",
        "zenoh",
        "--scene-package",
        "/scenes/xarm_table",
        "--record-topics",
        "color_image,camera_info,coordinator_joint_state,tf,sim_truth",
        "--scene-spawn",
        "0.1, 0.2, 0.3, 0.4",
    ]
    assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "false"
    assert proc.extra_env["MUJOCO_GL"] == "egl"

    proc = DimosCliCall()
    environment(scene="kitchen", module_env={"SIMULATIONMODULE__VIEWER": "true"}).configure_launch(
        proc
    )
    assert "--scene-spawn" not in proc.global_args
    assert proc.global_args[2:4] == ["--scene-package", "kitchen"]
    assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "true"


def test_the_shell_opens_the_simulator_window(shell):
    shell(SIMULATIONMODULE__VIEWER="true")
    proc = DimosCliCall()
    environment().configure_launch(proc)
    assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "true"
    # A case's own module_env still decides over the shell.
    proc = DimosCliCall()
    environment(module_env={"SIMULATIONMODULE__VIEWER": "false"}).configure_launch(proc)
    assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "false"


def test_preflight_needs_the_zenoh_bus(monkeypatch):
    agent = SimpleNamespace(config=SimpleNamespace(modules=[]))
    monkeypatch.setattr(global_config, "transport", "lcm")
    with pytest.raises(RuntimeError, match="zenoh"):
        environment().preflight(agent)


def test_setup_scene_enables_truth_on_the_running_simulation(monkeypatch):
    calls: list[bool] = []
    stopped: list[bool] = []
    app = SimpleNamespace(
        get_module=lambda name: SimpleNamespace(set_truth_enabled=calls.append),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setattr("dimos.porcelain.dimos.Dimos.connect", lambda: app)
    environment().setup_scene()
    assert calls == [True] and stopped == [True]


def test_truth_helper_reads_first_and_last_entity_pose():
    with MemoryStore() as store:
        with pytest.raises(LookupError, match="sim_truth"):
            first_entity_pose(store, "cup")
        stream = store.stream("sim_truth", SceneState)
        stream.append(truth(1.0, cup=0.19, apple=0.17), ts=1.0)
        stream.append(truth(2.0, cup=0.25, apple=0.17), ts=2.0)
        stream.append(truth(3.0, cup=0.31, apple=0.17), ts=3.0)
        assert first_entity_pose(store, "cup").position.z == pytest.approx(0.19)
        assert last_entity_pose(store, "cup").position.z == pytest.approx(0.31)
        assert last_entity_pose(store, "apple").position.z == pytest.approx(0.17)
        with pytest.raises(LookupError, match="orange"):
            last_entity_pose(store, "orange")


def test_ready_needs_fresh_streams_and_truth_entities():
    env = environment(truth_entities=("cup",))
    with MemoryStore() as store:
        with pytest.raises(TimeoutError, match="cup"):
            env.wait_ready(store, deadline=time.monotonic() + 0.3)
        now = time.time()
        store.stream("color_image", Image).append(
            Image(data=np.zeros((1, 1, 3), dtype=np.uint8), format=ImageFormat.RGB, ts=now)
        )
        store.stream("coordinator_joint_state", JointState).append(
            JointState(ts=now, name=["j1"], position=[0.0], velocity=[0.0])
        )
        store.stream("sim_truth", SceneState).append(truth(now, apple=0.17), ts=now)
        with pytest.raises(TimeoutError, match="cup"):
            env.wait_ready(store, deadline=time.monotonic() + 0.3)
        later = now + 0.1
        store.stream("sim_truth", SceneState).append(truth(later, apple=0.17, cup=0.19), ts=later)
        env.wait_ready(store, deadline=time.monotonic() + 2.0)


def test_latest_pose_needs_odom():
    env = environment()
    with MemoryStore() as store:
        with pytest.raises(LookupError):
            env.latest_pose(store)
        store.stream("odom", PoseStamped).append(PoseStamped(ts=5, frame_id="world"))
        assert env.latest_pose(store).ts == 5


def test_launch_and_cleanup(tmp_path, mocker):
    proc = mocker.patch("dimos.evals.environments.sim.DimosCliCall").return_value
    proc.extra_env = {}
    proc.global_args = []
    mocker.patch(
        "dimos.evals.environments.sim.McpAdapter"
    ).return_value.wait_for_ready.return_value = True
    store = mocker.patch("dimos.memory.store.sqlite.SqliteStore").return_value
    env = environment(truth_entities=("apple",))
    mocker.patch.object(env, "_wait_recording", return_value=tmp_path / "memory.db")
    truth_on = mocker.patch.object(env, "setup_scene")
    ready = mocker.patch.object(env, "wait_ready")
    try:
        result = env.start(("observe-skill",))
        assert proc.global_args[:2] == ["--transport", "zenoh"]
        assert proc.global_args[-1] == "--record"
        assert proc.demo_args == [
            "run",
            "xarm7-planner-coordinator",
            "mcp-server",
            "manipulation-skills",
            "observe-skill",
            "--disable",
            "rerun-bridge-module",
        ]
        # With no viewer settings in the shell: Rerun off, simulator window shut.
        assert proc.global_args[-5:-1] == ["--viewer", "none", "--rerun-open", "none"]
        assert proc.extra_env["SIMULATIONMODULE__VIEWER"] == "false"
        assert proc.extra_env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        truth_on.assert_called_once()
        ready.assert_called_once()
        assert set(result.artifacts) == {"recording"}
        assert env.config.agent_artifacts == (), "the agent must not be handed the recording"
    finally:
        env.stop()
    proc.stop.assert_called_once()
    store.stop.assert_called_once()
