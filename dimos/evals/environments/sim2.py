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

"""sim2 launch, ground truth, readiness and settling for live manipulation evals."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field

from dimos.core.coordination.blueprint_config.sources import configuration_environment
from dimos.core.global_config import global_config
from dimos.evals.environments.lib.sim_truth import last_entity_pose
from dimos.evals.environments.sim import Sim, SimConfig

if TYPE_CHECKING:
    from pathlib import Path

    from dimos.e2e_tests.dimos_cli_call import DimosCliCall
    from dimos.evals.agents.base import Agent
    from dimos.memory.store.base import Store
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped


class Sim2EnvironmentConfig(SimConfig):
    # A scene name from the sim2 data package, or an absolute path to a scene.xml
    # or to the directory holding it.
    scene: str
    # Where the robot base goes, as "x, y, z, yaw" in metres and radians, world
    # frame. Unset, the blueprint uses the scene's named spawn.
    scene_spawn: str | None = None
    # Scene entity ids (from the scene's scene.json) that sim_truth must report
    # before the case starts.
    truth_entities: tuple[str, ...] = ()
    at_rest_rad_s: float = 0.02
    module_env: dict[str, str] = Field(default_factory=dict)
    # The recording holds the true entity poses on sim_truth, which the agent is
    # meant to find through the camera, so only the grader opens it.
    agent_artifacts: tuple[str, ...] = ()
    ready_streams: tuple[str, ...] = ("color_image", "coordinator_joint_state", "sim_truth")
    recorded_topics: tuple[str, ...] = (
        "color_image",
        "camera_info",
        "coordinator_joint_state",
        "tf",
        "sim_truth",
    )


class Sim2Environment(Sim):
    """Run agent evaluations in a sim2 scene, with ground-truth object poses recorded on
    ``sim_truth``.

    The agent is never handed the recording: it holds the true world poses of every
    scene entity, which the agent is meant to find through the camera.
    """

    config: Sim2EnvironmentConfig

    def preflight(self, agent: Agent) -> None:
        # Truth is switched on through the running dimos's bus, so this process
        # must be on the same transport as the one it launches.
        if global_config.transport != "zenoh":
            raise RuntimeError(
                "Sim2Environment needs this process on the zenoh bus to enable sim_truth; "
                "run with --transport zenoh or DIMOS_TRANSPORT=zenoh"
            )
        super().preflight(agent)

    def configure_launch(self, proc: DimosCliCall) -> None:
        proc.simulator = "mujoco"
        proc.global_args = [
            "--transport",
            "zenoh",
            "--scene-package",
            self.config.scene,
            "--record-topics",
            ",".join(self.config.recorded_topics),
        ]
        if self.config.scene_spawn:
            proc.global_args += ["--scene-spawn", self.config.scene_spawn]
        proc.extra_env.update(self.config.module_env)
        # The simulator's own window stays shut unless the shell or .env sets
        # SIMULATIONMODULE__VIEWER=true.
        proc.extra_env.setdefault(
            "SIMULATIONMODULE__VIEWER",
            configuration_environment(None).get("SIMULATIONMODULE__VIEWER", "false"),
        )
        # Without an X display MuJoCo can only render off-screen through EGL.
        if "DISPLAY" not in os.environ:
            proc.extra_env.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

    def setup_scene(self) -> None:
        """Switch on the privileged ``sim_truth`` stream; sim2 keeps it off by default.

        MCP answers before the coordinator is on the bus, so connecting is retried until
        the launch timeout.
        """
        from dimos.porcelain.dimos import Dimos

        deadline = time.monotonic() + self.config.launch_timeout_s
        while True:
            try:
                app = Dimos.connect()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)
        try:
            simulation: Any = app.get_module("SimulationModule")  # handle type depends on imports
            simulation.set_truth_enabled(True)
        finally:
            app.stop()

    def prepare_recording(self, recording: Store, path: Path, deadline: float) -> dict[str, Path]:
        self.wait_ready(recording, deadline=deadline)
        return {}

    def wait_ready(self, recording: Store, *, deadline: float) -> None:
        """Wait for fresh samples on every ready stream and a truth pose for every entity."""
        while time.monotonic() < deadline:
            try:
                ages = [
                    time.time() - getattr(recording.streams, name).last().data.ts
                    for name in self.config.ready_streams
                    if name in recording.streams
                ]
                for entity in self.config.truth_entities:
                    last_entity_pose(recording, entity)
            except (LookupError, AttributeError):
                ages = []
            if len(ages) == len(self.config.ready_streams) and all(age < 10.0 for age in ages):
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"sim2 did not publish fresh {', '.join(self.config.ready_streams)}"
            + (
                f" and truth for {', '.join(self.config.truth_entities)}"
                if self.config.truth_entities
                else ""
            )
        )

    def latest_pose(self, recording: Store) -> PoseStamped:
        """Base pose from ``odom`` for floating-base robots; a fixed-base arm has none."""
        if "odom" not in recording.streams:
            raise LookupError("No sim2 odometry recorded")
        return cast("PoseStamped", recording.streams.odom.last().data)

    def settle(self, budget_s: float) -> None:
        """At rest once every joint is slower than ``at_rest_rad_s`` for ``at_rest_s``."""
        recording = self._recording
        if recording is None or "coordinator_joint_state" not in recording.streams:
            super().settle(budget_s)
            return
        rest_since: float | None = None
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            try:
                state = recording.streams.coordinator_joint_state.last().data
            except LookupError:
                return
            now = time.monotonic()
            if any(abs(v) > self.config.at_rest_rad_s for v in state.velocity):
                rest_since = None
            elif rest_since is None:
                rest_since = now
            elif now - rest_since >= self.config.at_rest_s:
                return
            time.sleep(self.config.settle_poll_s)
