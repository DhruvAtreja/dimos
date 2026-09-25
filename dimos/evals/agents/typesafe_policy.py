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

"""TypeSafe (Jev) as a closed-loop manipulation policy for a fixed-base arm.

The model plans nothing and calls no skill itself. Each tick: code assembles
the state (the end-effector pose in the world, every object's position in the
end-effector frame, and the static scene: table, floor and objects as they are
in the MuJoCo model), Jev picks one command (a fixed step along one of the
end-effector's own axes, open or close the gripper, or stop), and code runs it
as one blocking planner call over MCP (``move_to_pose`` / ``set_gripper``).
Poses come live off ``/tf``: the simulator publishes the tracked bodies, the
planner publishes its tip link, and both are in the same ``world`` frame.

Two contracts are meant to be edited: :class:`ArmState` (what Jev sees) and
:data:`STEP_CRITERIA` / :func:`build_questions` (what Jev answers). Everything
else is plumbing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from dimos.agents.mcp.mcp_adapter import McpAdapter
from dimos.evals.agents.base import Agent, AgentConfig
from dimos.evals.agents.lib.trajectory_builder import TrajectoryBuilder
from dimos.evals.types import EndedBy, Metrics, RunningEnvironment, ToolCall, Trajectory
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.tf2_msgs.TFMessage import TFMessage

if TYPE_CHECKING:
    from dimos.evals.environments.base import Environment

DEFAULT_SCENE = Path(__file__).parents[1] / "suites" / "scenes" / "xarm7_table.json"

# --- input contract: what Jev sees ------------------------------------------
# Positions are raw metres. The end effector is described in the world frame,
# every object in the end-effector frame, so "where is it relative to my
# gripper" needs no arithmetic from the model. The static scene block is the
# scene file verbatim: what an eye-in-hand camera would see, but as ground
# truth rather than pixels.


@dataclass(frozen=True, kw_only=True)
class ObjectInEE:
    label: str
    shape: str
    size_m: dict[str, float]
    xyz_ee: tuple[float, float, float]  # metres, in the end-effector frame


@dataclass(frozen=True, kw_only=True)
class ArmState:
    frame_id: str = "world"
    ee_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Last commanded opening, 0 closed .. 1 open; None until the first gripper command.
    gripper_open_fraction: float | None = None
    goal_label: str = ""
    objects_in_ee_frame: list[ObjectInEE] | None = None
    scene: dict[str, Any] | None = None  # the scene file, verbatim
    ticks_elapsed: int = 0

    def encode(self) -> dict[str, Any]:
        """The ``state`` payload sent to system_one."""
        return asdict(self)


# --- output contract: what Jev answers ---------------------------------------
# Moves are unit vectors in the end-effector frame, the frame the objects are
# reported in: the pick IS the direction, scaled by step_m. Code rotates it
# into the world and asks the planner for that pose; the wrist orientation is
# held, so the axes mean the same thing every tick.

STEPS: dict[str, tuple[float, float, float]] = {  # end-effector frame unit vectors
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
GRIPPER: dict[str, float] = {"open": 1.0, "close": 0.0}  # set_gripper fraction
STOP = "stop"

STEP_CRITERIA: dict[str, str] = {
    "+x": "Move the end effector one step along its own +x axis.",
    "-x": "Move the end effector one step along its own -x axis.",
    "+y": "Move the end effector one step along its own +y axis.",
    "-y": "Move the end effector one step along its own -y axis.",
    "+z": "Move one step along the end effector's +z axis, the direction the gripper points.",
    "-z": "Move one step along the end effector's -z axis, backing the gripper away.",
    "open": "Open the gripper.",
    "close": "Close the gripper on what is between its fingers.",
    STOP: "Do nothing this tick.",
}


def build_questions() -> dict[str, Any]:
    """One fan-out call per tick: the command plus the termination flag."""
    from typesafe_sdk import Choice, Noul

    return {
        "step": Choice(
            instructions=(
                "Which single command makes progress on the instruction. Object positions "
                "are in the end-effector frame in metres: an object at (0, 0, 0.05) is 5 cm "
                "ahead along the direction the gripper points, one at (0.1, 0, 0) is 10 cm "
                "along the gripper's +x axis. Every move is one fixed step. Close the "
                "gripper only once the goal object sits between the fingers, then carry it."
            ),
            criteria=STEP_CRITERIA,
        ),
        "reached": Noul(
            instructions="The goal object is held in the gripper, lifted off the table, "
            "and the instruction is complete"
        ),
    }


# --- scene file --------------------------------------------------------------
# dimos/evals/suites/scenes/xarm7_table.json: data/xarm7/scene.xml transcribed
# by hand (table top, floor, pedestal, and each free body with its shape, size
# and start position). Every object must be a MujocoEnvironment tracked body,
# since its live pose comes from tf.


@dataclass(frozen=True, kw_only=True)
class SceneObject:
    label: str
    aliases: tuple[str, ...]
    shape: str
    size_m: dict[str, float]
    start_xyz: tuple[float, float, float]


@dataclass(frozen=True, kw_only=True)
class Scene:
    raw: dict[str, Any]
    objects: list[SceneObject]

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.objects]


def load_scene(path: Path) -> Scene:
    raw = json.loads(Path(path).expanduser().read_text())
    objects = [
        SceneObject(
            label=str(o["label"]),
            aliases=tuple(str(a) for a in o.get("aliases", ())),
            shape=str(o.get("shape", "")),
            size_m={k: float(v) for k, v in o.items() if k.endswith("_m")},
            start_xyz=tuple(float(v) for v in o["start_xyz"]),  # type: ignore[arg-type]
        )
        for o in raw.get("objects", [])
    ]
    if not objects:
        raise ValueError(f"{path} lists no objects")
    return Scene(raw=raw, objects=objects)


# Instruction words that are never an object's name but could sit inside one.
_STOP_WORDS = frozenset({"air", "and", "hold", "into", "onto", "pick", "the", "then", "up"})


def derive_goal_label(instruction: str, scene: Scene) -> str:
    """The object whose label or alias the instruction names, by its label.

    Jev does no language grounding here: "pick up the cylinder" resolves to
    the ``cup`` body through its alias, in code, so the goal in the state is
    always a real tracked body. The longest matching word wins.
    """
    words = [w for w in re.findall(r"[a-z][a-z-]{2,}", instruction.lower()) if w not in _STOP_WORDS]
    hits: list[tuple[str, str]] = []
    for obj in scene.objects:
        names = {obj.label.lower(), *(a.lower() for a in obj.aliases)}
        for word in words:
            if any(word == name or word in name.split() for name in names):
                hits.append((word, obj.label))
    if not hits:
        raise LookupError(
            f"no word of {instruction!r} names a scene object; "
            f"use one of {sorted({n for o in scene.objects for n in (o.label, *o.aliases)})}"
        )
    return max(hits, key=lambda h: len(h[0]))[1]


# --- the agent ---------------------------------------------------------------


class TypeSafePolicyConfig(AgentConfig):
    model: str = "jev-latest"  # typesafe_sdk.constants.DEFAULT_MODEL
    scene_json: Path = DEFAULT_SCENE
    goal_label: str = ""  # a scene object's label; empty derives it from the instruction
    max_ticks: int = 40
    # The SDK default is 10s; a tick that blocks longer than this is dead time.
    request_timeout_s: float = 10.0
    step_m: float = 0.03  # one move along an end-effector axis
    settle_s: float = 0.3  # after a planner call returns, let tf catch up
    lift_m: float = 0.05  # goal body this far above where it started ends the episode
    reached_noul: float = 0.8
    # The environment should already have waited for the sim; this is a backstop.
    pose_wait_s: float = 60.0
    # The planner's tip link on tf: the frame move_to_pose targets and reports.
    ee_frame: str = "link_tcp"
    tf_topic: str = "/tf"
    # Open once before the first tick so the gripper state is known.
    open_gripper_first: bool = True
    # Wrist orientation (roll, pitch, yaw, radians) sent with every move, so the
    # end-effector axes mean the same thing every tick; the default points the
    # gripper straight down. None keeps whatever the arm has, which drifts a few
    # degrees per planned move.
    rpy_rad: tuple[float, float, float] | None = (math.pi, 0.0, 0.0)


class TypeSafePolicy(Agent):
    """Drive a fixed-base arm with Jev decisions, one planner call per tick."""

    config: TypeSafePolicyConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tf: dict[str, Transform] = {}
        self._wanted: frozenset[str] = frozenset()
        self._tf_seen = threading.Event()
        self._gripper: float | None = None

    def preflight(self, environment: Environment) -> None:
        from dimos.evals.environments.mujoco_sim import MujocoEnvironment

        if not isinstance(environment, MujocoEnvironment):
            raise ValueError(
                f"TypeSafePolicy drives a MuJoCo arm, not {type(environment).__name__}"
            )
        if not self.config.scene_json.is_file():
            raise FileNotFoundError(f"scene_json not found: {self.config.scene_json}")
        scene = load_scene(self.config.scene_json)
        untracked = [o for o in scene.labels if o not in environment.config.tracked_bodies]
        if untracked:
            raise ValueError(
                f"scene objects {untracked} are not in the environment's tracked_bodies; "
                "the policy reads their poses from tf"
            )
        if self.config.goal_label and self.config.goal_label not in scene.labels:
            raise LookupError(f"goal_label {self.config.goal_label!r} is not one of {scene.labels}")

    def run(
        self, inputs: str, env: RunningEnvironment, run_dir: Path, *, timeout_s: float
    ) -> Trajectory:
        from typesafe_sdk import TypeSafeClient

        from dimos.core.transport_factory import make_transport

        raw = run_dir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        trajectory = TrajectoryBuilder(inputs, name=type(self).__name__, model=self.config.model)

        scene = load_scene(self.config.scene_json)
        goal = self.config.goal_label or derive_goal_label(inputs, scene)
        # api_key comes from TYPESAFE_API_KEY (typesafe_sdk.constants.API_KEY_ENV).
        client = TypeSafeClient(model=self.config.model, timeout=self.config.request_timeout_s)
        questions = build_questions()
        tools = McpAdapter(env.mcp_url)

        self._wanted = frozenset({self.config.ee_frame, *scene.labels})
        tf = make_transport(self.config.tf_topic, TFMessage)
        tf.start()
        tf.subscribe(self._on_tf)

        deadline = time.monotonic() + timeout_s
        ended: EndedBy = "max_steps"
        try:
            if not self._tf_seen.wait(min(self.config.pose_wait_s, timeout_s)):
                missing = sorted(self._wanted - self._tf.keys())
                raise TimeoutError(f"no tf for {missing} on {self.config.tf_topic}")
            if self.config.open_gripper_first:
                self._set_gripper(tools, GRIPPER["open"])
            start_z = self._tf[goal].translation.z
            for tick in range(self.config.max_ticks):
                if time.monotonic() >= deadline:
                    ended = "timeout"
                    break
                ee, state = self.observe(scene, goal, tick)
                if self._tf[goal].translation.z - start_z >= self.config.lift_m:
                    ended = "answer"  # the grader's own criterion, met
                    break

                started = time.time()
                response = client.system_one(state=state.encode(), questions=questions)
                action = self.act(tools, response.answers["step"], ee)
                self._trace(trajectory, raw, tick, state, response, action, started)

                if _noul(response.answers["reached"]) > self.config.reached_noul:
                    ended = "answer"
                    break
                time.sleep(self.config.settle_s)
        finally:
            tf.stop()
        return trajectory.build(ended)

    def _on_tf(self, message: TFMessage) -> None:
        """Keep the newest world transform of every frame the state needs."""
        for transform in message.transforms:
            if transform.child_frame_id in self._wanted:
                self._tf[transform.child_frame_id] = transform
        if self._wanted <= self._tf.keys():
            self._tf_seen.set()

    def observe(self, scene: Scene, goal_label: str, tick: int) -> tuple[Transform, ArmState]:
        """Live tf + the static scene -> the state Jev sees."""
        ee = self._tf.get(self.config.ee_frame)
        if ee is None:
            raise LookupError(f"no tf for {self.config.ee_frame} yet")
        world_to_ee = ee.inverse()  # ee <- world
        objects = []
        for obj in scene.objects:
            world_obj = self._tf.get(obj.label)
            if world_obj is None:
                raise LookupError(f"no tf for {obj.label} yet")
            in_ee = world_to_ee.apply(world_obj)
            objects.append(
                ObjectInEE(
                    label=obj.label,
                    shape=obj.shape,
                    size_m=obj.size_m,
                    xyz_ee=_xyz(in_ee.translation),
                )
            )
        rpy = ee.rotation.to_euler()
        return ee, ArmState(
            frame_id=ee.frame_id,
            ee_xyz=_xyz(ee.translation),
            ee_rpy_deg=(
                round(math.degrees(rpy.x), 1),
                round(math.degrees(rpy.y), 1),
                round(math.degrees(rpy.z), 1),
            ),
            gripper_open_fraction=self._gripper,
            goal_label=goal_label,
            objects_in_ee_frame=objects,
            scene=scene.raw,
            ticks_elapsed=tick,
        )

    def target_for(self, step_key: str, ee: Transform) -> Vector3:
        """Jev's pick is a unit vector in the end-effector frame; rotate it into the world."""
        ux, uy, uz = STEPS[step_key]
        delta_ee = Vector3(ux, uy, uz) * self.config.step_m
        return ee.translation + ee.rotation.rotate_vector(delta_ee)

    def act(self, tools: McpAdapter, step: Any, ee: Transform) -> dict[str, Any]:
        """Run the pick as one blocking skill call; what happened, for the trace.

        No confidence threshold: ``choice`` is the argmax and it is applied as
        is. ``stop`` does nothing; only the episode ending stops the arm.
        """
        key = str(step.choice)
        if key == STOP:
            return {"choice": key, "tool": None, "arguments": {}, "ok": True, "message": ""}
        if key in GRIPPER:
            ok, message = self._set_gripper(tools, GRIPPER[key])
            return {
                "choice": key,
                "tool": "set_gripper",
                "arguments": {"position": GRIPPER[key]},
                "ok": ok,
                "message": message,
            }
        target = self.target_for(key, ee)
        arguments: dict[str, Any] = {
            "x": round(target.x, 4),
            "y": round(target.y, 4),
            "z": round(target.z, 4),
        }
        if self.config.rpy_rad is not None:
            roll, pitch, yaw = self.config.rpy_rad
            arguments.update({"roll": roll, "pitch": pitch, "yaw": yaw})
        ok, message = _skill_outcome(tools.call_tool_text("move_to_pose", arguments))
        return {
            "choice": key,
            "tool": "move_to_pose",
            "arguments": arguments,
            "ok": ok,
            "message": message,
        }

    def _set_gripper(self, tools: McpAdapter, position: float) -> tuple[bool, str]:
        ok, message = _skill_outcome(tools.call_tool_text("set_gripper", {"position": position}))
        if ok:
            self._gripper = position
        return ok, message

    def _trace(
        self,
        trajectory: TrajectoryBuilder,
        raw: Path,
        tick: int,
        state: ArmState,
        response: Any,
        action: dict[str, Any],
        started: float,
    ) -> None:
        """Persist the call and record one ATIF step.

        The runner counts ``raw/NNN-request.json`` for request_attempts, so the
        naming matters.
        """
        request_path = raw / f"{tick:03d}-request.json"
        response_path = raw / f"{tick:03d}-response.json"
        request_path.write_text(
            json.dumps({"body": {"state": state.encode()}, "started_at": started}, indent=2)
        )
        answers = {name: _answer_json(a) for name, a in response.answers.items()}
        response_path.write_text(
            json.dumps(
                {"model": response.model, "answers": answers, "action": action},
                indent=2,
                default=str,
            )
        )
        usage = response.usage
        tool_calls: tuple[ToolCall, ...] = ()
        if action["tool"]:
            tool_calls = (
                ToolCall(
                    tool_call_id=f"tick-{tick:03d}",
                    function_name=action["tool"],
                    arguments=dict(action["arguments"]),
                ),
            )
        trajectory.step(
            message=json.dumps(answers),
            request=request_path,
            response=response_path,
            tool_calls=tool_calls,
            # What actually ran, as reported by the provider.
            model_name=str(response.model),
            metrics=Metrics(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            ),
            at=started,
            latency_s=time.time() - started,
        )


def _skill_outcome(text: str) -> tuple[bool, str]:
    """A SkillResult over MCP is JSON with ``success`` and ``message``; anything else failed."""
    try:
        payload = json.loads(text)
    except ValueError:
        return False, text
    if not isinstance(payload, dict) or "success" not in payload:
        return False, text
    return bool(payload["success"]), str(payload.get("message", ""))


def _noul(answer: Any) -> float:
    """The Noul answer's value; other answer kinds have none."""
    return float(getattr(answer, "noul", 0.0))


def _answer_json(answer: Any) -> dict[str, Any]:
    """Flatten a Jev answer for the trace; graders read these back."""
    return {
        name: getattr(answer, name)
        for name in ("choice", "score", "noul", "confidence", "probabilities")
        if getattr(answer, name, None) is not None
    }


def _xyz(v: Vector3) -> tuple[float, float, float]:
    return (round(v.x, 3), round(v.y, 3), round(v.z, 3))
