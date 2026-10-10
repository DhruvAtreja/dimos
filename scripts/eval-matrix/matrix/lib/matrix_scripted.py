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

"""A pick-and-place agent with no model, for the matrix's grasp bench.

Every trial runs the same fixed steps: scan for the case's target, pick it, and if
it is held, place it at the drop-off centre. With no model choosing what to do, a
difference between two configs comes from their grasp code alone.

    PYTHONPATH=/matrix dimos evals run matrix_case --agent matrix_scripted
    ... --agent matrix_scripted --set attempts=3   # pick up to 3 times, re-scanning between

It lives with the matrix instead of on a branch so every branch runs the identical
script. It needs only the scan_objects, pick_object and place_at skills.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field, JsonValue
import requests

from dimos.evals.agents.base import Agent, AgentConfig
from dimos.evals.agents.lib.trajectory_builder import TrajectoryBuilder
from dimos.evals.types import EndedBy, RunningEnvironment, ToolCall, Trajectory

if TYPE_CHECKING:
    from dimos.evals.environments.base import Environment

TOOLS = ("scan_objects", "pick_object", "place_at")
# The SceneReplica case prompt names its target as "Pick up the <label> and place it ...".
_TARGET = re.compile(r"\bPick up the (.+?) and place\b")


class MatrixScriptedConfig(AgentConfig):
    # Most pick_object calls per trial. After a pick that leaves nothing held, the
    # script scans again and picks the newly scanned object. 1 picks once.
    attempts: int = Field(default=1, ge=1)
    # Where place_at puts the held object, world frame, metres: the drop-off square's
    # centre (x = 0.42, y = 0.38), 1.5 cm above the table top (z = 0.745).
    place_xyz: tuple[float, float, float] = (0.42, 0.38, 0.76)


class _OutOfTimeError(Exception):
    """The case's time ran out before the next call."""


@dataclass
class _Reply:
    """One MCP tool reply: its text, the skill's message and metadata, and whether
    MCP flagged it as failed."""

    text: str
    message: str
    metadata: dict[str, Any]
    failed: bool


class MatrixScripted(Agent):
    """Scan, pick (up to ``attempts`` times) and place, through the environment's MCP server."""

    config: MatrixScriptedConfig

    def preflight(self, environment: Environment) -> None:
        if not environment.has_robot:
            raise RuntimeError(
                f"MatrixScripted needs a robot; {type(environment).__name__} has none"
            )

    def available_tools(self, environment_tools: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in TOOLS if name in environment_tools)

    def run(
        self, inputs: str, env: RunningEnvironment, run_dir: Path, *, timeout_s: float
    ) -> Trajectory:
        from dimos.agents.mcp.mcp_adapter import McpAdapter

        deadline = time.monotonic() + timeout_s
        calls_dir = run_dir / "calls"
        calls_dir.mkdir(parents=True, exist_ok=True)
        trajectory = TrajectoryBuilder(inputs, name=type(self).__name__, model="scripted")
        script: dict[str, JsonValue] = {}
        made: list[str] = []

        def call(tool: str, arguments: dict[str, Any]) -> _Reply:
            """Call one MCP tool and record the call and its reply as a step."""
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _OutOfTimeError(tool)
            made.append(tool)
            call_id = f"call_{len(made)}"
            request = calls_dir / f"{len(made):02d}-{tool}-call.json"
            response = calls_dir / f"{len(made):02d}-{tool}-result.json"
            request.write_text(json.dumps({"tool": tool, "arguments": arguments}))
            started = time.monotonic()
            try:
                result = McpAdapter(env.mcp_url, timeout=math.ceil(remaining)).call_tool(
                    tool, arguments
                )
            except requests.Timeout as exc:
                raise _OutOfTimeError(tool) from exc
            latency = time.monotonic() - started
            response.write_text(json.dumps({"result": result, "seconds": latency}))
            reply = _reply(result)
            trajectory.step(
                message="",
                request=request,
                response=response,
                tool_calls=(
                    ToolCall(tool_call_id=call_id, function_name=tool, arguments=arguments),
                ),
                latency_s=latency,
            )
            trajectory.observe(call_id, reply.text)
            return reply

        ended_by: EndedBy = "answer"
        error = ""
        try:
            summary = self._script(inputs, call, script)
        except _OutOfTimeError as exc:
            ended_by, summary = "timeout", f"Out of time before {exc}."
        except Exception as exc:
            ended_by, error = "error", f"{type(exc).__name__}: {exc}"
            summary = error
        final = calls_dir / "summary.json"
        final.write_text(json.dumps({"summary": summary, "scripted": script}))
        trajectory.step(message=summary, request=final, response=final)
        # Newer branches can carry agent details into the results row; older ones cannot.
        if "details" in inspect.signature(trajectory.build).parameters:
            return trajectory.build(ended_by, error=error, details={"scripted": script})
        return trajectory.build(ended_by, error=error)

    def _script(self, inputs: str, call: Any, script: dict[str, JsonValue]) -> str:
        """Scan, pick and place, filling ``script``; returns the trial's summary."""
        found = _TARGET.search(inputs)
        if found is None:
            raise ValueError("the case prompt names no target as 'Pick up the <label> and place'")
        label = found.group(1)
        script["label"] = label
        picks: list[JsonValue] = []
        script["picks"] = picks

        pick = None
        for attempt in range(1, self.config.attempts + 1):
            started = time.monotonic()
            scan = call("scan_objects", {"prompts": [label]})
            objects = [obj for obj in scan.metadata.get("objects", []) if isinstance(obj, dict)]
            if scan.failed or not objects:
                picks.append({"attempt": attempt, "found": False, "scan": scan.message})
                continue
            named = [obj for obj in objects if obj.get("name") == label]
            object_id = str((named or objects)[0]["object_id"])
            pick = call("pick_object", {"object_id": object_id})
            held = _held(pick)
            picks.append(
                {
                    "attempt": attempt,
                    "object_id": object_id,
                    "held": held,
                    "seconds": round(time.monotonic() - started, 3),
                    "message": pick.message,
                }
            )
            if held:
                break
        else:
            last = pick.message if pick is not None else "the scan found no target"
            return f"Pick of {label} failed after {self.config.attempts} attempt(s): {last}"

        x, y, z = self.config.place_xyz
        started = time.monotonic()
        place = call("place_at", {"x": x, "y": y, "z": z})
        script["place_s"] = round(time.monotonic() - started, 3)
        script["place"] = place.message
        return f"Picked {label}; place: {place.message}"


def _held(pick: _Reply) -> bool:
    """Whether pick_object reports the object held: M1 branches set ``grasped``,
    M0 branches answer "Pick complete"."""
    if pick.failed:
        return False
    grasped = pick.metadata.get("grasped")
    if grasped is not None:
        return grasped is True
    return pick.message.startswith("Pick complete")


def _reply(result: dict[str, Any]) -> _Reply:
    """Read an MCP ``tools/call`` result: a skill's reply is JSON text with
    ``message`` and ``metadata``."""
    content = result.get("content") or []
    text = str(content[0].get("text", "")) if content and isinstance(content[0], dict) else ""
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        body = {"message": text}
    metadata = body.get("metadata")
    return _Reply(
        text=text,
        message=str(body.get("message", "")),
        metadata=metadata if isinstance(metadata, dict) else {},
        failed=bool(result.get("isError")),
    )
