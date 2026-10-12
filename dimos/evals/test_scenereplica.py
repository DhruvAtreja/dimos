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


"""Offline tests for the SceneReplica suite: the weighted scorer, the truth checks, the
failure split, the suite's shape and the summary script."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from dimos.agents.skill_result import SkillResult
from dimos.evals.environments.lib.scene_checks import (
    displacement,
    lifted_by,
    resting_in_region,
    unmoved,
)
from dimos.evals.environments.sim2 import Sim2Environment
from dimos.evals.runner import EvalRunner
from dimos.evals.scorers import weighted
from dimos.evals.suites.lib.scenereplica_summary import Trial, load_trials, main, report
from dimos.evals.types import (
    AgentInfo,
    EvalCase,
    FinalMetrics,
    Graded,
    Observation,
    ObservationResult,
    Outcome,
    RunExtra,
    Step,
    ToolCall,
    Trajectory,
)
from dimos.manipulation.manipulation_skills import ManipulationSkills
from dimos.manipulation.manipulation_spec import (
    ExecutionResult,
    ExecutionStatus,
    PlanResult,
    PlanStatus,
)
from dimos.manipulation.pick_and_place_module import PickAndPlaceModule
from dimos.memory.store.memory import MemoryStore
from dimos.memory.store.sqlite import SqliteStore
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.sim2.scene_types import EntityState, RegionState, SceneState
from dimos.utils.data import get_data_dir

DROPOFF = Pose((0.42, 0.38, 0.745), (0.0, 0.0, 0.0, 1.0))
XYZ = tuple[float, float, float]


def _entity(x: float, y: float, z: float, *, half: float = 0.03, speed: float = 0.0) -> EntityState:
    """A 6 cm cube centred at (x, y, z), moving at ``speed`` along x."""
    return EntityState(
        pose=Pose((x, y, z), (0.0, 0.0, 0.0, 1.0)),
        velocity=(speed, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        bounds_min=(x - half, y - half, z - half),
        bounds_max=(x + half, y + half, z + half),
    )


def _state(tick: int, entities: dict[str, EntityState]) -> SceneState:
    return SceneState(
        world_id="w",
        scene_id="scenereplica-01",
        generation=0,
        tick=tick,
        sim_time=float(tick),
        ts=1000.0 + tick,
        entities=entities,
        robots={},
        joints={},
        regions={"dropoff/top": RegionState(pose=DROPOFF, size=(0.14, 0.14, 0.0))},
        contacts=(),
    )


# cup: on the table, lifted 8 cm, then set down in the drop-off. bowl: never moves.
LIFT_AND_PLACE = [
    {"cup": _entity(0.5, 0.0, 0.775), "bowl": _entity(0.6, -0.1, 0.775)},
    {"cup": _entity(0.5, 0.0, 0.855), "bowl": _entity(0.6, -0.1, 0.775)},
    {"cup": _entity(0.42, 0.38, 0.775), "bowl": _entity(0.6, -0.1, 0.775)},
]


def _fill(store: MemoryStore | SqliteStore, frames: list[dict[str, EntityState]]) -> None:
    stream = store.stream("sim_truth", SceneState)
    for tick, entities in enumerate(frames):
        stream.append(_state(tick, entities), ts=1000.0 + tick)


def _trajectory(*results: str, tool: str = "pick_object") -> Trajectory:
    """One Pi-style bash step per result, each calling ``tool`` through ``dimos mcp call``."""
    steps: list[Step] = [Step(step_id=1, timestamp="t", source="user", message="go")]
    for content in results:
        n = len(steps) + 1
        call = ToolCall(
            tool_call_id=f"c{n}",
            function_name="bash",
            arguments={"command": f"dimos mcp call {tool} --json-args '{{}}'"},
        )
        steps.append(
            Step(
                step_id=n,
                timestamp="t",
                source="agent",
                message="",
                tool_calls=(call,),
                observation=Observation(
                    results=(ObservationResult(source_call_id=f"c{n}", content=content),)
                ),
            )
        )
    return Trajectory(
        agent=AgentInfo(name="t", version="0", model_name="m"),
        steps=tuple(steps),
        final_metrics=FinalMetrics(
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_cached_tokens=0,
            total_cost_usd=None,
            total_steps=len(steps),
        ),
        extra=RunExtra(ended_by="answer"),
    )


def _scenereplica() -> ModuleType:
    if not (get_data_dir() / "scenereplica" / "reach_report.json").is_file():
        pytest.skip("the scenereplica data package is not unpacked")
    from dimos.evals.suites import scenereplica

    return scenereplica


# -- weighted scorer -----------------------------------------------------------------


def test_weighted_averages_by_weight_and_keeps_each_check() -> None:
    outcome = Outcome(trajectory=_trajectory(), artifacts={})
    grade = weighted({"a": (3.0, lambda o: 1.0), "b": (1.0, lambda o: 0.0)})
    graded = grade(outcome)
    assert isinstance(graded, Graded)
    assert graded.score == pytest.approx(0.75)
    assert graded.details == {"a": 1.0, "b": 0.0}


def test_weighted_clamps_checks_and_rejects_bad_weights() -> None:
    outcome = Outcome(trajectory=_trajectory(), artifacts={})
    assert weighted({"a": (1.0, lambda o: 7.0)})(outcome).score == 1.0
    assert weighted({"a": (1.0, lambda o: -2.0)})(outcome).score == 0.0
    with pytest.raises(ValueError):
        weighted({})
    with pytest.raises(ValueError):
        weighted({"a": (0.0, lambda o: 1.0)})


def test_runner_row_carries_the_details(tmp_path: Path) -> None:
    runner = EvalRunner(out_dir=tmp_path)
    runner._run_dir = tmp_path
    case = EvalCase(
        id="c",
        inputs="",
        environment=Sim2Environment(blueprint=[], scene="x"),
        grade=lambda o: Graded(score=1.0, details={"lifted": 1.0}),
    )
    result = runner._result(case, time.monotonic(), None, score=0.5, details={"lifted": 1.0})
    runner._write_artifacts([result])
    row = json.loads((tmp_path / "results.jsonl").read_text())
    assert row["details"] == {"lifted": 1.0} and row["score"] == 0.5


# -- truth checks ----------------------------------------------------------------------


def test_lifted_by_scores_the_peak_not_the_end() -> None:
    with MemoryStore() as store:
        _fill(store, LIFT_AND_PLACE)
        assert lifted_by(store, "cup", 0.05) == 1.0
        assert lifted_by(store, "cup", 0.16) == pytest.approx(0.5)
        assert lifted_by(store, "bowl", 0.05) == 0.0
        with pytest.raises(LookupError):
            lifted_by(store, "mug", 0.05)


@pytest.mark.parametrize(
    ("final", "expected"),
    [
        (_entity(0.42, 0.38, 0.775), 1.0),  # centred in the square, on the table
        (_entity(0.48, 0.32, 0.775), 1.0),  # near a corner, still inside
        (_entity(0.42, 0.46, 0.775), 0.0),  # 1 cm past the edge
        (_entity(0.42, 0.38, 0.875), 0.0),  # held 10 cm above the table
        (_entity(0.42, 0.38, 0.03), 0.0),  # fell to the floor below the square
        (_entity(0.42, 0.38, 0.775, speed=0.1), 0.0),  # still sliding
    ],
)
def test_resting_in_region(final: EntityState, expected: float) -> None:
    with MemoryStore() as store:
        _fill(store, [LIFT_AND_PLACE[0], {"cup": final}])
        assert resting_in_region(store, "cup", "dropoff/top") == expected


def test_unmoved_and_displacement() -> None:
    with MemoryStore() as store:
        _fill(store, LIFT_AND_PLACE)
        assert displacement(store, "bowl") == 0.0
        assert displacement(store, "cup") == pytest.approx(0.388, abs=1e-3)
        assert unmoved(store, ["bowl"]) == 1.0
        assert unmoved(store, ["bowl", "cup"]) == 0.5
        assert unmoved(store, []) == 1.0


# -- failure cause and the whole grader --------------------------------------------------


def _content(result: SkillResult | None) -> str:
    """A skill's answer as the agent reads it from the MCP tool call."""
    assert result is not None
    return str(result.agent_encode()[0]["text"])


def _scan(name: str) -> str:
    return _content(
        SkillResult(
            "Detected 1 object(s)",
            metadata={"prompts": [name], "objects": [{"object_id": "1", "name": name}]},
        )
    )


SCAN_OK = _scan("cracker box")
SCAN_OTHER = _scan("red box")
DETECT_OK = "Detected 1 object(s):\n  - cracker (object_id='1')"
NO_PATH = PlanResult(PlanStatus.FAILED, "IK failed: COLLISION")
# pick_object when the planner reached none of the grasp candidates.
PICK_NO_PATH = _content(
    SkillResult(
        "The planner found no path to any of the 5 grasp candidate(s) tried for object 1; "
        "last planner result FAILED: IK failed: COLLISION."
    )
)
# place_at when the planner found no way down to the drop-off.
PLACE_NO_PATH = _content(
    PickAndPlaceModule._stopped("Move to the pre-place pose above (0.42, 0.38, 0.80)", NO_PATH)
)
# A motion skill such as move_to_pose when the planner found no way to the pose.
MOVE_NO_PATH = _content(ManipulationSkills._planning_result(NO_PATH))
# A close that met nothing, and a motion that stopped part-way: execution, not planning.
GRASP_FAIL = _content(
    SkillResult(
        "Closed the gripper on object 1. Final gripper position 0.01. Reopened the gripper."
    )
)
DRIVE_FAULT = _content(
    PickAndPlaceModule._stopped(
        "Move to grasp candidate 0 for object 1",
        ExecutionResult(ExecutionStatus.FAULT, "drive fault"),
    )
)


def test_failure_cause_follows_scenereplica() -> None:
    cause = _scenereplica().failure_cause
    assert cause(_trajectory(), "cracker box", grasped=False) == "perception"
    assert (
        cause(_trajectory(SCAN_OTHER, PICK_NO_PATH), "cracker box", grasped=False) == "perception"
    )
    assert cause(_trajectory(SCAN_OK, PICK_NO_PATH), "cracker box", grasped=False) == "planning"
    assert cause(_trajectory(DETECT_OK, GRASP_FAIL), "cracker box", grasped=False) == "execution"
    assert cause(_trajectory(SCAN_OK, DRIVE_FAULT), "cracker box", grasped=False) == "execution"
    assert cause(_trajectory(SCAN_OK), "cracker box", grasped=False) == "execution"
    # A planning failure on the agent's own move does not make it a planning loss.
    moved = _trajectory(SCAN_OK, MOVE_NO_PATH, tool="move_to_pose")
    assert cause(moved, "cracker box", grasped=False) == "execution"
    # Once the object was lifted, only the place call's planning failure counts.
    assert cause(_trajectory(SCAN_OK, PICK_NO_PATH), "cracker box", grasped=True) == "execution"
    placed = _trajectory(SCAN_OK, PLACE_NO_PATH, tool="place_at")
    assert cause(placed, "cracker box", grasped=True) == "planning"
    assert cause(placed, "cracker box", grasped=False) == "execution"


def test_planning_label_reads_what_the_skills_say() -> None:
    """The label matches the words the skills use, so a reworded skill fails here first."""
    planner_failed = _scenereplica()._PLANNER_FAILED
    for content in (PICK_NO_PATH, PLACE_NO_PATH, MOVE_NO_PATH):
        assert planner_failed.search(content), content
    for content in (GRASP_FAIL, DRIVE_FAULT, SCAN_OK):
        assert not planner_failed.search(content), content


def test_pick_and_place_grader_end_to_end(tmp_path: Path) -> None:
    pick_and_place = _scenereplica().pick_and_place
    path = tmp_path / "memory.db"
    with SqliteStore(path=str(path)) as store:
        _fill(store, LIFT_AND_PLACE)
    outcome = Outcome(trajectory=_trajectory(SCAN_OK), artifacts={"recording": path})
    graded = pick_and_place("cup", "cup", ("bowl",))(outcome)
    assert graded.score == 1.0
    assert graded.details == {
        "lifted": 1.0,
        "placed": 1.0,
        "unmoved": 1.0,
        "disturbed": [],
        "cause": None,
    }
    # The bowl graded as the target: never lifted, never placed, and the cup was moved.
    graded = pick_and_place("bowl", "bowl", ("cup",))(outcome)
    assert graded.score == 0.0
    assert graded.details["disturbed"] == ["cup"] and graded.details["unmoved"] == 0.0
    assert graded.details["cause"] == "perception"


def test_setup_scene_waits_for_the_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP answers before the coordinator is on the bus; the first connects fail."""
    attempts: list[int] = []
    switched: list[bool] = []
    app = SimpleNamespace(
        get_module=lambda name: SimpleNamespace(set_truth_enabled=switched.append),
        stop=lambda: None,
    )

    def connect() -> SimpleNamespace:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("No running DimOS coordinator found")
        return app

    monkeypatch.setattr("dimos.porcelain.dimos.Dimos.connect", connect)
    monkeypatch.setattr("dimos.evals.environments.sim2.time.sleep", lambda s: None)
    Sim2Environment(blueprint=[], scene="x").setup_scene()
    assert len(attempts) == 3 and switched == [True]
    clock = iter([0.0] + [1e9] * 10)  # the deadline is set, then the clock jumps past it
    monkeypatch.setattr("dimos.evals.environments.sim2.time.monotonic", lambda: next(clock))
    attempts.clear()
    with pytest.raises(RuntimeError, match="No running DimOS coordinator"):
        Sim2Environment(blueprint=[], scene="x").setup_scene()
    assert len(attempts) == 1


# -- suite shape ---------------------------------------------------------------------------


def test_suite_has_100_trials_with_scene_object_and_order_tags() -> None:
    module = _scenereplica()
    suite = module.SUITE
    assert len(suite) == 100 and len({c.id for c in suite}) == 100
    for scene in range(1, 21):
        cases = [c for c in suite if f"scene-{scene:02d}" in c.tags]
        assert len(cases) == 5
        assert sorted(
            int(next(t for t in c.tags if t.startswith("order-"))[6:]) for c in cases
        ) == [1, 2, 3, 4, 5]
        assert len({id(c.environment) for c in cases}) == 5
        for case in cases:
            env = case.environment
            assert isinstance(env, Sim2Environment)
            assert env.config.blueprint == module.BLUEPRINT
            assert Path(env.config.scene).name == f"scene-{scene:02d}"
            assert len(env.config.truth_entities) == 5
            name = case.id.removeprefix(f"scenereplica-scene-{scene:02d}-")
            assert name in env.config.truth_entities and name in case.tags
            assert {"scenereplica", "perception"} <= case.tags
            assert case.timeout_s == 900.0 and case.threshold == 1.0
            assert case.inputs.startswith("Pick up the ") and "x=0.42 m, y=0.38 m" in case.inputs
    unreachable = [c.id for c in suite if "unreachable" in c.tags]
    assert unreachable == ["scenereplica-scene-11-cracker_box"]


def test_dan_suite_runs_the_same_trials_on_its_own_launch() -> None:
    baseline = _scenereplica().SUITE
    from dimos.evals.suites import scenereplica_dan

    assert scenereplica_dan.BLUEPRINT == ["xarm-perception-sim2-dan", "mcp-server", "observe-skill"]
    assert len(scenereplica_dan.SUITE) == len(baseline) == 100
    for dan, base in zip(scenereplica_dan.SUITE, baseline, strict=True):
        assert (dan.id, dan.inputs, dan.tags) == (base.id, base.inputs, base.tags)
        assert (dan.timeout_s, dan.threshold) == (base.timeout_s, base.threshold)
        assert isinstance(dan.environment, Sim2Environment)
        assert isinstance(base.environment, Sim2Environment)
        assert dan.environment.config.blueprint == scenereplica_dan.BLUEPRINT
        assert dan.environment.config.model_dump(exclude={"blueprint"}) == (
            base.environment.config.model_dump(exclude={"blueprint"})
        )


def test_prompt_never_gives_object_positions() -> None:
    module = _scenereplica()
    scene_dir = get_data_dir() / "scenereplica" / "scene-01"
    scene = json.loads((scene_dir / "scene.json").read_text())
    cases = module.scene_cases(scene_dir)
    labels = [scene["entities"][c.id.rsplit("-", 1)[1]]["label"] for c in cases]
    for case, label in zip(cases, labels, strict=True):
        assert f"Pick up the {label} " in case.inputs
        for pose in scene["initial"]["poses"].values():
            x, y, _ = pose["position"]
            assert f"{x:.2f}" not in case.inputs and f"{y:.2f}" not in case.inputs
    # Near-to-far: the first case's object is the closest to the arm base.
    first = cases[0].id.rsplit("-", 1)[1]
    distances = {
        name: pose["position"][0] ** 2 + pose["position"][1] ** 2
        for name, pose in scene["initial"]["poses"].items()
    }
    assert first == min(distances, key=distances.__getitem__)


# -- summary script -------------------------------------------------------------------------


def _row(case_id: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "score": 0.0,
        "passed": False,
        "error": "",
        "details": {},
    }
    row.update(overrides)
    return row


def _write_run(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


def test_summary_counts_the_two_scores_and_the_cause_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    placed = {"lifted": 1.0, "placed": 1.0, "unmoved": 1.0, "disturbed": [], "cause": None}
    dropped = {
        "lifted": 1.0,
        "placed": 0.0,
        "unmoved": 0.75,
        "disturbed": ["bowl"],
        "cause": "execution",
    }
    unseen = {"lifted": 0.0, "placed": 0.0, "unmoved": 1.0, "disturbed": [], "cause": "perception"}
    stuck = {"lifted": 0.0, "placed": 0.0, "unmoved": 1.0, "disturbed": [], "cause": "planning"}
    run_a = _write_run(
        tmp_path / "run-a",
        [
            _row("scenereplica-scene-01-cracker_box", score=1.0, passed=True, details=placed),
            _row("scenereplica-scene-01-bowl", score=0.5, details=dropped),
            _row("scenereplica-scene-01-mug", details=unseen),
            _row("xarm_pick_cylinder", score=1.0, passed=True),  # another suite's row
        ],
    )
    run_b = _write_run(
        tmp_path / "run-b",
        [
            _row("scenereplica-scene-02-mug", details=stuck),
            _row("scenereplica-scene-02-bowl", error="TimeoutError('sim2 did not start')"),
        ],
    )
    trials = load_trials([run_a, run_b, tmp_path / "missing"])
    assert len(trials) == 5
    assert trials[0] == Trial("scene-01", "cracker_box", True, True, "")
    assert trials[1] == Trial("scene-01", "bowl", True, False, "execution")
    assert trials[4] == Trial("scene-02", "bowl", False, False, "error")
    text = report(trials)
    assert "5 of 100 trials run, 2 scene(s)" in text
    assert "grasp success            2 / 5  (40%)" in text
    assert "pick-and-place success   1 / 5  (20%)" in text
    assert "perception 1, planning 1, execution 1, error 1" in text
    assert re.search(r"^scene-01\s+3\s+2\s+1$", text, re.MULTILINE)
    assert re.search(r"^mug\s+2\s+0\s+0$", text, re.MULTILINE)
    main([str(run_a)])
    assert "3 of 100 trials run" in capsys.readouterr().out
    assert report([]) == "No SceneReplica trials found."
    with pytest.raises(SystemExit):
        main([])
