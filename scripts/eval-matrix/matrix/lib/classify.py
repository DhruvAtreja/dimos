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

"""Find the stage where each SceneReplica matrix trial was lost, from its trajectory and grade.

"Lifted" and "placed" are the graded truth: lifted means the target rose 5 cm at some point,
placed means it ended at rest inside the drop-off. A grasp counts as executed when the arm
reached a grasp pose and closed the gripper there (a pick_object answer saying the gripper
closed, the lift check failed, or "Pick complete"). A hold is a "Pick complete" answer.
Each failed trial gets the first stage below whose rule matches:

  perception          Not lifted. The agent scanned, but no scan_objects or detect result
                      named the target (a name whose words all belong to the target's label,
                      the suite's own rule), unless a grasp moved the target at least 5 mm
                      (then it was found under another name); or no grasp was executed and
                      pick_object only ever answered that the object had no point cloud.
  grasp_unreachable   Not lifted. No grasp executed, no pick motion refused, and some
                      pick_object answered that the planner found no path to any grasp
                      candidate.
  motion_refused      Not lifted, no grasp executed, and a pick motion was refused on
                      execution ("execution returned REJECTED", gripper would not open). Or
                      a controller fault (faulted device, lost joint state, skill server
                      gone) after which no motion succeeded. Or held, never let go of, and
                      the last place_at was refused.
  grasp_missed        Not lifted and a grasp was executed: it closed on nothing, failed the
                      lift check, slipped, or said "Pick complete" for an object that never
                      rose 5 cm. Also lifted with no hold ever reported and no sign it came
                      up in the gripper (knocked or tipped up).
  placed_off_target   Lifted. After the last hold, place_at released it ("Place complete",
                      "Released ...", or opened the gripper at the place pose), but it did
                      not end at rest inside the drop-off.
  place_unreachable   Lifted and held, never released; a place_at answered that the planner
                      found no path or IK to the place.
  dropped_in_transit  Lifted, then lost before any release: place_at found the jaws empty,
                      pick_object's lift check said it fell out and let go, or the jaws
                      stopped part-way (holding it) while the pick reported a failed grasp.
  agent               The robot reported nothing that lost the object. The agent never
                      scanned, never called pick_object, or never got one started (wrong
                      arguments or ids); or held the object and never called place_at,
                      opened the gripper itself, or carried it with its own arm moves
                      after the robot's last reported drop.
  other               Anything else, such as an object placed without a 5 cm lift; the
                      stage detail keeps the last skill message.

Passed trials get "passed". placed_off_target, place_unreachable and dropped_in_transit
together are "lost after grasp" (see GROUP).

Run, reading the matrix under DIR/results/matrix (the box's ~/dimos-evals by default):
    python3 matrix/lib/classify.py --root <this folder on the laptop>
It prints how many trials each stage took per config, every trial that fell to other,
the stage details, and how often the agent wasted calls.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

STAGES = (
    "passed",
    "perception",
    "grasp_unreachable",
    "motion_refused",
    "grasp_missed",
    "placed_off_target",
    "place_unreachable",
    "dropped_in_transit",
    "agent",
    "other",
)
# Coarse group of each stage, for short tables.
GROUP = {
    "placed_off_target": "lost_after_grasp",
    "place_unreachable": "lost_after_grasp",
    "dropped_in_transit": "lost_after_grasp",
}

# Skill answers are JSON objects starting with "message", printed one per line.
_REPLY = re.compile(r'\{"message":')
_TOOL_ERROR = re.compile(r"Error running tool '(\w+)': ([^\n]*)")
_DETECT_LIST = re.compile(r"^Detected \d+ object\(s\):", re.MULTILINE)
# A name the perception stack reported: scan_objects' JSON or detect's bullet list.
_DETECTION = re.compile(r'"name":\s*"([^"]*)"|^\s*-\s*(.+?)\s*\(object_id=', re.MULTILINE)
_LABEL = re.compile(r"Pick up the (.+?) and place it")
# Controller or stack faults: the arm or the skill server stopped answering.
_FAULT = re.compile(
    r"device is faulted|Joint state is missing selected joints|first joint state"
    r"|End-effector pose for planning group '\w+' is unavailable|no running MCP server"
    r"|timed out after|GraspGenX inference failed|returned UNCERTAIN|ended with UNCERTAIN"
)
# The arm cannot start a motion from where it is.
_STUCK = re.compile(
    r"Start configuration is in collision|Invalid start configuration"
    r"|Joint state is missing selected joints|first joint state|device is faulted"
    r"|End-effector pose for planning group '\w+' is unavailable|no running MCP server"
)
# get_robot_state answers echo the last error; they are status reads, not new failures.
_STATUS = ("ManipulationSnapshot(", "manipulator: PlanningGroupState(")
_SKILL_CALL = re.compile(r"""mcp\s+call\s+(\w+)|['"]call['"]\s*,\s*['"](\w+)['"]""")
_PARTIAL_CLOSE = re.compile(r"Gripper position (\d\.\d+) had not settled")
_GRIPPER_SET = re.compile(r"Gripper set to (\d+)% open")


def _replies(content):
    """The skill answers in one tool result, in order, as dicts with a message."""
    out = []
    decoder = json.JSONDecoder()
    for match in _REPLY.finditer(content):
        try:
            reply, _ = decoder.raw_decode(content, match.start())
        except ValueError:
            continue
        if isinstance(reply, dict):
            out.append(reply)
    for match in _TOOL_ERROR.finditer(content):
        out.append({"message": match.group(2), "skill": match.group(1)})
    if "Error: no running MCP server" in content:
        out.append({"message": "no running MCP server"})
    for match in _DETECT_LIST.finditer(content):
        out.append({"message": match.group(0), "skill": "detect"})
    return out


def _skill_of(reply, called):
    """Which skill gave this answer, from its text, else the only skill the command called."""
    message, meta = reply.get("message", ""), reply.get("metadata") or {}
    if reply.get("skill"):
        return reply["skill"]
    if "grasp_sources" in meta:
        return "pick_object"
    if re.search(
        r"placed at \(|place pose|to release the object|^Place complete"
        r"|^Released |after releasing",
        message,
    ):
        return "place_at"
    if message.startswith("Detected "):
        return "scan_objects"
    if message.startswith(("ExecutionResult(", "No plan;", "Execution ended")):
        return "motion"
    if _GRIPPER_SET.match(message):
        return "gripper_cmd"
    return called[0] if len(called) == 1 else "other"


def _pick_kind(message):
    """What one pick_object answer says happened.

    ok (the robot holds it), missed (a grasp closed but did not hold), no_path (planner
    reached no candidate), refused (a planned motion was refused, or the gripper would not
    open first), no_cloud, fault, or not_started (wrong id or arguments, still holding).
    """
    if message.startswith("Pick complete"):
        return "ok"
    if "planner found no path to any of" in message:
        return "no_path"
    if message.startswith(
        (
            "Closed the gripper on object",
            "Pick not complete",
            "After lifting object",
            "After carrying object",
            "Lifted the gripper",
            "Retract after grasping",
        )
    ):
        return "missed"
    if message.startswith("Move to grasp candidate") or "before grasping" in message:
        return "refused"
    if "has no point cloud" in message:
        return "no_cloud"
    if _FAULT.search(message):
        return "fault"
    return "not_started"


def _place_kind(message):
    """What one place_at answer says happened: empty, released, gone, fault, refused,
    unreachable or other."""
    if message.startswith("Not holding any object"):
        return "empty"
    if message.startswith(("Place complete", "Released ", "Commanded the gripper open to release")):
        return "released"
    if message.startswith("Retract from") and "after releasing" in message:
        return "released"
    if "no longer in the gripper" in message or "no longer in the jaws" in message:
        return "gone"
    if _FAULT.search(message):
        return "fault"
    if "execution returned" in message:
        return "refused"
    if "planner returned" in message:
        return "unreachable"
    return "other"


def events(trajectory):
    """The trial's skill answers in order, with the target's label.

    Args:
        trajectory: The parsed trajectory.json of one trial.
    Returns (label, named, events). label is the target's name in the task prompt; named
    says whether a perception result named it; events lists (skill, reply) pairs, reply
    being a dict with the answer's "message" (and "metadata" when it had one). The agent's
    own gripper commands appear as ("gripper_cmd", {"message": "open" or "close"}), and a
    motion answer to a command that called move_to_pose or move_to_joints has
    "own_move": True.
    """
    steps = trajectory.get("steps") or []
    prompt = next((s.get("message") or "" for s in steps if s.get("source") == "user"), "")
    match = _LABEL.search(prompt)
    label = match.group(1).lower() if match else ""
    wanted = set(label.split())
    named = False
    out = []
    for step in steps:
        observation = step.get("observation") or {}
        results = {r.get("source_call_id"): r for r in observation.get("results") or []}
        for call in step.get("tool_calls") or []:
            args = call.get("arguments")
            command = (args.get("command") if isinstance(args, dict) else "") or ""
            called = [a or b for a, b in _SKILL_CALL.findall(command)]
            result = results.get(call.get("tool_call_id"))
            if result is None:
                continue  # the run ended before this call was answered
            content = str(result.get("content") or "")
            if "data:image" in content:
                continue
            for json_name, listed_name in _DETECTION.findall(content):
                words = set((json_name or listed_name).lower().split())
                named |= bool(words) and words <= wanted
            own_move = any(c in ("move_to_pose", "move_to_joints") for c in called)
            for reply in _replies(content):
                skill = _skill_of(reply, called)
                if skill == "motion":
                    reply = {**reply, "own_move": own_move}
                elif skill == "gripper_cmd":
                    opening = int(_GRIPPER_SET.match(reply["message"]).group(1))
                    reply = {"message": "close" if opening < 50 else "open"}
                out.append((skill, reply))
    return label, named, out


def _short(message, limit=160):
    """A skill message cut to one line, with object ids shortened."""
    message = re.sub(r"[0-9a-f]{32}", "<id>", " ".join(message.split()))
    return message if len(message) <= limit else message[: limit - 3] + "..."


def classify(trajectory, details, passed, error=""):
    """Stage and counts for one trial.

    Args:
        trajectory: Parsed trajectory.json, or None when the trial has none.
        details: The results row's ``details`` (lifted, placed, disturbed, ...).
        passed: Whether the trial passed.
        error: The results row's ``error`` text, empty when the run finished normally.
    Returns a dict: stage and stage_detail (a short raw message); pick_calls, pick_ok,
    place_calls, scan_calls; grasp_tries (grasps executed); retries (extra tries inside
    M1 picks); candidates_tried and candidates_rejected (per source, grasps the planner
    found no path to); knocked_others (how many other objects moved over 2 cm);
    knock_phase ("pick" when the target was never carried, "unknown" when it was, ""
    when nothing moved); arm_stuck (1 when the trial ended with the arm refusing every
    motion from where it was); and agent habits: place_empty (place_at with nothing
    held), bad_args (pick_object refused its arguments), manual_grasps (agent's own
    close_gripper commands).
    """
    lifted = float(details.get("lifted") or 0) >= 1.0
    placed = float(details.get("placed") or 0) >= 1.0
    disturbed = details.get("disturbed") or []
    label, named, evs = events(trajectory or {})
    out = {
        "pick_calls": 0,
        "pick_ok": 0,
        "place_calls": 0,
        "scan_calls": 0,
        "grasp_tries": 0,
        "retries": 0,
        "candidates_tried": 0,
        "candidates_rejected": {},
        "knocked_others": len(disturbed),
        "knock_phase": "",
        "arm_stuck": 0,
        "place_empty": 0,
        "bad_args": 0,
        "manual_grasps": 0,
    }
    picks, places, faults, own_moves = [], [], [], []
    last_ok, stuck = -1, 0
    for index, (skill, reply) in enumerate(evs):
        message = reply.get("message", "")
        meta = reply.get("metadata") or {}
        executed = 0
        if skill in ("scan_objects", "detect"):
            out["scan_calls"] += 1
        elif skill == "pick_object":
            kind = _pick_kind(message)
            out["pick_calls"] += 1
            out["pick_ok"] += kind == "ok"
            out["bad_args"] += kind == "not_started" and "planning_group" in message
            tries = meta.get("tries") or []
            executed = sum(t.get("grasped") is not None for t in tries)
            if kind in ("ok", "missed") and (not tries or tries[-1].get("grasped") is None):
                executed += 1
            out["grasp_tries"] += executed
            out["retries"] += max(0, len(tries) - 1)
            rejected = meta.get("rejected_by_source") or {}
            for source, count in rejected.items():
                out["candidates_rejected"][source] = out["candidates_rejected"].get(
                    source, 0
                ) + int(count or 0)
            moved_to = len(tries) if tries else int("source" in meta)
            out["candidates_tried"] += sum(int(c or 0) for c in rejected.values()) + moved_to
            picks.append((index, kind, message, executed))
        elif skill == "place_at":
            kind = _place_kind(message)
            out["place_calls"] += 1
            out["place_empty"] += kind == "empty"
            places.append((index, kind, message))
        elif skill == "gripper_cmd":
            out["manual_grasps"] += message == "close"
        if message.startswith(_STATUS):
            continue
        if _FAULT.search(message):
            faults.append((index, message))
        moved = message.startswith(
            ("ExecutionResult(COMPLETED", "Pick complete", "Place complete", "Released ")
        )
        if moved and reply.get("own_move"):
            own_moves.append(index)
        if moved or executed:
            last_ok, stuck = index, 0
        elif _STUCK.search(message):
            stuck += 1
    out["arm_stuck"] = int(stuck >= 2)
    terminal_fault = next((m for i, m in reversed(faults) if i > last_ok), None)
    held_from = max((i for i, k, _, _ in picks if k == "ok"), default=None)
    if disturbed:
        out["knock_phase"] = "unknown" if (held_from is not None or lifted) else "pick"

    def done(stage, detail):
        out["stage"], out["stage_detail"] = stage, _short(detail)
        return out

    if passed:
        return done("passed", "")
    if trajectory is None:
        return done("other", error or "no trajectory")
    executed = [m for _, _, m, e in picks if e]
    gripper = [(i, r["message"]) for i, (s, r) in enumerate(evs) if s == "gripper_cmd"]
    if not lifted:
        if placed:
            return done(
                "other",
                f"placed without a 5 cm lift (lifted {float(details.get('lifted') or 0):.2f})",
            )
        # A target the robot grasped and moved (rose 5 mm) was found under another name.
        touched = bool(executed) and float(details.get("lifted") or 0) >= 0.1
        if out["scan_calls"] and not named and not touched:
            return done("perception", f"no scan named the {label}")
        kinds = [k for _, k, _, _ in picks]
        if not executed and "no_cloud" in kinds and set(kinds) <= {"no_cloud", "not_started"}:
            return done("perception", next(m for _, k, m, _ in picks if k == "no_cloud"))
        refused = [m for _, k, m, _ in picks if k == "refused"]
        no_path = [m for _, k, m, _ in picks if k == "no_path"]
        if not executed and not refused and no_path:
            return done("grasp_unreachable", no_path[-1])
        if not executed and refused:
            return done("motion_refused", refused[-1])
        if terminal_fault:
            return done("motion_refused", terminal_fault)
        if executed:
            last = executed[-1]
            if last.startswith("Pick complete"):
                last = f"said '{last}' but the object rose under 5 cm"
            if not named:
                last = f"target only found under another name; {last}"
            return done("grasp_missed", last)
        if not out["scan_calls"]:
            return done("agent", "never scanned for the target")
        if not picks:
            return done("agent", "never called pick_object")
        return done("agent", "pick never started: " + picks[-1][2])
    # Lifted but not placed: follow the last object the robot reported held.
    if held_from is not None:
        after = [p for p in places if p[0] > held_from]
        released = [m for _, k, m in after if k == "released"]
        gone = [m for _, k, m in after if k == "gone"]
        unreachable = [m for _, k, m in after if k == "unreachable"]
        refused = [m for _, k, m in after if k in ("refused", "fault")]
        if released:
            return done("placed_off_target", released[0])
        if gone:
            return done("dropped_in_transit", gone[0])
        if refused and after[-1][1] in ("refused", "fault"):
            return done("motion_refused", refused[-1])
        if unreachable:
            return done("place_unreachable", unreachable[-1])
        if terminal_fault:
            return done("motion_refused", terminal_fault)
        if any(i > held_from and m == "open" for i, m in gripper):
            return done("agent", "opened the gripper itself while holding the object")
        if not after:
            return done("agent", "held the object but never called place_at")
        return done("other", after[-1][2])
    # No hold was ever reported. Whichever came last decides: the robot losing it after
    # a lift (pick's lift check, or place_at finding the jaws empty), or the agent
    # carrying it with its own moves (closing the gripper then moving the arm, or moving
    # it after a failed pick that left the jaws closed, then opening them).
    lost = [
        (i, m) for i, k, m, _ in picks if k == "missed" and re.search(r"lift|carrying|Lifted", m)
    ]
    lost += [(i, m) for i, k, m in places if k == "gone"]
    last_grasp = max(((i, m) for i, _, m, e in picks if e), default=(-1, ""))
    carried = [j for j in own_moves if any(m == "close" and i < j for i, m in gripper)]
    if not re.search(r"Reopened|Commanded the gripper open", last_grasp[1]):
        carried += [
            i for i, m in gripper if m == "open" and any(last_grasp[0] < j < i for j in own_moves)
        ]
    if carried and (not lost or max(carried) > max(lost)[0]):
        return done("agent", "carried it with its own arm moves and gripper commands")
    if lost:
        return done("dropped_in_transit", max(lost)[1])
    for message in executed:
        partial = _PARTIAL_CLOSE.search(message)
        if partial and 0.05 < float(partial.group(1)) < 0.95:
            return done(
                "dropped_in_transit", f"jaws stopped part-way, held it, pick said failed: {message}"
            )
    if executed:
        return done("grasp_missed", f"rose 5 cm with no hold reported: {executed[-1]}")
    if terminal_fault:
        return done("motion_refused", terminal_fault)
    return done("other", evs[-1][1].get("message", "") if evs else "")


def load_trajectory(case_dir, run):
    """The parsed trajectory.json of one case's run, or None when it has none.

    Args:
        case_dir: The case's folder, results/matrix/<config>/<case_id>.
        run: The run folder name inside its evals/, e.g. "run-20261008-221017-jt8ji1au".
    """
    paths = glob.glob(os.path.join(case_dir, "evals", run, "*", "trajectory.json"))
    if not paths:
        return None
    try:
        with open(paths[0]) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        default=os.environ.get("EVALS_DIR") or os.path.expanduser("~ubuntu/dimos-evals"),
        help="evals folder holding results/matrix and matrix/*.tsv",
    )
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import export

    export.set_root(args.root)
    rows = export.trials(full=True)
    configs = [name for name, *_ in export.config_lines()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    width = max(len(c) for c in configs) + 2

    def table(title, label_width, lines):
        print(f"\n{title}\n")
        print(f"{'':{label_width}s}" + "".join(f"{c:>{width}s}" for c in configs) + f"{'all':>7s}")
        for name, count in lines:
            cells = [count(by[c]) for c in configs]
            print(
                f"{name:{label_width}s}"
                + "".join(f"{n:>{width}}" for n in cells)
                + f"{sum(cells):>7}"
            )

    table(
        "Trials per stage",
        20,
        [(s, lambda rs, s=s: sum(r["stage"] == s for r in rs)) for s in STAGES] + [("total", len)],
    )
    other = [r for r in rows if r["stage"] == "other"]
    print(f"\nother: {len(other)} of {len(rows)} ({100 * len(other) / max(len(rows), 1):.1f}%)")
    for r in other:
        print(f"  {r['config']:14s} {r['case_id']:45s} {r['stage_detail']}")
    table(
        "Agent habits and robot state (trials affected)",
        34,
        [
            ("place_at with nothing held", lambda rs: sum(r["place_empty"] > 0 for r in rs)),
            ("pick_object with bad arguments", lambda rs: sum(r["bad_args"] > 0 for r in rs)),
            ("agent closed the gripper itself", lambda rs: sum(r["manual_grasps"] > 0 for r in rs)),
            ("ran out of steps", lambda rs: sum(r["ended_by"] == "max_steps" for r in rs)),
            ("ended with the arm stuck", lambda rs: sum(r["arm_stuck"] for r in rs)),
            ("knocked another object", lambda rs: sum(r["knocked_others"] > 0 for r in rs)),
            (
                "'Pick complete' but not lifted",
                lambda rs: sum(r["pick_ok"] > 0 and not r["lifted"] for r in rs),
            ),
        ],
    )
    print("\nStage details (numbers replaced by N, most common first)")
    details = collections.Counter(
        (r["stage"], re.sub(r"-?\d+(\.\d+)?", "N", r["stage_detail"])[:120])
        for r in rows
        if r["stage"] != "passed"
    )
    for (stage, detail), n in sorted(details.items(), key=lambda x: (STAGES.index(x[0][0]), -x[1])):
        print(f"  {n:3d}  {stage:18s} {detail}")


if __name__ == "__main__":
    main()
