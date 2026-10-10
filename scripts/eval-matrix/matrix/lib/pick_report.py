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

"""Count how scripted picks ended, per config, for a matrix config set.

Run on the evals box (sudo: finished case folders can be root-owned):
    sudo python3 ~/dimos-evals/matrix/lib/pick_report.py cc
    sudo python3 ~/dimos-evals/matrix/lib/pick_report.py cc --pairs cc-graspgen:cc-graspgen-nocheck
    sudo python3 ~/dimos-evals/matrix/lib/pick_report.py cc --trials > trials.json

Reads results/matrix-<SET>/<config>/<case>/ and each trial's pick_object replies (scripted
agent only); the pick columns describe the first reply. Trials where the simulator's physics stopped are left out.

Columns:
    n          trials counted
    grasped    the grader saw the object lifted
    placed     the grader saw it resting in the drop-off
    nothing    the first close ended with the jaws shut on nothing (gripper < 0.05)
    stuck      the first close never got going (gripper >= 0.95, or "had not settled")
    slipped    the object slipped out during the lift
    dropped    held after the pick, then lost while carrying it
    no_grasp   the first pick found no grasp left to try
    no_path    the planner reached none of the first pick's grasps
    timeout    the first pick call ran past its time limit and returned an error
    no_target  the scan found no target, so there was no pick
    rejected   grasps the contact check rejected, summed over all pick tries
"""

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export

COLUMNS = (
    "n",
    "grasped",
    "placed",
    "nothing",
    "stuck",
    "slipped",
    "dropped",
    "no_grasp",
    "no_path",
    "timeout",
    "no_target",
    "rejected",
)
_JAWS = re.compile(r"gripper position (\d+\.\d+)", re.IGNORECASE)


def first_pick(case_dir, run):
    """Every pick_object reply of one trial, in call order, as (message, metadata) pairs."""
    out = []
    for path in sorted(
        glob.glob(os.path.join(case_dir, "evals", run, "*", "calls", "*-pick_object-result.json"))
    ):
        text = json.load(open(path))["result"]["content"][0]["text"]
        try:
            body = json.loads(text)
        except ValueError:
            body = {"message": text}
        out.append((str(body.get("message", "")), body.get("metadata") or {}))
    return out


def outcome(message, metadata):
    """How one pick_object reply began: its first grasp try, read from the message's start.

    held, nothing (jaws shut on nothing), stuck (jaws never closed), slipped (lost in the
    lift), no_grasp (no grasp left to try), no_path (the planner reached none), timeout (the
    pick call ran past its time limit) or other.
    """
    if metadata.get("grasped") is True or message.startswith("Pick complete"):
        return "held"
    if "timed out" in message:
        return "timeout"
    if message.startswith("Generated"):
        return "no_grasp"
    if message.startswith("The planner found no path"):
        return "no_path"
    if (
        message.startswith(("After lifting", "Lifted the gripper"))
        or "slipped" in message.split(". ")[0]
    ):
        return "slipped"
    if "had not settled" in message.split(". ")[0]:
        return "stuck"
    jaws = _JAWS.search(message.split(". Reopened")[0])
    if jaws:
        value = float(jaws.group(1))
        return "stuck" if value >= 0.95 else "nothing" if value < 0.05 else "other"
    return "other"


def rejected(metadata):
    found = metadata.get("contact_rejected") or {}
    return int(found.get("rejected") or 0) if isinstance(found, dict) else 0


def report(name):
    """{config: {column: count}} and {config: {case_id: trial row}} for config set NAME."""
    export.set_root(export.BOX, name)
    counts, trials = {}, {}
    for config, *_ in export.config_lines():
        counts[config], trials[config] = dict.fromkeys(COLUMNS, 0), {}
    for row in export.trials():
        if row["sim_died"]:
            continue
        case_dir = os.path.join(export.RESULTS, row["config"], row["case_id"])
        picks = first_pick(case_dir, row["run"])
        trial = {
            "config": row["config"],
            "config_order": row["config_order"],
            "batch": row["batch"],
            "case_id": row["case_id"],
            "object": row["object"],
            "grasped": row["lifted"],
            "placed": row["placed"],
            "dropped": int(row["stage"] == "dropped_in_transit"),
            "first": outcome(*picks[0]) if picks else "no_target",
        }
        trials[row["config"]][row["case_id"]] = trial
        total = counts[row["config"]]
        total["n"] += 1
        for column in ("grasped", "placed", "dropped"):
            total[column] += trial[column]
        if trial["first"] in total:
            total[trial["first"]] += 1
        trial["rejected"] = sum(rejected(meta) for _, meta in picks)
        total["rejected"] += trial["rejected"]
    return counts, trials


def mcnemar(only_a, only_b):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("set", help="config set name, as in auto.sh set:NAME")
    parser.add_argument("--pairs", nargs="*", default=[], help="A:B config pairs to compare")
    parser.add_argument(
        "--trials", action="store_true", help="print one JSON row per trial instead"
    )
    args = parser.parse_args()
    counts, trials = report(args.set)
    if args.trials:
        json.dump(
            {"rows": [t for by_case in trials.values() for t in by_case.values()]},
            sys.stdout,
            indent=1,
        )
        sys.exit()
    width = max(len(c) for c in counts) if counts else 10
    print(f"{'config':<{width}} " + " ".join(f"{c:>9}" for c in COLUMNS))
    for config, row in counts.items():
        print(f"{config:<{width}} " + " ".join(f"{row[c]:>9}" for c in COLUMNS))
    for pair in args.pairs:
        a, b = pair.split(":")
        shared = sorted(set(trials[a]) & set(trials[b]))
        for metric in ("grasped", "placed"):
            only_a = sum(1 for k in shared if trials[a][k][metric] and not trials[b][k][metric])
            only_b = sum(1 for k in shared if trials[b][k][metric] and not trials[a][k][metric])
            print(
                f"{a} vs {b}, {len(shared)} shared trials, {metric}: "
                f"{sum(trials[a][k][metric] for k in shared)} vs "
                f"{sum(trials[b][k][metric] for k in shared)} "
                f"(only {a} {only_a}, only {b} {only_b}, p = {mcnemar(only_a, only_b):.3f})"
            )
