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

"""Print the matrix's finished trials and its configs as JSON, for the results dashboard.

Run on the evals box (with sudo, since finished case folders can be root-owned):
    sudo python3 ~/dimos-evals/matrix/lib/export.py trials
    sudo python3 ~/dimos-evals/matrix/lib/export.py configs
or on the laptop against the copies pull.sh made:
    python3 matrix/lib/export.py trials --root <this folder on the laptop>

Each trial row also carries where it was lost (stage, stage_detail) and pick counts from
its trajectory; classify.py defines them.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify

BOX = os.environ.get("EVALS_DIR") or os.path.expanduser("~ubuntu/dimos-evals")
# Trial fields from classify.py that go into the export; the rest stay in classify.py's report.
CLASSIFY_FIELDS = (
    "stage",
    "stage_detail",
    "pick_calls",
    "pick_ok",
    "grasp_tries",
    "candidates_tried",
    "candidates_rejected",
    "knocked_others",
    "knock_phase",
    "arm_stuck",
)


def set_root(root, name=""):
    """Read results/matrix and matrix/{configs,batches}.tsv under this evals folder.

    Args:
        root: The evals folder: ~ubuntu/dimos-evals on the box (or EVALS_DIR), this folder on the laptop.
        name: A named config set, as auto.sh's "set:NAME" step runs it: results from
            results/matrix-NAME, configs from matrix/configs-NAME.tsv. Empty reads the main set.
    """
    global RESULTS, CONFIGS, BATCHES
    root = os.path.expanduser(root)
    suffix = f"-{name}" if name else ""
    RESULTS = os.path.join(root, "results", f"matrix{suffix}")
    CONFIGS = os.path.join(root, "matrix", f"configs{suffix}.tsv")
    BATCHES = os.path.join(root, "matrix", "batches.tsv")


set_root(BOX)


def config_lines():
    """The active configs in run order, as (name, ref, suite, env) tuples."""
    out = []
    for line in open(CONFIGS):
        cols = line.split()
        if not cols or cols[0].startswith("#"):
            continue
        out.append((cols[0], cols[1], cols[2], " ".join(cols[5:])))
    return out


def batches():
    """case_id -> (batch number, scene, order, object)."""
    out = {}
    for line in open(BATCHES):
        if line.startswith("#") or not line.strip():
            continue
        batch, case_id, scene, order, obj = line.split()[:5]
        out[case_id] = (int(batch), scene, order, obj)
    return out


def latest_result(case_dir):
    """The newest scored result for one case, or None while it is unfinished."""
    rows = []
    for path in glob.glob(os.path.join(case_dir, "evals", "run-*", "results.jsonl")):
        for line in open(path):
            if line.strip():
                rows.append((os.path.basename(os.path.dirname(path)), json.loads(line)))
    scored = [r for r in rows if not r[1].get("error")]
    pick = sorted(scored or rows, key=lambda r: r[0])
    return pick[-1] if pick else None


SIM_DIED = "sim2 physics stopped"


def sim_died(case_dir, run):
    """Whether the simulator's physics stopped during this attempt.

    Once sim2 logs that line its physics thread is gone for good: joint states and the
    camera stop, so the rest of the trial measures the crash, not the robot.

    Args:
        case_dir: The case folder, holding one aN.log per attempt.
        run: The run folder name of the attempt being exported; its log names it.
    """
    for log in glob.glob(os.path.join(case_dir, "a*.log")):
        try:
            text = open(log, errors="replace").read()
        except OSError:
            continue
        if run in text:
            return SIM_DIED in text
    return False


def trials(full=False):
    """One row per finished case and config.

    Args:
        full: Keep every classify.py field (place_calls, retries, agent habit counts, ...)
            instead of only CLASSIFY_FIELDS.
    """
    index = batches()
    out = []
    for order_idx, (name, _ref, _suite, _env) in enumerate(config_lines()):
        for case_dir in sorted(glob.glob(os.path.join(RESULTS, name, "scenereplica-*"))):
            found = latest_result(case_dir)
            if found is None:
                continue
            run, r = found
            d = r.get("details") or {}
            batch, scene, order, obj = index.get(r["case_id"], (0, "", "", ""))
            lifted = float(d.get("lifted") or 0) >= 1.0
            placed = float(d.get("placed") or 0) >= 1.0
            row = {
                "config": name,
                "config_order": order_idx,
                "batch": batch,
                "case_id": r["case_id"],
                "scene": scene,
                "object": obj,
                "lifted": int(lifted),
                "placed": int(placed),
                "passed": int(bool(r.get("passed"))),
                "score": round(float(r.get("score") or 0), 3),
                "outcome": "placed"
                if placed
                else ("lifted, not placed" if lifted else "not lifted"),
                "cause": d.get("cause")
                or ("none" if r.get("passed") else "error" if r.get("error") else "unlabelled"),
                "grasp_mode": d.get("grasp_mode", ""),
                "settled": "" if d.get("settled") is None else str(d.get("settled")).lower(),
                "ended_by": r.get("ended_by", ""),
                "steps": r.get("steps"),
                "duration_s": round(float(r.get("duration_s") or 0), 1),
                "agent_s": round(float(r.get("agent_duration_s") or 0), 1),
                "cost_usd": round(float(r.get("cost_usd") or 0), 4),
                "run": run,
            }
            found = classify.classify(
                classify.load_trajectory(case_dir, run),
                d,
                bool(r.get("passed")),
                r.get("error") or "",
            )
            row.update(found if full else {k: found[k] for k in CLASSIFY_FIELDS})
            row["sim_died"] = int(sim_died(case_dir, run))
            if row["sim_died"] and not row["passed"]:
                row["stage"], row["stage_detail"] = "sim_died", SIM_DIED
            out.append(row)
    return out


def configs():
    out = []
    for order_idx, (name, ref, suite, env) in enumerate(config_lines()):
        meta = {}
        path = os.path.join(RESULTS, name, "config.json")
        if os.path.exists(path):
            meta = json.load(open(path))
        out.append(
            {
                "config": name,
                "config_order": order_idx,
                "branch": ref,
                "commit": (meta.get("sha") or "")[:9],
                "perception": "Dan" if suite.endswith("_dan") else "OSR",
                "grasp_mode": env.replace("SCENEREPLICA_GRASP=", "") if env != "-" else "default",
                "agent": meta.get("agent", ""),
            }
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", nargs="?", default="trials", choices=("trials", "configs"))
    parser.add_argument(
        "--root", default=BOX, help="evals folder holding results/matrix and matrix/*.tsv"
    )
    parser.add_argument("--set", default="", help="named config set, e.g. scripted")
    args = parser.parse_args()
    set_root(args.root, args.set)
    json.dump({"rows": trials() if args.what == "trials" else configs()}, sys.stdout, indent=1)
