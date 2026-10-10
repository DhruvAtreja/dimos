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

"""Host-side helpers for matrix.sh: the batch list, config checks, progress and the report.

    python3 matrix.py batches LISTING OUT       write the batch list from a case listing
    python3 matrix.py check CONFIG LISTING     compare a config's case listing with the batches
    python3 matrix.py cases N                  case ids of batch N, one per line
    python3 matrix.py status-case CONFIG CASE  done / error / missing / new for one case
    python3 matrix.py progress [N]             per config: trials done of 100 (and of batch N)
    python3 matrix.py report [CONFIG]          the side-by-side table, or one config's detail

A LISTING is the output of `python matrix_case.py list SUITE` run in a container:
"case_id<TAB>tags<TAB>blueprint" per line. Paths come from the environment matrix.sh sets:
MATRIX_HOME (the dimos-evals folder) and MATRIX_CONFIGS (the configs file).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time

HOME = Path(os.environ.get("MATRIX_HOME", Path(__file__).resolve().parents[2]))
CONFIGS = Path(os.environ.get("MATRIX_CONFIGS", HOME / "matrix" / "configs.tsv"))
BATCHES = Path(os.environ.get("MATRIX_BATCHES") or HOME / "matrix" / "batches.tsv")
RESULTS = Path(os.environ.get("MATRIX_RESULTS", HOME / "results" / "matrix"))
N_BATCHES = 10
MAX_ATTEMPTS = 2  # a case whose trial ended in an error is run once more


# ---------------------------------------------------------------- configs and batches


def configs() -> list[dict]:
    """Every config line, in file order."""
    out = []
    for line in CONFIGS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, ref, suite, parallel, blueprint, *env = line.split()
        env = [] if env in ([], ["-"]) else env
        out.append(
            {
                "name": name,
                "ref": ref,
                "suite": suite,
                "parallel": int(parallel),
                "blueprint": "" if blueprint == "-" else blueprint,
                "env": env,
            }
        )
    return out


def batches() -> dict[int, list[str]]:
    """Batch number -> its case ids, from batches.tsv."""
    out: dict[int, list[str]] = {}
    for line in BATCHES.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        batch, case_id = line.split("\t")[:2]
        out.setdefault(int(batch), []).append(case_id)
    return out


def read_listing(path: Path) -> dict[str, tuple[set[str], str]]:
    """case id -> (tags, blueprint) from a case listing."""
    cases = {}
    for line in path.read_text().splitlines():
        if "\t" not in line:
            continue
        case_id, tags, blueprint = ([*line.split("\t"), ""])[:3]
        cases[case_id] = (set(tags.split(",")), blueprint)
    return cases


def scene_order(tags: set[str]) -> tuple[int, int]:
    scene = [int(t[6:]) for t in tags if re.fullmatch(r"scene-\d+", t)]
    order = [int(t[6:]) for t in tags if re.fullmatch(r"order-\d+", t)]
    if len(scene) != 1 or len(order) != 1:
        raise SystemExit(f"case tags without one scene-NN and one order-K: {sorted(tags)}")
    return scene[0], order[0]


def make_batches(listing: Path, out: Path) -> None:
    """Batch 2k-1 is order-k of scenes 01-10, batch 2k is order-k of scenes 11-20."""
    cases = read_listing(listing)
    by_slot = {}
    for case_id, (tags, _) in cases.items():
        slot = scene_order(tags)
        if slot in by_slot:
            raise SystemExit(f"two cases for scene {slot[0]} order {slot[1]}")
        by_slot[slot] = case_id
    rows = []
    for order in range(1, 6):
        for half in (0, 1):
            batch = 2 * order - 1 + half
            for scene in range(1 + 10 * half, 11 + 10 * half):
                case_id = by_slot.pop((scene, order), None)
                if case_id is None:
                    raise SystemExit(f"no case for scene {scene:02d} order-{order}")
                obj = case_id.split(f"scene-{scene:02d}-", 1)[-1]
                rows.append(f"{batch}\t{case_id}\tscene-{scene:02d}\torder-{order}\t{obj}")
    if by_slot or len(rows) != 100:
        raise SystemExit(f"cases left over or missing: {sorted(by_slot.values())}")
    header = (
        "# batch\tcase_id\tscene\torder\tobject\n"
        "# Batch 2k-1 = the k-th nearest object (order-k) of scenes 01-10, batch 2k = the same\n"
        "# for scenes 11-20. All 100 SceneReplica cases, each exactly once; every config runs\n"
        "# the same list, so results pair case by case.\n"
    )
    out.write_text(header + "\n".join(rows) + "\n")
    print(f"wrote {out}: {len(rows)} cases in {N_BATCHES} batches")


def check(config_name: str, listing: Path) -> int:
    """Print problems with a config's case listing; exit status 1 if there are any."""
    cfg = next((c for c in configs() if c["name"] == config_name), None)
    if cfg is None:
        print(f"{config_name}: not in {CONFIGS}")
        return 1
    cases = read_listing(listing)
    problems = []
    wanted = [case_id for ids in batches().values() for case_id in ids]
    missing = [c for c in wanted if c not in cases]
    if missing:
        problems.append(f"suite lacks {len(missing)} batch case(s), e.g. {missing[0]}")
    # The main list puts order-k cases in batches 2k-1 and 2k; a set's own list (MATRIX_BATCHES)
    # groups cases its own way.
    for batch, ids in [] if os.environ.get("MATRIX_BATCHES") else batches().items():
        for case_id in ids:
            if case_id in cases:
                order = scene_order(cases[case_id][0])[1]
                if batch not in (2 * order - 1, 2 * order):
                    problems.append(f"{case_id} is order-{order} here, not in batch {batch}")
    env = dict(e.split("=", 1) for e in cfg["env"])
    mode = env.get("SCENEREPLICA_GRASP")
    any_tags = next(iter(cases.values()))[0] if cases else set()
    if mode and f"grasp-{mode}" not in any_tags:
        problems.append(
            f"SCENEREPLICA_GRASP={mode} is set but this checkout's suite does not read it "
            "(no grasp-<mode> tags): the trials would run the suite's fixed grasp setup"
        )
    blueprints = {bp for _, bp in cases.values()}
    print(f"{config_name}: {len(cases)} cases, blueprint {' | '.join(sorted(blueprints))}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return 1 if problems else 0


# ---------------------------------------------------------------- results


def rows(case_dir: Path) -> list[tuple[Path, dict]]:
    """(run dir, results row) for every attempt of a case that wrote a row, oldest first."""
    out = []
    for results in sorted(case_dir.glob("evals/run-*/results.jsonl")):
        try:
            text = results.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.strip():
                out.append((results.parent, json.loads(line)))
    return out


def attempts(case_dir: Path) -> int:
    return len(list(case_dir.glob("a*.log")))


def case_state(config_name: str, case_id: str) -> str:
    """done: has a clean row, or an error row after MAX_ATTEMPTS tries. error: an error row
    with tries left. missing: tried, no row. new: never tried."""
    case_dir = RESULTS / config_name / case_id
    found = rows(case_dir)
    if any(not row.get("error") for _, row in found):
        return "done"
    if found:
        return "done" if attempts(case_dir) >= MAX_ATTEMPTS else "error"
    return "missing" if attempts(case_dir) else "new"


def chosen(config_name: str, case_id: str) -> tuple[Path, dict] | None:
    """The row a case counts with: its first clean row, else its last error row."""
    found = rows(RESULTS / config_name / case_id)
    clean = [r for r in found if not r[1].get("error")]
    if clean:
        return clean[0]
    return found[-1] if found else None


def progress(batch: int | None) -> None:
    all_batches = batches()
    for cfg in configs():
        name = cfg["name"]
        total = sum(case_state(name, c) == "done" for ids in all_batches.values() for c in ids)
        line = f"{name}: {total}/{sum(len(ids) for ids in all_batches.values())}"
        if batch:
            ids = all_batches[batch]
            line += f", batch {batch}: {sum(case_state(name, c) == 'done' for c in ids)}/{len(ids)}"
        print(line)


def _summary_module():
    """dimos.evals.suites.lib.scenereplica_summary from the first checkout that has it."""
    for cfg in configs():
        src = HOME / "src" / "matrix" / cfg["name"]
        if (src / "dimos/evals/suites/lib/scenereplica_summary.py").is_file():
            sys.path.insert(0, str(src))
            from dimos.evals.suites.lib import scenereplica_summary

            return scenereplica_summary
    raise SystemExit("no checkout with dimos/evals/suites/lib/scenereplica_summary.py")


def _meta(config_name: str) -> dict:
    # Written by `matrix.sh prepare`, always under results/matrix.
    path = HOME / "results" / "matrix" / config_name / "config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def report(only: str | None) -> None:
    summary = _summary_module()
    all_batches = batches()
    cfgs = [c for c in configs() if only in (None, c["name"])]
    if not cfgs:
        raise SystemExit(f"no config {only!r}")
    # trials[config][case] = Trial; errors counted apart.
    trials: dict[str, dict[str, object]] = {}
    shas: dict[str, set[str]] = {}
    modes: dict[str, set[str]] = {}
    for cfg in cfgs:
        name = cfg["name"]
        trials[name], shas[name], modes[name] = {}, set(), set()
        for ids in all_batches.values():
            for case_id in ids:
                if case_state(name, case_id) != "done":
                    continue
                run_dir, row = chosen(name, case_id)
                found = summary.load_trials([run_dir])
                if found:
                    trials[name][case_id] = found[0]
                manifest = run_dir / "manifest.json"
                if manifest.is_file():
                    sha = (json.loads(manifest.read_text()).get("code") or {}).get("git_sha")
                    if sha:
                        shas[name].add(sha[:9])
                mode = (row.get("details") or {}).get("grasp_mode")
                modes[name].add(mode or "fixed by suite")

    print(f"SceneReplica matrix, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    agents = {_meta(c["name"]).get("agent", "pi (code defaults)") for c in cfgs}
    if len(agents) == 1:
        print(f"agent: {agents.pop()}")
    width = max(11, *(len(c["name"]) for c in cfgs))
    for cfg in cfgs:
        name, meta = cfg["name"], _meta(cfg["name"])
        settings = " ".join(cfg["env"]) or "-"
        if cfg["blueprint"]:
            settings += f" blueprint={cfg['blueprint']}"
        if cfg["suite"] != "dimos.evals.suites.scenereplica":
            settings += f" suite={cfg['suite'].rsplit('.', 1)[-1]}"
        print(f"  {name:<{width}} {cfg['ref']} @ {meta.get('sha', '?')[:9]}  {settings}")
        ran = sorted(shas[name])
        if ran and ran != [meta.get("sha", "")[:9]]:
            print(f"  {'':<{width}} NOTE: trials ran on commit(s) {', '.join(ran)}")
        if trials[name]:
            print(
                f"  {'':<{width}} grasp mode recorded in results: {', '.join(sorted(modes[name]))}"
            )
        if len(agents) > 1:
            print(f"  {'':<{width}} agent: {meta.get('agent', '?')}")

    if only:
        print()
        print(summary.report(list(trials[only].values())))
        return

    def cell(name: str, ids: list[str]) -> str:
        got = [trials[name][c] for c in ids if c in trials[name]]
        if not got:
            return "-"
        return f"{sum(t.grasped for t in got)}/{sum(t.placed for t in got)} of {len(got)}"

    print()
    print("grasped/placed of trials finished (grasped = lifted 5 cm, placed = in the drop-off)")
    print("batch   " + "".join(f"{c['name']:>{width + 2}}" for c in cfgs))
    for batch in range(1, N_BATCHES + 1):
        ids = all_batches[batch]
        cells = [cell(c["name"], ids) for c in cfgs]
        if all(x == "-" for x in cells):
            continue
        print(f"{batch:<8}" + "".join(f"{x:>{width + 2}}" for x in cells))
    every = [c for ids in all_batches.values() for c in ids]
    print(f"{'total':<8}" + "".join(f"{cell(c['name'], every):>{width + 2}}" for c in cfgs))
    shared = [c for c in every if all(c in trials[cfg["name"]] for cfg in cfgs)]
    if len(cfgs) > 1:
        print(
            f"{'paired':<8}"
            + "".join(f"{cell(c['name'], shared):>{width + 2}}" for c in cfgs)
            + f"   (the {len(shared)} cases every config has finished)"
        )
    for cfg in cfgs:
        got = list(trials[cfg["name"]].values())
        errors = sum(t.cause == "error" for t in got)
        if errors:
            print(f"  {cfg['name']}: {errors} trial(s) ended in an error (counted as failed)")


def main(argv: list[str]) -> int:
    if argv[:1] == ["batches"] and len(argv) == 3:
        make_batches(Path(argv[1]), Path(argv[2]))
    elif argv[:1] == ["check"] and len(argv) == 3:
        return check(argv[1], Path(argv[2]))
    elif argv[:1] == ["cases"] and len(argv) == 2:
        print("\n".join(batches()[int(argv[1])]))
    elif argv[:1] == ["status-case"] and len(argv) == 3:
        print(case_state(argv[1], argv[2]))
    elif argv[:1] == ["progress"]:
        progress(int(argv[1]) if len(argv) > 1 else None)
    elif argv[:1] == ["report"]:
        report(argv[1] if len(argv) > 1 else None)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
