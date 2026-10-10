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

"""Tell real differences between the eval matrix's configs from run-to-run noise.

Every config runs the same SceneReplica cases, so two configs are compared case by case,
on the cases both have finished. For the grasp rate (object lifted 5 cm) and the place
rate (object at rest in the drop-off) it reports:
  - per config: the rate with a 95% Wilson interval;
  - per config pair: the difference, how many cases each side won, the exact McNemar
    p-value, a 95% interval for the difference (Newcombe's method for paired data), the
    smallest difference the test would catch 4 times in 5 at n=50 and n=100, and a
    plain verdict;
  - per config: how many failed trials each stage (classify.py) took.

    python3 matrix/lib/stats.py --root <this folder on the laptop>

prints a markdown report and writes it to DIR/dash/analysis.md, with the numbers in
DIR/dash/analysis.json. Standard library only.
"""

import argparse
import datetime
import functools
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify
import export

Z95 = 1.959963984540054
ALPHA = 0.05
POWER = 0.8
METRICS = (("grasp", "lifted"), ("place", "placed"))
# (A, B, what the comparison shows); a positive difference means A did better.
PAIRS = (
    ("m1-heuristic", "m0-heuristic", "M1 vs M0, heuristic grasps"),
    ("m1-default", "m0-default", "M1 vs M0, GraspGenX first"),
    ("m1-heuristic", "m1-default", "heuristic vs GraspGenX-first, in M1"),
    ("m1-heuristic", "m1-graspgen", "heuristic vs GraspGenX-only, in M1"),
    ("m1-default", "m1-graspgen", "GraspGenX-first vs GraspGenX-only, in M1"),
    ("m0-heuristic", "m0-default", "heuristic vs GraspGenX-first, in M0"),
)
# Earlier runs of one commit on two machines, kept only to show how much single cases
# flip between identical runs; never mixed into the tests.
RERUN_A = os.path.expanduser(
    "~/.local/state/dimos/evals/run-20261007-235042-7n81tosy/results.jsonl"
)
RERUN_B = "results/baseline/evals/*/run-*/results.jsonl"


def wilson(k, n, z=Z95):
    """95% Wilson interval (low, high) for k successes out of n trials; (0, 1) when n is 0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_pmf(k, n, p):
    """Chance of exactly k successes in n tries that each succeed with chance p."""
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def mcnemar_p(a_won, b_won):
    """Exact two-sided McNemar p-value: how often a split of the discordant cases at least
    this lopsided happens when neither config is better.

    Args:
        a_won: Cases A succeeded on and B failed.
        b_won: Cases B succeeded on and A failed.
    """
    k = a_won + b_won
    if k == 0:
        return 1.0
    tail = sum(binom_pmf(i, k, 0.5) for i in range(min(a_won, b_won) + 1))
    return min(1.0, 2 * tail)


def paired_interval(both, a_won, b_won, neither, z=Z95):
    """95% interval (low, high) for rate(A) - rate(B) on paired cases, Newcombe 1998 method 10.

    Args:
        both, a_won, b_won, neither: Case counts: both succeeded, only A, only B, neither.
    """
    n = both + a_won + b_won + neither
    if n == 0:
        return -1.0, 1.0
    p1, p2 = (both + a_won) / n, (both + b_won) / n
    l1, u1 = wilson(both + a_won, n, z)
    l2, u2 = wilson(both + b_won, n, z)
    denominator = (both + a_won) * (b_won + neither) * (both + b_won) * (a_won + neither)
    phi = (both * neither - a_won * b_won) / math.sqrt(denominator) if denominator else 0.0
    d = p1 - p2
    low = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    high = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, d - low), min(1.0, d + high)


@functools.cache
def _rejects(k):
    """Numbers of A-wins out of k discordant cases that the exact McNemar test calls real."""
    return tuple(b for b in range(k + 1) if mcnemar_p(b, k - b) <= ALPHA)


def mcnemar_power(n, discordance, difference):
    """Chance the exact McNemar test (alpha 0.05) finds a real difference.

    Args:
        n: Paired cases.
        discordance: Share of cases (0-1) where the two configs disagree.
        difference: True rate(A) - rate(B), as a share (0-1), at most the discordance.
    """
    if discordance <= 0 or difference <= 0:
        return 0.0
    a_share = min(1.0, (discordance + difference) / (2 * discordance))
    power = 0.0
    for k in range(n + 1):
        weight = binom_pmf(k, n, discordance)
        if weight < 1e-12:
            continue
        power += weight * sum(binom_pmf(b, k, a_share) for b in _rejects(k))
    return power


def detectable(n, discordance):
    """Smallest true difference (share, 0-1) the test catches with 80% chance at n paired
    cases, or None when no difference within the discordance gets there."""
    if n <= 0 or discordance <= 0 or mcnemar_power(n, discordance, discordance) < POWER:
        return None
    low, high = 0.0, discordance
    for _ in range(30):
        mid = (low + high) / 2
        if mcnemar_power(n, discordance, mid) >= POWER:
            high = mid
        else:
            low = mid
    return high


def holm(p_values):
    """Holm-adjusted p-values, same order, for checking several comparisons at once."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def verdict(name_a, name_b, diff, p, p_holm, need_now):
    """One plain sentence on whether a paired difference is real or noise."""
    pts = abs(diff) * 100
    ahead = name_a if diff > 0 else name_b
    if p_holm < ALPHA:
        return f"Real: {ahead} is ahead by {pts:.0f} points."
    if p < ALPHA:
        return (
            f"Probably real, not yet sure: {ahead} ahead by {pts:.0f} points; "
            "it does not hold up once all comparisons are counted."
        )
    floor = (
        f" (at this n only gaps of about {need_now * 100:.0f}+ points show reliably)"
        if need_now
        else ""
    )
    if p < 0.2:
        return f"A hint only: {ahead} ahead by {pts:.0f} points, could be noise{floor}."
    if diff == 0:
        return f"Noise: no difference{floor}."
    return f"Noise: {ahead} ahead by {pts:.0f} points, well within chance{floor}."


def _pct(x):
    return f"{x * 100:.0f}%"


def compare(rows):
    """Per-config rates and paired comparisons, as JSON-ready dicts.

    Args:
        rows: Trial rows from export.trials(), one per config and case.
    Returns (configs, pairs, floor): per-config summaries, per-pair-and-metric results
    and, per metric, the noise floor at the median discordance of its comparisons.
    Trials whose every run ended in an error (never scored) are left out.
    """
    by = {}
    for r in rows:
        if r["cause"] != "error":
            by.setdefault(r["config"], {})[r["case_id"]] = r
    names = [name for name, *_ in export.config_lines() if name in by]
    configs = []
    for name in names:
        cases = list(by[name].values())
        entry = {"config": name, "n": len(cases), "batches": sorted({r["batch"] for r in cases})}
        for metric, field in METRICS:
            k = sum(r[field] for r in cases)
            low, high = wilson(k, len(cases))
            entry[metric] = {"k": k, "rate": k / len(cases), "lo": low, "hi": high}
        entry["stages"] = {s: sum(r["stage"] == s for r in cases) for s in classify.STAGES}
        configs.append(entry)
    pairs = []
    for a, b, label in PAIRS:
        if a not in by or b not in by:
            continue
        shared = sorted(set(by[a]) & set(by[b]))
        n = len(shared)
        for metric, field in METRICS:
            sa = [by[a][c][field] for c in shared]
            sb = [by[b][c][field] for c in shared]
            both = sum(x and y for x, y in zip(sa, sb, strict=False))
            a_won = sum(x and not y for x, y in zip(sa, sb, strict=False))
            b_won = sum(y and not x for x, y in zip(sa, sb, strict=False))
            neither = n - both - a_won - b_won
            diff = (a_won - b_won) / n if n else 0.0
            low, high = paired_interval(both, a_won, b_won, neither)
            discordance = (a_won + b_won) / n if n else 0.0
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "label": label,
                    "metric": metric,
                    "n": n,
                    "rate_a": sum(sa) / n if n else 0.0,
                    "rate_b": sum(sb) / n if n else 0.0,
                    "diff": diff,
                    "lo": low,
                    "hi": high,
                    "a_won": a_won,
                    "b_won": b_won,
                    "discordance": discordance,
                    "p": mcnemar_p(a_won, b_won),
                    "detectable": {str(m): detectable(m, discordance) for m in (n, 50, 100)},
                }
            )
    for pair, p_holm in zip(pairs, holm([p["p"] for p in pairs]), strict=False):
        pair["p_holm"] = p_holm
        pair["verdict"] = verdict(
            pair["a"],
            pair["b"],
            pair["diff"],
            pair["p"],
            p_holm,
            pair["detectable"][str(pair["n"])],
        )
    floor = {}
    for metric, _ in METRICS:
        shares = [p["discordance"] for p in pairs if p["metric"] == metric]
        median = statistics.median(shares) if shares else 0.0
        floor[metric] = {
            "discordance": median,
            "detectable": {str(m): detectable(m, median) for m in (50, 100)},
        }
    return configs, pairs, floor


def rerun_flips(root):
    """How many shared cases changed outcome between two runs of one commit, or None.

    Args:
        root: The evals folder; reads RERUN_B under it and RERUN_A on this laptop.
    """

    def load(paths):
        rows = {}
        for path in sorted(paths):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error"):
                        rows[r["case_id"]] = r.get("details") or {}
        return rows

    first = load(glob.glob(RERUN_A))
    second = load(glob.glob(os.path.join(root, RERUN_B)))
    shared = sorted(set(first) & set(second))
    if not shared:
        return None
    out = {"n": len(shared)}
    for metric, field in METRICS:
        a = [float(first[c].get(field) or 0) >= 1 for c in shared]
        b = [float(second[c].get(field) or 0) >= 1 for c in shared]
        out[metric] = {
            "first": sum(a),
            "second": sum(b),
            "flipped": sum(x != y for x, y in zip(a, b, strict=False)),
        }
    return out


def report(configs, pairs, floor, rerun):
    """The markdown report."""
    lines = [
        "# SceneReplica matrix: real differences or noise?",
        "",
        f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Grasp = object lifted 5 cm; "
        "place = object at rest in the drop-off. Points = percentage points.",
        "",
        "## Rates per config (95% interval)",
        "",
        "| config | trials | batches | grasp | place |",
        "|---|---:|---|---|---|",
    ]
    for c in configs:
        cells = [
            f"{c[m]['k']}/{c['n']} = {_pct(c[m]['rate'])} ({_pct(c[m]['lo'])}-{_pct(c[m]['hi'])})"
            for m, _ in METRICS
        ]
        lines.append(
            f"| {c['config']} | {c['n']} | {','.join(map(str, c['batches']))} | {cells[0]} | {cells[1]} |"
        )
    lines += [
        "",
        "## Paired comparisons (same cases, A minus B)",
        "",
        '"A won" = cases only A got right; "B won" = only B. p = exact McNemar; '
        "Holm p = the same after allowing for all comparisons at once. "
        '"Detectable" = smallest true gap the test catches 4 times in 5 at that many paired cases, '
        "given how often the two configs disagree.",
        "",
    ]
    for metric, _ in METRICS:
        lines += [
            f"### {metric.capitalize()} rate",
            "",
            "| A vs B | n | A | B | A-B (95% interval) | A won / B won | p | Holm p | detectable now / n=50 / n=100 | verdict |",
            "|---|---:|---:|---:|---|---|---:|---:|---|---|",
        ]
        for p in (p for p in pairs if p["metric"] == metric):
            det = " / ".join(
                "—" if p["detectable"][k] is None else f"{p['detectable'][k] * 100:.0f}"
                for k in (str(p["n"]), "50", "100")
            )
            lines.append(
                f"| {p['a']} vs {p['b']} ({p['label']}) | {p['n']} | {_pct(p['rate_a'])} | {_pct(p['rate_b'])} "
                f"| {p['diff'] * 100:+.0f} ({p['lo'] * 100:+.0f} to {p['hi'] * 100:+.0f}) | {p['a_won']} / {p['b_won']} "
                f"| {p['p']:.2f} | {p['p_holm']:.2f} | {det} | {p['verdict']} |"
            )
        lines.append("")
    lines += ["## Noise floor", ""]
    for metric, _ in METRICS:
        det = floor[metric]["detectable"]
        lines.append(
            f"- {metric.capitalize()}: two configs disagree on {_pct(floor[metric]['discordance'])} of shared cases "
            f"(median of the comparisons above). A real gap must be about {_fmt(det['50'])} points to be caught "
            f"reliably with 50 paired cases, about {_fmt(det['100'])} with 100."
        )
    lines += ["", "Smaller gaps can be real, but this matrix cannot tell them from noise.", ""]
    if rerun:
        g, p = rerun["grasp"], rerun["place"]
        lines += [
            f"Same-code reruns (not used in the tests): one commit run twice on {rerun['n']} shared cases "
            f"lifted {g['first']} vs {g['second']} and placed {p['first']} vs {p['second']} in total, yet "
            f"{g['flipped']} cases flipped their grasp result and {p['flipped']} their place result. "
            "Single cases flip often between identical runs; only the totals over many cases mean something.",
            "",
        ]
    stages = [s for s in classify.STAGES if any(c["stages"][s] for c in configs)]
    lines += [
        "## Where trials were lost (classify.py stages)",
        "",
        "| stage | " + " | ".join(c["config"] for c in configs) + " |",
        "|---|" + "---:|" * len(configs),
    ]
    for s in stages:
        lines.append(f"| {s} | " + " | ".join(str(c["stages"][s]) for c in configs) + " |")
    return "\n".join(lines) + "\n"


def _fmt(share):
    return "—" if share is None else f"{share * 100:.0f}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.dirname(here)),
        help="evals folder holding results/matrix and matrix/*.tsv (default: this checkout)",
    )
    parser.add_argument(
        "--out", help="folder for analysis.md and analysis.json (default: ROOT/dash)"
    )
    args = parser.parse_args()
    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out) if args.out else os.path.join(root, "dash")
    export.set_root(root)
    configs, pairs, floor = compare(export.trials())
    rerun = rerun_flips(root)
    text = report(configs, pairs, floor, rerun)
    print(text)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "analysis.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out, "analysis.json"), "w") as f:
        json.dump(
            {
                "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                "configs": configs,
                "pairs": pairs,
                "noise_floor": floor,
                "same_code_rerun": rerun,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
