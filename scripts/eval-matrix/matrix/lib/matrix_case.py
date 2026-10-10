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

"""One case of an eval suite, so the matrix runs exactly one trial per container.

`dimos evals run --tags` keeps a case that has any of the given tags, so it cannot pick a
single case. This module is a suite of its own that holds just one case of another:

    MATRIX_SUITE=dimos.evals.suites.scenereplica MATRIX_CASE=scenereplica-scene-03-bowl \
        PYTHONPATH=/matrix dimos evals run matrix_case --agent dimos.evals.agents.pi

MATRIX_SUITE   the suite module to take the case from.
MATRIX_CASE    the case id.
MATRIX_BLUEPRINT  optional: a blueprint name that replaces the case's first blueprint (the
                  perception stack), for a branch that has the blueprint but no setting
                  to choose it. Empty or unset keeps the suite's own.
MATRIX_TIMEOUT_S  seconds the agent gets in one trial (default 300). The trial then ends
                  and is graded as it stands. 0 keeps the suite's own limit.

    python matrix_case.py list SUITE   prints "case_id<TAB>tag,tag,...<TAB>blueprints" for
                                       every case, with MATRIX_BLUEPRINT applied.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys

# Agent time per trial, in seconds: 95% of successful trials finish within about 5 min of
# agent time, while failing ones otherwise run to the suite's 15 min.
DEFAULT_TIMEOUT_S = 300.0


def _suite(name: str):
    return importlib.import_module(name).SUITE


def _one_case() -> list:
    suite_name = os.environ["MATRIX_SUITE"]
    case_id = os.environ["MATRIX_CASE"]
    cases = [case for case in _suite(suite_name) if case.id == case_id]
    if len(cases) != 1:
        raise SystemExit(f"{suite_name} has {len(cases)} cases with id {case_id!r}, expected 1")
    timeout_s = float(os.environ.get("MATRIX_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    if timeout_s > 0:
        cases = [dataclasses.replace(cases[0], timeout_s=timeout_s)]
    blueprint = os.environ.get("MATRIX_BLUEPRINT", "")
    if blueprint:
        # The list object is shared by the suite's cases; only this one case is loaded.
        cases[0].environment.config.blueprint[0] = blueprint
    print(
        f"matrix case: {case_id} from {suite_name}, "
        f"blueprint {' '.join(cases[0].environment.config.blueprint)}, "
        f"agent time limit {cases[0].timeout_s:.0f} s",
        file=sys.stderr,
        flush=True,
    )
    return cases


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "list":
        override = os.environ.get("MATRIX_BLUEPRINT", "")
        for case in _suite(sys.argv[2]):
            blueprint = list(case.environment.config.blueprint)
            if override:
                blueprint[0] = override
            print(f"{case.id}\t{','.join(sorted(case.tags))}\t{' '.join(blueprint)}")
    else:
        raise SystemExit(__doc__)
else:
    SUITE = _one_case()
