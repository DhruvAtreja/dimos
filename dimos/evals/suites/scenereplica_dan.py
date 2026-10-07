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


"""The SceneReplica suite on ``xarm-perception-sim2-dan``, where the localize memory finds
the objects. The 100 trials, case ids, tags, prompt and grader are those of
``dimos.evals.suites.scenereplica``; only the perception stack differs, so the two runs'
rows line up case by case.

    dimos evals run dimos.evals.suites.scenereplica_dan --agent dimos.evals.agents.pi --tags scene-01
    python -m dimos.evals.suites.lib.scenereplica_summary ~/.local/state/dimos/evals/run-*
"""

from __future__ import annotations

from dimos.evals.suites.scenereplica import suite
from dimos.evals.types import Suite
from dimos.utils.data import get_data

BLUEPRINT = ["xarm-perception-sim2-dan", "mcp-server", "observe-skill"]

SUITE: Suite = suite(get_data("scenereplica"), BLUEPRINT)
