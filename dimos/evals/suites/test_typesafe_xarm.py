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

"""The TypeSafe xArm suite is consistent with the agent and the scene file."""

from __future__ import annotations

from dimos.evals.agents.typesafe_policy import (
    DEFAULT_SCENE,
    TypeSafePolicy,
    derive_goal_label,
    load_scene,
)
from dimos.evals.suites.typesafe_xarm import BODIES, CASES, SUITE


def test_every_instruction_names_the_graded_body() -> None:
    """The agent and the grader must agree on the goal, resolved the same way."""
    scene = load_scene(DEFAULT_SCENE)
    for _, instruction, tag in CASES:
        assert derive_goal_label(instruction, scene) == BODIES[tag]


def test_cases_are_unique_and_preflight_with_the_agent() -> None:
    ids = [case.id for case in SUITE]
    assert len(ids) == len(set(ids))
    for case in SUITE:
        TypeSafePolicy().preflight(case.environment)
        assert "typesafe" in case.tags
