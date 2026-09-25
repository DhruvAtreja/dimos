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

"""Pick-up cases on the xArm7 table scene, driven by the TypeSafe (Jev) policy.

No perception stack: the policy reads ground-truth poses from ``tf`` (the
simulator publishes the tracked bodies, the planner its tip link) and acts
through the planner skills. Full credit once the goal body ends the episode
5 cm or more above where it started, the same bar as ``mujoco_xarm``.

    dimos evals run dimos.evals.suites.typesafe_xarm \\
        --agent dimos.evals.agents.typesafe_policy \\
        --tags cylinder            # or: apple; omit to run both
    # optional: --set step_m=0.02 --set 'rpy_rad=[3.1416, 0, 0]' --set max_ticks=60
"""

from __future__ import annotations

from dimos.evals.suites.mujoco_xarm import arm_only_environment, lifted
from dimos.evals.types import EvalCase, Suite

LIFT_M = 0.05

# (case id, instruction, tag) — the instruction must name the goal by a label
# or alias from scenes/xarm7_table.json; the agent resolves it in code.
CASES: tuple[tuple[str, str, str], ...] = (
    ("typesafe_xarm_cylinder", "Pick up the cylinder and hold it in the air.", "cylinder"),
    (
        "typesafe_xarm_apple",
        "Pick up the apple and hold it in the air above the table.",
        "apple",
    ),
)
BODIES = {"cylinder": "cup", "apple": "apple"}

SUITE: Suite = [
    EvalCase(
        id=case_id,
        inputs=instruction,
        # Rerun stays on: on macOS the sim is headless and Rerun is the only view.
        environment=arm_only_environment(rerun=True),
        grade=lifted(BODIES[tag], by_m=LIFT_M),
        timeout_s=600.0,
        tags=frozenset({"mujoco", "manipulation", "pick", "typesafe", tag}),
    )
    for case_id, instruction, tag in CASES
]
