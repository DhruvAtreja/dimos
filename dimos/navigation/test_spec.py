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

from dimos_lcm.actionlib_msgs import GoalStatus
import pytest

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.spec import PREEMPT_REPLACED, goal_ended, goal_id, goal_status


def _goal(ts: float) -> PoseStamped:
    return PoseStamped(ts=ts, position=(1.0, 0.0, 0.0))


_MINE = _goal(1000.5)


@pytest.mark.parametrize("status", [GoalStatus.PENDING, GoalStatus.ACTIVE])
def test_a_goal_the_planner_still_holds_is_not_over(status: int) -> None:
    assert goal_ended(_MINE, goal_status(_MINE, status, "")) is None


@pytest.mark.parametrize(
    "status",
    [GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.PREEMPTED, GoalStatus.REJECTED],
)
def test_a_report_that_ends_the_goal_is_returned(status: int) -> None:
    report = goal_status(_MINE, status, "why")
    assert goal_ended(_MINE, report) is report


@pytest.mark.parametrize(
    "newer",
    [
        pytest.param(_goal(1003.0), id="later second"),
        pytest.param(_goal(1000.75), id="same second, later nanosecond"),
    ],
)
@pytest.mark.parametrize("status", [GoalStatus.ACTIVE, GoalStatus.SUCCEEDED])
def test_a_report_for_a_newer_goal_means_this_one_was_replaced(
    newer: PoseStamped, status: int
) -> None:
    ended = goal_ended(_MINE, goal_status(newer, status, ""))
    assert ended is not None
    assert (ended.goal_id.id, ended.status, ended.text) == (
        goal_id(_MINE),
        GoalStatus.PREEMPTED,
        PREEMPT_REPLACED,
    )


@pytest.mark.parametrize(
    "older",
    [
        pytest.param(_goal(900.0), id="earlier second"),
        pytest.param(_goal(1000.25), id="same second, earlier nanosecond"),
    ],
)
def test_a_report_for_an_older_goal_says_nothing(older: PoseStamped) -> None:
    assert goal_ended(_MINE, goal_status(older, GoalStatus.SUCCEEDED, "reached")) is None
