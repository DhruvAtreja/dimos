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

import asyncio
from collections.abc import Awaitable, Generator
from typing import NamedTuple, cast
from unittest.mock import Mock

from dimos_lcm.actionlib_msgs import GoalStatus
import pytest
from pytest_mock import MockerFixture

from dimos.agents.skills.point_nav import PointNavSkillContainer
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.spec import PREEMPT_CANCELED, PREEMPT_REPLACED, goal_status

_GOAL = PoseStamped(ts=1000.0, position=(1.0, 0.0, 0.0))


class _Rig(NamedTuple):
    module: PointNavSkillContainer
    navigation: Mock
    pose: Mock
    tool_update: Mock

    def updates(self) -> list[str]:
        return [call.args[1] for call in self.tool_update.call_args_list]


def _result(called: Awaitable[str]) -> str:
    """The return of an async skill, which has already run when called off the module loop."""
    return cast("str", called)


@pytest.fixture()
def rig(mocker: MockerFixture) -> Generator[_Rig, None, None]:
    module = PointNavSkillContainer(base_height_m=0.3)
    navigation = mocker.Mock()
    mocker.patch.object(module, "_navigation", navigation, create=True)
    pose = mocker.patch.object(module, "_pose", return_value=None)
    tool_update = mocker.patch.object(module, "tool_update")
    mocker.patch.object(module, "start_tool")
    mocker.patch.object(module, "stop_tool")
    try:
        yield _Rig(module, navigation, pose, tool_update)
    finally:
        module._goal = None
        module._close_module()


@pytest.mark.parametrize(
    ("status", "text", "outcome"),
    [
        (GoalStatus.SUCCEEDED, "reached", "Reached the target"),
        (GoalStatus.PREEMPTED, PREEMPT_REPLACED, "Replaced by another goal"),
        (GoalStatus.PREEMPTED, PREEMPT_CANCELED, "Canceled"),
        (GoalStatus.REJECTED, "bad goal", "Failed: bad goal"),
        (GoalStatus.ABORTED, "no safe path", "Blocked: no safe path"),
    ],
)
def test_a_report_that_ends_the_goal_ends_the_go_to(
    rig: _Rig, status: int, text: str, outcome: str
) -> None:
    rig.module._goal = _GOAL
    asyncio.run(rig.module.handle_nav_status(goal_status(_GOAL, status, text)))

    assert rig.updates() == [f"{outcome}; robot position unknown."]
    assert rig.module._goal is None
    # The planner ended the goal itself, so there is nothing to cancel.
    rig.navigation.cancel_goal.assert_not_called()


@pytest.mark.parametrize(
    "report",
    [
        pytest.param(goal_status(_GOAL, GoalStatus.ACTIVE, "following the path"), id="active"),
        pytest.param(
            goal_status(PoseStamped(ts=900.0), GoalStatus.SUCCEEDED, "reached"), id="older goal"
        ),
    ],
)
def test_a_report_that_does_not_end_the_goal_is_ignored(rig: _Rig, report: GoalStatus) -> None:
    rig.module._goal = _GOAL
    asyncio.run(rig.module.handle_nav_status(report))

    assert rig.updates() == []
    assert rig.module._goal is _GOAL


def test_go_to_sends_the_point_at_the_height_of_the_floor_under_the_robot(rig: _Rig) -> None:
    rig.pose.return_value = PoseStamped(position=(0.0, 0.0, 1.5))

    _result(rig.module.go_to(4.0, 2.0))

    (goal,) = (call.args[0] for call in rig.navigation.set_goal.call_args_list)
    assert (goal.x, goal.y, goal.z) == pytest.approx((4.0, 2.0, 1.2))
    assert rig.module._goal is goal


def test_go_to_sends_nothing_without_a_pose(rig: _Rig) -> None:
    assert _result(rig.module.go_to(4.0, 2.0)).startswith("No goal sent")
    rig.navigation.set_goal.assert_not_called()


def test_a_second_go_to_is_refused_while_one_is_under_way(rig: _Rig) -> None:
    rig.module._goal = _GOAL

    assert _result(rig.module.go_to(4.0, 2.0)).startswith("Already navigating")
    rig.navigation.set_goal.assert_not_called()


def test_stop_navigation_cancels_the_goal_and_reports_how_far_it_was(rig: _Rig) -> None:
    rig.module._goal = _GOAL
    rig.pose.return_value = PoseStamped(position=(1.0, 3.0, 0.0))

    assert _result(rig.module.stop_navigation()) == "Stopped."

    assert rig.updates() == ["Canceled; robot at (1.00, 3.00), 3.00 m from the target."]
    rig.navigation.cancel_goal.assert_called_once_with()
