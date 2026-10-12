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
from collections.abc import Generator
from unittest.mock import Mock

from dimos_lcm.actionlib_msgs import GoalStatus
import pytest
from pytest_mock import MockerFixture

from dimos.agents.skills.point_nav import PointNavSkillContainer
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.navigation.spec import PREEMPT_CANCELED, PREEMPT_REPLACED, goal_status

_GOAL = PoseStamped(ts=1000.0, position=(1.0, 0.0, 0.0))


@pytest.fixture()
def container(mocker: MockerFixture) -> Generator[PointNavSkillContainer, None, None]:
    module = PointNavSkillContainer()
    mocker.patch.object(module, "_navigation", mocker.Mock(), create=True)
    mocker.patch.object(module, "_pose", return_value=None)
    mocker.patch.object(module, "tool_update")
    mocker.patch.object(module, "stop_tool")
    module._goal = _GOAL
    try:
        yield module
    finally:
        module._goal = None
        module._close_module()


def _updates(module: PointNavSkillContainer) -> list[str]:
    tool_update: Mock = module.tool_update  # type: ignore[assignment]
    return [call.args[1] for call in tool_update.call_args_list]


@pytest.mark.parametrize(
    ("status", "text", "outcome"),
    [
        (GoalStatus.SUCCEEDED, "reached", "Reached the target"),
        (GoalStatus.PREEMPTED, PREEMPT_REPLACED, "Replaced by another goal"),
        (GoalStatus.PREEMPTED, PREEMPT_CANCELED, "Cancelled"),
        (GoalStatus.REJECTED, "bad goal", "Failed: bad goal"),
        (GoalStatus.ABORTED, "no safe path", "Blocked: no safe path"),
    ],
)
def test_a_report_that_ends_the_goal_ends_the_go_to(
    container: PointNavSkillContainer, status: int, text: str, outcome: str
) -> None:
    asyncio.run(container.handle_nav_status(goal_status(_GOAL, status, text)))

    assert _updates(container) == [f"{outcome}; robot position unknown."]
    assert container._goal is None
    # The planner ended the goal itself, so there is nothing to cancel.
    container._navigation.cancel_goal.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "report",
    [
        pytest.param(goal_status(_GOAL, GoalStatus.ACTIVE, "following the path"), id="active"),
        pytest.param(
            goal_status(PoseStamped(ts=900.0), GoalStatus.SUCCEEDED, "reached"), id="older goal"
        ),
    ],
)
def test_a_report_that_does_not_end_the_goal_is_ignored(
    container: PointNavSkillContainer, report: GoalStatus
) -> None:
    asyncio.run(container.handle_nav_status(report))

    assert _updates(container) == []
    assert container._goal is _GOAL
