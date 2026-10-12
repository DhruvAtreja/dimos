# Copyright 2025-2026 Dimensional Inc.
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

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
import difflib
import math
from typing import Any
from unittest.mock import Mock

from dimos_lcm.actionlib_msgs import GoalStatus
from langchain_core.messages import HumanMessage
import pytest
from pytest_mock import MockerFixture

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out, Stream, Transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.spec import PREEMPT_CANCELED, PREEMPT_REPLACED, goal_status
from dimos.robot.unitree.unitree_skill_container import (
    _UNITREE_COMMANDS,
    UnitreeSkillContainer,
    _goal_pose,
)


class StubNavigation(Module):
    nav_status: Out[GoalStatus]

    @rpc
    def set_goal(self, goal: PoseStamped) -> bool:
        return True

    @rpc
    def get_status(self) -> GoalStatus | None:
        return None

    @rpc
    def cancel_goal(self) -> bool:
        return True


class StubGO2Connection(Module):
    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        return {}


class MockedUnitreeSkill(UnitreeSkillContainer):
    pass


def test_pounce(agent_setup) -> None:
    history = agent_setup(
        blueprints=[
            MockedUnitreeSkill.blueprint(),
            StubNavigation.blueprint(),
            StubGO2Connection.blueprint(),
        ],
        messages=[HumanMessage("Pounce! Use the execute_sport_command tool.")],
    )

    response = history[-1].content.lower()
    assert "pounce" in response


def test_did_you_mean() -> None:
    suggestions = difflib.get_close_matches("Pounce", _UNITREE_COMMANDS.keys(), n=3, cutoff=0.6)
    assert "FrontPounce" in suggestions
    assert "Pose" in suggestions


def _pose(x: float, y: float, yaw_deg: float) -> PoseStamped:
    return PoseStamped(
        position=Vector3(x, y, 0.3),
        orientation=Quaternion.from_euler(Vector3(0, 0, math.radians(yaw_deg))),
    )


def _xy(pose: PoseStamped) -> tuple[float, float]:
    return pose.position.x, pose.position.y


def _yaw_deg(pose: PoseStamped) -> float:
    return math.degrees(pose.orientation.to_euler().yaw)


def test_move_to_world_keeps_heading() -> None:
    goal = _goal_pose(_pose(1, 1, 90), 4, 1, None, relative=False)
    assert _xy(goal) == pytest.approx((4, 1))
    assert goal.position.z == pytest.approx(0.3)
    assert _yaw_deg(goal) == pytest.approx(90)


def test_move_to_world_explicit_heading_is_absolute() -> None:
    goal = _goal_pose(_pose(0, 0, 45), 2, 2, 90, relative=False)
    assert _xy(goal) == pytest.approx((2, 2))
    assert _yaw_deg(goal) == pytest.approx(90)


def test_move_to_relative_offset_rotates_into_world() -> None:
    # Facing north, 2 m forward and 1 m left lands at (-1, 2), still facing north.
    goal = _goal_pose(_pose(0, 0, 90), 2, 1, None, relative=True)
    assert _xy(goal) == pytest.approx((-1, 2))
    assert _yaw_deg(goal) == pytest.approx(90)


def test_move_to_relative_turn_in_place() -> None:
    goal = _goal_pose(_pose(0, 0, 90), 0, 0, -90, relative=True)
    assert _xy(goal) == pytest.approx((0, 0))
    assert _yaw_deg(goal) == pytest.approx(0)


class _StatusTransport(Transport[GoalStatus]):
    """Hands reports to the subscribers still held."""

    def __init__(self) -> None:
        self.subscribers: list[Callable[[GoalStatus], object]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def broadcast(self, selfstream: Stream[GoalStatus] | None, value: GoalStatus) -> None:
        for subscriber in self.subscribers:
            subscriber(value)

    def subscribe(
        self,
        callback: Callable[[GoalStatus], object],
        selfstream: Stream[GoalStatus] | None = None,
    ) -> Callable[[], None]:
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)


@pytest.fixture()
def navigating(
    mocker: MockerFixture,
) -> Generator[tuple[UnitreeSkillContainer, _StatusTransport, Mock], None, None]:
    transport = _StatusTransport()
    container = UnitreeSkillContainer()
    container.nav_status.transport = transport
    navigation = mocker.Mock()
    mocker.patch.object(container, "_navigation", navigation, create=True)
    try:
        yield container, transport, navigation
    finally:
        container._close_module()


def _navigate(container: UnitreeSkillContainer, goal: PoseStamped) -> str:
    """Run _navigate, failing instead of hanging when it never returns."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(container._navigate, goal).result(timeout=5.0)


@pytest.mark.parametrize(
    ("status", "text", "outcome"),
    [
        (GoalStatus.SUCCEEDED, "reached", "Navigation goal reached"),
        (GoalStatus.PREEMPTED, PREEMPT_REPLACED, "Navigation was replaced by another goal"),
        (GoalStatus.PREEMPTED, PREEMPT_CANCELED, "Navigation was canceled"),
        (GoalStatus.ABORTED, "no safe path", "Navigation failed: no safe path"),
        (GoalStatus.REJECTED, "bad goal", "Navigation failed: bad goal"),
    ],
)
def test_navigate_returns_how_the_goal_ended(
    navigating: tuple[UnitreeSkillContainer, _StatusTransport, Mock],
    status: int,
    text: str,
    outcome: str,
) -> None:
    container, transport, navigation = navigating
    earlier = PoseStamped(ts=900.0, position=(5.0, 0.0, 0.0))

    def set_goal(sent: PoseStamped) -> bool:
        # The heartbeat still repeats how the goal before this one ended.
        transport.broadcast(None, goal_status(earlier, GoalStatus.ABORTED, "no safe path"))
        transport.broadcast(None, goal_status(sent, GoalStatus.ACTIVE, "navigating"))
        transport.broadcast(None, goal_status(sent, status, text))
        return True

    navigation.set_goal.side_effect = set_goal

    assert _navigate(container, _pose(1, 0, 0)) == outcome
    assert transport.subscribers == []
