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

"""Application-side reset orchestration using existing module RPCs."""

from collections.abc import Callable, Sequence
import time
from typing import cast

from dimos.control.coordinator import ControlCoordinator
from dimos.porcelain.dimos import Dimos
from dimos.sim2.scene_types import SceneControlSpec, SceneState, SceneUpdate


def reset_scene(
    app: Dimos,
    initial: SceneUpdate | None = None,
    *,
    simulation: str = "SimulationModule",
    coordinators: Sequence[str] = ("ControlCoordinator",),
    before_reset: Sequence[Callable[[Dimos], None]] = (),
    after_reset: Sequence[Callable[[Dimos], None]] = (),
) -> SceneState:
    """Reset a whole world and its explicitly selected application owners.

    Supply cancellation/history hooks for the chosen blueprint, including any
    navigation, mapping or perception state. This function cannot infer those
    histories. Failure leaves physics paused. A reset does not move an arm's
    planner base frame: retain its configured spawn when using manipulation.
    """
    sim = cast("SceneControlSpec", app.get_module(simulation))
    controllers = [cast("ControlCoordinator", app.get_module(name)) for name in coordinators]
    deadline = time.monotonic() + 30
    while any(controller.get_tick_count() == 0 for controller in controllers):
        if time.monotonic() >= deadline:
            raise TimeoutError("controllers have not started ticking; no reset performed")
        time.sleep(0.05)
    sim.set_paused(True)
    for controller in controllers:
        if not controller.set_estop(True):
            raise RuntimeError("connection halt failed; world remains paused")
        controller.cancel_trajectory()
        controller.set_activated(False)
    for hook in before_reset:
        hook(app)
    state = sim.reset(initial)
    for controller in controllers:
        results = controller.reset_runtime_state(reactivate=False)
        if not all(results.values()):
            raise RuntimeError(f"controller reset failed; world remains paused: {results}")
    for hook in after_reset:
        hook(app)
    for controller in controllers:
        controller.set_activated(True)
        if not controller.set_estop(False):
            raise RuntimeError("connection resume failed; world remains paused")
    sim.set_paused(False)
    return state
