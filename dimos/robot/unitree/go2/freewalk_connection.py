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

"""SONIC-style policy connection: Twist references in, complete PD motor frames out.

ControlCoordinator arbitrates references; this module alone owns policy history,
the 50 Hz inference clock, arming and feedback freshness. No simulator dependency.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dimos.control.go2_freewalk.models import load_weights
from dimos.control.go2_freewalk.policy import JOINT_NAMES, FreePolicy, Proprioception
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.utils.logging_config import setup_logger
from dimos.utils.realtime import make_current_thread_periodic

logger = setup_logger()


class Go2FreewalkConnectionConfig(ModuleConfig):
    model_path: Path | None = None
    # MotorCommandArray has implicit ordering. This must match the motor device.
    joint_names: tuple[str, ...] = JOINT_NAMES
    auto_arm: bool = False
    ramp_seconds: float = Field(default=1.0, gt=0)
    command_timeout: float = Field(default=0.5, gt=0)
    feedback_timeout: float = Field(default=0.15, gt=0)
    maximum_forward: float = Field(default=1.0, gt=0)
    maximum_lateral: float = Field(default=0.5, gt=0)
    maximum_yaw: float = Field(default=1.0, gt=0)
    # The FREE policy barely turns below ~0.25 rad/s (measured in sim2 over 5 s: 0.3 -> 0.08, 0.6 -> 0.33,
    # 1.0 -> 0.7 rad/s), so a planner's gentle turns stall. A requested turn gets this much added; 0 disables it.
    yaw_deadband: float = Field(default=0.25, ge=0)


class Go2FreewalkConnection(Module):
    config: Go2FreewalkConnectionConfig
    dedicated_worker = True
    base_command: In[Twist]
    motor_states: In[JointState]
    imu: In[Imu]
    motor_command: Out[MotorCommandArray]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if len(self.config.joint_names) != 12 or set(self.config.joint_names) != set(JOINT_NAMES):
            raise ValueError("Go2 motor order must contain each of the twelve FREE joints once")
        self._output_order = [JOINT_NAMES.index(name) for name in self.config.joint_names]
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._policy: FreePolicy | None = None
        self._states: dict[float, tuple[JointState, float]] = {}
        self._imus: dict[float, tuple[Imu, float]] = {}
        self._feedback: tuple[Proprioception, float] | None = None
        self._stamp = -float("inf")
        self._requested = np.zeros(3)
        self._command = np.zeros(3)
        self._command_time = -float("inf")
        self._armed_at: float | None = None
        self._ramp_start = np.zeros(12)
        self._fault: str | None = None
        self._auto_arm_pending = self.config.auto_arm
        self._ticks = 0
        self._inference_ms: deque[float] = deque(maxlen=3000)

    @rpc
    def start(self) -> None:
        self._policy = FreePolicy(load_weights(self.config.model_path))
        super().start()
        self.register_disposable(Disposable(self.base_command.subscribe(self._receive_command)))
        self.register_disposable(Disposable(self.motor_states.subscribe(self._receive_states)))
        self.register_disposable(Disposable(self.imu.subscribe(self._receive_imu)))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="go2-freewalk", daemon=True)
        self._thread.start()

    def _receive_command(self, msg: Twist) -> None:
        command = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        with self._lock:
            if not np.isfinite(command).all():
                self._latch_fault("non-finite velocity command")
                return
            limits = np.array(
                [self.config.maximum_forward, self.config.maximum_lateral, self.config.maximum_yaw]
            )
            if abs(command[2]) > 0.02:
                command[2] += np.sign(command[2]) * self.config.yaw_deadband
            self._requested = np.clip(command, -limits, limits)
            self._command_time = time.monotonic()

    def _receive_states(self, msg: JointState) -> None:
        with self._lock:
            self._states[msg.ts] = (msg, time.monotonic())
            self._pair(msg.ts)

    def _receive_imu(self, msg: Imu) -> None:
        with self._lock:
            self._imus[msg.ts] = (msg, time.monotonic())
            self._pair(msg.ts)

    def _pair(self, stamp: float) -> None:
        # Both messages come from one low-level sample, even if delivery is reordered.
        if stamp > self._stamp and stamp in self._states and stamp in self._imus:
            state, joint_time = self._states.pop(stamp)
            imu, imu_time = self._imus.pop(stamp)
            try:
                if len(state.name) != 12 or set(state.name) != set(JOINT_NAMES):
                    raise ValueError("feedback must name all twelve Go2 joints")
                if len(state.position) != 12 or len(state.velocity) != 12:
                    raise ValueError("incomplete Go2 motor feedback")
                order = [state.name.index(name) for name in JOINT_NAMES]
                q = np.asarray(state.position)[order]
                dq = np.asarray(state.velocity)[order]
                gyro = np.array(imu.angular_velocity.to_tuple())
                quat = np.array(imu.orientation.to_tuple())
                if not np.isfinite(np.concatenate([q, dq, gyro, quat, [stamp]])).all():
                    raise ValueError("non-finite Go2 feedback")
                if abs(float(np.linalg.norm(quat)) - 1) > 0.05:
                    raise ValueError("invalid Go2 IMU quaternion")
                gravity = Rotation.from_quat(quat).inv().apply([0.0, 0.0, -1.0])
                self._feedback = (Proprioception(gyro, gravity, q, dq), min(joint_time, imu_time))
                self._stamp = stamp
            except ValueError as error:
                self._latch_fault(str(error))
        for pending in (self._states, self._imus):
            while len(pending) > 8:
                pending.pop(next(iter(pending)))

    def _latch_fault(self, reason: str) -> None:
        if self._fault != reason:
            logger.error("Go2 FREE policy fault", reason=reason)
        self._fault = reason
        self._armed_at = None
        self._auto_arm_pending = False
        self._command.fill(0)

    @rpc
    def arm(self) -> bool:
        """Clear a latched fault and ramp to standing, only with fresh coherent feedback."""
        with self._lock:
            now = time.monotonic()
            if (
                self._policy is None
                or self._feedback is None
                or now - self._feedback[1] > self.config.feedback_timeout
            ):
                return False
            self._policy.reset()
            self._command.fill(0)
            self._requested.fill(0)
            self._command_time = -float("inf")
            self._ramp_start = self._feedback[0].joint_position.copy()
            self._armed_at = now
            self._fault = None
            self._auto_arm_pending = False
            return True

    @rpc
    def disarm(self) -> None:
        """Latch passive damping; only an explicit arm call resumes the policy."""
        with self._lock:
            self._armed_at = None
            self._auto_arm_pending = False
            self._command.fill(0)
            self.motor_command.publish(self._damping())

    def _damping(self) -> MotorCommandArray:
        return MotorCommandArray(q=[0.0] * 12, kd=[1.0] * 12)

    def _step(self, now: float) -> MotorCommandArray:
        if self._auto_arm_pending and self._feedback is not None:
            self.arm()
        if self._armed_at is None or self._policy is None:
            return self._damping()
        if self._feedback is None or now - self._feedback[1] > self.config.feedback_timeout:
            self._latch_fault("coherent motor/IMU feedback timed out")
            return self._damping()
        obs = self._feedback[0]
        if obs.gravity[2] > -0.5:
            self._latch_fault("robot tilt exceeds 60 degrees")
            return self._damping()
        fraction = (now - self._armed_at) / self.config.ramp_seconds
        if fraction < 1:
            targets = self._ramp_start + np.clip(fraction, 0, 1) * (
                self._policy.default_pose - self._ramp_start
            )
        else:
            requested = (
                self._requested
                if now - self._command_time <= self.config.command_timeout
                else np.zeros(3)
            )
            # Match go2web's bounded command approach: 50 Hz, no burst catch-up.
            self._command += np.clip(
                requested - self._command, [-0.05, -0.04, -0.10], [0.05, 0.04, 0.10]
            )
            before = time.perf_counter()
            targets = self._policy.act(obs, self._command)
            self._inference_ms.append((time.perf_counter() - before) * 1000)
        order = self._output_order
        return MotorCommandArray(
            q=targets[order].tolist(),
            kp=self._policy.kp[order].tolist(),
            kd=self._policy.kd[order].tolist(),
        )

    def _run(self) -> None:
        make_current_thread_periodic(0.02)
        deadline = time.monotonic()
        while not self._stop.is_set():
            before = time.monotonic()
            with self._lock:
                try:
                    command = self._step(before)
                except Exception as error:
                    self._latch_fault(f"inference failed: {error}")
                    command = self._damping()
                self.motor_command.publish(command)
                self._ticks += 1
            # Absolute deadlines avoid cumulative OS sleep overshoot. Skip missed
            # periods rather than evaluating several history frames in a burst.
            deadline += 0.02
            now = time.monotonic()
            if deadline <= now:
                deadline = now + 0.02
            self._stop.wait(deadline - now)

    @rpc
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "armed": self._armed_at is not None,
                "fault": self._fault,
                "policy_ticks": self._ticks,
                "command": self._command.tolist(),
                "feedback_age_seconds": time.monotonic() - self._feedback[1]
                if self._feedback
                else None,
                "inference_p95_ms": float(np.percentile(self._inference_ms, 95))
                if self._inference_ms
                else None,
            }

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Go2 inference did not stop; motor transport retained")
            self._thread = None
        self.disarm()
        super().stop()
