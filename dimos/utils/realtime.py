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

"""Exempt the calling thread from macOS timer coalescing, which stretches short waits ~7x."""

import ctypes
import sys

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_THREAD_TIME_CONSTRAINT_POLICY = 2
_THREAD_TIME_CONSTRAINT_POLICY_COUNT = 4


class _TimeConstraint(ctypes.Structure):
    _fields_ = [
        ("period", ctypes.c_uint32),
        ("computation", ctypes.c_uint32),
        ("constraint", ctypes.c_uint32),
        ("preemptible", ctypes.c_int),
    ]


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def make_current_thread_periodic(period_seconds: float) -> bool:
    """Give a fixed-rate loop thread a time-constraint policy so its waits wake on time.

    A no-op off macOS. Without it, a background process's `Event.wait(0.005)` takes ~35 ms,
    so a 200 Hz loop runs at ~30 Hz.
    """
    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
    timebase = _Timebase()
    libc.mach_timebase_info(ctypes.byref(timebase))
    period = min(period_seconds, 0.05)

    def ticks(seconds: float) -> int:
        return int(seconds * 1e9 * timebase.denom / timebase.numer)

    policy = _TimeConstraint(ticks(period), ticks(period / 4), ticks(period), 1)
    libc.mach_thread_self.restype = ctypes.c_uint32
    result = libc.thread_policy_set(
        libc.mach_thread_self(),
        _THREAD_TIME_CONSTRAINT_POLICY,
        ctypes.byref(policy),
        _THREAD_TIME_CONSTRAINT_POLICY_COUNT,
    )
    if result != 0:
        logger.warning("thread_policy_set failed; waits stay coalesced", result=result)
    return result == 0
