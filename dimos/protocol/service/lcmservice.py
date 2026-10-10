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

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
import platform
import threading
import traceback
from typing import Any, ClassVar

import lcm as lcm_mod

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.global_config import TransportBackend
from dimos.protocol.service.spec import Service, SessionConfig
from dimos.protocol.service.system_configurator.base import configure_system
from dimos.protocol.service.system_configurator.lcm_config import lcm_configurators
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DEFAULT_LCM_HOST = "239.255.76.67"
_DEFAULT_LCM_PORT = "7667"
# LCM_DEFAULT_URL is used by LCM (we didn't pick that env var name)
_DEFAULT_LCM_URL = os.getenv(
    "LCM_DEFAULT_URL", f"udpm://{_DEFAULT_LCM_HOST}:{_DEFAULT_LCM_PORT}?ttl=0"
)


def autoconf(check_only: bool = False) -> None:
    checks = lcm_configurators()
    if not checks:
        logger.error(f"System configuration not supported on {platform.system()}")
        return
    configure_system(checks, check_only=check_only)


class LCMConfig(SessionConfig):
    transport: ClassVar[TransportBackend] = "lcm"

    ttl: int = 0
    url: str = _DEFAULT_LCM_URL
    lcm: lcm_mod.LCM | None = None


_LCM_LOOP_TIMEOUT = 50


# this class just sets up cpp LCM instance
# and runs its handle loop in a thread
# higher order stuff is done by pubsub/impl/lcmpubsub.py
class LCMService(Service):
    config: LCMConfig
    l: lcm_mod.LCM | None
    _stop_event: threading.Event
    _startup_complete: threading.Event
    _l_lock: threading.Lock
    _pending_unsubs: list[lcm_mod.LCMSubscription]
    _start_lock: threading.Lock
    _thread: threading.Thread | None
    _call_thread_pool: ThreadPoolExecutor | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # we support passing an existing LCM instance
        if self.config.lcm:
            self.l = self.config.lcm
        else:
            self.l = lcm_mod.LCM(self.config.url) if self.config.url else lcm_mod.LCM()

        self._l_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_unsubs = []
        self._startup_complete = threading.Event()
        self._thread = None
        self._stop_calls = 0
        self._call_thread_pool = None
        self._call_thread_pool_max_workers = 4
        self._pool_worker = threading.local()

    def __getstate__(self):  # type: ignore[no-untyped-def]
        """Exclude unpicklable runtime attributes when serializing."""
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        state.pop("l", None)
        state.pop("_stop_event", None)
        state.pop("_startup_complete", None)
        state.pop("_thread", None)
        state.pop("_l_lock", None)
        state.pop("_pending_unsubs", None)
        state.pop("_start_lock", None)
        state.pop("_call_thread_pool", None)
        state.pop("_call_thread_pool_lock", None)
        state.pop("_stop_calls", None)
        state.pop("_pool_worker", None)
        return state

    def __setstate__(self, state) -> None:  # type: ignore[no-untyped-def]
        """Restore object from pickled state."""
        self.__dict__.update(state)
        # Reinitialize runtime attributes
        self.l = None
        self._stop_event = threading.Event()
        self._startup_complete = threading.Event()
        self._thread = None
        self._l_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._pending_unsubs = []
        self._call_thread_pool = None
        self._stop_calls = 0
        self._pool_worker = threading.local()

    def start(self) -> None:
        """Start dispatch, or raise if a previous shutdown is still in progress.

        Wait for the first poll to finish (or for its callback to stop dispatch).
        After callback shutdown or a join timeout, retry only once dispatch exits.
        """
        with self._start_lock:
            if self._stop_calls or (
                self._thread is not None and self._thread.is_alive() and self._stop_event.is_set()
            ):
                raise RuntimeError("LCM service is still stopping; cannot restart yet")
            if self._thread is None or not self._thread.is_alive():
                # Reinitialize LCM if it's None (e.g., after unpickling)
                if self.l is None:
                    # Any pending unsubscribes reference the destroyed instance.
                    self._pending_unsubs.clear()
                    if self.config.lcm:
                        self.l = self.config.lcm
                    else:
                        self.l = lcm_mod.LCM(self.config.url) if self.config.url else lcm_mod.LCM()

                self._stop_event = threading.Event()
                # Waiters retain their own run's event across a later restart.
                self._startup_complete = threading.Event()
                self._thread = threading.Thread(target=self._lcm_loop)
                self._thread.daemon = True
                self._thread.start()
            thread = self._thread
            startup_complete = self._startup_complete
        # The first dispatch may call stop(), which also needs _start_lock.
        if threading.current_thread() is not thread and not startup_complete.wait(timeout=5.0):
            raise RuntimeError("LCM handler thread failed to start within 5s")

    def _defer_unsubscribe(self, subscription: lcm_mod.LCMSubscription) -> None:
        """Queue an unsubscribe for the loop thread to apply between polls.

        lcm-python's unsubscribe is not safe against a concurrently
        dispatching handle_timeout (it frees the subscription's state under
        the handler → segfault), and excluding the loop with a lock would
        block the caller for up to a poll cycle — a deadlock hazard for
        callers on event-loop threads. Deferring to the dispatch thread
        makes unsubscribe non-blocking and race-free; the subscription may
        receive callbacks for at most one more poll.
        """
        if self.l is None:
            return
        self._pending_unsubs.append(subscription)

    def _lcm_loop(self) -> None:
        """LCM message handling loop."""
        primed = False
        try:
            while not self._stop_event.is_set():
                try:
                    with self._l_lock:
                        if self.l is None:
                            break
                        while self._pending_unsubs:
                            self.l.unsubscribe(self._pending_unsubs.pop())
                        self.l.handle_timeout(_LCM_LOOP_TIMEOUT)
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    print(f"Error in LCM handling: {e}\n{stack_trace}")
                if not primed:
                    # Signal start() only after one full poll cycle, so callers
                    # don't race the first handle_timeout dispatch.
                    primed = True
                    self._startup_complete.set()
        finally:
            # A callback may request stop while handle_timeout holds _l_lock.
            # Only release the native instance after dispatch has unwound.
            self._release_lcm()
            self._startup_complete.set()

    def _release_lcm(self) -> None:
        if not self.config.lcm:
            with self._l_lock:
                # Destroying the instance frees its subscriptions; drop
                # pendings so a restart doesn't apply them to a new one.
                self._pending_unsubs.clear()
                self.l = None

    @contextmanager
    def _stop_scope(self) -> Iterator[None]:
        """Exclude restart and pool creation throughout cooperative stop methods."""
        with self._start_lock:
            self._stop_calls += 1
        try:
            yield
        finally:
            with self._start_lock:
                self._stop_calls -= 1

    def stop(self) -> None:
        """Request shutdown, joining dispatch unless called from its callback.

        A join timeout leaves the native instance alive until dispatch exits.
        Callbacks and pool workers request pool shutdown without waiting for it.
        Every other caller waits for the handler pool to drain before stopping.
        Restart is rejected while a stop call or stopping dispatch is still active.
        """
        with self._stop_scope():
            with self._start_lock:
                self._stop_event.set()
                thread = self._thread
                if thread is None:
                    # Construction opens LCM even if start() was never called.
                    self._release_lcm()

            self._shutdown_thread_pool()
            if thread is not None and threading.current_thread() is not thread:
                # Callbacks can stop while another caller joins, without a lock cycle.
                thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    logger.warning("LCM thread did not stop cleanly within timeout")
                else:
                    with self._start_lock:
                        if self._thread is thread:
                            self._thread = None

    def _shutdown_thread_pool(self) -> None:
        with self._start_lock:
            pool = self._call_thread_pool
            is_dispatch_thread = threading.current_thread() is self._thread
        if pool is None:
            return
        # Neither a pool worker nor dispatch can wait for handlers that
        # might themselves be stopping this service.
        is_pool_thread = getattr(self._pool_worker, "active", False)
        # The pool stays attached while it drains, so a concurrent external
        # stop waits for the same handlers instead of skipping the drain.
        pool.shutdown(wait=not (is_dispatch_thread or is_pool_thread))
        with self._start_lock:
            if self._call_thread_pool is pool:
                self._call_thread_pool = None

    def _mark_pool_worker(self) -> None:
        self._pool_worker.active = True

    def _get_call_thread_pool(self) -> ThreadPoolExecutor:
        with self._start_lock:
            if self._stop_calls or self._stop_event.is_set():
                raise RuntimeError("LCM service is stopping or stopped; cannot accept calls")
            if self._call_thread_pool is None:
                self._call_thread_pool = ThreadPoolExecutor(
                    max_workers=self._call_thread_pool_max_workers,
                    initializer=self._mark_pool_worker,
                )
            return self._call_thread_pool
