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

from collections.abc import Generator
from contextlib import contextmanager
import threading

import pytest

from dimos.constants import LCM_MAX_CHANNEL_NAME_LENGTH
from dimos.protocol.pubsub.impl.lcmpubsub import Topic
from dimos.protocol.rpc.pubsubrpc import LCMRPC
from dimos.protocol.rpc.spec import DEFAULT_RPC_TIMEOUT


@pytest.fixture
def lcmrpc() -> Generator[LCMRPC, None, None]:
    ret = LCMRPC(rpc_timeouts={}, default_rpc_timeout=DEFAULT_RPC_TIMEOUT)
    ret.start()
    yield ret
    ret.stop()


def test_short_name(lcmrpc) -> None:
    actual = lcmrpc.topicgen("Hello/say", req_or_res=True)
    assert actual.topic == "/rpc/Hello/say/res"


def test_long_name(lcmrpc) -> None:
    long = "GreatyLongComplexExampleClassNameForTestingStuff/create"
    long_topic = lcmrpc.topicgen(long, req_or_res=True).topic
    assert long_topic == "/rpc/2cudPuFGMJdWxM5KZb/res"

    less_long = long[:-1]
    less_long_topic = lcmrpc.topicgen(less_long, req_or_res=True).topic
    assert less_long_topic == "/rpc/GreatyLongComplexExampleClassNameForTestingStuff/creat/res"

    assert len(less_long_topic) == LCM_MAX_CHANNEL_NAME_LENGTH


@contextmanager
def _memory_rpc():
    rpc = LCMRPC(url="memq://")
    try:
        yield rpc
    finally:
        rpc.stop()


@pytest.mark.parametrize("from_callback", [False, True], ids=["external", "callback"])
def test_stop_can_race_a_stopping_rpc_handler(run_in_subprocess, from_callback):
    run_in_subprocess(_check_concurrent_rpc_stop, from_callback)


def _check_concurrent_rpc_stop(from_callback):
    with _memory_rpc() as rpc:
        worker_entered = threading.Event()
        shutdown_entered = threading.Event()
        worker_stopped = threading.Event()
        initiator_stopped = threading.Event()

        def worker():
            worker_entered.set()
            assert shutdown_entered.wait(timeout=2.0)
            rpc.stop()
            worker_stopped.set()

        def initiate_stop(*args):
            rpc.stop()
            initiator_stopped.set()

        rpc.serve_rpc(worker, "worker")
        rpc.subscribe(Topic("shutdown"), initiate_stop)
        rpc.start()
        rpc.call_nowait("worker", ([], {}))
        assert worker_entered.wait(timeout=2.0)
        pool = rpc._get_call_thread_pool()
        real_shutdown = pool.shutdown

        def observe_shutdown(wait):
            shutdown_entered.set()
            real_shutdown(wait=wait)

        with pytest.MonkeyPatch.context() as patches:
            patches.setattr(pool, "shutdown", observe_shutdown)
            initiator = threading.Thread(target=initiate_stop)
            try:
                if from_callback:
                    rpc.publish(Topic("shutdown"), "stop")
                else:
                    initiator.start()
                assert initiator_stopped.wait(timeout=2.0)
                assert worker_stopped.wait(timeout=2.0)
                assert rpc.l is None
            finally:
                shutdown_entered.set()
                if initiator.ident is not None:
                    initiator.join(timeout=2.0)
                real_shutdown(wait=True)


def _start_nested_call(rpc):
    """Serve outer -> inner, call outer, and return (release_inner, results)."""
    inner_entered = threading.Event()
    release_inner = threading.Event()
    results = []

    def inner():
        inner_entered.set()
        assert release_inner.wait(timeout=2.0)
        return 42

    def outer():
        result, _ = rpc.call_sync("inner", ([], {}), rpc_timeout=2.0)
        results.append(result)

    rpc.serve_rpc(inner, "inner")
    rpc.serve_rpc(outer, "outer")
    rpc.start()
    rpc.call_nowait("outer", ([], {}))
    assert inner_entered.wait(timeout=2.0)
    return release_inner, results


def test_external_stop_drains_in_flight_nested_rpc(run_in_subprocess):
    run_in_subprocess(_check_rpc_drain)


def _check_rpc_drain():
    with _memory_rpc() as rpc:
        release_inner, results = _start_nested_call(rpc)
        draining = threading.Event()
        stopped = threading.Event()
        pool = rpc._get_call_thread_pool()
        real_shutdown = pool.shutdown

        def observe_shutdown(wait):
            draining.set()
            real_shutdown(wait=wait)

        def stop():
            rpc.stop()
            stopped.set()

        stopper = threading.Thread(target=stop)
        with pytest.MonkeyPatch.context() as patches:
            patches.setattr(pool, "shutdown", observe_shutdown)
            try:
                stopper.start()
                assert draining.wait(timeout=2.0)
                with pytest.raises(RuntimeError, match="still stopping"):
                    rpc.start()
                with pytest.raises(RuntimeError, match="cannot accept calls"):
                    rpc._get_call_thread_pool()
                release_inner.set()
                assert stopped.wait(timeout=2.0)
                assert results == [42]
                assert rpc.l is None
            finally:
                release_inner.set()
                stopper.join(timeout=2.0)


def test_concurrent_external_stops_both_drain_in_flight_nested_rpc(run_in_subprocess):
    run_in_subprocess(_check_concurrent_rpc_drain)


def _check_concurrent_rpc_drain():
    with _memory_rpc() as rpc:
        release_inner, results = _start_nested_call(rpc)
        drains = []
        both_draining = threading.Event()
        pool = rpc._get_call_thread_pool()
        real_shutdown = pool.shutdown

        def observe_shutdown(wait):
            drains.append(wait)
            if len(drains) == 2:
                both_draining.set()
            real_shutdown(wait=wait)

        stoppers = [threading.Thread(target=rpc.stop) for _ in range(2)]
        with pytest.MonkeyPatch.context() as patches:
            patches.setattr(pool, "shutdown", observe_shutdown)
            try:
                for stopper in stoppers:
                    stopper.start()
                # The second stop must wait on the same pool, not skip the drain
                # and tear down dispatch under the first stop's handlers.
                assert both_draining.wait(timeout=2.0)
                assert drains == [True, True]
                release_inner.set()
                for stopper in stoppers:
                    stopper.join(timeout=2.0)
                    assert not stopper.is_alive()
                assert results == [42]
                assert rpc.l is None
            finally:
                release_inner.set()
                for stopper in stoppers:
                    stopper.join(timeout=2.0)
