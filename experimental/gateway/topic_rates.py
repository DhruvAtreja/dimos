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

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
import re
import threading
import time
from typing import Any

WINDOW_S = 2.0
# samples are counted into buckets this long, so a topic keeps a few counters, not a record per sample
BUCKET_S = 0.25
# the subscription (every sample on the bus, payloads too) stops this long after the last request
IDLE_S = 60.0
LIVELINESS_EVERY_S = 5.0
KIND = re.compile(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+")


def parse_key(key: str) -> tuple[str, str] | None:
    parts = key.split("/")
    if len(parts) < 3 or parts[0] != "dimos" or "*" in key or "/@" in key or parts[1] == "rpc":
        return None
    if not KIND.fullmatch(parts[-1]):
        return None
    return "/" + "/".join(parts[1:-1]), parts[-1]


@dataclass
class Topic:
    topic: str
    type: str
    messages: int = 0
    bytes: int = 0
    last: float | None = None
    declared: bool = False
    # [bucket start, samples, bytes], oldest first, covering at most the last WINDOW_S
    window: deque[list[float]] = field(default_factory=deque)

    def trim(self, now: float) -> None:
        while self.window and now - self.window[0][0] > WINDOW_S:
            self.window.popleft()


class TopicWatch:
    def __init__(
        self, open_session: Callable[[], Any], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.open_session = open_session
        self.clock = clock
        self.topics: dict[tuple[str, str], Topic] = {}
        self.lock = threading.Lock()
        self.session: Any = None
        self.subscriber: Any = None
        self.liveliness_at = 0.0
        self.requested_at = 0.0
        self.error: str | None = None

    def heard(self, key: str, size: int) -> None:
        parsed = parse_key(key)
        if parsed is None:
            return
        now = self.clock()
        with self.lock:
            entry = self.topics.setdefault(parsed, Topic(*parsed))
            entry.messages += 1
            entry.bytes += size
            entry.last = now
            start = now - now % BUCKET_S
            if entry.window and entry.window[-1][0] == start:
                entry.window[-1][1] += 1
                entry.window[-1][2] += size
            else:
                entry.window.append([start, 1, size])
                # trimmed here, not only in snapshot(): unread, it kept every sample (4 GB in 12 h on the R1)
                entry.trim(now)

    def declared(self, key: str) -> None:
        parsed = parse_key(key.split("/@adv/", 1)[0])
        if parsed is not None:
            with self.lock:
                self.topics.setdefault(parsed, Topic(*parsed)).declared = True

    def start(self) -> None:
        if self.subscriber is not None:
            return
        self.requested_at = self.clock()
        try:
            self.session = self.session or self.open_session()
            self.subscriber = self.session.declare_subscriber(
                "dimos/**", lambda sample: self.heard(str(sample.key_expr), len(sample.payload))
            )
            self.error = None
        except Exception as error:
            self.error = f"can't listen to the bus: {error}"
            return
        threading.Thread(target=self._stop_when_idle, name="topic-rates-idle", daemon=True).start()

    def stop_if_idle(self) -> bool:
        """Drops the subscription when nobody asked for IDLE_S; True if it stopped."""
        if self.subscriber is None or self.clock() - self.requested_at <= IDLE_S:
            return False
        subscriber, self.subscriber = self.subscriber, None
        try:
            subscriber.undeclare()
        except Exception:
            pass
        with self.lock:
            self.topics.clear()
        return True

    def _stop_when_idle(self) -> None:
        while self.subscriber is not None and not self.stop_if_idle():
            time.sleep(5.0)

    def _ask_liveliness(self) -> None:
        if self.session is None:
            return
        try:
            for reply in self.session.liveliness().get("dimos/**/@adv/pub/**", timeout=0.3):
                ok = getattr(reply, "ok", None)
                if ok is not None:
                    self.declared(str(ok.key_expr))
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        self.requested_at = now
        self.start()
        if now - self.liveliness_at > LIVELINESS_EVERY_S:
            self.liveliness_at = now
            self._ask_liveliness()
        rows: list[dict[str, Any]] = []
        with self.lock:
            for entry in self.topics.values():
                entry.trim(now)
                count = sum(bucket[1] for bucket in entry.window)
                rows.append(
                    {
                        "topic": entry.topic,
                        "type": entry.type,
                        "hz": round(count / WINDOW_S, 2),
                        "bps": round(sum(bucket[2] for bucket in entry.window) / WINDOW_S, 1),
                        "messages": entry.messages,
                        "lastSeen": None if entry.last is None else round(now - entry.last, 1),
                        "declared": entry.declared,
                    }
                )
        rows.sort(key=lambda row: (-float(row["hz"]), str(row["topic"])))
        return {"up": self.error is None, "error": self.error, "topics": rows}
