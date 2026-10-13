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

"""Camera window for realsense-stereo-record: live color view, telemetry, Record and Stop.

Record (Space or R) starts writing. Stop (S, Esc or closing the window) stops writing and ends
the run. There is no pause: tool_mesh needs one continuous walk.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Protocol

import numpy as np
from reactivex.disposable import Disposable

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT, DIMOS_PROJECT_ROOT
from dimos.core.coordination.coordinator_rpc import CoordinatorRPC
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.Image import Image
from dimos.spec.utils import Spec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

VIEW = (848, 480)
BAR_HEIGHT = 56
# tool_mesh's default depth range, and the share of it below which tool_mesh drops a frame.
DEPTH_RANGE_MM = (300, 3000)
MIN_FILL = 0.15
BACKGROUND = (30, 30, 30)
TEXT = (235, 235, 235)
DIM = (90, 90, 90)
RED = (220, 50, 50)
GREEN = (90, 200, 90)
YELLOW = (230, 190, 60)


class RecorderControlSpec(Spec, Protocol):
    def start_recording(self) -> None: ...
    def stop_recording(self) -> None: ...


class RecordWindowConfig(ModuleConfig):
    db_path: str


class RecordWindow(Module):
    """Shows the camera while recording and drives the recorder's start and stop."""

    config: RecordWindowConfig

    color_image: In[Image]
    depth_image: In[Image]

    _recorder: RecorderControlSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._color: Image | None = None
        self._depth: Image | None = None
        self._arrivals: deque[float] = deque(maxlen=30)
        self._state = "preview"
        self._frames = 0
        self._started = 0.0
        self._ended: float | None = None

    @rpc
    def start(self) -> None:
        if pygame is None:
            raise ImportError("The record window needs pygame: uv pip install pygame")
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color)))
        self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))
        self._thread = threading.Thread(target=self._window_thread, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(DEFAULT_THREAD_JOIN_TIMEOUT)
        super().stop()

    def _on_color(self, image: Image) -> None:
        self._color = image
        self._arrivals.append(time.monotonic())
        if self._state == "recording":
            self._frames += 1

    def _on_depth(self, image: Image) -> None:
        self._depth = image

    def _window_thread(self) -> None:
        try:
            self._window()
        except Exception:
            logger.error("Record window failed", exc_info=True)

    def _window(self) -> None:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((VIEW[0], VIEW[1] + BAR_HEIGHT))
        pygame.display.set_caption(f"Recording to {self._shown_path()}")
        font = pygame.font.Font(None, 28)
        clock = pygame.time.Clock()
        top = VIEW[1] + (BAR_HEIGHT - 36) // 2
        record_button = pygame.Rect(VIEW[0] - 210, top, 95, 36)
        stop_button = pygame.Rect(VIEW[0] - 105, top, 95, 36)
        shown: Image | None = None
        view: pygame.Surface | None = None
        fill: float | None = None
        written = free = ""
        next_slow = 0.0

        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._finish()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_r):
                    self._record()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_s, pygame.K_ESCAPE):
                    self._finish()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if record_button.collidepoint(event.pos):
                        self._record()
                    elif stop_button.collidepoint(event.pos):
                        self._finish()

            now = time.monotonic()
            if now >= next_slow:
                next_slow = now + 0.5
                fill = self._depth_fill()
                written, free = self._disk()
            color = self._color
            if color is not None and color is not shown:
                shown, view = color, _surface(color)

            screen.fill(BACKGROUND)
            if view is None:
                screen.blit(font.render("Waiting for the camera...", True, TEXT), (20, 20))
            else:
                screen.blit(view, (0, 0))
            self._draw_status(screen, font, fill, written, free)
            if self._state == "preview":
                _draw_button(screen, font, record_button, "Record", True, RED)
            _draw_button(screen, font, stop_button, "Stop", self._state != "stopping", DIM)
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def _draw_status(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        fill: float | None,
        written: str,
        free: str,
    ) -> None:
        middle = VIEW[1] + BAR_HEIGHT // 2
        fps = self._fps()
        parts: list[tuple[str, tuple[int, int, int]]] = []
        if self._state == "preview":
            parts.append(("PREVIEW", TEXT))
        else:
            elapsed = int((self._ended or time.monotonic()) - self._started)
            pygame.draw.circle(
                screen, RED if self._state == "recording" else YELLOW, (22, middle), 8
            )
            state = "REC" if self._state == "recording" else "SAVING"
            parts.append((f"{state} {elapsed // 60:02d}:{elapsed % 60:02d}", TEXT))
            parts.append((f"{self._frames} frames", TEXT))
            parts.append((written, TEXT))
        parts.append((f"{fps:.0f} fps", TEXT if fps > 0 else RED))
        if fill is not None:
            parts.append((f"depth {fill:.0%}", GREEN if fill >= MIN_FILL else RED))
        parts.append((f"{free} free", TEXT))

        x = 14 if self._state == "preview" else 38
        for text, color in parts:
            label = font.render(text, True, color)
            screen.blit(label, (x, middle - label.get_height() // 2))
            x += label.get_width() + 18

    def _record(self) -> None:
        if self._state != "preview":
            return
        self._frames = 0
        self._started = time.monotonic()
        self._recorder.start_recording()
        self._state = "recording"
        logger.info(f"Recording to {self._shown_path()}")

    def _finish(self) -> None:
        if self._state == "stopping":
            return
        if self._state == "recording":
            self._recorder.stop_recording()
            self._ended = time.monotonic()
            logger.info(f"Recorded {self._frames} frames to {self._shown_path()}")
        self._state = "stopping"
        # Off the window thread: the module is stopped while the shutdown runs.
        threading.Thread(target=self._request_shutdown, daemon=True).start()

    def _request_shutdown(self) -> None:
        try:
            client = CoordinatorRPC.connect(timeout=10.0)
            try:
                client.call("shutdown")
            finally:
                client.stop()
        except Exception:
            logger.error("Failed to request coordinator shutdown", exc_info=True)

    def _fps(self) -> float:
        times = list(self._arrivals)
        if len(times) < 2 or time.monotonic() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def _depth_fill(self) -> float | None:
        if self._depth is None:
            return None
        depth = self._depth.data
        lo, hi = DEPTH_RANGE_MM
        return float(np.count_nonzero((depth >= lo) & (depth < hi))) / depth.size

    def _disk(self) -> tuple[str, str]:
        db = Path(self.config.db_path)
        files = (db, db.with_name(db.name + "-wal"))
        written = sum(path.stat().st_size for path in files if path.exists())
        folder = db.parent if db.parent.exists() else DIMOS_PROJECT_ROOT
        return _size(written), _size(shutil.disk_usage(folder).free)

    def _shown_path(self) -> str:
        db = Path(self.config.db_path)
        return str(
            db.relative_to(DIMOS_PROJECT_ROOT) if db.is_relative_to(DIMOS_PROJECT_ROOT) else db
        )


def _surface(image: Image) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(image.to_rgb().data.swapaxes(0, 1))
    if surface.get_size() != VIEW:
        surface = pygame.transform.smoothscale(surface, VIEW)
    return surface


def _draw_button(
    screen: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    text: str,
    enabled: bool,
    color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(screen, color if enabled else (50, 50, 50), rect, border_radius=6)
    label = font.render(text, True, TEXT if enabled else DIM)
    screen.blit(label, label.get_rect(center=rect.center))


def _size(n: float) -> str:
    gb = n / 1e9
    if gb >= 10:
        return f"{gb:.0f} GB"
    return f"{gb:.1f} GB" if gb >= 1 else f"{n / 1e6:.0f} MB"
