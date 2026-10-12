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

"""Native RGB/depth camera capture in a dedicated sensor worker."""

import os
import pickle
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
from numpy.typing import NDArray

# Matches dimos.sim2.runtime.STATE; restated so the render child imports only mujoco.
_STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class _Renderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._options = mujoco.MjvOption()
        # Source assets can put visible geometry in any MuJoCo group.
        self._options.geomgroup[:] = 1

    def capture(
        self, data: mujoco.MjData, camera: int, depth: bool
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32] | None]:
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=camera, scene_option=self._options)
        rgb = self._renderer.render().copy()
        distances = None
        if depth:
            self._renderer.enable_depth_rendering()
            distances = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()
        return rgb, distances

    def close(self) -> None:
        self._renderer.close()


def _serve(model_path: str, width: int, height: int) -> None:
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    renderer = _Renderer(model, width, height)
    out = sys.stdout.buffer
    pickle.dump(None, out)
    out.flush()
    try:
        while True:
            try:
                request = pickle.load(sys.stdin.buffer)
            except EOFError:  # the sensor worker exited without saying goodbye
                break
            if request is None:
                break
            state, camera, depth = request
            mujoco.mj_setState(model, data, state, _STATE)
            mujoco.mj_forward(model, data)
            pickle.dump(
                renderer.capture(data, camera, depth), out, protocol=pickle.HIGHEST_PROTOCOL
            )
            out.flush()
    finally:
        renderer.close()


class _SpawnedRenderer:
    """Renders in a fork+exec child: Apple's OpenGL-on-Metal aborts when sibling processes forked
    without exec (dimos' forkserver workers) compile shaders concurrently."""

    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        self._model = model
        self._state = np.empty(mujoco.mj_stateSize(model, _STATE))
        file = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
        file.close()
        self._model_path = file.name
        mujoco.mj_saveModel(model, self._model_path, None)
        self._process = subprocess.Popen(
            [sys.executable, "-m", __name__, self._model_path, str(width), str(height)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert self._process.stdin is not None and self._process.stdout is not None
        self._send, self._receive = self._process.stdin, self._process.stdout
        pickle.load(self._receive)

    def capture(
        self, data: mujoco.MjData, camera: int, depth: bool
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32] | None]:
        mujoco.mj_getState(self._model, data, self._state, _STATE)
        pickle.dump((self._state, camera, depth), self._send)
        self._send.flush()
        result: tuple[NDArray[np.uint8], NDArray[np.float32] | None] = pickle.load(self._receive)
        return result

    def close(self) -> None:
        try:
            pickle.dump(None, self._send)
            self._send.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        os.unlink(self._model_path)


def MujocoCamera(model: mujoco.MjModel, width: int, height: int) -> _Renderer | _SpawnedRenderer:
    return (_SpawnedRenderer if sys.platform == "darwin" else _Renderer)(model, width, height)


if __name__ == "__main__":
    _serve(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
