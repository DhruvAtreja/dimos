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

"""A lunar landing site: cratered regolith, scattered rocks and a lander under a black sky.

TARS walks quasi-statically on flat feet, so the ground along `ROUTE` stays nearly flat
(slopes under ~3%); craters and rocks are kept off it. Run
`python -m tars_sdk.scenes.moon [out_dir]` to write the scene, or use
`tars_sdk.scenes.scene_path("moon")`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import math
from pathlib import Path
import sys

import numpy as np

SIZE = 90.0  # terrain side (m), centered on the spawn point
N = 900  # heightfield samples per side (0.1 m)
# a walk that skirts the craters; TARS spawns at the first point facing the second
ROUTE = [(0.0, 0.0), (7.0, 1.5), (13.0, -1.0), (19.0, 2.5), (25.0, 0.5)]
ROUTE_CLEARANCE = 3.5  # walking strip: flat to 2 m, blended out to this; no rocks closer (m)
RIM_CLEARANCE = 2.5  # crater relief (fades out by 1.5 radii) stays this far from the route (m)
LANDER_XY = (3.0, -4.8)  # beside the route, inside the lidar's view as TARS walks past


@dataclass
class Crater:
    x: float
    y: float
    radius: float
    depth: float


CRATERS = [
    Crater(11.5, 13.5, 6.5, 1.6),
    Crater(6.5, -11.5, 4.0, 0.9),
    Crater(21.0, -9.5, 5.0, 1.2),
    Crater(-9.0, -6.0, 3.0, 0.6),
    Crater(29.0, 9.0, 4.5, 1.0),
    Crater(16.0, 6.8, 1.6, 0.35),
    Crater(9.5, -4.6, 1.1, 0.25),
    Crater(24.0, -4.2, 1.0, 0.22),
    Crater(2.5, 5.5, 1.3, 0.3),
    Crater(-2.0, -4.2, 0.9, 0.2),
]


def _route_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.full(x.shape, np.inf)
    for (x0, y0), (x1, y1) in pairwise(ROUTE):
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        d = np.minimum(d, np.hypot(x - (x0 + t * dx), y - (y0 + t * dy)))
    return d


def _blur(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian blur via FFT (periodic edges are fine for noise)."""
    k = np.fft.fftfreq(a.shape[0])[:, None] ** 2 + np.fft.fftfreq(a.shape[1])[None, :] ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * (math.pi * sigma_cells) ** 2 * k)))


def heights(seed: int = 7) -> np.ndarray:
    """Terrain height (m), shape (N, N); row i is y = -SIZE/2 + i*SIZE/(N-1).

    The spawn point (grid center) is at z = 0.
    """
    rng = np.random.default_rng(seed)
    c = np.linspace(-SIZE / 2, SIZE / 2, N)
    x, y = np.meshgrid(c, c)
    # long, gentle swells: the only relief along the walking strip
    swell = np.zeros_like(x)
    for wavelength, amp in ((45.0, 0.10), (28.0, 0.06), (17.0, 0.03)):
        a, ph = rng.uniform(0, 2 * math.pi, 2)
        swell += amp * np.sin(2 * math.pi * (x * math.cos(a) + y * math.sin(a)) / wavelength + ph)
    h = swell.copy()
    rough = _blur(rng.normal(size=x.shape), 2.0)
    h += 0.005 * rough / rough.std()  # regolith grain (the texture carries the rest)
    for cr in CRATERS:  # parabolic bowl + raised rim
        r = np.hypot(x - cr.x, y - cr.y) / cr.radius
        h += np.where(r < 1.0, -cr.depth * (1.0 - r**2), 0.0)
        h += 0.22 * cr.depth * np.exp(-(((r - 1.0) / 0.28) ** 2))
    for _ in range(140):  # small pits
        px, py = rng.uniform(-SIZE / 2 + 3, SIZE / 2 - 3, 2)
        rad, dep = rng.uniform(0.3, 0.9), rng.uniform(0.04, 0.15)
        r = np.hypot(x - px, y - py) / rad
        h += np.where(r < 1.0, -dep * (1.0 - r**2), 0.0)
        h += 0.2 * dep * np.exp(-(((r - 1.0) / 0.3) ** 2))
    # the walking strip keeps only the swell, blended back to full relief beyond it
    w = np.clip((_route_distance(x, y) - 2.0) / (ROUTE_CLEARANCE - 2.0), 0.0, 1.0)
    h = swell + (w * w * (3 - 2 * w)) * (h - swell)
    return h - h[N // 2, N // 2]


def _rocks(seed: int = 3) -> list[tuple[float, float, float, float, float, float]]:
    """(x, y, sx, sy, sz, yaw) ellipsoid rocks off the route and out of the craters."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < 70:
        x, y = rng.uniform(-30, 38), rng.uniform(-28, 28)
        if _route_distance(np.array(x), np.array(y)) < ROUTE_CLEARANCE:
            continue
        if math.hypot(x - LANDER_XY[0], y - LANDER_XY[1]) < 3.5:
            continue
        if any(math.hypot(x - c.x, y - c.y) < c.radius * 0.9 for c in CRATERS):
            continue
        s = rng.choice([0.12, 0.2, 0.3, 0.5], p=[0.4, 0.3, 0.2, 0.1]) * rng.uniform(0.7, 1.4)
        out.append(
            (
                x,
                y,
                s * rng.uniform(1, 1.8),
                s * rng.uniform(0.8, 1.4),
                s * 0.7,
                rng.uniform(0, math.pi),
            )
        )
    return out


def _lander(x0: float, y0: float) -> str:
    """A four-legged descent stage with a gold-foil body and a crew module on top."""
    legs = []
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        hx, hy = x0 + 1.0 * math.cos(a), y0 + 1.0 * math.sin(a)
        fx, fy = x0 + 2.3 * math.cos(a), y0 + 2.3 * math.sin(a)
        legs.append(
            f'<geom type="capsule" fromto="{hx:.3f} {hy:.3f} 1.3 {fx:.3f} {fy:.3f} 0.12" '
            f'size="0.06" material="strut"/>'
            f'<geom type="cylinder" pos="{fx:.3f} {fy:.3f} 0.05" size="0.32 0.05" material="strut"/>'
        )
    return f"""
    <body name="lander" pos="0 0 0">
      <geom type="box" pos="{x0} {y0} 1.55" size="1.15 1.15 0.55" euler="0 0 0.785" material="foil"/>
      <geom type="cylinder" pos="{x0} {y0} 2.55" size="0.85 0.45" material="hull"/>
      <geom type="box" pos="{x0} {y0} 3.15" size="0.55 0.55 0.18" material="hull"/>
      <geom type="cylinder" pos="{x0 + 0.35} {y0 - 0.2} 3.7" size="0.03 0.4" material="strut"/>
      <geom type="ellipsoid" pos="{x0 + 0.35} {y0 - 0.2} 4.15" size="0.32 0.32 0.08" euler="0.5 0.3 0" material="hull"/>
      <geom type="box" pos="{x0 + 0.8} {y0 - 0.8} 0.75" size="0.04 0.3 0.75" euler="0 0.35 0.785" material="strut"/>
      <geom type="cylinder" pos="{x0} {y0} 0.9" size="0.4 0.15" material="strut"/>
      {"".join(legs)}
    </body>"""


def scene_xml(hfield_file: str, h: np.ndarray) -> str:
    lo, hi = float(h.min()), float(h.max())
    rocks = "\n".join(
        f'<geom type="ellipsoid" pos="{x:.3f} {y:.3f} {_ground(h, x, y) + sz * 0.45:.3f}" '
        f'size="{sx:.3f} {sy:.3f} {sz:.3f}" euler="0.15 0.1 {yaw:.3f}" material="rock"/>'
        for x, y, sx, sy, sz, yaw in _rocks()
    )
    # low sun from the side: long shadows across the route. A directional light's shadow
    # map is centered on its position, so put it up-sun of the scene center.
    d = np.array([-0.35, -0.8, -0.32])
    d /= np.linalg.norm(d)
    sun_dir = " ".join(f"{v:.3f}" for v in d)
    sun_pos = " ".join(f"{v:.2f}" for v in np.array([10.0, 0.0, 0.0]) - 60.0 * d)
    return f"""<mujoco model="moon">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <statistic extent="40" center="10 0 0"/>
  <visual>
    <global offwidth="1920" offheight="1440" azimuth="200" elevation="-12"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.01" zfar="30" haze="0.0" shadowclip="0.7" shadowscale="0.9"/>
    <headlight ambient="0.06 0.06 0.07" diffuse="0.08 0.08 0.08" specular="0 0 0"/>
    <rgba haze="0 0 0 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="32" height="32"/>
    <texture name="regolith" type="2d" builtin="flat" rgb1="0.55 0.54 0.52" rgb2="0.5 0.5 0.48"
             mark="random" random="0.5" markrgb="0.46 0.45 0.44" width="1024" height="1024"/>
    <material name="regolith" texture="regolith" texrepeat="150 150" texuniform="false"
              specular="0.02" shininess="0.02" reflectance="0"/>
    <material name="rock" rgba="0.38 0.37 0.36 1" specular="0.05" shininess="0.05"/>
    <material name="foil" rgba="0.6 0.47 0.2 1" specular="0.7" shininess="0.6" reflectance="0.05"/>
    <material name="hull" rgba="0.82 0.83 0.85 1" specular="0.5" shininess="0.5"/>
    <material name="strut" rgba="0.3 0.3 0.32 1" specular="0.4" shininess="0.4"/>
    <hfield name="terrain" file="{hfield_file}" size="{SIZE / 2} {SIZE / 2} {hi - lo:.4f} 0.5"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="{sun_pos}" dir="{sun_dir}"
           diffuse="1.35 1.32 1.25" specular="0.3 0.3 0.3" castshadow="true"/>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 {lo:.4f}" material="regolith"/>
{rocks}
{_lander(*LANDER_XY)}
  </worldbody>
</mujoco>
"""


def _ground(h: np.ndarray, x: float, y: float) -> float:
    i = round((y + SIZE / 2) / SIZE * (N - 1))
    j = round((x + SIZE / 2) / SIZE * (N - 1))
    return float(h[min(max(i, 0), N - 1), min(max(j, 0), N - 1)])


def check_route() -> None:
    """TARS can't climb crater walls: refuse a layout whose rims reach the route."""
    for cr in CRATERS:
        edge = float(_route_distance(np.array(cr.x), np.array(cr.y))) - 1.5 * cr.radius
        if edge < RIM_CLEARANCE:
            raise ValueError(f"{cr} rim is {edge:.1f} m from the route (< {RIM_CLEARANCE})")


def write(out_dir: Path) -> Path:
    """Write moon.xml + moon_terrain.bin into out_dir; returns the scene path."""
    check_route()
    out_dir.mkdir(parents=True, exist_ok=True)
    h = heights()
    with open(out_dir / "moon_terrain.bin", "wb") as f:
        np.array([N, N], dtype=np.int32).tofile(f)
        h.astype(np.float32).tofile(f)
    path = out_dir / "moon.xml"
    path.write_text(scene_xml("moon_terrain.bin", h))
    return path


if __name__ == "__main__":
    print(write(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("moon_scene")))
