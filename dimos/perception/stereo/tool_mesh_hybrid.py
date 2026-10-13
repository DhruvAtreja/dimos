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

"""Mesh a recording like tool_mesh, with fragments from Open3D's tensor reconstruction system.

Run: uv run python -m dimos.perception.stereo.tool_mesh_hybrid <recording.db> <out.rrd>
         [--depth-min <m>] [--depth-max <m>] [--min-fill <fraction>] [--work <dir>]

The t_reconstruction_system's GPU odometry, loop closure and integration build the fragments,
then tool_mesh's reconstruction_system registers, refines and integrates them unmodified.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
import open3d.core as o3c  # type: ignore[import-untyped]

from dimos.constants import CACHE_DIR
from dimos.perception.stereo.tool_mesh import (
    OPEN3D_CONFIG,
    camera_path,
    export,
    reconstruction_system,
    write_rrd,
)

TENSOR_SCRIPTS = "examples/python/t_reconstruction_system"
# The reconstruction_system's n_frames_per_fragment, which its later stages assume.
FRAGMENT_FRAMES = 100
# t_reconstruction_system/default_config.yml, for its fragment odometry and integration.
TENSOR_CONFIG = {
    "device": "CUDA:0",
    "integrate_color": True,
    "depth_scale": 1000.0,
    "odometry_loop_interval": 10,
}


def tensor_reconstruction_system(scripts: Path) -> Path:
    """Open3D's t_reconstruction_system example, added once to tool_mesh's clone."""
    root = scripts.parents[2]  # scripts is <root>/examples/python/reconstruction_system
    tensor = root / TENSOR_SCRIPTS
    if not (tensor / "run_system.py").exists():
        checkout = ["git", "-C", str(root), "sparse-checkout", "add", TENSOR_SCRIPTS]
        subprocess.run(checkout, check=True)
    return tensor


def tensor_fragments(tensor: Path, dataset: Path, depth_max: float) -> None:
    """Fragments from the t_reconstruction_system's GPU odometry, loop closure and integration,
    saved under the names the reconstruction_system's later stages read."""
    # Example scripts, not a package: they import each other by bare module name.
    sys.path.insert(0, str(tensor))
    from integrate import integrate  # type: ignore[import-not-found]
    from pose_graph_optim import PoseGraphWrapper  # type: ignore[import-not-found]
    from rgbd_odometry import rgbd_loop_closure, rgbd_odometry  # type: ignore[import-not-found]

    config = SimpleNamespace(**TENSOR_CONFIG, depth_max=depth_max)
    column_major = json.loads((dataset / "intrinsic.json").read_text())["intrinsic_matrix"]
    intrinsic = o3c.Tensor(np.array(column_major, dtype=np.float64).reshape(3, 3).T)
    colors = sorted(str(path) for path in (dataset / "color").glob("*.jpg"))
    depths = sorted(str(path) for path in (dataset / "depth").glob("*.png"))
    (dataset / "fragments").mkdir()
    for fragment, start in enumerate(range(0, len(depths), FRAGMENT_FRAMES)):
        depth_list = depths[start : start + FRAGMENT_FRAMES]
        color_list = colors[start : start + FRAGMENT_FRAMES]
        # As t_reconstruction_system/run_system.py --fragment_odometry builds it.
        graph = PoseGraphWrapper()
        pose = np.eye(4)
        graph.add_node(0, pose.copy())
        _, steps, infos = rgbd_odometry(depth_list, color_list, intrinsic, config)
        for i, (step, info) in enumerate(zip(steps, infos, strict=True)):
            pose = pose @ np.linalg.inv(step)
            graph.add_node(i + 1, pose.copy())
            graph.add_edge(i, i + 1, step.copy(), info.copy(), False)
        loops, steps, infos = rgbd_loop_closure(depth_list, color_list, intrinsic, config)
        for (i, j), step, info in zip(loops, steps, infos, strict=True):
            graph.add_edge(i, j, step.copy(), info.copy(), True)
        graph.solve_()
        graph.save(str(dataset / "fragments" / f"fragment_optimized_{fragment:03d}.json"))
        grid = integrate(
            depth_list, color_list, intrinsic, intrinsic, graph.export_extrinsics(), config
        )
        cloud = grid.extract_point_cloud().to_legacy()
        o3d.io.write_point_cloud(str(dataset / "fragments" / f"fragment_{fragment:03d}.ply"), cloud)
        del grid
        o3c.cuda.release_cache()


def register(scripts: Path, config: Path, log: Path) -> None:
    """The reconstruction_system's stages after --make, on the tensor fragments."""
    command = [sys.executable, "run_system.py", "--config", str(config)]
    with log.open("w") as out:
        done = subprocess.run(
            [*command, "--register", "--refine", "--integrate"],
            cwd=scripts,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if done.returncode:
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise SystemExit(f"Open3D reconstruction failed, see {log}:\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording", type=Path, help="realsense-stereo-record .db")
    parser.add_argument("out", type=Path, help="output .rrd")
    parser.add_argument("--depth-min", type=float, default=0.3, help="nearest depth used (m)")
    parser.add_argument("--depth-max", type=float, default=3.0, help="farthest depth used (m)")
    parser.add_argument(
        "--min-fill",
        type=float,
        default=0.15,
        help="frames with less of their depth in range than this are dropped",
    )
    parser.add_argument("--work", type=Path, help="keep the frames, fragments and log here")
    args = parser.parse_args()

    if args.work is not None and (args.work / "dataset").exists():
        raise SystemExit(f"{args.work / 'dataset'} exists: pass an empty --work")
    start = time.perf_counter()
    scripts = reconstruction_system()
    tensor = tensor_reconstruction_system(scripts)
    # On disk and out of the repo: frames and fragments run to gigabytes, and /tmp may be RAM.
    (CACHE_DIR / "tool_mesh").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE_DIR / "tool_mesh") as scratch:
        work = args.work or Path(scratch)
        dataset = work / "dataset"
        kept, dropped = export(
            args.recording, dataset, args.depth_min, args.depth_max, args.min_fill
        )
        print(
            f"{kept} frames kept, {dropped} dropped with under {args.min_fill:.0%} of their depth "
            f"in {args.depth_min:g}-{args.depth_max:g} m"
        )
        config = {
            "name": args.recording.parent.name,
            "path_dataset": str(dataset),
            "path_intrinsic": str(dataset / "intrinsic.json"),
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            **OPEN3D_CONFIG,
        }
        (work / "config.json").write_text(json.dumps(config, indent=2))
        tick = time.perf_counter()
        tensor_fragments(tensor, dataset, args.depth_max)
        print(f"tensor fragments in {time.perf_counter() - tick:.0f} s")
        register(scripts, work / "config.json", work / "open3d.log")
        mesh = o3d.io.read_triangle_mesh(str(dataset / "scene" / "integrated.ply"))
        path = camera_path(dataset / "scene" / "trajectory.log")
        write_rrd(mesh, args.out)
    moved = np.linalg.norm(path - path[0], axis=1)
    print(
        f"camera path: farthest {moved.max():.2f} m from start, ends {moved[-1]:.2f} m from it; "
        f"mesh {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )
    print(f"wrote {args.out} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
