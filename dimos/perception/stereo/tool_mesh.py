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

"""Mesh a recording's color + aligned depth with Open3D's reconstruction system, into an .rrd.

Run: uv run python -m dimos.perception.stereo.tool_mesh <recording.db> <out.rrd>
         [--depth-min <m>] [--depth-max <m>] [--min-fill <fraction>] [--voxel <m>] [--work <dir>]

Made for realsense-stereo-record recordings. Open3D's own make/register/refine/integrate
pipeline runs unmodified, cloned once into the dimos cache at the installed open3d version.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from numpy.typing import NDArray
import open3d as o3d  # type: ignore[import-untyped]

from dimos.constants import CACHE_DIR
from dimos.memory.cli.dataset import open_dataset

OPEN3D_REPO = "https://github.com/isl-org/Open3D.git"
SCRIPTS = "examples/python/reconstruction_system"
PAIR_TOLERANCE_S = 0.002
DEPTH_UNITS_PER_M = 1000  # DEPTH16 millimeters
# Open3D tracks on color too; its default 95 changes the trajectory.
JPEG_QUALITY = 100
# Open3D's config/realsense.json, less the paths, the depth range and the voxel.
OPEN3D_CONFIG = {
    "voxel_size": 0.05,
    "depth_diff_max": 0.07,
    "preference_loop_closure_odometry": 0.1,
    "preference_loop_closure_registration": 5.0,
    "icp_method": "color",
    "global_registration": "ransac",
    "python_multi_threading": True,
}
# Open3D's TSDF voxel is tsdf_cubic_size / 512; realsense.json's 3.0 makes it 5.9 mm.
TSDF_VOXELS_PER_CUBE = 512
DEFAULT_VOXEL_M = 0.01


def reconstruction_system() -> Path:
    """Open3D's reconstruction_system example at the installed open3d version, cloned once."""
    version = o3d.__version__.split("+")[0]
    root = CACHE_DIR / f"open3d-{version}"
    scripts = root / SCRIPTS
    if not (scripts / "run_system.py").exists():
        shutil.rmtree(root, ignore_errors=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        clone = ["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1"]
        sparse = ["--filter=blob:none", "--sparse", "--branch", f"v{version}"]
        subprocess.run([*clone, *sparse, OPEN3D_REPO, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", SCRIPTS], check=True)
    return scripts


def export(
    recording: Path, dataset: Path, depth_min: float, depth_max: float, min_fill: float
) -> tuple[int, int]:
    """Color/depth pairs in Open3D's dataset layout, returning (kept, dropped).

    A frame with too little depth in range is dropped: Open3D's odometry runs off into
    kilometers on one, and every later pose inherits it.
    """
    (dataset / "color").mkdir(parents=True)
    (dataset / "depth").mkdir()
    lo, hi = depth_min * DEPTH_UNITS_PER_M, depth_max * DEPTH_UNITS_PER_M
    kept = dropped = 0
    with open_dataset(str(recording)) as store:
        info = store.streams.camera_info.first().data
        depths = list(store.streams.depth_image)
        if depths[0].data.frame_id != info.frame_id:
            frame = depths[0].data.frame_id
            raise SystemExit(f"depth is in {frame}, not aligned to color in {info.frame_id}")
        depth_ts = np.array([obs.ts for obs in depths])
        for color in store.streams.color_image:
            k = int(np.abs(depth_ts - color.ts).argmin())
            if abs(depth_ts[k] - color.ts) > PAIR_TOLERANCE_S:
                continue
            depth = np.asarray(depths[k].data.data, dtype=np.uint16)
            if np.mean((depth >= lo) & (depth < hi)) < min_fill:
                dropped += 1
                continue
            cv2.imwrite(
                str(dataset / "color" / f"{kept:06d}.jpg"),
                color.data.to_opencv(),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            cv2.imwrite(str(dataset / "depth" / f"{kept:06d}.png"), depth)
            kept += 1
    k = info.K
    intrinsic = {
        "width": info.width,
        "height": info.height,
        "intrinsic_matrix": [k[0], 0, 0, 0, k[4], 0, k[2], k[5], 1],  # column-major
    }
    (dataset / "intrinsic.json").write_text(json.dumps(intrinsic))
    return kept, dropped


def reconstruct(scripts: Path, config: Path, log: Path) -> None:
    command = [sys.executable, "run_system.py", "--config", str(config)]
    with log.open("w") as out:
        done = subprocess.run(
            [*command, "--make", "--register", "--refine", "--integrate"],
            cwd=scripts,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if done.returncode:
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise SystemExit(f"Open3D reconstruction failed, see {log}:\n{tail}")


def camera_path(trajectory: Path) -> NDArray[np.float64]:
    """Camera positions from Open3D's trajectory.log: per frame an index line, then a 4x4 pose."""
    rows = [line.split() for line in trajectory.read_text().splitlines() if line.strip()]
    poses = np.array([rows[i + 1 : i + 5] for i in range(0, len(rows), 5)], dtype=np.float64)
    return poses[:, :3, 3]


def write_rrd(mesh: o3d.geometry.TriangleMesh, out: Path) -> None:
    import rerun as rr

    from dimos.visualization.rerun.init import rerun_init

    rerun_init("tool-mesh")
    rr.save(str(out))
    # Open3D's world is the first color camera's optical frame: x right, y down, z forward.
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "map",
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
            vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
            triangle_indices=np.asarray(mesh.triangles, dtype=np.uint32),
        ),
        static=True,
    )


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
    parser.add_argument(
        "--voxel",
        type=float,
        default=DEFAULT_VOXEL_M,
        help="TSDF voxel (m); smaller is sharper but takes much more RAM, see README",
    )
    parser.add_argument("--work", type=Path, help="keep the frames, fragments and log here")
    args = parser.parse_args()

    if args.work is not None and (args.work / "dataset").exists():
        raise SystemExit(f"{args.work / 'dataset'} exists: pass an empty --work")
    start = time.perf_counter()
    scripts = reconstruction_system()
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
            "tsdf_cubic_size": args.voxel * TSDF_VOXELS_PER_CUBE,
            **OPEN3D_CONFIG,
        }
        (work / "config.json").write_text(json.dumps(config, indent=2))
        reconstruct(scripts, work / "config.json", work / "open3d.log")
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
