# RealSense D455f -> 3D map

Walk around with a RealSense D455f in your hand, get a 3D map of the place as an .rrd artifact.
The map is a colored mesh. Open3D's reconstruction system builds it from the color and depth
streams and works out the camera poses itself, no SLAM module needed.

1. Record
2. Point tool_mesh to the recording, get the .rrd artifact
3. Inspect it in rerun (dimos-viewer)

Use tool_mesh. The GPU variant, tool_mesh_hybrid, is experimental and its maps come out
incorrectly aligned, see [below](#tool_mesh_hybrid-experimental-incorrectly-aligned).

## 1. Record

Plug in the camera, then:

```bash
uv run dimos run realsense-stereo-record
```

A window opens with the live camera view. Aim, hit Record (or Space), walk. Hit Stop (or S, Esc,
or close the window): the recording is saved and the run ends. No pause, tool_mesh needs one
continuous walk.

The bar at the bottom shows REC with the elapsed time, frames, file size, fps, how much of the
depth is in tool_mesh's 0.3-3 m range (red under 15%, tool_mesh drops those frames) and free disk.
Red 0 fps means the camera stopped sending.

The recording lands in `recordings/<date>_<time>-<zone>/stereo.db` in the repo root (also in the
window title). About 15 MB/s on disk, so 30 min is about 27 GB. To name the folder yourself:

```bash
uv run dimos run realsense-stereo-record --stereorecorder.db-path recordings/my_walk/stereo.db --recordwindow.db-path recordings/my_walk/stereo.db
```

First run builds the camera driver if it is missing, see [RealSense](../../hardware/sensors/camera/realsense/README.md).

## 2. Produce the .rrd

```bash
REC=$(ls -t recordings/*/stereo.db | head -1)  # the newest recording, or put your own path
time uv run python -m dimos.perception.stereo.tool_mesh "$REC" map.rrd
```

About 4 min for an 85 s indoor walk, see [Timing](#timing). The map is built from 1 cm voxels,
read [Voxel](#voxel) before you change that. First run clones Open3D's example scripts into
`~/.cache/dimos`, needs internet once.

## 3. Inspect

```bash
uv run dimos-viewer map.rrd
```

All in one go:

```bash
time uv run python -m dimos.perception.stereo.tool_mesh "$REC" map.rrd && uv run dimos-viewer map.rrd
```

## Voxel

tool_mesh builds the map out of small cubes, voxels, then cuts the mesh out of them. `--voxel` is
the cube size in meters, default `0.01` (1 cm). This one param decides the detail, the RAM and the
file size:

- Detail: nothing smaller than a voxel makes it into the map. The colors sit on the mesh points,
  so they blur the same way.
- RAM: this is what kills runs. From 2 cm to 1 cm took 1.6x the RAM indoors and 2.3x outdoors.
  From 1 cm to 6 mm took 2.3x indoors, outdoors the old 5.9 mm default ran out of RAM.
- Mesh and .rrd size: about 5x for every halving of the voxel.
- Time: grows much less than the RAM, see the table. Most of it goes to tracking the camera, the
  voxel plays no role there.
- Outdoors costs more than indoors at the same voxel: you keep passing new things, indoors you
  see the same room again and again.

Measured on the same two recordings, the machine is in [Timing](#timing):

| Recording | Voxel | Time | Peak RAM | Mesh vertices | .rrd |
|---|---|---|---|---|---|
| indoor walk, 85 s | 0.02 | 3.7 min | 6.3 GB | 0.46M | 11 MB |
| indoor walk, 85 s | 0.01 (default) | 4.1 min | 10.1 GB | 2.5M | 59 MB |
| indoor walk, 85 s | 0.006 | 4.8 min | 22.9 GB | 8.2M | 196 MB |
| outdoor walk, 104 s | 0.02 | 4.9 min | 8.7 GB | 2.3M | 54 MB |
| outdoor walk, 104 s | 0.01 (default) | 6.8 min | 19.7 GB | 12.6M | 294 MB |
| outdoor walk, 104 s | 0.0059 (old default) | ran out of RAM | | | |

Testing on a new place or a new kind of walk? Start coarse and go finer step by step on the same
recording, then flip between the maps in the viewer (Sources panel, top left):

```bash
time uv run python -m dimos.perception.stereo.tool_mesh "$REC" map_2cm.rrd --voxel 0.02
time uv run python -m dimos.perception.stereo.tool_mesh "$REC" map_1cm.rrd --voxel 0.01
time uv run python -m dimos.perception.stereo.tool_mesh "$REC" map_6mm.rrd --voxel 0.006
uv run dimos-viewer map_2cm.rrd map_1cm.rrd map_6mm.rrd
```

One run at a time, and watch the RAM while it runs (htop). Each finer step took up to 2.3x the RAM
of the one before, over 3x outdoors at 6 mm: if a run peaked over a third of your RAM, stop there.

## Timing

Measured on a laptop with an AMD Ryzen AI 7 350 (8 cores, 16 threads) and 64 GB RAM. tool_mesh
runs on the CPU, Open3D spreads it over all cores.

- The walks above take 4-7 min, about 3-4x the walk itself.
- Longer walks take a lot longer than that. tool_mesh cuts the walk into fragments of 100 frames,
  about 7 s of walk each, then:
  - builds the fragments 15 at a time, about 3 min per 15 (100 s of walk)
  - matches every pair of fragments, so 2x the walk is 4x the pairs. 78 pairs took 20 s indoors,
    91 pairs took 2.5 min outdoors: pairs that don't overlap are the slow ones
  - puts the whole map together, about 35-50 s per 100 s of walk
- A 30 min walk, estimated, not measured: about 50 min to build 270 fragments, then about 36,000
  pairs to match, 3 h at the indoor rate and 16 h at the outdoor rate, then 10-15 min to put the
  map together, if it fits in RAM.
- RAM on long walks: building the fragments stays around 10-20 GB at 1 cm. The last step holds the
  whole map, so outdoors it grows with the distance walked: 16 GB at 1 cm and 4.6 GB at 2 cm for the
  104 s outdoor walk (about 100 m). Estimated from that, 64 GB fills after about 6 min of outdoor
  walking at 1 cm, about 20 min at 2 cm.

## Params

| Flag | Default | What it does |
|---|---|---|
| `--voxel` | `0.01` | map cube size, m, see [Voxel](#voxel) |
| `--depth-min` | `0.3` | nearest depth used, m |
| `--depth-max` | `3.0` | farthest depth used, m |
| `--min-fill` | `0.15` | drops frames with less depth in range than this, like a hand over the lens |
| `--work <dir>` | temp, deleted | keeps the frames, fragments, Open3D log and camera poses (`dataset/scene/trajectory.log`) in a new or empty dir |

```bash
time uv run python -m dimos.perception.stereo.tool_mesh "$REC" map.rrd --depth-max 4 --work ~/mesh_work
```

## Good to know

- Walk slow, turn slow. Blank walls and fast pans are the hardest part for the tracking.
- Come back to where you started. The tool prints how far the camera path ends from its start, our good walk ended 0.20 m off.
- Two runs on the same recording can come out different: Open3D's loop closure matches at random.
- One run at a time. Two runs at once used up 61 GB of RAM and killed the desktop session.
- Temp files go to `~/.cache/dimos/tool_mesh`, deleted at the end: 0.7 GB for the 85 s indoor walk at 1 cm, 1.8 GB for the 104 s outdoor one.
- The map starts at the first camera pose: x right, y down, z forward.

## tool_mesh_hybrid: experimental, incorrectly aligned

**Its maps come out incorrectly aligned.** It is the tensor (GPU) path: same as
tool_mesh, but the fragments are built on the GPU with Open3D's tensor code. Needs an NVIDIA GPU.

On our test walk it was about 2x faster, but the camera path ended 9.4 m from its start (tool_mesh:
0.2 m) and went 11 m away on a walk that spans 5.4 m. It has no `--voxel` and always builds at the
old 5.9 mm. Kept only to work on it.

```bash
time uv run python -m dimos.perception.stereo.tool_mesh_hybrid "$REC" map_hybrid.rrd && uv run dimos-viewer map_hybrid.rrd
```

## When recording fails

- `RealSense: no frames within 1s - retrying`: USB autosuspend put the camera to sleep. Replug it.
- `error while loading shared libraries: librealsense2.so...`: nix cleaned up the library the driver was built against. Rebuild it: `uv run dimos run realsense-stereo-record --build-native`.
- IMU `Permission denied` on `scan_elements`: the udev rules call bash and chmod from nix store paths that are gone. Point them at `/bin`, then replug:

  ```bash
  sudo sed -i -E 's#/nix/store/[a-z0-9]+-bash-[^/]+/bin/#/bin/#g; s#/nix/store/[a-z0-9]+-coreutils-[^/]+/bin/#/bin/#g' /etc/udev/rules.d/99-realsense-libusb.rules
  sudo udevadm control --reload-rules
  ```

## Files

- `blueprints/realsense_stereo_record.py`: the `realsense-stereo-record` blueprint, the camera, the camera window and a recorder that only writes between Record and Stop. Records color, depth, the infrared pair and the IMU, infrared and depth lossless.
- `record_window.py`: the camera window, live view, telemetry, Record and Stop.
- `tool_mesh.py`: recording -> .rrd with Open3D's reconstruction_system, unmodified.
- `tool_mesh_hybrid.py`: the experimental tensor (GPU) variant, maps come out incorrectly aligned.
