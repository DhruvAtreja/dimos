# Optimizing VQA Generation

Plan for improving the deterministic VQA generator in this package. It starts from measured
baseline runs on `main` and lists changes in priority order. Usage docs live in
[docs/usage/vqa.md](/docs/usage/vqa.md).

## Baseline

Recording: `2026-08-27_sf_office_8mins_moshi/recording_go2.db` (2043 images, Go2 `lidar` stream).
Machine: RTX 5070 laptop (8 GB). Code: `main` at `1e7b4918bc`.

| Frames | Total time | Proposed | Accepted | Unique | Presence + count | LiDAR questions | Always-"yes" score |
|---|---|---|---|---|---|---|---|
| 1 | 22.9 s | 6 | 4 (67%) | 4 | 100% | 0 | 75% |
| 10 | 58.2 s | 62 | 41 (66%) | 35 | 98% | 0 | 56% |
| 100 | 352.6 s | 444 | 286 (64%) | 153 | 86% | 6 | 53% |

- Bootup is about 15 s; generation is about 3.4 s per frame (median, 100-frame run).
- Unique counts distinct question + choices. LiDAR questions are `object_distance` + `closest_object`.

Family split, 100-frame run (accepted / proposed):

| Family | Accepted / proposed | Main rejection reason |
|---|---|---|
| presence | 153 / 197 | detector found nothing (44) |
| object_count | 94 / 100 | detector found nothing (6) |
| largest_visible_area | 19 / 80 | more than one detection for a name (58) |
| horizontal_direction | 12 / 14 | more than one detection (2) |
| closest_object | 5 / 48 | more than one detection (29), < 5 LiDAR points (8), ranges overlap (6) |
| image_coverage | 2 / 3 | |
| object_distance | 1 / 2 | |

## What limits yield

1. **Family choice.** The author mostly proposes presence and count. Direction, coverage and
   distance were 19 of 444 proposals.
2. **One-detection rule.** Multi-object families need exactly one Moondream detection per name.
   This caused 87 rejections, more than any other cause.
3. **LiDAR coverage.** The Go2 `lidar` stream is a local voxel map in `world`: about 22k points,
   about 6 m around the robot, heights clipped to -0.2 to 1.0 m. Only 2k to 3.5k points land in a
   1280x720 image (about 0.3% of pixels), and the farthest in-image point is 3.1 to 3.8 m. Objects
   above 1 m or beyond about 4 m get 0 points, and every accepted range was 1.3 to 2.9 m, so
   "3 meters or more" is effectively unreachable.
4. **Label balance.** Presence is always "yes". Answering "yes" to everything scores 53%.
5. **Repetition.** Only 153 of 286 accepted questions are unique; neighbouring frames repeat the
   same scene.
6. **Reproducibility.** The author call sets no temperature or seed, so each run proposes a
   different question set.

## Plan

### 1. Author proposes the right families

- Ask for at least one proposal per available family, or run one author call per family group.
- Add one example per family to the prompt.
- For single-instance families, ask for distinctive references ("red chair on the left").
- Set temperature to 0 and record a prompt version in `audit/run.json`.

### 2. Relax the one-detection rule safely

- Accept multiple detections when one clearly dominates (for example, area at least 2x the next).
- Use the author's spatial qualifier ("left", "right") to pick one instance deterministically.
- Add category-level variants that use all instances: "the nearest chair", "the largest chair".
- On more than one match, ask the author once for a more specific reference.

### 3. Windowed LiDAR map

- Merge clouds within about 5 to 10 s of the frame. They are already in `world`, so projection
  uses the existing `world` to camera transform with no new calibration.
- Deduplicate by voxel and draw each point as a patch sized by voxel size and depth, so near
  surfaces hide what is behind them in the z-buffer.
- Re-run the same 100 frames and compare: LiDAR-stage rejections, supporting points per mask, and
  answers at 3 m or more.

### 4. Full map and better sensors

- Build a full-recording map only after checking odometry drift, and drop moving objects
  (people, moved chairs).
- Prefer Mid-360 recordings (`pointlio_lidar`) to remove the 1 m height cap and range limit.
- Optional: a monocular depth model scaled by LiDAR, used only where the two agree.

### 5. Balanced, less repetitive datasets

- Negative presence: propose objects that are not in the frame and keep "no" only when the
  detector finds nothing.
- Select frames by robot motion (odometry distance or yaw change) or image similarity instead of a
  fixed stride, and skip blurry frames.
- Drop identical question + answer pairs within a window of frames, and cap each family's share.
- Use adaptive distance buckets like `image_coverage`, and add buckets past 3 m once the map lands.

### 6. Label quality

- Write a review image per answered question into `audit/`: box, mask and projected points.
- Hand-check 30 to 50 questions per family and track precision.
- Optional: cross-check presence and count with a second detector.

### 7. Speed and tooling

- Cache detections per image and object name. Presence, count and the mask pipeline each call
  Moondream separately for the same name today.
- Overlap the author call for the next frame with perception on the current frame.
- Record bootup and per-frame timings in `audit/run.json`.
- Add CLI flags for families, models and thresholds.

### 8. Evaluation

- Run `dimos evals vqa run` on the 100-frame set with a few models.
- Report per-family accuracy next to always-"yes" and random-choice baselines.

## Targets

Proposed numbers for the 100-frame run:

| Metric | Baseline | Target |
|---|---|---|
| Presence + count share | 86% | < 60% |
| LiDAR questions share | 2% | >= 15% |
| Accepted / proposed | 64% | >= 75% |
| Unique / accepted | 53% | >= 80% |
| Always-"yes" score | 53% | <= 30% |
| Time per frame | 3.4 s | <= 5 s |

## Reproduce the baseline

```bash skip
dimos evals vqa generate /path/to/recording_go2.db --start 0 --stop 2000 --stride 20 \
  --output ~/.local/state/dimos/datasets/vqa/sf-office-100frames
```

Counts are in `audit/run.json`. Per-question accept and reject reasons are in
`audit/frame-*/ground_truth.json`.
