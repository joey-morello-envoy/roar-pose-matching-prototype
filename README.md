# Pose Matching

Silhouette-based pose matching prototype for a live camera installation. A
person stands in front of a depth camera, and the system reports how well
their silhouette matches a reference pose, in real time.

The matcher works entirely on **binary mask shape** — no skeleton tracking,
no keypoint detection. This makes it cheap, deterministic, and forgiving of
lighting / clothing / depth noise.

The project runs inside **TouchDesigner** as a set of Script TOPs that wrap
NumPy + OpenCV code, plus one GLSL helper. The Femto Bolt depth stream is
the eventual live input; in the current state the pipeline is driven by
still reference images for tuning and validation.

---

## What's in the repo

```
pose_matching/
├── pose_matching.toe                   TouchDesigner project (current)
├── pose_matching.35.toe                Backup of the above (TD 2025 build 35)
├── Backup/                             Project auto-saves
├── requirements.txt                    pip deps for the .venv
├── TDPyEnvManagerContext.yaml          TDPyEnvManager: bind .venv to TD
├── .venv/                              Python 3.11 venv (created by TDPyEnvManager)
├── assets/
│   ├── code/
│   │   ├── python/
│   │   │   ├── pose_preprocess.py      Phase B/C/D — clean → split → normalize → atlas
│   │   │   ├── pose_match_iou.py       Matcher: pixel IoU + optimal assignment
│   │   │   ├── pose_match.py           Matcher: manual Hu-moments distance (multi-slot)
│   │   │   └── pose_matchShapes.py     Matcher: cv2.matchShapes (single tile)
│   │   └── glsl/
│   │       └── crop_to_edge.frag       Optional auto-crop to silhouette bbox
│   └── images/
│       └── pose_ref/                   Reference silhouettes + debug dumps
└── docs/
    ├── pose-matching-system-reference.md     Technical reference (start here)
    ├── pose-matching-plan.md                  Original implementation plan
    ├── pose_preprocess_and_iou_explained.md   Deep-dive on the two main scripts
    └── hu_moments_pose_matching_engineering_plan.md  Original Hu-moments plan
```

---

## How it works

The pipeline has three logical stages. Both the live (candidate) silhouette
and the reference silhouette pass through the same preprocessing stage
before reaching a matcher.

```
[Live silhouette]  ──►  pose_preprocess.py  ──┐
                                              ├──►  matcher  ──►  match_score
[Ref silhouette]   ──►  pose_preprocess.py  ──┘
```

An optional GLSL crop step (`crop_to_edge.frag`) can be inserted upstream of
preprocessing to strip excess black padding from a reference image.

### Stage 0 — Optional GLSL crop (`crop_to_edge.frag`)

Auto-crops a thresholded silhouette to its tight bounding box and fits the
result into the output frame with aspect ratio preserved (letterboxed).
Useful for reference images that have a lot of empty padding.

Cost is `O(SCAN_RES²)` texture taps per output pixel. `SCAN_RES = 64` is
fine for static reference images. For live every-frame input either drop
`SCAN_RES` to ~24–32 or replace the scan with bbox uniforms from an
`analyzeTOP` (see the comments at the bottom of the shader).

### Stage 1 — Preprocessing (`pose_preprocess.py`)

A Script TOP that takes a raw silhouette image (white = subject, black =
background) and produces a **canonical atlas** of per-person masks. Runs
four internal phases:

- **Phase A — Input parsing.** Read the red channel of the TOP input as a
  float32 0..1 image, promote to uint8 0..255.
- **Phase B — Mask cleanup.** Hard-threshold, morphological CLOSE with an
  elliptical kernel sized as a percentage of the image diagonal (so tuning
  is resolution-agnostic), Gaussian blur + re-threshold to smooth jagged
  edges, drop connected components smaller than a configurable percentage
  of the frame.
- **Phase C — Person separation.** `connectedComponentsWithStats` gives
  one blob per visually-isolated person. Two people touching produce a
  fused blob, so each component is tested against three heuristics
  (wide aspect ratio, low solidity vs convex hull, vertical pinch in the
  column profile). Flagged components are split via distance-transform
  watershed; pure vertical cut at the pinch column is the fallback when
  watershed only finds one peak.
- **Phase D — Per-person normalization.** Each individual mask is tight-
  cropped, aspect-preserving resized into a `Canonsize × Canonsize` square,
  letterbox-padded, and optionally translated so its binary-moments
  centroid lands exactly at the tile center. After this step, an adult
  and a child holding the same pose look nearly identical.

The normalized tiles are sorted (left-to-right by centroid, or largest-
first) and packed into a single RGBA atlas image whose grid is
approximately square. The atlas layout and per-slot metadata are published
via `scriptOp.store()` so downstream ops can slice tiles back out without
re-deriving the layout:

| Store key       | Description                                          |
|-----------------|------------------------------------------------------|
| `num_people`    | Number of populated slots                            |
| `atlas_grid`    | `[cols, rows, canon]` grid geometry                  |
| `slot_bboxes`   | Source-frame bbox of each person `[[x,y,w,h], ...]`  |
| `slot_centroids`| Source-frame centroid of each person                 |
| `slot_areas`    | Source-frame pixel area of each person               |

### Stage 2 — Matching

Three matchers ship in the repo. They all consume the atlas format from
`pose_preprocess.py` and differ in how they measure similarity. Pick one
per use case — they are not meant to be chained.

| Script                | Method                              | Score direction      | Multi-person                                  |
|-----------------------|-------------------------------------|----------------------|-----------------------------------------------|
| `pose_match_iou.py`   | Pixel IoU + optimal assignment      | 0 → 1, higher better | Yes, any M targets × N candidates             |
| `pose_match.py`       | Hu-moments distance (log-scaled)    | 0+, lower better     | Yes, index-matched slot-by-slot               |
| `pose_matchShapes.py` | `cv2.matchShapes` I1 / I2 / I3      | 0+, lower better     | No, operates on a single tile                 |

#### `pose_match_iou.py` (recommended)

For every (target_i, candidate_j) pair, compute `|A ∩ B| / |A ∪ B|` on the
normalized binary masks. When **Mirror Invariant** is on, each candidate is
also tested against its horizontal flip and the higher IoU wins (so a
right-facing T-pose matches a left-facing T-pose).

The result is an M×N IoU matrix. A bitmask-DP solver finds the one-to-one
permutation that maximizes total IoU. Unequal counts are padded to square
with zero-IoU dummies, which naturally penalizes scenes where person counts
don't match. The final `match_score` is the mean IoU across the chosen
assignment.

Score bands (higher = better):

| Score range  | Meaning                                                 |
|--------------|---------------------------------------------------------|
| ≥ 0.90       | Excellent — near-identical silhouettes                  |
| 0.75 – 0.90  | Good match — same pose, minor size/edge differences     |
| 0.50 – 0.75  | Borderline — similar shape family but not a clear match |
| < 0.50       | No match                                                |

The op also publishes `iou_matrix`, `assignment`, `assignment_scores`, and
`assignment_mirrored` to `store()` for downstream visualization.

#### `pose_match.py` (Hu moments, manual)

`cv2.HuMoments(cv2.moments(mask))` returns seven moment-derived values that
are invariant to translation, scale, and rotation. They're log-scaled with
sign preserved before comparison, then per-slot distances are summed and
averaged across slot pairs. Multi-slot atlas-aware. Useful when you want
rotation invariance "for free" without doing it yourself.

#### `pose_matchShapes.py` (Hu moments, OpenCV built-in)

Thin wrapper around `cv2.matchShapes()` with `I1` / `I2` / `I3` selectable.
Simpler than the manual Hu script but does **not** understand the atlas —
it operates on a single tile.

---

## Choosing a method

| Property                   | IoU + assignment    | Hu (manual)             | matchShapes        |
|----------------------------|---------------------|-------------------------|--------------------|
| Score direction            | ↑ higher (0..1)     | ↓ lower (0 = identical) | ↓ lower            |
| Mirror invariance          | Toggle per-run      | Inherent in Hu          | Inherent in Hu     |
| Multi-person support       | Optimal assignment  | Index-matched slots     | Single tile only   |
| Sensitivity to alignment   | High (pixel-level)  | Medium (distribution)   | Medium             |
| Empty-mask handling        | Returns 0           | Returns inf, excluded   | Returns inf        |
| Debug output               | RGBA overlay        | Live passthrough        | Live passthrough   |
| Best for                   | Precise matching    | Multi-slot Hu distance  | Single-person demo |

Default choice for the installation is `pose_match_iou.py` — it's the most
discriminating and gives a clean, intuitive 0..1 score.

---

## Setup

### Prerequisites

- Windows 10/11
- TouchDesigner 2025 (currently building against build 35)
- The `TDPyEnvManager` palette component
- Python 3.11 (inherited from TD's bundled interpreter)

### Install dependencies

1. Open `pose_matching.toe` in TouchDesigner.
2. `TDPyEnvManagerContext.yaml` is already configured to bind a local
   `.venv` to the project. With the TDPyEnv Manager component, **pulse
   "Create from requirements.txt"** to provision the venv and install
   dependencies.
3. Alternatively, from a shell:
   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

### Femto Bolt depth capture (Phase 2)

`pyorbbecsdk` is not on PyPI. Grab the matching wheel from
[orbbec/pyorbbecsdk releases](https://github.com/orbbec/pyorbbecsdk/releases)
(`cp311-win_amd64`) and install it into the venv:

```powershell
.venv\Scripts\python.exe -m pip install path\to\pyorbbecsdk-X.Y.Z-cp311-cp311-win_amd64.whl
```

---

## Tuning notes

The most common knobs:

- **`Threshold`** on `pose_preprocess` — cutoff applied to the red channel
  before binarization. Bump this up if the silhouette has soft edges.
- **`Closepct`** — morphological CLOSE kernel as % of image diagonal.
  Higher fills bigger gaps (between fingers, between legs).
- **`Minareapct`** — drops blobs below this % of frame area. Raise if
  sensor speckle is showing up; lower if you want to keep small subjects
  like children.
- **`Canonsize`** — side length of each per-person tile. 256 is the
  default; 128 is fast and usually enough for IoU.
- **`Mirrorinvariant`** (on `pose_match_iou`) — turn on so left-facing and
  right-facing versions of the same pose match.
- **`Threshold`** on `pose_match_iou` — 0.75 is the default `match_pass`
  cutoff. Lower if matches feel too strict, raise if false positives leak
  through.

IoU is **sensitive to alignment** — a silhouette shifted by 5% of the tile
width loses meaningful overlap. The centroid alignment step in Phase D
exists specifically to mitigate this; if matches feel jumpy, verify that
step is enabled.

---

## Status

| Stage                                       | Status     |
|---------------------------------------------|------------|
| Reference-image driven preprocessing        | Working    |
| Mask cleanup (Phase B)                      | Working    |
| Person separation / fused-blob split (Phase C) | Working |
| Per-person normalization + atlas (Phase D)  | Working    |
| IoU + assignment matcher                    | Working    |
| Hu-moments matcher (manual + matchShapes)   | Working    |
| GLSL bbox auto-crop                         | Working    |
| Femto Bolt live depth capture               | Pending    |
| End-to-end live → match → UI feedback       | Pending    |

---

## Further reading

- [`docs/pose-matching-system-reference.md`](docs/pose-matching-system-reference.md)
  — concise technical reference for the whole pipeline. **Start here.**
- [`docs/pose_preprocess_and_iou_explained.md`](docs/pose_preprocess_and_iou_explained.md)
  — function-by-function deep dive into `pose_preprocess.py` and
  `pose_match_iou.py`.
- [`docs/pose-matching-plan.md`](docs/pose-matching-plan.md) — original
  implementation plan; useful for the rationale behind each phase.
- [`docs/hu_moments_pose_matching_engineering_plan.md`](docs/hu_moments_pose_matching_engineering_plan.md)
  — the project's original Hu-moments-only engineering plan, before IoU
  was added.
