# Pose Matching System — Technical Reference

Silhouette-based pose comparison pipeline running as TouchDesigner Script TOPs.
No skeleton or keypoint detection — pure binary mask shape analysis.

---

## Pipeline Overview

```
[Live Silhouette]  ──→  [pose_preprocess.py]  ──┐
                                                  ├──→  [Matching Script]  ──→  match_score
[Ref Silhouette]   ──→  [pose_preprocess.py]  ──┘
```

Both silhouette inputs follow the same preprocessing path before reaching the matcher.
An optional GLSL crop step (`crop_to_edge.frag`) can be inserted before preprocessing
to remove excess empty padding from the input image.

**Three matching scripts are available — choose one:**

| Script | Method | Score direction |
|---|---|---|
| `pose_match_iou.py` | Pixel IoU + optimal assignment | 0–1, higher = better |
| `pose_match.py` | Hu moments (manual, multi-slot) | 0+, lower = better |
| `pose_matchShapes.py` | `cv2.matchShapes()` I1/I2/I3 | 0+, lower = better |

---

## Stage 0 — Optional GLSL Crop (`crop_to_edge.frag`)

Before preprocessing, a GLSL fragment shader can auto-crop the silhouette to its tight
bounding box. Useful when the input has heavy black padding.

**How it works:**

1. Scan the input on a 64×64 grid (`SCAN_RES`) to find the UV bounding box of all
   pixels above threshold (default 0.5).
2. Pad the bbox by 1% per side (`kPadding`).
3. Fit the bbox into the output frame with aspect ratio preserved — black letterboxing
   fills unused area.

Cost is O(SCAN_RES²) texture samples per output pixel (~4,096 at SCAN_RES=64). Fine
for static reference images. For live every-frame input, either reduce SCAN_RES to
24–32 or replace the scan loop with uniforms from an analyzeTOP (`uBBoxMin`/`uBBoxMax`).

---

## Stage 1 — Silhouette Preprocessing (`pose_preprocess.py`)

Accepts a raw silhouette image (white = subject, black = background) and produces a
normalized atlas image. Runs four internal phases.

### Phase B — Mask Cleanup

The raw input may have holes, edge fringing, and small noise blobs from the depth sensor.

| Step | Operation | Purpose |
|---|---|---|
| 1 | `cv2.threshold` (binary) | Convert float mask to hard 0/255 using the Threshold parameter |
| 2 | `cv2.morphologyEx CLOSE` — elliptical kernel, ~1.5% of image diagonal | Fill internal holes: gaps between fingers, eyes, space between legs |
| 3 | `cv2.GaussianBlur (9×9)` → re-threshold at 127 | Smooth jagged depth-sensor edges before re-binarizing |
| 4 | `cv2.connectedComponentsWithStats` → drop blobs < 0.5% of frame area | Remove sensor speckle and stray pixels |

### Phase C — Person Separation (Fused-Blob Splitting)

Connected components after Phase B may contain multiple people standing close together
or touching. Phase C identifies and splits these fused blobs.

**Fusion detection — a component is flagged if any trigger fires:**

| Heuristic | Condition | Rationale |
|---|---|---|
| Aspect ratio | bbox width / height > 1.3 | A single standing person is taller than wide |
| Solidity drop | area / convex_hull_area < 0.7 | A fused pair has a concave waist the convex hull does not |
| Vertical pinch | column pixel-count dips to < 70% of flanking max in middle 60% of bbox | Physical gap between bodies creates a narrow waist |

**Splitting strategy:**

| Step | Detail |
|---|---|
| Distance transform | `cv2.distanceTransform(component, DIST_L2, 5)` — each interior pixel receives its distance to the nearest background pixel |
| Seed extraction | Threshold at 50% of max distance → connected regions become seeds; each seed is one skeleton core of a person |
| Watershed | If ≥ 2 seeds found: label background=1, seeds=2..N+1, unknown interior=0; run `cv2.watershed` on inverted distance transform |
| Vertical cut fallback | If watershed produces only 1 seed but a pinch column was detected, hard-cut the component at that column into left and right halves |

### Phase D — Per-Person Normalization

Each individual person mask is transformed into a canonical square so that an adult and
a child in the same pose are directly comparable.

| Step | Operation | Effect |
|---|---|---|
| 1 — Tight crop | `np.where` to find min/max pixel coordinates | Removes all empty border padding |
| 2 — Aspect-preserving scale | `scale = min(canon/w, canon/h)` → resize with `INTER_AREA` | Fits the person inside the canon square without stretching |
| 3 — Letterbox pad | `cv2.copyMakeBorder` to reach canon×canon, centered | Keeps the silhouette centered in a fixed-size tile |
| 4 — Centroid alignment | Shift so `cv2.moments` centroid lands at (canon/2, canon/2) | Corrects slight off-center silhouettes from asymmetric poses |

Default canon size is 256×256 pixels. Each resize step re-thresholds at 127 to keep
the mask strictly binary.

**What normalization removes:** translation, scale, centroid offset, aspect variation.

**What normalization preserves:** pose shape, left/right handedness (not flipped — mirror
invariance is handled separately in the matcher), relative limb proportions, contour topology.

> **Note:** IoU is very sensitive to alignment — a silhouette shifted by 5% of the tile
> width loses significant overlap. Centroid alignment is therefore more important for IoU
> accuracy than for Hu moments, which are translation-invariant by construction.

### Atlas Packing and Metadata

Normalized tiles are packed into a grid:
- `cols = ceil(sqrt(maxpeople))`
- `rows = ceil(maxpeople / cols)`
- Slot 0 = top-left, incrementing left-to-right, wrapping to next row.

Slot ordering: left-to-right by centroid x (default), or largest-first by area.

The following values are published via `scriptOp.store()` for downstream ops:

| Key | Type | Description |
|---|---|---|
| `num_people` | int | Number of detected people (populated slots) |
| `atlas_grid` | [cols, rows, canon] | Grid geometry for downstream slicing |
| `slot_bboxes` | list[[x,y,w,h]] | Source-frame bounding box of each person |
| `slot_centroids` | list[[cx,cy]] | Source-frame centroid of each person |
| `slot_areas` | list[int] | Pixel area of each person in source frame |

---

## Stage 2 — Shape Matching

Three Script TOPs implement different comparison strategies. They all consume the same
atlas format but differ in how they measure silhouette similarity.

---

### Method A: IoU + Optimal Assignment (`pose_match_iou.py`)

#### Intersection over Union

For a single pair of binary masks A and B (after normalization to the same size):

```
IoU = |A ∩ B| / |A ∪ B|
```

Pixel counts are used. Result is in [0, 1]. If the union is empty (both masks blank),
IoU returns 0.0.

#### Mirror Invariance

When the **Mirror Invariant** parameter is enabled, each candidate mask is also tested
against its horizontal flip (`cv2.flip(candidate, 1)`). The higher IoU wins:

```
best_iou = max(iou(target, candidate), iou(target, flip(candidate)))
```

This makes a left-facing T-pose match a right-facing T-pose without any preprocessing.
The script records which orientation was chosen per pair.

#### Pairwise IoU Matrix

Given M target people and N candidate people, every (target_i, candidate_j) pair is
evaluated, producing an M×N matrix. This feeds the assignment solver.

#### Optimal One-to-One Assignment

The assignment problem maximizes total IoU across all matched pairs. The solver pads
the matrix to square with zeros (unmatched dummy people) and uses bitmask dynamic
programming:

```
score(row, used_cols) = max over col ∉ used: matrix[row][col] + score(row+1, used|col)
```

Unmatched people (assigned to dummy slots) contribute 0 IoU, naturally penalizing
scenes where person counts differ.

#### Aggregate Score

`match_score` = mean of assigned IoU values across all pairs (including 0 for dummy
pairs). Sits in [0, 1]. `match_pass` = true when `match_score ≥ threshold` (default 0.75).

#### Score Interpretation (higher = better)

| Score range | Meaning |
|---|---|
| ≥ 0.90 | Excellent — near-identical silhouettes |
| 0.75 – 0.90 | Good match — same pose, minor size/edge differences |
| 0.50 – 0.75 | Borderline — similar shape family but not a clear match |
| < 0.50 | No match — different pose or very different proportions |

#### Multi-Person Scenarios

| Scenario | Behavior |
|---|---|
| M targets, N candidates (M = N) | Square IoU matrix → optimal one-to-one assignment → mean score |
| M targets, N candidates (M ≠ N) | Padded to max(M,N)² with zeros → unmatched pairs score 0, lowering the mean |
| One or both inputs empty | Returns match_score=0, match_pass=False |

The assignment solver uses bitmask DP with `@lru_cache`. Up to 4 people = 24 states.
Up to 8 people = 40,320 states — still instantaneous.

#### Stored Values

| Key | Type | Description |
|---|---|---|
| `match_score` | float | 0–1, mean IoU over assigned pairs (higher = better) |
| `match_threshold` | float | Configured threshold parameter |
| `match_pass` | bool | True if match_score ≥ threshold |
| `iou_matrix` | list[list[float]] | Full M×N pairwise IoU values |
| `assignment` | list[(int,int)] | Optimal (target_idx, candidate_idx) pairs |
| `assignment_scores` | list[float] | IoU for each assigned pair |
| `assignment_mirrored` | list[bool] | Whether each pair used the mirrored candidate |
| `num_target_people` | int | Number of target slots extracted |
| `num_candidate_people` | int | Number of candidate slots extracted |

---

### Method B: Hu Moments — Manual Multi-Slot (`pose_match.py`)

#### The Hu Moment Vector

`cv2.HuMoments(cv2.moments(mask))` returns seven values derived from the statistical
moments of the pixel distribution. These are invariant to translation, scale, and rotation
(including reflection for most). Raw values span many orders of magnitude, so they are
log-scaled before comparison:

```
log_hu[i] = −sign(hu[i]) × log₁₀(|hu[i]| + ε)
```

The sign is preserved so positive and negative Hu values map to opposite sides of zero
rather than collapsing toward zero. ε = 1×10⁻³⁰ prevents log(0).

#### Per-Slot Comparison

For each matched slot pair (live[i], ref[i]):

```
score = Σ |log_hu_live[k] − log_hu_ref[k]|   for k in 0..6
```

If either slot's mask is empty (`countNonZero == 0`), the slot returns `inf` and is
excluded from the aggregate mean.

#### Multi-Slot Aggregation

Slots are matched by index: live[0] vs ref[0], live[1] vs ref[1], up to
`min(num_live, num_ref)` pairs. Final score = mean of all finite per-slot scores.

#### Score Interpretation (lower = better)

| Band | Score range | Meaning |
|---|---|---|
| MATCH | < 0.50 | Essentially the same shape |
| CLOSE | 0.50 – 2.00 | Same pose, minor differences |
| SIMILAR | 2.00 – 5.00 | Similar but clearly different pose |
| DIFFERENT | 5.00 – 15.00 | Different pose / very different silhouette |
| INVALID | > 15.00 | One side likely empty or noisy |

#### Stored Values

| Key | Type | Description |
|---|---|---|
| `match_score` | float | Mean of finite per-slot Hu distances (lower = better) |
| `match_scores` | list[float] | Per-slot scores (may include inf for empty slots) |
| `num_pairs` | int | Number of slot pairs compared |
| `hu_live` | list[list[float]] | Log-scaled Hu vectors for each live slot |
| `hu_ref` | list[list[float]] | Log-scaled Hu vectors for each ref slot |

---

### Method C: matchShapes (`pose_matchShapes.py`)

Delegates directly to OpenCV's `cv2.matchShapes()`, which computes Hu-moment descriptors
for each shape and compares them using one of three formulas. Simpler than Method B but
does not support multi-slot atlas awareness — operates on a single tile.

#### The Three Methods

| Method | Formula | Typical match range | Typical non-match range |
|---|---|---|---|
| I1 | Σ \|1/mᵢᴬ − 1/mᵢᴮ\| | < 0.05 | > 0.20 |
| I2 | Σ \|mᵢᴬ − mᵢᴮ\|  (log-scaled) | < 0.50 | > 2.00 |
| I3 | Σ \|mᵢᴬ − mᵢᴮ\| / \|mᵢᴬ\| | < 0.05 | > 0.20 |

mᵢ = log-scaled Hu moment i. I2 uses absolute differences so its magnitude is larger.
I1 and I3 normalize by moment magnitude and stay below 1 for typical silhouettes.

> **Mirror-image ambiguity:** Hu moments are rotation-invariant, which includes reflections.
> A left-arm-raised and a right-arm-raised silhouette will score nearly identically across
> all three methods. If handedness matters, add a secondary centroid-offset check on the
> x-axis.

#### Score Interpretation (I2 defaults, lower = better)

| Band | Score range | Meaning |
|---|---|---|
| MATCH | < 0.50 | Essentially the same shape |
| CLOSE | 0.50 – 2.00 | Same pose, minor differences |
| SIMILAR | 2.00 – 5.00 | Similar but clearly different pose |
| DIFFERENT | 5.00 – 15.00 | Different pose / very different silhouette |
| INVALID | > 15.00 | One side likely empty or noisy |

#### Stored Values

| Key | Type | Description |
|---|---|---|
| `match_score` | float | Raw cv2.matchShapes score (lower = better) |
| `match_method` | str | Which method was active (I1/I2/I3) |

---

## Choosing a Matching Method

| Property | IoU + Assignment | Hu Moments (manual) | matchShapes |
|---|---|---|---|
| Score direction | ↑ higher = better (0–1) | ↓ lower = better (0 = identical) | ↓ lower = better (0 = identical) |
| Mirror invariance | Toggle per-run | None (inherent in Hu) | Inherent in Hu moments |
| Multi-person support | Optimal assignment (any M×N) | Index-matched slots | Single tile only |
| Sensitivity to overlap | High — pixel-level | Medium — distribution-based | Medium — distribution-based |
| Empty mask handling | Returns 0 gracefully | Returns inf, excluded from mean | Returns inf, script handles |
| Debug output | Overlay RGBA (red=ref, green=live) | Live passthrough | Live passthrough |
| Best for | Precise multi-person matching | Multi-slot Hu distance | Single-person quick prototype |

---

## Source Files

| File | Role |
|---|---|
| `assets/code/glsl/crop_to_edge.frag` | Optional GLSL bounding-box crop |
| `assets/code/python/pose_preprocess.py` | Mask cleanup, person separation, normalization, atlas packing |
| `assets/code/python/pose_match_iou.py` | IoU + optimal assignment matcher |
| `assets/code/python/pose_match.py` | Manual Hu moments matcher (multi-slot) |
| `assets/code/python/pose_matchShapes.py` | cv2.matchShapes() matcher |
