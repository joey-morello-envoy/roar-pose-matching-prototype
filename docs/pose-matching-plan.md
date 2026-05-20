# Pose Matching Pipeline — Implementation Plan

## Goal

Build a Python script that takes two silhouette images (a **target/reference pose** and a **candidate pose**) and returns a per-person match score that is invariant to:
- Person size (adult vs child)
- Absolute position in frame
- Whether people are touching or separated
- Left/right facing direction

No skeleton/keypoint detection. Pure silhouette shape comparison.

---

## Tech Stack

- Python 3.10+
- `opencv-python` (image ops, morphology, connected components, watershed)
- `numpy`
- `scipy` (Hungarian assignment via `scipy.optimize.linear_sum_assignment`)

Install: `pip install opencv-python numpy scipy`

---

## Project Structure

```
pose_match/
├── main.py              # entry point, CLI
├── preprocess.py        # Phase B: clean silhouettes
├── separate.py          # Phase C: split into per-person masks
├── normalize.py         # Phase D: canonical resize
├── compare.py           # Phase E: IoU + Hungarian matching
├── debug.py             # writes intermediate images
└── debug_out/           # output folder for intermediate images
```

---

## Phase A — CLI and I/O

**Input:** two image paths (target, candidate). Images are white silhouettes on black background, any resolution.

**Output:**
- Printed match score (0.0–1.0) per person pair
- Printed overall match score (mean of pairs)
- A boolean `match` based on a configurable threshold (default 0.75)
- Debug images written to `debug_out/` (one subfolder per input)

**CLI:**
```bash
python main.py --target path/to/target.png --candidate path/to/candidate.png [--threshold 0.75] [--debug]
```

**Acceptance:** Running the command on the two original example images produces a score and exits cleanly.

---

## Phase B — Per-Image Preprocessing

Goal: turn raw silhouette images into clean binary masks where each person is a solid white blob.

Apply to **both** target and candidate independently:

1. **Load** with `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`.
2. **Threshold** to binary using Otsu: `cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`. Result: 0 = background, 255 = person.
3. **Morphological close** to fill small holes (eyes, gaps between fingers/legs):
   - Kernel: elliptical, size ~ 1.5% of image diagonal, e.g. `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))` for a ~1500px image.
   - Operation: `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)`.
4. **Smooth edges**: apply `cv2.GaussianBlur(mask, (9, 9), 0)` then re-threshold at 127.
5. **Remove tiny noise components**: connected components, drop any with area < 0.5% of image area.

**Debug output:** save the binary mask after this phase as `debug_out/<name>/01_preprocessed.png`.

**Acceptance:** preprocessed masks visually look like clean, solid white blobs with smooth edges and no holes inside the body.

---

## Phase C — Person Separation

Goal: turn a binary mask (which may contain N people, possibly touching) into a list of N separate single-person masks.

1. **Initial split** with `cv2.connectedComponentsWithStats`. Each component is a candidate person.
2. **Detect "fused" components** — components that likely contain more than one person. Heuristics (use all three, flag if any triggers):
   - **Aspect ratio**: bounding box `width / height > 1.3` (a single standing person is typically taller than wide).
   - **Solidity drop**: `area / convex_hull_area < 0.7` is suspicious for a single human shape.
   - **Vertical pinch**: scan columns of the component, compute the vertical extent (top-to-bottom white pixel count) per column. If there is a local minimum in the middle 60% of the bbox where extent drops to less than 70% of the surrounding maxima, it's a pinch point.
3. **Split fused components** using distance transform + watershed:
   - Compute distance transform: `cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)`.
   - Find peaks (markers): threshold the distance transform at e.g. 0.5 × max, label connected regions as seeds.
   - If exactly 2 seeds emerge, run `cv2.watershed` with these seeds to split the component.
   - If only 1 seed emerges despite a pinch trigger, fall back to a **vertical cut** at the pinch column.
4. **Final list:** all non-fused components plus split products. Filter again by minimum area.

**Debug output:**
- `02_components_raw.png` — initial connected components, color-coded.
- `03_components_split.png` — after fusion-splitting, color-coded.

**Acceptance:**
- Target example (two adults back-to-back) → 2 separate person masks.
- Candidate example (adult + separate child) → 2 separate person masks.

---

## Phase D — Per-Person Normalization

Goal: take each individual person mask and turn it into a canonical 256×256 image so that adults and children become directly comparable.

For each person mask from Phase C:

1. **Crop to tight bounding box** of the white region.
2. **Compute aspect-preserving resize target**: fit inside 256×256 leaving the longer dimension at 256.
   - If cropped is 400×800 → resize to 128×256.
   - If cropped is 600×400 → resize to 256×170.
3. **Resize** with `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
4. **Pad to 256×256** with black, centered. Use `cv2.copyMakeBorder` with `cv2.BORDER_CONSTANT, value=0`.
5. **Optional centroid alignment**: compute centroid of the white pixels, shift so the centroid sits at (128, 128). This is cheap insurance against off-center silhouettes.

Each person becomes a 256×256 binary image.

**Debug output:** `04_normalized_personN.png` for each extracted person, both target and candidate.

**Acceptance:** all normalized masks are 256×256, single-channel, values in {0, 255}. Adult and child T-poses should look visually nearly identical when laid side by side.

---

## Phase E — Comparison

Goal: compute a match score for each candidate-person vs each target-person and produce a final score.

1. **Pairwise IoU**: for every (target_i, candidate_j) pair compute
   ```
   iou = intersection / union
   ```
   on the normalized masks. Both binary, both 256×256.
2. **Mirror invariance**: also compute IoU with `cv2.flip(candidate_j, 1)` (horizontal flip) and take `max(iou, iou_mirrored)`. This way a left-facing T-pose still matches a right-facing T-pose.
3. **Assignment**: if there are M target people and N candidate people:
   - Build an M×N cost matrix where `cost[i][j] = 1 - iou[i][j]`.
   - Use `scipy.optimize.linear_sum_assignment(cost)` to find the optimal one-to-one matching.
   - If `M != N`, pad the cost matrix with high-cost dummy rows/columns so unmatched people incur a penalty.
4. **Aggregate**: overall score = mean of the IoU scores of the matched pairs. Unmatched (dummy) pairs contribute 0.
5. **Threshold**: `match = overall_score >= threshold` (default 0.75; tune empirically).

**Debug output:**
- `05_pair_overlay_TtoC.png` for each matched pair: target mask in red channel, candidate mask in green channel, overlap in yellow.
- A printed table of all IoU scores plus the chosen assignment.

**Acceptance:**
- Two identical T-pose silhouettes → IoU > 0.9.
- T-pose vs. arms-down silhouette → IoU < 0.6.
- Target example vs candidate example (both contain T-poses at different scales) → overall score above threshold.

---

## Phase F — Debugging and Tuning Knobs

Expose these as CLI flags or constants at the top of `main.py`:

- `CANONICAL_SIZE` (default 256)
- `MIN_COMPONENT_AREA_PCT` (default 0.5)
- `CLOSE_KERNEL_PCT` (default 1.5) — kernel size as percent of image diagonal
- `MATCH_THRESHOLD` (default 0.75)
- `MIRROR_INVARIANT` (default True)

Always write debug images when `--debug` is passed. The folder should make it possible to eyeball every stage of the pipeline.

---

## Test Plan

Create a `tests/` folder with at least these cases:

1. **Identity test**: feed the same image as both target and candidate. Overall score should be ≥ 0.95.
2. **Mirror test**: feed an image and its horizontal flip. Should still score ≥ 0.95 because of mirror-invariance.
3. **Scale test**: feed an image and a half-size version of itself. Should score ≥ 0.9 (normalization handles this).
4. **Different pose test**: T-pose vs. arms-down. Should score below threshold.
5. **Original example pair**: the two example images from the conversation. Both contain T-poses; should score above threshold.
6. **Touching-vs-separate test**: same two people, one image with them touching and another with them apart. Per-person scores should remain high.

---

## Stretch Goals (do not implement in v1, but design so they can be added)

- **Radial signature comparison** as a secondary score: from the centroid of each normalized mask, cast 72 rays and record the distance to the silhouette edge. Compare with normalized cross-correlation. Adds robustness when IoU is borderline.
- **Rotation invariance**: try a few candidate rotations (±15°, ±30°) and take the best. Only enable if real input data shows tilted poses.
- **Pose library**: store many normalized reference masks per named pose ("t_pose", "arms_up", etc.) and classify candidates by nearest-neighbor.

---

## Deliverables

1. Runnable `python main.py --target ... --candidate ...` script.
2. `--debug` flag that produces every intermediate image.
3. A short `README.md` documenting how to run it, what each phase does, and what to tune if the scores look wrong.
4. Test cases from the Test Plan section, runnable with `pytest`.
