# Pose Preprocessing and IoU Pose Matching: Developer Reference

This document explains, in detail, what the two TouchDesigner Script TOP callback scripts do:

- `assets/code/python/pose_preprocess.py`
- `assets/code/python/pose_match_iou.py`

Together, these scripts implement a silhouette-based pose matching pipeline:

1. `pose_preprocess.py` takes a raw silhouette image, cleans it up, detects one or more people, normalizes each person into a fixed-size square mask, and packs those masks into an atlas image.
2. `pose_match_iou.py` takes two atlas images, compares every target person against every candidate person using Intersection over Union, optionally handles mirrored poses, chooses the best one-to-one matching between people, and stores a final match score plus diagnostics.

The system is designed for TouchDesigner. Both files are Script TOP callback modules, meaning TouchDesigner calls functions like `onSetupParameters`, `onCook`, and `onGetCookLevel` at specific moments in the operator lifecycle.

---

## Big-picture pipeline

At a high level, the intended operator graph looks like this:

```text
candidate silhouette TOP ──> pose_preprocess.py ──┐
                                                  ├──> pose_match_iou.py ──> score + debug image
reference silhouette TOP ──> pose_preprocess.py ──┘
```

The preprocessing stage converts arbitrary silhouette images into a consistent format. The matching stage can then compare those consistent masks.

The core assumption is:

- White pixels represent the subject or body silhouette.
- Black pixels represent background.
- The image can contain one person or multiple people.
- The downstream matcher should compare people slot-by-slot, but not necessarily in the same order; it solves the best assignment.

---

# Part 1: `pose_preprocess.py`

## Purpose

`pose_preprocess.py` prepares raw silhouette input for pose comparison.

Its responsibilities are:

1. Read a silhouette TOP input.
2. Convert it into a binary mask.
3. Clean the mask with morphological and smoothing operations.
4. Remove tiny noise components.
5. Detect connected foreground blobs.
6. Optionally split fused people into separate masks.
7. Normalize each detected person to a fixed square size.
8. Pack all normalized person masks into a grid atlas.
9. Store metadata that downstream operators can fetch.

The output is not a single person mask. It is an atlas: a larger RGBA image divided into equal square slots. Each populated slot contains one normalized person silhouette.

For example, if `Maxpeople` is `4` and `Canonsize` is `256`, the script creates a 2-by-2 atlas with dimensions `512 x 512`. Each cell is `256 x 256`.

---

## TouchDesigner callback structure

The file defines three TouchDesigner callbacks:

- `onSetupParameters(scriptOp)`
- `onPulse(par)`
- `onCook(scriptOp)`
- `onGetCookLevel(scriptOp)`

### `onSetupParameters`

This function creates the custom parameter page and parameters for the Script TOP. TouchDesigner calls it when you press the operator's `Setup Parameters` button.

The page is named `Pose Preprocess`.

The parameters are:

### `Maxpeople`

Maximum number of detected people to keep and pack into the atlas.

- Default: `4`
- Minimum: `1`
- Used to determine the atlas grid size.

If more people are detected than `Maxpeople`, the extras are discarded after sorting.

### `Canonsize`

The square size, in pixels, for each normalized person slot.

- Default: `256`
- Minimum: `16`

Every detected person is resized and padded into a `Canonsize x Canonsize` mask.

### `Threshold`

The binary threshold used to convert the input image to a foreground/background mask.

- Default: `0.5`
- Range: `0.0` to `1.0`

TouchDesigner TOP pixel data usually arrives as floats in the range `0.0` to `1.0`. The script converts this threshold to `0..255` by multiplying by `255`.

Pixels above the threshold become foreground. Pixels below the threshold become background.

### `Closepct`

Controls the size of the morphological closing kernel as a percentage of the image diagonal.

- Default: `1.5`
- Range: `0.0` to `5.0`

Morphological closing helps fill small gaps and connect nearby foreground areas. The kernel is elliptical.

### `Minareapct`

Minimum connected-component area as a percentage of the full frame.

- Default: `0.5`
- Range: `0.0` to `5.0`

Detected blobs smaller than this percentage of the frame area are removed as noise.

### `Splitfused`

Toggle that enables or disables fused-person splitting.

If enabled, the script attempts to detect whether a connected component likely contains more than one person and then split it.

### `Aspectfusedmin`

Width-to-height aspect ratio threshold used as one fused-blob heuristic.

- Default: `1.3`

Despite the parameter name ending in `min`, the code treats this as an aspect ratio maximum for a normal person-shaped component. If a blob is wider than this threshold, it may be considered fused.

### `Solidityfusedmax`

Solidity threshold used as another fused-blob heuristic.

- Default: `0.7`

Solidity is approximately:

```text
component area / convex hull area
```

A low-solidity shape has concavities or gaps. Two touching people often form a shape with a more concave outline than a single person, so low solidity can indicate a fused blob.

### `Centroidalign`

Whether the normalized person mask should be shifted so its centroid lands at the center of the canonical square.

- Default: `True`

This can make comparisons more stable when the subject's body is off-center inside their bounding box.

### `Sortby`

Determines the order in which detected people are assigned to atlas slots.

Options:

- `xcent`: sort left-to-right by source-frame centroid x-coordinate.
- `area`: sort largest-first by component area.

The matcher later solves assignments, so exact order is not fatal, but stable ordering is still useful for debugging and deterministic output.

---

## Atlas geometry helpers

### `_grid_dims(maxpeople)`

This function computes the number of columns and rows in the atlas.

The goal is to create a compact, roughly square grid that has enough slots for `maxpeople`.

```text
cols = ceil(sqrt(maxpeople))
rows = ceil(maxpeople / cols)
```

Examples:

```text
Maxpeople = 1 -> cols = 1, rows = 1
Maxpeople = 2 -> cols = 2, rows = 1
Maxpeople = 3 -> cols = 2, rows = 2
Maxpeople = 4 -> cols = 2, rows = 2
Maxpeople = 5 -> cols = 3, rows = 2
```

The function clamps both dimensions to at least `1`.

### `_empty_atlas(maxpeople, canon)`

This function creates a blank RGBA atlas.

Steps:

1. Calls `_grid_dims(maxpeople)` to determine the atlas layout.
2. Allocates a NumPy array with shape:

```text
(rows * canon, cols * canon, 4)
```

3. Initializes all RGB values to `0`, meaning black.
4. Sets the alpha channel to `255`, meaning fully opaque.

The returned image is a black, opaque RGBA image with enough square slots to hold all possible people.

---

## Phase B: mask preprocessing

The function `_preprocess_mask` performs image cleanup.

Signature:

```python
def _preprocess_mask(img_u8: np.ndarray, thresh_u8: int,
                     close_pct: float, min_area_pct: float) -> np.ndarray:
```

Inputs:

- `img_u8`: a single-channel `uint8` image in the range `0..255`.
- `thresh_u8`: binary threshold in the range `0..255`.
- `close_pct`: morphological close kernel size as a percentage of diagonal length.
- `min_area_pct`: minimum component area as a percentage of frame area.

Output:

- A cleaned binary `uint8` mask where foreground is `255` and background is `0`.

### Step 1: binary threshold

The function starts with:

```python
_, m = cv2.threshold(img_u8, thresh_u8, 255, cv2.THRESH_BINARY)
```

Every pixel greater than `thresh_u8` becomes `255`. Every other pixel becomes `0`.

This turns the grayscale source into a strict foreground/background mask.

### Step 2: compute closing kernel size

The script reads the mask dimensions:

```python
h, w = m.shape[:2]
diag = math.hypot(w, h)
```

`math.hypot(w, h)` computes the diagonal length of the image:

```text
sqrt(width^2 + height^2)
```

The close kernel size is:

```python
k = int(round(diag * (close_pct / 100.0)))
```

This means the kernel scales with image resolution. A `Closepct` of `1.5` produces a larger kernel for a 4K image than for a 720p image.

The kernel size is then forced to be:

- at least `3`
- odd

OpenCV morphology kernels generally behave best with odd dimensions because they have a clear center pixel.

### Step 3: morphological closing

The script creates an elliptical kernel:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
```

Then applies closing:

```python
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
```

Morphological closing is dilation followed by erosion.

It tends to:

- fill small holes inside silhouettes
- bridge small gaps between nearby white regions
- smooth tiny breaks in body contours

This is useful when the input silhouette has small missing regions due to segmentation noise.

### Step 4: Gaussian blur and re-threshold

The script blurs the mask:

```python
m = cv2.GaussianBlur(m, (9, 9), 0)
```

Then thresholds again at `127`:

```python
_, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
```

The blur softens jagged edges and small artifacts. Re-thresholding converts the softened image back into a binary mask.

This sequence can smooth rough silhouette edges while preserving binary output.

### Step 5: connected components and area filtering

The script identifies connected foreground regions:

```python
num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
```

With `connectivity=8`, pixels are considered connected if they touch horizontally, vertically, or diagonally.

The returned values are:

- `num`: number of labels, including background label `0`.
- `labels`: image where each pixel stores its component label.
- `stats`: per-component statistics such as bounding box and area.
- `_`: centroids, ignored here.

The minimum area is calculated from the full image area:

```python
frame_area = float(h * w)
min_area = frame_area * (min_area_pct / 100.0)
```

The script creates a blank output mask and copies only sufficiently large components into it:

```python
keep = np.zeros_like(m)
for i in range(1, num):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        keep[labels == i] = 255
```

The loop starts at `1` because component `0` is the background.

This removes small specks, shadows, segmentation fragments, and other noise.

---

## Phase C: fused-person detection and splitting

Sometimes multiple people touch or overlap in the source silhouette and are detected as a single connected component. The script has optional logic to detect and split these fused blobs.

This behavior is controlled by `Splitfused`.

The relevant functions are:

- `_vertical_pinch_col`
- `_is_fused`
- `_split_fused`

### `_vertical_pinch_col(comp_local)`

This helper looks for a vertical pinch point in a component.

Input:

- `comp_local`: a binary mask cropped to one connected component's bounding box.

Output:

- A local x-coordinate column index if a pinch is found.
- `None` otherwise.

The intuition is that two side-by-side people touching at arms or shoulders may create a shape that is wide on the left and right but narrow somewhere between them.

The function works as follows:

1. If the component is too narrow (`w < 8`), return `None`.
2. Count the number of foreground pixels in each column:

```python
col_count = (comp_local > 0).sum(axis=0).astype(np.int32)
```

3. Inspect only the middle 60% of the width:

```text
lo = 20% of width
hi = 80% of width
```

4. Compute the largest column count on the left and right flanks.
5. Find the minimum column count in the middle region.
6. If the minimum middle column has less than `70%` of the flank maximum, treat it as a pinch.

In practical terms, a pinch is a vertical column where the silhouette becomes much thinner than the surrounding left/right body masses.

### `_is_fused(comp_local, area, aspect_max, solidity_max)`

This function decides whether a connected component might contain multiple people.

It returns:

```python
(flagged, pinch)
```

Where:

- `flagged` is `True` if the component appears fused.
- `pinch` is either a local x-column from `_vertical_pinch_col` or `None`.

The script uses three heuristics.

### Heuristic 1: wide aspect ratio

```python
if (w / float(h)) > aspect_max:
    flagged = True
```

If a component is unusually wide compared to its height, it may contain multiple side-by-side people.

For example, two people standing next to each other can create a much wider silhouette than one person.

### Heuristic 2: low solidity

The script finds the outer contour:

```python
cnts, _ = cv2.findContours(comp_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Then computes its convex hull:

```python
hull = cv2.convexHull(c)
hull_area = float(cv2.contourArea(hull))
```

Solidity is:

```text
area / hull_area
```

If solidity is below `solidity_max`, the blob is flagged:

```python
if hull_area > 0 and (area / hull_area) < solidity_max:
    flagged = True
```

A single compact person silhouette often has higher solidity. Two touching people may produce concave gaps between bodies, lowering solidity.

### Heuristic 3: vertical pinch

The function calls `_vertical_pinch_col`.

If a pinch exists, the component is flagged as fused.

This is also important because `_split_fused` can use the pinch column as a fallback split location.

### `_split_fused(comp_local, pinch_col)`

This function attempts to split a fused component into multiple component-local masks.

It returns a list of masks, all with the same shape as `comp_local`.

If splitting fails, it returns a list containing the original component:

```python
[comp_local]
```

The primary method is watershed segmentation using the distance transform.

### Step 1: distance transform

```python
dist = cv2.distanceTransform(comp_local, cv2.DIST_L2, 5)
```

The distance transform assigns each foreground pixel a value equal to its distance from the nearest background pixel.

Inside a body-shaped blob:

- boundary pixels have low values
- central thick regions have high values

For two touching people, each person's torso may produce its own distance peak.

### Step 2: find high-distance peaks

```python
dmax = float(dist.max())
peaks = (dist > 0.5 * dmax).astype(np.uint8) * 255
```

Pixels whose distance value is greater than half the maximum are treated as seed regions.

Then connected components are found in the peak mask:

```python
num_seeds, seed_labels = cv2.connectedComponents(peaks)
num_seeds -= 1
```

The subtraction removes the background label.

If there are at least two seed regions, watershed can attempt to grow those seeds into separated person masks.

### Step 3: prepare watershed markers

OpenCV watershed requires a marker image where different positive labels represent different seed regions.

The script converts seed labels:

```python
markers = seed_labels.astype(np.int32).copy()
markers[markers > 0] += 1
markers[comp_local == 0] = 1
```

After this:

- Background is label `1`.
- Seed regions become labels `2..N+1`.
- Unknown foreground interior remains `0`.

### Step 4: run watershed

Watershed needs a 3-channel image, so the script creates one from the inverted normalized distance transform:

```python
dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
inv_dist_bgr = cv2.cvtColor(255 - dist_u8, cv2.COLOR_GRAY2BGR)
cv2.watershed(inv_dist_bgr, markers)
```

The distance transform is inverted so that the separation saddle between blobs acts like a ridge in the watershed landscape.

After `cv2.watershed`, the marker image is modified in place. Regions assigned to each seed keep their seed labels.

### Step 5: convert watershed regions into masks

The script iterates over seed labels:

```python
for sid in range(2, num_seeds + 2):
    sub = np.zeros_like(comp_local)
    sub[(markers == sid) & (comp_local > 0)] = 255
```

Each watershed region becomes a new binary sub-mask. Empty sub-masks are discarded.

If watershed creates at least two sub-masks, those are returned.

### Step 6: fallback vertical cut

If watershed cannot split the component and a pinch column exists, the script performs a simple left/right cut:

```python
left = comp_local.copy()
right = comp_local.copy()
left[:, pinch_col:] = 0
right[:, :pinch_col] = 0
```

If both halves contain foreground pixels, it returns:

```python
[left, right]
```

This fallback is less sophisticated than watershed, but it gives the system a deterministic split when the pinch heuristic strongly suggests two side-by-side people.

---

## Phase D: per-person normalization

The function `_normalize_mask` converts an arbitrary source-frame person mask into a canonical square mask.

Signature:

```python
def _normalize_mask(full_mask: np.ndarray, canon: int, centroid_align: bool) -> np.ndarray:
```

Inputs:

- `full_mask`: a binary mask the same size as the source frame, containing one person.
- `canon`: target square size.
- `centroid_align`: whether to center the silhouette's centroid.

Output:

- A binary `canon x canon` mask.

### Step 1: find foreground pixels

```python
ys, xs = np.where(full_mask > 0)
```

If there are no foreground pixels, the function returns an all-black canonical mask.

### Step 2: tight crop

The bounding box of the foreground pixels is computed:

```python
x0, x1 = int(xs.min()), int(xs.max()) + 1
y0, y1 = int(ys.min()), int(ys.max()) + 1
cropped = full_mask[y0:y1, x0:x1]
```

This removes empty background around the detected person.

### Step 3: aspect-preserving resize

The function computes a scale factor that fits the crop inside a `canon x canon` square without distortion:

```python
scale = min(canon / float(cw), canon / float(ch))
```

This uses the smaller of the width scale and height scale.

The new dimensions are:

```python
new_w = max(1, int(round(cw * scale)))
new_h = max(1, int(round(ch * scale)))
```

The crop is resized:

```python
resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
```

`INTER_AREA` is generally appropriate for downsampling binary-ish images because it reduces aliasing.

The resized image is thresholded again:

```python
_, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
```

This ensures the output remains strictly binary.

### Step 4: letterbox padding

The resized person is centered in a square canvas:

```python
top = (canon - new_h) // 2
bottom = canon - new_h - top
left = (canon - new_w) // 2
right = canon - new_w - left
```

Then padded:

```python
padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=0)
```

This preserves the person's aspect ratio while producing a fixed-size square image.

### Step 5: optional centroid alignment

If `Centroidalign` is enabled, the script computes image moments:

```python
mom = cv2.moments(padded, binaryImage=True)
```

For a binary mask:

- `m00` is the total foreground mass.
- `m10 / m00` is the x-coordinate of the centroid.
- `m01 / m00` is the y-coordinate of the centroid.

The script computes the translation needed to move the centroid to the center of the canonical square:

```python
dx = int(round(canon / 2.0 - cx))
dy = int(round(canon / 2.0 - cy))
```

Then applies an affine translation:

```python
M = np.float32([[1, 0, dx], [0, 1, dy]])
padded = cv2.warpAffine(padded, M, (canon, canon),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
```

`INTER_NEAREST` preserves hard binary edges during translation.

After translation, the mask is thresholded again to guarantee binary output.

This makes the normalized representation less sensitive to asymmetries in the tight crop. For example, if an arm is stretched far to one side, the bounding-box center and body centroid may differ. Centroid alignment makes comparison more about the body mass position than the raw crop bounds.

---

## Main preprocessing cook: `onCook`

`onCook` is the function TouchDesigner calls when the Script TOP needs to produce output.

### Step 1: read parameters

The script reads and clamps core parameters:

```python
maxpeople = max(1, int(scriptOp.par.Maxpeople.eval()))
canon = max(16, int(scriptOp.par.Canonsize.eval()))
cols, rows = _grid_dims(maxpeople)
```

It always ensures:

- at least one person slot
- canonical size at least `16`

### Step 2: handle missing input

If no TOP input is connected:

1. Create an empty atlas.
2. Output the empty atlas.
3. Store empty metadata.
4. Return.

Stored metadata in this case:

```text
num_people      = 0
slot_bboxes     = []
slot_centroids  = []
slot_areas      = []
atlas_grid      = [cols, rows, canon]
```

This is important because downstream scripts can still fetch valid metadata even when the input is missing.

### Step 3: read source TOP image

```python
src = scriptOp.inputs[0].numpyArray(delayed=False)
src_u8 = (src[:, :, 0] * 255).astype(np.uint8)
```

TouchDesigner provides TOP data as a NumPy array. The script uses only the red channel, assuming the silhouette is grayscale or at least encoded consistently in channel `0`.

The red channel is converted from float `0..1` to `uint8` `0..255`.

### Step 4: read processing parameters

The script reads:

- binary threshold
- close kernel percentage
- minimum component area percentage
- fused splitting toggle
- fused aspect heuristic
- fused solidity heuristic
- centroid alignment toggle
- sort mode

The binary threshold is explicitly clamped to `0..1` before converting to `0..255`.

### Step 5: preprocess the mask

```python
mask = _preprocess_mask(src_u8, thresh_u8, close_pct, min_area_pct)
```

At this point, `mask` is a cleaned binary image.

### Step 6: find connected components

```python
num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
```

Each connected foreground blob becomes a possible person or fused group.

### Step 7: iterate over components

For each foreground component label `i`, the script extracts:

- bounding box x
- bounding box y
- bounding box width
- bounding box height
- component area

It skips invalid empty components.

Then it creates a component-local mask:

```python
comp_local = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
```

This mask is cropped to the component's bounding box.

### Step 8: optionally split fused components

The default sub-mask list is just the original component:

```python
sub_masks_local = [comp_local]
```

If `Splitfused` is enabled:

1. `_is_fused` checks whether the component appears to contain multiple people.
2. If fused, `_split_fused` attempts to split it.
3. The resulting sub-masks replace the original list.

Each sub-mask is then treated as its own person.

### Step 9: convert local sub-masks back to full-frame masks

For every sub-mask:

1. Find its foreground pixels.
2. Create a full-frame blank mask.
3. Place the local sub-mask back into the source-frame location:

```python
full = np.zeros_like(mask)
full[y:y + h, x:x + w] = sub_local
```

The full-frame mask is needed by `_normalize_mask`, which tight-crops it later.

### Step 10: compute metadata for each person

The script computes:

- `sub_area`: number of foreground pixels in the local sub-mask
- `scx`: source-frame centroid x-coordinate
- `scy`: source-frame centroid y-coordinate
- source-frame bounding box

The centroid is calculated as the mean of foreground pixel coordinates, offset by the component's original source-frame location:

```python
scx = float(xs2.mean()) + x
scy = float(ys2.mean()) + y
```

The person record is appended as:

```python
{
    'mask': full,
    'bbox': [sx0, sy0, sx1 - sx0, sy1 - sy0],
    'area': sub_area,
    'centroid': [scx, scy],
}
```

### Step 11: sort detected people

If `Sortby` is `area`, largest components come first:

```python
persons.sort(key=lambda p: -p['area'])
```

Otherwise, people are sorted left-to-right:

```python
persons.sort(key=lambda p: p['centroid'][0])
```

Sorting controls which detected person goes into which atlas slot.

### Step 12: enforce `Maxpeople`

```python
persons = persons[:maxpeople]
```

Only the first `Maxpeople` after sorting are kept.

### Step 13: build the output atlas

The script starts with an empty atlas:

```python
atlas = _empty_atlas(maxpeople, canon)
```

Then for each person:

1. Normalize the person mask:

```python
norm = _normalize_mask(person['mask'], canon, centroid_align)
```

2. Compute the atlas slot row and column:

```python
row = idx // cols
col = idx % cols
```

3. Compute the top-left atlas pixel for that slot:

```python
y0a = row * canon
x0a = col * canon
```

4. Copy the normalized mask into RGB channels:

```python
atlas[y0a:y0a + canon, x0a:x0a + canon, 0] = norm
atlas[y0a:y0a + canon, x0a:x0a + canon, 1] = norm
atlas[y0a:y0a + canon, x0a:x0a + canon, 2] = norm
atlas[y0a:y0a + canon, x0a:x0a + canon, 3] = 255
```

Each populated slot is a grayscale silhouette stored identically in red, green, and blue.

### Step 14: output and store metadata

The atlas is sent to the Script TOP output:

```python
scriptOp.copyNumpyArray(atlas)
```

The script stores metadata:

```text
num_people      = number of populated person slots
slot_bboxes     = source-frame bounding boxes
slot_centroids  = source-frame centroids
slot_areas      = source-frame foreground areas
atlas_grid      = [cols, rows, canon]
```

Downstream operators can fetch these values from the Script TOP. `pose_match_iou.py` depends especially on:

- `num_people`
- `atlas_grid`

---

## Preprocessor output contract

`pose_preprocess.py` outputs:

1. An RGBA atlas image.
2. Metadata stored on the Script TOP.

The atlas layout is:

```text
slot 0  slot 1  slot 2 ...
slot N ...
```

Each slot is:

```text
Canonsize x Canonsize
```

The slot position is:

```text
row = slot_index // cols
col = slot_index % cols
x0  = col * Canonsize
y0  = row * Canonsize
```

Empty slots remain black.

---

# Part 2: `pose_match_iou.py`

## Purpose

`pose_match_iou.py` compares a candidate silhouette atlas against a target/reference silhouette atlas.

Its responsibilities are:

1. Read two TOP inputs:
   - input `0`: candidate/live atlas
   - input `1`: target/reference atlas
2. Extract populated person slots from each atlas.
3. Compute pairwise Intersection over Union between every target and every candidate.
4. Optionally test mirrored versions of candidate masks.
5. Solve the best one-to-one target/candidate assignment.
6. Compute an overall match score.
7. Compare the score to a threshold.
8. Store diagnostics.
9. Output either a debug overlay, the reference input, or the candidate input.

The final score is stored as `match_score`, with range `0..1`.

Higher is better.

---

## Intersection over Union concept

Intersection over Union, usually abbreviated IoU, measures how much two masks overlap.

For two binary masks `A` and `B`:

```text
intersection = pixels where A and B are both foreground
union        = pixels where A or B is foreground
IoU          = intersection / union
```

Interpretation:

```text
IoU = 1.0 -> masks overlap perfectly
IoU = 0.0 -> masks do not overlap at all
```

For pose silhouettes, IoU is a direct shape-overlap score. If the candidate person and target person have very similar normalized silhouettes, their IoU will be high.

---

## TouchDesigner parameters

### `Threshold`

The required overall score for a match.

- Default: `0.75`
- Range: `0.0` to `1.0`

After scoring, the script checks:

```python
passed = overall >= threshold
```

### `Mirrorinvariant`

If enabled, each candidate mask is compared both normally and horizontally flipped.

This allows a left-facing and right-facing version of the same pose to match.

Example:

- Target person raises left arm.
- Candidate person raises right arm.
- Without mirror invariance, IoU may be low.
- With mirror invariance, the candidate is flipped before comparison, so the silhouettes can match.

### `Debugpair`

Selects which assigned target/candidate pair to visualize in the debug overlay.

If there are multiple matched people, `Debugpair = 0` shows the first real pair, `Debugpair = 1` shows the second, and so on. Values beyond the available pair count are clamped to the last pair.

### `Debugprint`

If enabled, the script prints the IoU matrix, assignment, and overall match state every cook.

If disabled, the script still prints when the high-level debug signature changes. The signature is:

```text
(passed, number of target masks, number of candidate masks)
```

This avoids printing constantly when nothing important changes.

### `Passthroughref`

If enabled, the Script TOP outputs the reference input directly instead of the debug overlay.

If disabled and there are real matched pairs, the Script TOP outputs a colored overlay for the selected pair.

---

## Module-level debug state

The script defines:

```python
_last_debug_signature: tuple[bool, int, int] | None = None
```

This remembers the last printed high-level state:

- whether the score passed threshold
- number of target people
- number of candidate people

It is used to avoid repeated debug printing unless the state changes.

---

## Reading atlas metadata

### `_resolve_grid(top_input)`

This function determines the atlas grid layout for an input TOP.

It returns:

```python
(cols, rows, canon)
```

Where:

- `cols`: atlas column count
- `rows`: atlas row count
- `canon`: square slot size

The function first tries to fetch metadata stored by `pose_preprocess.py`:

```python
grid = top_input.fetch('atlas_grid', None)
```

If `atlas_grid` exists and contains three values, it returns them as integers.

If metadata is missing, it falls back to single-tile behavior:

```python
return 1, 1, max(h, w)
```

This fallback makes the matcher compatible with a plain binary mask input that did not come from `pose_preprocess.py`.

If `top_input` itself is `None`, it returns:

```python
(1, 1, 0)
```

### `_resolve_num_people(top_input, default)`

This function determines how many atlas slots are populated.

It tries to fetch:

```python
top_input.fetch('num_people', None)
```

If the metadata exists, it returns that number.

If it does not exist, it returns the supplied `default`.

This fallback allows non-atlas inputs to be treated as one-person masks.

---

## Extracting binary masks from atlas slots

### `_tile_bin(img_u8, cols, canon, slot, thresh_u8)`

This function extracts one slot from an atlas and thresholds it into a binary mask.

Inputs:

- `img_u8`: single-channel atlas image in `0..255`.
- `cols`: number of atlas columns.
- `canon`: slot size.
- `slot`: slot index to extract.
- `thresh_u8`: threshold.

If `canon <= 0`, the function treats the whole image as one mask:

```python
_, t = cv2.threshold(img_u8, thresh_u8, 255, cv2.THRESH_BINARY)
return t
```

Otherwise, it computes the atlas tile position:

```python
row = slot // cols
col = slot % cols
y0 = row * canon
x0 = col * canon
```

Then it safely crops the tile:

```python
y1 = min(y0 + canon, h)
x1 = min(x0 + canon, w)
tile = img_u8[y0:y1, x0:x1]
```

Finally, it thresholds the tile and returns a binary mask.

### `_extract_masks(img_u8, top_input, thresh_u8)`

This function extracts all populated person masks from an input atlas.

Steps:

1. Read atlas grid:

```python
cols, rows, canon = _resolve_grid(top_input)
```

2. Compute maximum possible slot count:

```python
max_slot = max(1, cols * rows)
```

3. Resolve the number of actual people:

```python
num_people = max(0, min(_resolve_num_people(top_input, 1), max_slot))
```

This clamps the fetched person count to the valid range.

4. Extract each populated slot:

```python
return [
    _tile_bin(img_u8, max(cols, 1), canon, slot, thresh_u8)
    for slot in range(num_people)
]
```

The result is a list of binary masks.

---

## Shape compatibility helper

### `_same_shape(candidate, target_shape)`

IoU requires both masks to have the same dimensions.

This function checks whether a candidate mask already has the desired shape. If so, it returns the candidate unchanged.

If not, it resizes using nearest-neighbor interpolation:

```python
return cv2.resize(candidate, (target_shape[1], target_shape[0]),
                 interpolation=cv2.INTER_NEAREST)
```

Nearest-neighbor interpolation is used because these are binary masks. It avoids introducing gray antialiasing values.

---

## Computing IoU

### `_iou(mask_a, mask_b)`

This function computes Intersection over Union between two binary masks.

Steps:

1. Resize `mask_b` if its shape differs from `mask_a`.
2. Convert both masks into boolean foreground arrays:

```python
a = mask_a > 0
b = mask_b > 0
```

3. Compute union:

```python
union = np.logical_or(a, b).sum()
```

4. If union is zero, return `0.0`.

This avoids division by zero. A union of zero means both masks are empty. The script treats that as no match.

5. Compute intersection:

```python
intersection = np.logical_and(a, b).sum()
```

6. Return:

```python
float(intersection / union)
```

---

## Pairwise target/candidate scoring

### `_pairwise_iou(target_masks, candidate_masks, mirror_invariant)`

This function compares every target mask to every candidate mask.

If there are `m` target masks and `n` candidate masks, it creates:

```python
iou_matrix = np.zeros((m, n), dtype=np.float64)
```

Rows are targets. Columns are candidates.

It also creates a parallel matrix of booleans:

```python
mirrored = [[False for _ in range(n)] for _ in range(m)]
```

Each `mirrored[i][j]` records whether the best score for target `i` and candidate `j` came from a horizontally flipped candidate.

For every target/candidate pair:

1. Compute direct IoU:

```python
direct = _iou(target, candidate)
best = direct
```

2. If mirror invariance is enabled, flip the candidate horizontally:

```python
flipped = cv2.flip(candidate, 1)
mirrored_iou = _iou(target, flipped)
```

The second argument `1` means horizontal flip.

3. If the mirrored IoU is better than the direct IoU, use it:

```python
if mirrored_iou > direct:
    best = mirrored_iou
    mirrored[i][j] = True
```

4. Store the best score in the matrix:

```python
iou_matrix[i, j] = best
```

The function returns:

```python
(iou_matrix, mirrored)
```

---

## Best one-to-one assignment

### Why assignment is needed

When multiple people are present, the script should not simply compare slot `0` to slot `0`, slot `1` to slot `1`, and so on.

For example:

```text
Targets:    A, B
Candidates: B, A
```

If the people appear in a different order, slot-by-slot comparison would be wrong.

Instead, the matcher computes all pairwise IoUs and finds the assignment that maximizes total IoU:

```text
target 0 -> candidate 1
target 1 -> candidate 0
```

Each target can be matched to at most one candidate, and each candidate can be matched to at most one target.

### `_best_assignment(iou_matrix)`

This function solves the maximum-score one-to-one assignment problem.

Rows are targets. Columns are candidates.

The function returns:

```python
(assignment, scores)
```

Where:

- `assignment` is a list of `(row, col)` pairs.
- `scores` is the IoU score for each assigned pair.

### Step 1: handle empty case

```python
m, n = iou_matrix.shape[:2]
size = max(m, n)
if size == 0:
    return [], []
```

### Step 2: pad to a square matrix

Assignment is easiest when the matrix is square.

If there are different numbers of targets and candidates, the script pads with zeros:

```python
padded = np.zeros((size, size), dtype=np.float64)
padded[:m, :n] = iou_matrix
```

Dummy rows or columns represent unmatched people and contribute `0` score.

This means mismatched counts are naturally penalized when the final score averages all assignment scores.

Example:

```text
2 targets, 1 candidate
```

The padded assignment has size `2`, so one target must match a dummy candidate with score `0`.

### Step 3: recursive dynamic programming search

The script defines a cached recursive function:

```python
@lru_cache(maxsize=None)
def solve(row: int, used_cols: int) -> tuple[float, tuple[int, ...]]:
```

The state is:

- `row`: which target/dummy row is currently being assigned.
- `used_cols`: a bitmask of columns already assigned.

The function returns:

- best total score from this row onward
- tuple of chosen columns

The bitmask works like this:

```text
column 0 used -> bit 0 is 1
column 1 used -> bit 1 is 1
column 2 used -> bit 2 is 1
```

For each available column:

1. Skip it if already used.
2. Recursively solve the remaining rows.
3. Add current pair score to the recursive score.
4. Keep the column that produces the best total.

The cache prevents repeated recomputation of the same `(row, used_cols)` state.

This is effectively a small-scale assignment solver. It is practical here because `Maxpeople` is expected to be small, such as `4` or `8`.

### Step 4: build assignment and scores

After solving:

```python
_, cols = solve(0, 0)
assignment = [(row, col) for row, col in enumerate(cols)]
```

The score for each assignment is:

```python
float(iou_matrix[row, col]) if row < m and col < n else 0.0
```

If the row or column is a dummy, the score is `0.0`.

---

## Combining scores

### `_score_masks(target_masks, candidate_masks, mirror_invariant)`

This function wraps the full scoring process.

Steps:

1. Compute the pairwise IoU matrix and mirror flags:

```python
iou_matrix, mirrored = _pairwise_iou(target_masks, candidate_masks, mirror_invariant)
```

2. Solve the best assignment:

```python
assignment, assignment_scores = _best_assignment(iou_matrix)
```

3. Compute overall score as the mean of assignment scores:

```python
overall = float(np.mean(assignment_scores))
```

If there are no assignment scores, `overall` is `0.0`.

Using the mean is important. It makes extra or missing people reduce the overall score because dummy assignments contribute `0.0`.

Example:

```text
target/candidate score = 0.9
missing second person  = 0.0
overall                = (0.9 + 0.0) / 2 = 0.45
```

4. Build `assignment_mirrored`, a list showing whether each assigned real pair used mirroring.

5. Return a diagnostics dictionary:

```python
{
    'overall': overall,
    'iou_matrix': iou_matrix,
    'assignment': assignment,
    'assignment_scores': assignment_scores,
    'assignment_mirrored': assignment_mirrored,
    'pairwise_mirrored': mirrored,
}
```

---

## Debug overlay generation

### `_overlay_pair(target, candidate, mirrored)`

This function creates a visual comparison image for one matched pair.

Inputs:

- `target`: target binary mask.
- `candidate`: candidate binary mask.
- `mirrored`: whether to horizontally flip the candidate first.

Steps:

1. Flip candidate if needed:

```python
if mirrored:
    candidate = cv2.flip(candidate, 1)
```

2. Resize candidate if needed.
3. Convert target and candidate to boolean masks.
4. Create an RGBA output image.
5. Write:

```text
red channel   = target mask
green channel = candidate mask
alpha channel = 255
```

The resulting colors are useful:

```text
red only      -> target foreground not covered by candidate
green only    -> candidate foreground not in target
yellow        -> overlap, because red + green = yellow
black         -> neither mask
```

This lets a developer visually inspect where the candidate pose matches or differs from the reference.

### `_blank_overlay(size=256)`

This returns a black opaque RGBA image of shape:

```text
size x size x 4
```

It is used when the Script TOP lacks the required inputs.

---

## Debug printing

### `_print_debug(...)`

This function prints:

1. The IoU matrix.
2. The chosen assignment.
3. The final match state.

The IoU matrix is printed with rows as targets and columns as candidates:

```text
IoU matrix target(row) x candidate(col):
  target 0: 0.812  0.104
  target 1: 0.097  0.775
```

The assignment section shows which target is matched to which candidate:

```text
target 0 -> candidate 0: IoU=0.812
target 1 -> candidate 1: IoU=0.775 mirrored
```

If dummy rows or columns are involved, the printout labels them as unmatched dummy entries.

Finally, it prints:

```text
MATCH | overall=0.794
```

or:

```text
NO MATCH | overall=0.422
```

---

## Main matcher cook: `onCook`

### Step 1: handle missing inputs

The matcher requires two inputs:

- candidate/live atlas
- target/reference atlas

If fewer than two inputs are connected, the script:

1. Outputs a tiny blank overlay.
2. Stores a `0.0` match score.
3. Stores the threshold.
4. Stores `match_pass = False`.
5. Stores empty diagnostics.
6. Stores zero target/candidate counts.
7. Returns.

This keeps downstream operators from reading stale values.

### Step 2: read inputs

```python
candidate_in = scriptOp.inputs[0]
target_in = scriptOp.inputs[1]

candidate = candidate_in.numpyArray(delayed=False)
target = target_in.numpyArray(delayed=False)
```

Input `0` is treated as the live/candidate pose. Input `1` is treated as the target/reference pose.

### Step 3: convert to single-channel `uint8`

```python
candidate_u8 = (candidate[:, :, 0] * 255).astype(np.uint8)
target_u8 = (target[:, :, 0] * 255).astype(np.uint8)
```

Only the red channel is used. This matches the preprocessor's output, where all RGB channels contain the same silhouette mask.

The matcher uses a fixed threshold:

```python
thresh_u8 = 127
```

So pixels above the halfway point are foreground.

### Step 4: extract masks from atlases

```python
candidate_masks = _extract_masks(candidate_u8, candidate_in, thresh_u8)
target_masks = _extract_masks(target_u8, target_in, thresh_u8)
```

These functions use the upstream metadata:

- `atlas_grid`
- `num_people`

If the input did not come from `pose_preprocess.py`, the fallback behavior treats the whole image as a single mask.

### Step 5: read matcher parameters

```python
threshold = float(scriptOp.par.Threshold.eval())
mirror_invariant = bool(scriptOp.par.Mirrorinvariant.eval())
```

`threshold` determines pass/fail. `mirror_invariant` controls whether candidate masks are also tested horizontally flipped.

### Step 6: score masks

```python
result = _score_masks(target_masks, candidate_masks, mirror_invariant)
```

The result contains:

- final overall score
- pairwise IoU matrix
- best assignment
- assignment scores
- mirrored flags

The script then computes pass/fail:

```python
overall = float(result['overall'])
passed = overall >= threshold
```

### Step 7: store downstream diagnostics

The script stores:

```text
match_score           = overall score
match_threshold       = current threshold
match_pass            = True or False
iou_matrix            = pairwise target/candidate IoUs
assignment            = chosen row/column assignment
assignment_scores     = IoUs for chosen assignment
assignment_mirrored   = whether each chosen pair used mirroring
pairwise_mirrored     = whether each pair's best IoU used mirroring
num_target_people     = number of extracted target masks
num_candidate_people  = number of extracted candidate masks
```

These stored values can be read by downstream DATs, CHOPs, or other scripts to drive UI, feedback, scoring, or logging.

### Step 8: debug printing

The script reads:

```python
debug_print = bool(scriptOp.par.Debugprint.eval())
debug_signature = (passed, len(target_masks), len(candidate_masks))
```

It prints diagnostics if either:

- `Debugprint` is enabled, or
- the debug signature changed since the previous cook.

This means it prints when the pass/fail state or person counts change even if continuous debug printing is disabled.

### Step 9: select debug pair

The assignment may include dummy rows or columns. The script filters to real target/candidate pairs only:

```python
real_pairs = [
    (row, col, assignment_mirrored[idx])
    for idx, (row, col) in enumerate(assignment)
    if row < len(target_masks) and col < len(candidate_masks)
]
```

Then it reads `Debugpair` and clamps it to the available real-pair range.

### Step 10: choose output image

There are three possible output modes:

### Mode 1: debug overlay

If real pairs exist and `Passthroughref` is off:

```python
overlay = _overlay_pair(target_masks[row], candidate_masks[col], mirrored)
scriptOp.copyNumpyArray(overlay)
```

This is the normal debug output.

### Mode 2: reference passthrough

If `Passthroughref` is on:

```python
scriptOp.copyNumpyArray(target)
```

This lets the operator display the reference image directly.

### Mode 3: candidate passthrough

If no real pairs exist and reference passthrough is off:

```python
scriptOp.copyNumpyArray(candidate)
```

This gives some visible output even when no assignment exists.

---

## Matcher output contract

`pose_match_iou.py` outputs:

1. A TOP image:
   - usually a red/green/yellow debug overlay
   - or the reference input
   - or the candidate input
2. Stored scalar and diagnostic values:
   - final score
   - threshold
   - pass/fail
   - matrix and assignment details

The most important stored values are:

```text
match_score
match_pass
iou_matrix
assignment
assignment_scores
assignment_mirrored
num_target_people
num_candidate_people
```

---

# How the two scripts work together

## Shared atlas metadata

The preprocessor stores:

```text
atlas_grid = [cols, rows, canon]
num_people = len(persons)
```

The matcher fetches those same values to know:

- how many slots exist
- how big each slot is
- how many slots are populated

This avoids hard-coding atlas layout in the matcher.

## Data normalization before comparison

The preprocessor handles differences in:

- original image resolution
- person position in frame
- bounding-box size
- body scale
- optional person ordering
- small segmentation noise

By the time the matcher runs, each person is represented as a standardized binary square mask.

This makes IoU comparison meaningful.

## Multi-person matching

For multiple people, the system does not rely on slot order alone.

The preprocessor creates slots. The matcher compares all slots against all slots, then solves for the best one-to-one assignment.

This makes the system robust when the live and reference people appear in different left-to-right order or when sorting differs.

## Mirror invariance

The matcher can make the score mirror-invariant by flipping candidate masks horizontally during pairwise comparison.

This is useful if the experience wants a pose to count as correct regardless of left/right orientation.

If left/right specificity matters, `Mirrorinvariant` should be turned off.

---

# Important implementation assumptions

## The input silhouette is read from the red channel

Both scripts use:

```python
src[:, :, 0]
```

or equivalent.

That means they ignore green, blue, and alpha for mask extraction.

If an upstream TOP encodes the silhouette in alpha only, these scripts will not see it unless the channel is copied into red first.

## Foreground is bright, background is dark

Both scripts assume:

```text
foreground/person = white or high value
background        = black or low value
```

If the input is inverted, the pipeline will interpret the background as the person.

## Person count is small

The assignment solver uses recursive dynamic programming over a bitmask. This is fine for small `Maxpeople` values like `4` or `8`.

For very large person counts, a dedicated Hungarian algorithm implementation would be more scalable.

## The final score averages over dummy assignments

When the number of target and candidate people differs, the assignment matrix is padded with dummy rows or columns that score `0`.

This intentionally penalizes missing or extra people.

---

# Practical tuning guidance

## If silhouettes are noisy

Increase `Closepct` slightly to fill gaps and smooth small holes.

Increase `Minareapct` if tiny blobs are being detected as people.

Be careful: too much closing can merge nearby people into one fused blob.

## If separate people are merging

Enable `Splitfused`.

Tune:

- `Aspectfusedmin`
- `Solidityfusedmax`

Lowering the solidity threshold can make fused detection less sensitive. Raising it can make the script flag more components as fused.

## If pose comparison is too strict

Lower the matcher `Threshold`.

Also consider whether `Centroidalign` should be enabled. Centroid alignment can improve stability when people are off-center, but in some cases it can also remove meaningful positional differences.

## If mirrored poses should not count

Disable `Mirrorinvariant`.

With mirror invariance enabled, left/right pose differences may be treated as matches.

## If debug overlay is confusing

Remember the colors:

```text
red    = target only
green  = candidate only
yellow = overlap
black  = background
```

The more yellow, the stronger the overlap.

---

# End-to-end example

Suppose the reference atlas contains two target people and the candidate atlas contains two live people.

The matcher may compute a matrix like:

```text
             candidate 0   candidate 1
target 0        0.20          0.82
target 1        0.76          0.10
```

A slot-by-slot comparison would average:

```text
(0.20 + 0.10) / 2 = 0.15
```

But the best assignment is:

```text
target 0 -> candidate 1 = 0.82
target 1 -> candidate 0 = 0.76
```

So the actual score is:

```text
(0.82 + 0.76) / 2 = 0.79
```

If the match threshold is `0.75`, this passes.

This illustrates why the pairwise matrix and assignment step are essential for multi-person pose matching.

---

# Summary

`pose_preprocess.py` turns messy raw silhouette images into clean, normalized, per-person atlas slots.

`pose_match_iou.py` compares two such atlases using IoU, supports multi-person assignment, optionally handles mirrored poses, and stores a final match score plus detailed diagnostics.

Together, they form a practical silhouette-based pose matching system for TouchDesigner:

```text
raw silhouette -> cleaned normalized person atlas -> IoU matrix -> best assignment -> final score
```