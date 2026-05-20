"""
Script TOP Callbacks - Pose preprocessing atlas.

me - this DAT
scriptOp - the OP which is cooking

PURPOSE
-------
This Script TOP turns a noisy silhouette image into a *canonicalized atlas*
of per-person masks. Downstream comparison ops (e.g. `pose_match_iou.py`)
can then iterate the atlas tiles instead of having to redo person detection,
cropping, and normalization themselves.

PIPELINE OVERVIEW (the four phases referenced in section headers below)
-----------------------------------------------------------------------
Phase A - Input parsing:
    Read the single TOP input as a numpy array, take the red channel as a
    silhouette (TouchDesigner gives RGBA float32 0..1; we promote to uint8
    0..255 so OpenCV is happy).

Phase B - Mask cleanup (`_preprocess_mask`):
    Binarize at the user threshold, close holes with an elliptical kernel
    sized relative to the image diagonal (so tuning is resolution-agnostic),
    smooth + re-threshold to soften the closed edges, then drop tiny
    connected components that are almost certainly noise.

Phase C - Per-person separation (`_is_fused`, `_split_fused`):
    `connectedComponentsWithStats` gives one blob per visually-isolated
    person, but two people touching/overlapping produce a single fused
    blob. If "Split Fused" is on we test each component against three
    heuristics (wide aspect, low solidity vs convex hull, vertical pinch
    in the column profile) and, when flagged, run a distance-transform
    watershed to break it into >=2 sub-masks. Pure column-cut at the
    detected pinch is the fallback when watershed only finds one peak.

Phase D - Per-person normalization (`_normalize_mask`):
    Tight-crop to the person's bbox, aspect-preserving resize into a
    `Canonsize` x `Canonsize` square, letterbox-pad black, and optionally
    translate so the binary moments centroid lands at the tile center.
    The result is a per-person mask in canonical coordinates - same
    resolution, same centering - so downstream IoU/shape compares can
    treat tiles as directly comparable.

Final pack:
    Sort persons (left-to-right or largest-first), keep the first
    `Maxpeople`, and paint each into one slot of an RGBA atlas where the
    grid is approximately square (cols = ceil(sqrt(Maxpeople))).

INPUT
-----
One TOP input: a silhouette image (any resolution, white = subject on
black). Only the red channel is read.

OUTPUT
------
RGBA uint8 atlas at (rows * Canonsize, cols * Canonsize, 4) where
    cols = ceil(sqrt(Maxpeople))
    rows = ceil(Maxpeople / cols)
Each populated slot holds a binarized 0/255 person mask replicated across
RGB. Alpha is always 255. Empty slots stay fully black RGB with alpha 255.

PER-SLOT METADATA (published via `scriptOp.store`, fetched downstream)
----------------------------------------------------------------------
    op('pose_preprocess1').fetch('num_people', 0)
    op('pose_preprocess1').fetch('slot_bboxes', [])     # [[x, y, w, h], ...] source px
    op('pose_preprocess1').fetch('slot_centroids', [])  # [[cx, cy], ...]    source px
    op('pose_preprocess1').fetch('slot_areas', [])      # [int, ...]         source px
    op('pose_preprocess1').fetch('atlas_grid', [1, 1, 256])  # [cols, rows, canon]

`atlas_grid` is the contract that lets a downstream op slice the atlas
back into individual tiles without re-deriving the layout.
"""

import math
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Parameter page
# ---------------------------------------------------------------------------

def onSetupParameters(scriptOp: scriptTOP):
	"""
	Called to setup custom parameters for the Script TOP.

	Parameter reference (each maps to a single tunable in the op UI):
	    Maxpeople        Hard cap on slots in the atlas. Drives grid size.
	    Canonsize        Side length (px) of each per-person tile.
	    Threshold        0..1 cutoff applied to the red channel before
	                     binarization (Phase B).
	    Closepct         Morphological-close kernel size as % of image
	                     diagonal. Resolution-independent way to fill
	                     small holes / bridge tiny gaps in the silhouette.
	    Minareapct       Drop connected components smaller than this % of
	                     the full frame area. Kills speckle noise.
	    Splitfused       Toggle Phase C watershed splitting of fused
	                     people-blobs.
	    Aspectfusedmin   When Splitfused is on, components whose width/
	                     height exceeds this ratio are flagged as
	                     potentially-fused (people standing side by side).
	    Solidityfusedmax When Splitfused is on, components whose
	                     area / convex-hull-area is BELOW this value are
	                     flagged as fused (concave silhouettes from two
	                     bodies touching).
	    Centroidalign    Phase D shift so the binary centroid lands at
	                     the canonical tile center. Removes any leftover
	                     translation bias from letterboxing.
	    Sortby           How slots are ordered into the atlas. Left-to-
	                     right keeps slot indices stable when people
	                     don't cross over; largest-first prioritizes the
	                     dominant subject when Maxpeople < detected count.
	"""
	page = scriptOp.appendCustomPage('Pose Preprocess')

	p = page.appendInt('Maxpeople', label='Max People')
	p[0].default = 4
	p[0].min = 1
	p[0].clampMin = True
	p[0].normMin = 1
	p[0].normMax = 8

	p = page.appendInt('Canonsize', label='Canonical Size')
	p[0].default = 256
	p[0].min = 16
	p[0].clampMin = True
	p[0].normMin = 64
	p[0].normMax = 512

	p = page.appendFloat('Threshold', label='Binary Threshold')
	p[0].default = 0.5
	p[0].normMin = 0.0
	p[0].normMax = 1.0

	p = page.appendFloat('Closepct', label='Close Kernel %')
	p[0].default = 1.5
	p[0].normMin = 0.0
	p[0].normMax = 5.0

	p = page.appendFloat('Minareapct', label='Min Component Area %')
	p[0].default = 0.5
	p[0].normMin = 0.0
	p[0].normMax = 5.0

	page.appendToggle('Splitfused', label='Split Fused')

	p = page.appendFloat('Aspectfusedmin', label='Fused: Aspect W/H Min')
	p[0].default = 1.3
	p[0].normMin = 1.0
	p[0].normMax = 3.0

	p = page.appendFloat('Solidityfusedmax', label='Fused: Solidity Max')
	p[0].default = 0.7
	p[0].normMin = 0.3
	p[0].normMax = 1.0

	t = page.appendToggle('Centroidalign', label='Centroid Align')
	t[0].default = True

	m = page.appendMenu('Sortby', label='Slot Order')
	m[0].menuNames = ['xcent', 'area']
	m[0].menuLabels = ['Left-to-Right (centroid x)', 'Largest First (area)']
	m[0].default = 'xcent'

	return


def onPulse(par: Par):
	return


# ---------------------------------------------------------------------------
# Atlas geometry helpers
# ---------------------------------------------------------------------------

def _grid_dims(maxpeople: int) -> tuple[int, int]:
	"""Return (cols, rows) for the atlas grid given max slot count.

	We pick the most-square layout that can hold `maxpeople` slots so the
	atlas stays compact and downstream slicing is easy. For Maxpeople=4
	this yields 2x2, for 6 it yields 3x2, etc.
	"""
	cols = max(1, int(math.ceil(math.sqrt(maxpeople))))
	rows = max(1, int(math.ceil(maxpeople / float(cols))))
	return cols, rows


def _empty_atlas(maxpeople: int, canon: int) -> np.ndarray:
	"""Allocate a fully-black RGBA atlas large enough for `maxpeople` tiles.

	Alpha is pre-filled to 255 so that empty slots are still considered
	opaque - downstream ops should treat the RGB channels (all zero) as
	"no person here", not the alpha.
	"""
	cols, rows = _grid_dims(maxpeople)
	atlas = np.zeros((rows * canon, cols * canon, 4), dtype=np.uint8)
	atlas[..., 3] = 255
	return atlas


# ---------------------------------------------------------------------------
# Phase B: per-image cleanup
# ---------------------------------------------------------------------------

def _preprocess_mask(img_u8: np.ndarray, thresh_u8: int,
                     close_pct: float, min_area_pct: float) -> np.ndarray:
	"""Binarize -> elliptical close -> gaussian smooth + re-threshold ->
	drop components below min_area_pct of frame area. Returns uint8 0/255.

	Why this sequence:
	  1. Threshold gives us a hard 0/255 silhouette to work on.
	  2. Morphological close fills small holes (e.g. inside-the-body
	     speckles) and bridges hairline gaps (e.g. a thin gap between an
	     arm and the torso) without dilating the outer boundary much.
	     The kernel scales with image diagonal so the same close_pct
	     behaves consistently across resolutions.
	  3. Gaussian blur + re-threshold rounds off the jagged edges left by
	     the close kernel - softer contours give more stable moments and
	     IoU later in the pipeline.
	  4. The connected-components pass keeps only blobs large enough to
	     plausibly be a person; everything smaller is treated as noise.
	"""
	# 1) Hard binarize on the (uint8) red channel.
	_, m = cv2.threshold(img_u8, thresh_u8, 255, cv2.THRESH_BINARY)

	# 2) Resolution-independent elliptical close. Kernel must be odd and
	#    at least 3px to be a valid structuring element.
	h, w = m.shape[:2]
	diag = math.hypot(w, h)
	k = int(round(diag * (close_pct / 100.0)))
	k = max(3, k)
	if k % 2 == 0:
		k += 1
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
	m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

	# 3) Smooth-then-rethreshold: keeps blobs at roughly the same area
	#    while making contours less staircase-y. The 9x9 kernel is small
	#    relative to typical person blobs so we don't melt features.
	m = cv2.GaussianBlur(m, (9, 9), 0)
	_, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

	# 4) Drop components below min_area_pct of total frame area. Label 0
	#    is background, so we start the loop at 1.
	num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
	frame_area = float(h * w)
	min_area = frame_area * (min_area_pct / 100.0)
	keep = np.zeros_like(m)
	for i in range(1, num):
		if stats[i, cv2.CC_STAT_AREA] >= min_area:
			keep[labels == i] = 255
	return keep


# ---------------------------------------------------------------------------
# Phase C: per-person separation (fused-blob splitting)
# ---------------------------------------------------------------------------

def _vertical_pinch_col(comp_local: np.ndarray):
	"""Return the column index (in component-local coords) of a vertical
	pinch in the middle 60% of the component, or None.

	Intuition: if two people stand side-by-side and their silhouettes
	just touch, the column-wise pixel count profile will dip sharply
	where the bodies meet. We look at the middle 60% of the bbox (to
	avoid pinches caused by the natural taper at limbs/feet) and call
	a "pinch" only when the minimum there is significantly smaller than
	the flanks (< 70% of the larger flank's max).
	"""
	h, w = comp_local.shape[:2]
	if w < 8:
		# Too narrow to meaningfully look for a middle pinch.
		return None

	# Column-wise count of foreground pixels = vertical density profile.
	col_count = (comp_local > 0).sum(axis=0).astype(np.int32)
	lo = int(w * 0.2)
	hi = int(w * 0.8)
	if hi - lo < 3:
		return None
	mid = col_count[lo:hi]
	if mid.size == 0 or mid.max() == 0:
		return None
	# Compare middle minimum against the brighter of the two flanks; a
	# real pinch should be a clear local valley relative to the bodies.
	left_max = int(col_count[:lo].max()) if lo > 0 else 0
	right_max = int(col_count[hi:].max()) if hi < w else 0
	flank_max = max(left_max, right_max)
	if flank_max == 0:
		return None
	min_idx_local = int(np.argmin(mid))
	min_val = int(mid[min_idx_local])
	if min_val < 0.7 * flank_max:
		# Convert from middle-slice index back to component-local x.
		return lo + min_idx_local
	return None


def _is_fused(comp_local: np.ndarray, area: int,
              aspect_max: float, solidity_max: float):
	"""Apply the three Phase-C heuristics. Returns (flagged, pinch_col_local|None).

	Heuristics (any one trips the flag):
	  - Aspect:   bbox width/height > aspect_max. People standing
	              shoulder-to-shoulder produce wide blobs.
	  - Solidity: blob_area / convex_hull_area < solidity_max. Two
	              touching bodies leave concave gaps that the convex
	              hull fills in, dropping solidity well below 1.
	  - Pinch:    a vertical pinch column inside the middle of the bbox
	              (see `_vertical_pinch_col`).

	The pinch column, when found, is also returned so that the downstream
	splitter can use it as a fallback vertical cut location if watershed
	doesn't find two seeds.
	"""
	h, w = comp_local.shape[:2]
	if h <= 0 or w <= 0:
		return False, None
	flagged = False

	# Heuristic 1: aspect ratio of the local (already-tightly-cropped) bbox.
	if (w / float(h)) > aspect_max:
		flagged = True

	# Heuristic 2: solidity vs convex hull. We use the largest external
	# contour as the blob outline (there's effectively one because we're
	# already inside a single connected component).
	cnts, _ = cv2.findContours(comp_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if cnts:
		c = max(cnts, key=cv2.contourArea)
		hull = cv2.convexHull(c)
		hull_area = float(cv2.contourArea(hull))
		if hull_area > 0 and (area / hull_area) < solidity_max:
			flagged = True

	# Heuristic 3: vertical pinch in the column-density profile.
	pinch = _vertical_pinch_col(comp_local)
	if pinch is not None:
		flagged = True

	return flagged, pinch


def _split_fused(comp_local: np.ndarray, pinch_col):
	"""Watershed-split a fused component into >=2 sub-masks (same shape as
	comp_local). Falls back to vertical-cut at pinch_col if watershed only
	produces a single seed. Returns [comp_local] if no split is possible.

	Algorithm:
	  1. Distance transform of the component: each foreground pixel is
	     replaced with its distance to the nearest background pixel.
	     Body centers become the highest peaks.
	  2. Threshold at 50% of the global max distance to get "seed" blobs
	     near each body center. If there's only one peak, the heuristic
	     was probably a false alarm (or the bodies overlap too heavily
	     for watershed) - we bail out to the column-cut fallback.
	  3. With >=2 seeds, run cv2.watershed on the *inverted* distance
	     transform so the saddles between bodies appear as ridges that
	     watershed will trace as boundaries. Each seed expands until it
	     meets another seed's territory.
	  4. Reassemble per-seed masks, clipped to the original component so
	     we never leak into background pixels watershed may have
	     "labeled" incidentally.
	"""
	# 1) Distance transform - peaks land at the "thickest" interior points.
	dist = cv2.distanceTransform(comp_local, cv2.DIST_L2, 5)
	dmax = float(dist.max())
	if dmax <= 0.0:
		return [comp_local]

	# 2) Seed regions = pixels well inside the body, away from the edge.
	#    The 0.5 * dmax threshold is empirical; high enough to separate
	#    two people without breaking a single body into multiple seeds.
	peaks = (dist > 0.5 * dmax).astype(np.uint8) * 255
	num_seeds, seed_labels = cv2.connectedComponents(peaks)
	num_seeds -= 1  # connectedComponents counts background as label 0.

	if num_seeds >= 2:
		# cv2.watershed marker convention:
		#   0      = "unknown, please flood here"
		#   1      = explicit background
		#   2..N+1 = explicit foreground seeds
		# So we shift our seed labels up by 1 and pin background to 1.
		markers = seed_labels.astype(np.int32).copy()
		markers[markers > 0] += 1
		markers[comp_local == 0] = 1

		# Watershed wants an actual image to ride the gradients of. We
		# feed it the inverted distance transform: bright = far from edge
		# (body interiors), dark = pinch/saddle between bodies, which is
		# exactly the ridge we want watershed to settle on.
		dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		inv_dist_bgr = cv2.cvtColor(255 - dist_u8, cv2.COLOR_GRAY2BGR)
		cv2.watershed(inv_dist_bgr, markers)

		# Extract one sub-mask per seed, clipped to the original blob so
		# the watershed's boundary label (-1) and any out-of-blob fill
		# stay out.
		sub_masks = []
		for sid in range(2, num_seeds + 2):
			sub = np.zeros_like(comp_local)
			sub[(markers == sid) & (comp_local > 0)] = 255
			if cv2.countNonZero(sub) > 0:
				sub_masks.append(sub)
		if len(sub_masks) >= 2:
			return sub_masks

	# Fallback: just cut the blob in half at the detected pinch column.
	# This is safer than guessing - if the pinch heuristic flagged this
	# blob there's an obvious vertical seam to cut along.
	if pinch_col is not None and 0 < pinch_col < comp_local.shape[1] - 1:
		left = comp_local.copy()
		right = comp_local.copy()
		left[:, pinch_col:] = 0
		right[:, :pinch_col] = 0
		if cv2.countNonZero(left) > 0 and cv2.countNonZero(right) > 0:
			return [left, right]

	# Neither watershed nor a column cut produced a valid split. Treat
	# the blob as a single person.
	return [comp_local]


# ---------------------------------------------------------------------------
# Phase D: per-person normalization
# ---------------------------------------------------------------------------

def _normalize_mask(full_mask: np.ndarray, canon: int, centroid_align: bool) -> np.ndarray:
	"""Tight-crop -> aspect-preserving resize into canon x canon ->
	letterbox pad -> optional centroid-to-center shift. Returns canon x canon uint8.

	The output of this function is what makes downstream comparison
	meaningful: every person across every input frame ends up at the same
	resolution, the same aspect-preserved scale, and (with centroid_align)
	roughly centered in their tile. Two such tiles can then be compared
	directly via IoU, Hu moments, etc. without any further alignment.

	Steps:
	  1. Tight-crop to the person's bbox so the silhouette fills the
	     working space.
	  2. Aspect-preserving resize so the longer side maps to `canon`. We
	     re-threshold after INTER_AREA because the area filter can
	     produce intermediate gray values along edges.
	  3. Letterbox padding centers the resized mask in the canonical
	     square; both axes are split equally.
	  4. (Optional) Centroid alignment: compute the binary image's first
	     moments and translate so the center of mass lands at the tile
	     center. This removes any leftover bias from a person being, e.g.,
	     bottom-heavy (legs only) which letterboxing alone can't fix.
	"""
	# 1) Tight crop. np.where gives pixel coords of all foreground pixels.
	ys, xs = np.where(full_mask > 0)
	if xs.size == 0:
		# No person in this mask - hand back an empty canonical tile.
		return np.zeros((canon, canon), dtype=np.uint8)

	x0, x1 = int(xs.min()), int(xs.max()) + 1
	y0, y1 = int(ys.min()), int(ys.max()) + 1
	cropped = full_mask[y0:y1, x0:x1]
	ch, cw = cropped.shape[:2]
	if ch == 0 or cw == 0:
		return np.zeros((canon, canon), dtype=np.uint8)

	# 2) Aspect-preserving resize. `scale` is the smaller of the two
	#    axis-scales so the longer side hits `canon` exactly and the
	#    shorter side ends up <= canon.
	scale = min(canon / float(cw), canon / float(ch))
	new_w = max(1, int(round(cw * scale)))
	new_h = max(1, int(round(ch * scale)))
	resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
	# INTER_AREA returns smoothed values; re-threshold to a hard 0/255 mask.
	_, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

	# 3) Letterbox pad. Any leftover odd pixel goes to `bottom`/`right`.
	top = (canon - new_h) // 2
	bottom = canon - new_h - top
	left = (canon - new_w) // 2
	right = canon - new_w - left
	padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
	                            cv2.BORDER_CONSTANT, value=0)

	# 4) Centroid alignment via affine translation. cv2.moments on a
	#    binary image gives:
	#       m00 = total foreground area
	#       m10 = sum of x coords  -> centroid_x = m10/m00
	#       m01 = sum of y coords  -> centroid_y = m01/m00
	#    We shift by (canvas_center - centroid) so the center of mass
	#    lands at the middle of the canonical tile.
	if centroid_align:
		mom = cv2.moments(padded, binaryImage=True)
		if mom['m00'] > 0:
			cx = mom['m10'] / mom['m00']
			cy = mom['m01'] / mom['m00']
			dx = int(round(canon / 2.0 - cx))
			dy = int(round(canon / 2.0 - cy))
			if dx != 0 or dy != 0:
				M = np.float32([[1, 0, dx], [0, 1, dy]])
				padded = cv2.warpAffine(padded, M, (canon, canon),
				                        flags=cv2.INTER_NEAREST,
				                        borderMode=cv2.BORDER_CONSTANT,
				                        borderValue=0)
				# warpAffine on NEAREST is already crisp, but threshold
				# again as a safety net so downstream IoU sees only 0/255.
				_, padded = cv2.threshold(padded, 127, 255, cv2.THRESH_BINARY)

	return padded


# ---------------------------------------------------------------------------
# Main cook
# ---------------------------------------------------------------------------

def onCook(scriptOp: scriptTOP):
	"""
	Called when the Script TOP needs to cook.

	Orchestrates the full pipeline:
	    Phase A : pull image + params -> uint8 silhouette
	    Phase B : `_preprocess_mask` -> cleaned binary mask
	    Phase C : per-connected-component splitting (optional)
	    Phase D : `_normalize_mask` per person -> canonical tile
	    Pack    : sort, cap to Maxpeople, paint into RGBA atlas
	    Publish : copy atlas to TOP output, store per-slot metadata
	"""
	# Atlas dimensions are derived from Maxpeople only (not detected
	# people) so the output resolution stays constant frame-to-frame.
	maxpeople = max(1, int(scriptOp.par.Maxpeople.eval()))
	canon = max(16, int(scriptOp.par.Canonsize.eval()))
	cols, rows = _grid_dims(maxpeople)

	# Defensive: no input wired up. Still publish a valid (empty) atlas
	# and zeroed metadata so downstream ops don't error out.
	if len(scriptOp.inputs) < 1:
		atlas = _empty_atlas(maxpeople, canon)
		scriptOp.copyNumpyArray(atlas)
		scriptOp.store('num_people', 0)
		scriptOp.store('slot_bboxes', [])
		scriptOp.store('slot_centroids', [])
		scriptOp.store('slot_areas', [])
		scriptOp.store('atlas_grid', [cols, rows, canon])
		return

	# Phase A: pull image. TouchDesigner returns float32 RGBA in [0, 1];
	# OpenCV wants uint8, and we only need the red channel.
	src = scriptOp.inputs[0].numpyArray(delayed=False)
	src_u8 = (src[:, :, 0] * 255).astype(np.uint8)

	# Snapshot all param evals up front so the cook is consistent even
	# if a param changes mid-cook (rare but real in TD timewires).
	thresh_u8 = int(max(0.0, min(1.0, float(scriptOp.par.Threshold.eval()))) * 255)
	close_pct = float(scriptOp.par.Closepct.eval())
	min_area_pct = float(scriptOp.par.Minareapct.eval())
	split_fused = bool(scriptOp.par.Splitfused.eval())
	aspect_max = float(scriptOp.par.Aspectfusedmin.eval())
	solidity_max = float(scriptOp.par.Solidityfusedmax.eval())
	centroid_align = bool(scriptOp.par.Centroidalign.eval())
	sort_mode = str(scriptOp.par.Sortby.eval())

	# Phase B: clean the silhouette into a single 0/255 binary mask.
	mask = _preprocess_mask(src_u8, thresh_u8, close_pct, min_area_pct)

	# Per-component scan. stats[i] = [LEFT, TOP, WIDTH, HEIGHT, AREA].
	# Label 0 is background; everything from 1..num-1 is a real blob.
	num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

	# Accumulate detected persons (possibly more than Maxpeople; we cap
	# after sorting so the most-relevant ones survive).
	persons = []
	for i in range(1, num):
		x = int(stats[i, cv2.CC_STAT_LEFT])
		y = int(stats[i, cv2.CC_STAT_TOP])
		w = int(stats[i, cv2.CC_STAT_WIDTH])
		h = int(stats[i, cv2.CC_STAT_HEIGHT])
		area = int(stats[i, cv2.CC_STAT_AREA])
		if w <= 0 or h <= 0:
			continue
		# Isolate this single component as a local-coord binary tile.
		# Working in local coords keeps the splitter / heuristics fast.
		comp_local = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255

		# Phase C: try to split into multiple people if the component
		# looks fused. If splitting is disabled or the heuristics don't
		# fire, we treat the whole component as a single person.
		sub_masks_local = [comp_local]
		if split_fused:
			fused, pinch = _is_fused(comp_local, area, aspect_max, solidity_max)
			if fused:
				sub_masks_local = _split_fused(comp_local, pinch)

		# Each sub-mask becomes a person record. Coordinates are
		# converted back to *source-frame* pixel coords because the
		# downstream consumer expects those (the atlas tile is just a
		# normalized rendering of the person, not their position).
		for sub_local in sub_masks_local:
			ys2, xs2 = np.where(sub_local > 0)
			if xs2.size == 0:
				continue
			# Paint the sub-mask back into a full-frame canvas so the
			# normalizer (Phase D) sees the person in original-resolution
			# coordinates.
			full = np.zeros_like(mask)
			full[y:y + h, x:x + w] = sub_local
			sub_area = int(xs2.size)
			# Centroid in source pixel coords (offset by component origin).
			scx = float(xs2.mean()) + x
			scy = float(ys2.mean()) + y
			sx0 = int(xs2.min()) + x
			sy0 = int(ys2.min()) + y
			sx1 = int(xs2.max()) + x + 1
			sy1 = int(ys2.max()) + y + 1
			persons.append({
				'mask': full,
				'bbox': [sx0, sy0, sx1 - sx0, sy1 - sy0],
				'area': sub_area,
				'centroid': [scx, scy],
			})

	# Order persons into slot indices. Left-to-right gives stable slot
	# assignments frame-to-frame when subjects don't cross paths; area
	# sort is useful when Maxpeople caps off smaller subjects.
	if sort_mode == 'area':
		persons.sort(key=lambda p: -p['area'])
	else:
		persons.sort(key=lambda p: p['centroid'][0])

	# Hard cap so we never overflow the atlas grid.
	persons = persons[:maxpeople]

	# Phase D + pack: normalize each person and paste into its tile.
	# Slot layout matches `_grid_dims`: row-major, top-left origin.
	atlas = _empty_atlas(maxpeople, canon)
	for idx, person in enumerate(persons):
		norm = _normalize_mask(person['mask'], canon, centroid_align)
		row = idx // cols
		col = idx % cols
		y0a = row * canon
		x0a = col * canon
		# Replicate the binary mask into RGB so the atlas reads as a
		# grayscale silhouette in any consumer. Alpha is forced opaque.
		atlas[y0a:y0a + canon, x0a:x0a + canon, 0] = norm
		atlas[y0a:y0a + canon, x0a:x0a + canon, 1] = norm
		atlas[y0a:y0a + canon, x0a:x0a + canon, 2] = norm
		atlas[y0a:y0a + canon, x0a:x0a + canon, 3] = 255

	scriptOp.copyNumpyArray(atlas)

	# Publish per-slot metadata. The atlas alone tells you *what* each
	# person looks like canonically; these arrays tell you *where* each
	# person was in the source frame, in slot order.
	scriptOp.store('num_people', len(persons))
	scriptOp.store('slot_bboxes', [p['bbox'] for p in persons])
	scriptOp.store('slot_centroids', [p['centroid'] for p in persons])
	scriptOp.store('slot_areas', [p['area'] for p in persons])
	# atlas_grid is the contract downstream ops use to slice tiles back
	# out: [cols, rows, canon_size]. Keep it in sync with the atlas
	# dimensions you copy out above.
	scriptOp.store('atlas_grid', [cols, rows, canon])
	return


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.
	"""
	return CookLevel.AUTOMATIC
