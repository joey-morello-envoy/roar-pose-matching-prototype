"""
Script TOP Callbacks - IoU pose match (multi-slot aware).

me - this DAT
scriptOp - the OP which is cooking

PURPOSE
-------
Given two silhouette *atlases* from `pose_preprocess.py` (a live/candidate
atlas and a target/reference atlas), score how well the live people match
the reference people. The result is a single 0..1 `match_score` plus a
collection of diagnostic stores (per-pair IoU matrix, chosen assignment,
mirror flags) that downstream DATs/CHOPs can pull from.

The score is "the average IoU of the best one-to-one pairing", which
means:
    score = 1.0    every live person perfectly overlaps a unique target
    score = 0.0    no overlap at all (or a missing input)
A `Match Threshold` parameter turns the score into a boolean
`match_pass`.

PIPELINE OVERVIEW
-----------------
1. `_resolve_grid` / `_resolve_num_people`
       Read the atlas layout published by the upstream op via `store`.
       Falls back to "single-tile, no metadata" so this op still works
       when fed a plain binary mask TOP that didn't come from
       pose_preprocess.

2. `_tile_bin` / `_extract_masks`
       Slice each populated atlas slot back out as a 0/255 binary mask.
       The number of slots actually read is `num_people` (not the total
       grid capacity) so we never compare empty tiles.

3. `_pairwise_iou`
       Build an M x N IoU matrix (M target masks, N candidate masks).
       When Mirror Invariant is on, each cell takes the max of (direct,
       horizontally-flipped) and we remember which orientation won.

4. `_best_assignment`
       Bitmask-DP that finds the one-to-one permutation maximizing the
       sum of IoU values. Padding to a square matrix lets the solver
       deal with unequal counts (extra targets or extra candidates) by
       pairing the surplus to dummy zero-IoU rows/columns, which
       naturally penalizes the average.

5. `_score_masks`
       Glue: builds the IoU matrix, runs the assignment, and produces a
       result dict with the overall mean score plus all diagnostics.

6. `onCook`
       Wires steps 1-5 together, publishes scores via `store`, prints
       human-readable debug output, and renders one of three TOP
       outputs depending on UI toggles:
           - matched-pair overlay (target = red, candidate = green)
           - reference passthrough
           - candidate passthrough

INPUTS
------
    0 - Live/candidate silhouette atlas (from pose_preprocess.py, or a
        plain mask if used standalone)
    1 - Target/reference silhouette atlas (same)

OUTPUT TOP
----------
RGBA uint8. Contents depend on UI:
    - default      : R = selected target mask, G = selected candidate
                     mask (post-mirror), A = 255. Yellow = overlap.
    - Show Reference toggle on : passthrough of input 1.
    - else / no real pairs     : passthrough of input 0.

STORES (read downstream via `op('pose_match_iou1').fetch(...)`)
----------------------------------------------------------------
    match_score             float, mean IoU of chosen assignment
    match_threshold         float, current Threshold param value
    match_pass              bool, match_score >= threshold
    iou_matrix              [[float]], target rows x candidate cols
    assignment              [(target_idx, candidate_idx), ...]
                            (may include dummy indices when counts differ)
    assignment_scores       [float], per-pair IoU in assignment order
    assignment_mirrored     [bool],  did the candidate win flipped?
    pairwise_mirrored       [[bool]], per-cell mirror flag from step 3
    num_target_people       int
    num_candidate_people    int
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import cv2


# Module-level cache so we only spam the debug print when the result
# *changes* (pass/fail flip or person-count change), not on every cook.
# When Debug Print is on we print every cook regardless.
_last_debug_signature: tuple[bool, int, int] | None = None


# press 'Setup Parameters' in the OP to call this function to re-create
# the parameters.
def onSetupParameters(scriptOp: scriptTOP):
	"""
	Called to setup custom parameters for the Script TOP.

	Parameter reference:
	    Threshold        Minimum mean IoU required for `match_pass` to
	                     be True. 0.75 is a reasonable default for full-
	                     body silhouettes after preprocessing.
	    Mirrorinvariant  If on, each candidate mask is also compared
	                     flipped horizontally and the better orientation
	                     wins. Useful when "pose facing left" should
	                     match "pose facing right".
	    Debugpair        Index into the chosen assignment used to pick
	                     which target/candidate pair the overlay output
	                     visualizes. Clamped to the number of real pairs.
	    Debugprint       Force the human-readable debug block to print
	                     every cook, not just when the result changes.
	    Passthroughref   Output the reference atlas (input 1) directly
	                     instead of an overlay. Handy when wiring this
	                     into a viewer.
	"""
	page = scriptOp.appendCustomPage('Pose IoU')

	p = page.appendFloat('Threshold', label='Match Threshold')
	p[0].default = 0.75
	p[0].normMin = 0.0
	p[0].normMax = 1.0

	t = page.appendToggle('Mirrorinvariant', label='Mirror Invariant')
	t[0].default = True

	p = page.appendInt('Debugpair', label='Debug Pair')
	p[0].default = 0
	p[0].min = 0
	p[0].clampMin = True
	p[0].normMin = 0
	p[0].normMax = 8

	page.appendToggle('Debugprint', label='Debug Print')
	page.appendToggle('Passthroughref', label='Show Reference')
	return


def onPulse(par: Par):
	"""
	Called when a custom pulse parameter is pushed.
	"""
	return


def _resolve_grid(top_input) -> tuple[int, int, int]:
	"""Pull (cols, rows, canon) from the upstream op store.

	Falls back to single-tile behavior if the upstream did not publish
	`atlas_grid`, which keeps this compatible with a plain binary TOP input.

	Return values:
	    (cols, rows, canon)  - atlas slot grid + per-tile size
	    canon == 0           - "no atlas, treat whole input as one tile"
	"""
	if top_input is None:
		return 1, 1, 0
	# fetch() raises if the key isn't there; we just want a clean fallback.
	grid = None
	try:
		grid = top_input.fetch('atlas_grid', None)
	except Exception:
		grid = None
	if grid and len(grid) == 3:
		return int(grid[0]), int(grid[1]), int(grid[2])
	# Plain-mask input path: pretend the whole image is one big tile so
	# the downstream tile logic still produces something sensible.
	try:
		h = int(top_input.height)
		w = int(top_input.width)
	except Exception:
		h = w = 0
	return 1, 1, max(h, w)


def _resolve_num_people(top_input, default: int) -> int:
	"""Read `num_people` from the upstream store, or fall back to `default`.

	Used to know how many *populated* slots an atlas has, so we don't
	waste cycles comparing empty tiles.
	"""
	if top_input is None:
		return 0
	try:
		n = top_input.fetch('num_people', None)
	except Exception:
		n = None
	if n is None:
		return default
	return int(n)


def _tile_bin(img_u8: np.ndarray, cols: int, canon: int, slot: int,
              thresh_u8: int) -> np.ndarray:
	"""Return slot `slot` from an atlas as a uint8 0/255 binary mask.

	If `canon <= 0` we treat the whole image as a single tile (plain
	mask compatibility mode). Otherwise we index row-major into the
	atlas using the same (cols, canon) layout that pose_preprocess.py
	wrote.
	"""
	if canon <= 0:
		# Whole-frame mode.
		_, t = cv2.threshold(img_u8, thresh_u8, 255, cv2.THRESH_BINARY)
		return t

	# Row-major: slot 0 is top-left, advancing left-to-right then top-to-
	# bottom. Matches the pack order in pose_preprocess.onCook.
	row = slot // cols
	col = slot % cols
	y0 = row * canon
	x0 = col * canon
	# Clamp to image bounds in case the atlas was sized slightly off.
	h, w = img_u8.shape[:2]
	y1 = min(y0 + canon, h)
	x1 = min(x0 + canon, w)
	tile = img_u8[y0:y1, x0:x1]
	_, t = cv2.threshold(tile, thresh_u8, 255, cv2.THRESH_BINARY)
	return t


def _extract_masks(img_u8: np.ndarray, top_input, thresh_u8: int) -> list[np.ndarray]:
	"""Slice an atlas into per-person binary masks.

	Reads layout + count metadata from the upstream op and returns one
	0/255 mask per populated slot, in slot order.
	"""
	cols, rows, canon = _resolve_grid(top_input)
	# Clamp num_people to the grid capacity so a stale/wrong upstream
	# store can never make us index past the atlas.
	max_slot = max(1, cols * rows)
	num_people = max(0, min(_resolve_num_people(top_input, 1), max_slot))
	return [
		_tile_bin(img_u8, max(cols, 1), canon, slot, thresh_u8)
		for slot in range(num_people)
	]


def _same_shape(candidate: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
	"""Resize `candidate` to `target_shape` (H, W) using nearest-neighbor.

	Nearest-neighbor is used on purpose: these are binary masks, and any
	smoothing interpolation would introduce gray pixels that bias the IoU
	count after re-thresholding. If shapes already match this is a no-op.
	"""
	if candidate.shape[:2] == target_shape:
		return candidate
	# cv2.resize takes (W, H), our shapes are (H, W).
	return cv2.resize(candidate, (target_shape[1], target_shape[0]),
	                 interpolation=cv2.INTER_NEAREST)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
	"""Intersection-over-Union for two binary masks.

	    IoU = |A and B| / |A or B|

	IoU is 0 when masks are disjoint, 1 when identical, and is sensitive
	to both missed and extra foreground pixels - which is exactly what
	we want for "how well do these two silhouettes overlap".

	If the two masks aren't the same shape (e.g. one came from a custom
	upstream with a different canon size) we resample to match. Pre-
	atlas resampling would be cheaper, but doing it lazily here keeps
	the rest of the code path simple.
	"""
	if mask_a.shape[:2] != mask_b.shape[:2]:
		mask_b = _same_shape(mask_b, mask_a.shape[:2])

	a = mask_a > 0
	b = mask_b > 0
	union = np.logical_or(a, b).sum()
	if union == 0:
		# Two empty masks - undefined IoU, return 0 so they don't get
		# matched preferentially.
		return 0.0
	intersection = np.logical_and(a, b).sum()
	return float(intersection / union)


def _pairwise_iou(target_masks: list[np.ndarray],
                  candidate_masks: list[np.ndarray],
                  mirror_invariant: bool) -> tuple[np.ndarray, list[list[bool]]]:
	"""Compute the M x N IoU matrix (M targets, N candidates).

	With `mirror_invariant=True` each cell becomes
	    max(IoU(target, candidate), IoU(target, flip(candidate)))
	and we remember which orientation produced the chosen score in the
	`mirrored` 2D list. The assignment solver and the debug overlay both
	use that flag to draw / report the correct candidate orientation.
	"""
	m = len(target_masks)
	n = len(candidate_masks)
	iou_matrix = np.zeros((m, n), dtype=np.float64)
	mirrored = [[False for _ in range(n)] for _ in range(m)]

	for i, target in enumerate(target_masks):
		for j, candidate in enumerate(candidate_masks):
			direct = _iou(target, candidate)
			best = direct
			if mirror_invariant:
				# Horizontal flip (flipCode=1 mirrors left/right).
				flipped = cv2.flip(candidate, 1)
				mirrored_iou = _iou(target, flipped)
				if mirrored_iou > direct:
					best = mirrored_iou
					mirrored[i][j] = True
			iou_matrix[i, j] = best

	return iou_matrix, mirrored


def _best_assignment(iou_matrix: np.ndarray) -> tuple[list[tuple[int, int]], list[float]]:
	"""Maximize total IoU over a square padded one-to-one assignment.

	Rows are targets and columns are candidates. Dummy rows/columns contribute
	0 IoU, so unmatched people are naturally penalized in the aggregate.

	Algorithm:
	    Pad the M x N IoU matrix to a square (size x size) with zeros so
	    every target gets a column and every candidate gets a row. Then
	    pick the permutation of column-per-row that maximizes the sum of
	    selected cells. We solve it with a bitmask DP:

	        solve(row, used_cols) -> (best_score_for_remaining, picks)

	    `used_cols` is a bitmask of already-assigned columns; lru_cache
	    memoizes by (row, used_cols) so each state is computed once.

	    Complexity is O(size * 2^size * size). For Maxpeople up to ~8
	    this is trivially fast (2^8 = 256 states per row); for much
	    larger atlases we'd swap this out for the Hungarian algorithm.

	Returns:
	    assignment       List of (target_row, candidate_col) pairs in
	                     target-row order. Indices >= M or >= N are
	                     "dummies" representing unmatched people.
	    scores           Per-pair IoU values; dummy pairs score 0.
	"""
	m, n = iou_matrix.shape[:2]
	size = max(m, n)
	if size == 0:
		return [], []

	# Pad to square. Real cells keep their IoU; dummy cells are zero so
	# the solver never prefers a fake match over a real one.
	padded = np.zeros((size, size), dtype=np.float64)
	padded[:m, :n] = iou_matrix

	@lru_cache(maxsize=None)
	def solve(row: int, used_cols: int) -> tuple[float, tuple[int, ...]]:
		# Base case: every row assigned, nothing else to pick.
		if row >= size:
			return 0.0, ()

		best_score = -1.0
		best_cols: tuple[int, ...] = ()
		for col in range(size):
			mask = 1 << col
			if used_cols & mask:
				# Column already taken by an earlier row.
				continue
			rest_score, rest_cols = solve(row + 1, used_cols | mask)
			score = float(padded[row, col]) + rest_score
			if score > best_score:
				best_score = score
				best_cols = (col,) + rest_cols
		return best_score, best_cols

	_, cols = solve(0, 0)
	assignment = [(row, col) for row, col in enumerate(cols)]
	# Score per pair: real IoU if both indices are in the real range,
	# otherwise 0 (this is a dummy slot used to balance counts).
	scores = [
		float(iou_matrix[row, col]) if row < m and col < n else 0.0
		for row, col in assignment
	]
	return assignment, scores


def _score_masks(target_masks: list[np.ndarray],
                 candidate_masks: list[np.ndarray],
                 mirror_invariant: bool) -> dict:
	"""Run the full IoU + assignment pipeline and bundle diagnostics.

	The aggregate `overall` score is the *mean* IoU across all
	assignment pairs - including dummy (zero) pairs. That means an
	atlas with 2 candidates but 3 targets has its best two pairs
	averaged with a 0, so missing people genuinely hurt the score.
	"""
	# Step 3 (per module docstring): build the IoU matrix.
	iou_matrix, mirrored = _pairwise_iou(target_masks, candidate_masks, mirror_invariant)
	# Step 4: solve one-to-one assignment.
	assignment, assignment_scores = _best_assignment(iou_matrix)

	# Mean of the chosen pair scores (dummies included so unmatched
	# people pull the overall down).
	if assignment_scores:
		overall = float(np.mean(assignment_scores))
	else:
		overall = 0.0

	# Per-assignment mirror flag, gated to real pairs. Dummy pairs
	# always report False since there's no candidate to mirror.
	assignment_mirrored = []
	for row, col in assignment:
		if row < len(target_masks) and col < len(candidate_masks):
			assignment_mirrored.append(bool(mirrored[row][col]))
		else:
			assignment_mirrored.append(False)

	return {
		'overall': overall,
		'iou_matrix': iou_matrix,
		'assignment': assignment,
		'assignment_scores': assignment_scores,
		'assignment_mirrored': assignment_mirrored,
		'pairwise_mirrored': mirrored,
	}


def _overlay_pair(target: np.ndarray, candidate: np.ndarray, mirrored: bool) -> np.ndarray:
	"""Render an RGBA debug overlay for one matched pair.

	Channel layout:
	    R = target mask  -> reads red where only target is set
	    G = candidate    -> reads green where only candidate is set
	    R + G = yellow   -> areas where they overlap (good)
	    A = 255          -> always opaque

	The candidate is flipped first if the chosen assignment used the
	mirrored orientation, so what you see is the orientation that
	actually scored.
	"""
	if mirrored:
		candidate = cv2.flip(candidate, 1)
	# If the two tiles ended up different sizes (mixed atlas configs)
	# we normalize the candidate to the target's resolution before XOR.
	if target.shape[:2] != candidate.shape[:2]:
		candidate = _same_shape(candidate, target.shape[:2])

	t = target > 0
	c = candidate > 0
	h, w = target.shape[:2]
	overlay = np.zeros((h, w, 4), dtype=np.uint8)
	overlay[..., 0] = t.astype(np.uint8) * 255
	overlay[..., 1] = c.astype(np.uint8) * 255
	overlay[..., 3] = 255
	return overlay


def _blank_overlay(size: int = 256) -> np.ndarray:
	"""Tiny opaque-black overlay used when inputs aren't connected yet."""
	blank = np.zeros((size, size, 4), dtype=np.uint8)
	blank[..., 3] = 255
	return blank


def _print_debug(iou_matrix: np.ndarray,
                 assignment: list[tuple[int, int]],
                 assignment_scores: list[float],
                 assignment_mirrored: list[bool],
                 overall: float,
                 passed: bool) -> None:
	"""Pretty-print the match result to the TD textport.

	Shows:
	    - The raw IoU matrix (targets are rows, candidates are columns)
	    - The chosen one-to-one assignment with per-pair IoU and mirror
	      flag. Dummy assignments (used when target/candidate counts
	      differ) are labeled explicitly.
	    - The final pass/fail verdict and overall score.
	"""
	clear()  # wipe textport before printing

	print('IoU matrix target(row) x candidate(col):')
	if iou_matrix.size == 0:
		print('  <empty>')
	else:
		for i, row in enumerate(iou_matrix):
			cells = '  '.join(f'{v:0.3f}' for v in row)
			print(f'  target {i}: {cells}')

	print('Assignment:')
	m, n = iou_matrix.shape[:2]
	for idx, (row, col) in enumerate(assignment):
		score = assignment_scores[idx]
		mirror = ' mirrored' if assignment_mirrored[idx] else ''
		# Indices past the real count are dummies inserted by the
		# square-padding step in `_best_assignment`.
		target_label = f'target {row}' if row < m else 'unmatched target dummy'
		candidate_label = f'candidate {col}' if col < n else 'unmatched candidate dummy'
		print(f'  {target_label} -> {candidate_label}: IoU={score:0.3f}{mirror}')

	state = 'MATCH' if passed else 'NO MATCH'
	print(f'{state} | overall={overall:0.3f}')


def onCook(scriptOp: scriptTOP):
	"""
	Called when the Script TOP needs to cook.

	Cook order:
	    1. Bail out early with sane empty stores if both inputs aren't wired.
	    2. Read both atlases as uint8 silhouettes and slice them into
	       per-person masks using the upstream `atlas_grid` / `num_people`
	       metadata.
	    3. Score the masks (mirror-invariant IoU + best one-to-one match).
	    4. Publish everything to scriptOp.store so DATs/CHOPs downstream
	       can drive UI, audio cues, gameplay, etc.
	    5. Print human-readable debug when the result changes (or every
	       cook if Debug Print is on).
	    6. Render the TOP output: pair overlay, ref passthrough, or
	       candidate passthrough.
	"""
	global _last_debug_signature

	# (1) Defensive early-out: need BOTH inputs wired before we have
	# anything to compare. Publish neutral stores so downstream is sane.
	if len(scriptOp.inputs) < 2:
		blank = _blank_overlay(2)
		scriptOp.copyNumpyArray(blank)
		scriptOp.store('match_score', 0.0)
		scriptOp.store('match_threshold', float(scriptOp.par.Threshold.eval()))
		scriptOp.store('match_pass', False)
		scriptOp.store('iou_matrix', [])
		scriptOp.store('assignment', [])
		scriptOp.store('assignment_scores', [])
		scriptOp.store('assignment_mirrored', [])
		scriptOp.store('pairwise_mirrored', [])
		scriptOp.store('num_target_people', 0)
		scriptOp.store('num_candidate_people', 0)
		return

	# Convention: input 0 = candidate (live), input 1 = target (reference).
	# Keep the references around so we can fetch upstream stores AND so
	# we can passthrough the raw RGBA later if requested.
	candidate_in = scriptOp.inputs[0]
	target_in = scriptOp.inputs[1]

	candidate = candidate_in.numpyArray(delayed=False)
	target = target_in.numpyArray(delayed=False)

	# Atlases from pose_preprocess come in pre-binarized as 0/1 floats in
	# the red channel; we promote to uint8 and use 127 as the threshold
	# (anything > 0.5 in the float input becomes foreground).
	thresh_u8 = 127
	candidate_u8 = (candidate[:, :, 0] * 255).astype(np.uint8)
	target_u8 = (target[:, :, 0] * 255).astype(np.uint8)

	# (2) Slice both atlases into per-person mask lists. Lengths reflect
	# the upstream `num_people` stores, not grid capacity.
	candidate_masks = _extract_masks(candidate_u8, candidate_in, thresh_u8)
	target_masks = _extract_masks(target_u8, target_in, thresh_u8)

	# (3) Score. Snapshot params first so the result is deterministic
	# even if the user wiggles a slider mid-cook.
	threshold = float(scriptOp.par.Threshold.eval())
	mirror_invariant = bool(scriptOp.par.Mirrorinvariant.eval())
	result = _score_masks(target_masks, candidate_masks, mirror_invariant)

	overall = float(result['overall'])
	passed = overall >= threshold
	assignment = result['assignment']
	assignment_scores = result['assignment_scores']
	assignment_mirrored = result['assignment_mirrored']
	iou_matrix = result['iou_matrix']

	# (4) Publish stores. numpy arrays need .tolist() so TD's storage
	# (which serializes via JSON-ish paths in some contexts) is happy.
	scriptOp.store('match_score', overall)
	scriptOp.store('match_threshold', threshold)
	scriptOp.store('match_pass', passed)
	scriptOp.store('iou_matrix', iou_matrix.tolist())
	scriptOp.store('assignment', assignment)
	scriptOp.store('assignment_scores', assignment_scores)
	scriptOp.store('assignment_mirrored', assignment_mirrored)
	scriptOp.store('pairwise_mirrored', result['pairwise_mirrored'])
	scriptOp.store('num_target_people', len(target_masks))
	scriptOp.store('num_candidate_people', len(candidate_masks))

	# (5) Debug print. We dedupe by (pass/fail, target count, candidate
	# count) so the textport doesn't flood when nothing meaningful has
	# changed between cooks. Toggling Debug Print bypasses the dedupe.
	debug_print = bool(scriptOp.par.Debugprint.eval())
	debug_signature = (passed, len(target_masks), len(candidate_masks))
	if debug_print or debug_signature != _last_debug_signature:
		_print_debug(iou_matrix, assignment, assignment_scores,
		             assignment_mirrored, overall, passed)
		_last_debug_signature = debug_signature

	# (6) Choose what to send out of the TOP.
	#
	#   real_pairs is the assignment filtered down to (target, candidate)
	#   pairs that both reference *real* people (no dummies). Debug Pair
	#   indexes into this list to pick which overlay to render.
	debug_pair = max(0, int(scriptOp.par.Debugpair.eval()))
	real_pairs = [
		(row, col, assignment_mirrored[idx])
		for idx, (row, col) in enumerate(assignment)
		if row < len(target_masks) and col < len(candidate_masks)
	]

	if real_pairs and not scriptOp.par.Passthroughref.eval():
		# Render the requested matched pair as an R=target / G=candidate
		# overlay. Yellow pixels = overlap, which is what we maximize.
		row, col, mirrored = real_pairs[min(debug_pair, len(real_pairs) - 1)]
		overlay = _overlay_pair(target_masks[row], candidate_masks[col], mirrored)
		scriptOp.copyNumpyArray(overlay)
	elif scriptOp.par.Passthroughref.eval():
		# Send the raw reference atlas straight through.
		scriptOp.copyNumpyArray(target)
	else:
		# Nothing matched (or no real pairs); show the live candidate atlas.
		scriptOp.copyNumpyArray(candidate)
	return


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.
	"""
	return CookLevel.AUTOMATIC
