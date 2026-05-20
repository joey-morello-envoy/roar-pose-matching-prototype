"""
Script TOP Callbacks - Hu-moments pose match (multi-slot aware).

me - this DAT
scriptOp - the OP which is cooking

Inputs:
    0 - Live silhouette atlas    (from pose_preprocess.py, or a plain mask)
    1 - Reference silhouette atlas (same)

When the upstream op publishes `atlas_grid` and `num_people` via store(),
the script slices each input into tiles of `canon x canon`, compares slot i
of live to slot i of ref for i in 0..min(num_live, num_ref), and aggregates
the per-slot Hu scores into a single `match_score` (mean of valid pairs).

When `atlas_grid` is not published (e.g. the preprocessor is not in front
of this op), the input is treated as a single tile - same behavior as the
original Hu script.

Output: passthrough of either live or ref (TD requires the op to emit an
image). `match_score` and per-slot diagnostics are published via store().
"""

import numpy as np
import cv2

# Tracks last interpretation band so we only print on changes
_last_band: str | None = None


# press 'Setup Parameters' in the OP to call this function to re-create 
# the parameters.
def onSetupParameters(scriptOp: scriptTOP):
	"""
	Called to setup custom parameters for the Script TOP.
	"""
	page = scriptOp.appendCustomPage('Pose Match')
	p = page.appendFloat('Threshold', label='Binary Threshold')
	p[0].default = 0.5

	m = page.appendMenu('Method', label='Match Method')
	m[0].menuNames  = ['I1', 'I2', 'I3']
	m[0].menuLabels = ['I1', 'I2', 'I3']

	page.appendToggle('Passthroughref', label='Show Reference')
	return


def onPulse(par: Par):
	"""
	Called when a custom pulse parameter is pushed.
	
	Args:
		par: The parameter that was pulsed
	"""
	return


def _resolve_grid(top_input) -> tuple[int, int, int]:
	"""Pull (cols, rows, canon) from the upstream op's store, falling back
	to single-tile behavior if the upstream did not publish atlas_grid."""
	if top_input is None:
		return 1, 1, 0
	grid = None
	try:
		grid = top_input.fetch('atlas_grid', None)
	except Exception:
		grid = None
	if grid and len(grid) == 3:
		return int(grid[0]), int(grid[1]), int(grid[2])
	# Single-tile fallback: whole image is one shape.
	try:
		h = int(top_input.height)
		w = int(top_input.width)
	except Exception:
		h = w = 0
	return 1, 1, max(h, w)


def _resolve_num_people(top_input, default: int) -> int:
	if top_input is None:
		return 0
	try:
		n = top_input.fetch('num_people', None)
	except Exception:
		n = None
	if n is None:
		return default
	return int(n)


def _tile_bin(img_u8: np.ndarray, cols: int, rows: int, canon: int,
              slot: int, thresh_u8: int) -> np.ndarray:
	"""Return the slot-i tile from an atlas as a uint8 0/255 binary mask."""
	if canon <= 0:
		# Single-tile fallback: re-threshold the whole image.
		_, t = cv2.threshold(img_u8, thresh_u8, 255, cv2.THRESH_BINARY)
		return t
	row = slot // cols
	col = slot % cols
	y0 = row * canon
	x0 = col * canon
	h, w = img_u8.shape[:2]
	y1 = min(y0 + canon, h)
	x1 = min(x0 + canon, w)
	tile = img_u8[y0:y1, x0:x1]
	_, t = cv2.threshold(tile, thresh_u8, 255, cv2.THRESH_BINARY)
	return t


def _log_hu(h: np.ndarray) -> np.ndarray:
	return -np.sign(h) * np.log10(np.abs(h) + 1e-30)


def _hu_score_pair(live_bin: np.ndarray, ref_bin: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
	"""Compute the existing sum-abs-diff Hu score for one tile pair plus the
	per-tile log-scaled Hu vectors (useful for downstream debugging)."""
	if cv2.countNonZero(live_bin) == 0 or cv2.countNonZero(ref_bin) == 0:
		empty = np.zeros(7, dtype=np.float64)
		return float('inf'), empty, empty
	hu_live = cv2.HuMoments(cv2.moments(live_bin)).flatten()
	hu_ref  = cv2.HuMoments(cv2.moments(ref_bin)).flatten()
	hu_live_log = _log_hu(hu_live)
	hu_ref_log  = _log_hu(hu_ref)
	score = float(np.sum(np.abs(hu_live_log - hu_ref_log)))
	return score, hu_live_log, hu_ref_log


def onCook(scriptOp: scriptTOP):
	"""
	Called when the Script TOP needs to cook.
	"""
	# Need both inputs to do a comparison
	if len(scriptOp.inputs) < 2:
		blank = np.zeros((2, 2, 4), dtype=np.uint8)
		scriptOp.copyNumpyArray(blank)
		scriptOp.store('match_score', float('inf'))
		scriptOp.store('match_scores', [])
		scriptOp.store('num_pairs', 0)
		return

	live_in = scriptOp.inputs[0]
	ref_in  = scriptOp.inputs[1]

	live = live_in.numpyArray(delayed=False)
	ref  = ref_in.numpyArray(delayed=False)

	thresh = float(scriptOp.par.Threshold.eval())
	thresh_u8 = int(max(0.0, min(1.0, thresh)) * 255)

	live_u8 = (live[:, :, 0] * 255).astype(np.uint8)
	ref_u8  = (ref[:,  :, 0] * 255).astype(np.uint8)

	# Resolve atlas geometry from upstream stores (falls back to single-tile).
	cols_l, rows_l, canon_l = _resolve_grid(live_in)
	cols_r, rows_r, canon_r = _resolve_grid(ref_in)

	# Use the matched grid; if they differ, prefer the smaller canon and the
	# smaller cols/rows so we never read past either input.
	cols = min(cols_l, cols_r)
	rows = min(rows_l, rows_r)
	canon = min(canon_l, canon_r) if (canon_l > 0 and canon_r > 0) else 0

	# How many slots are actually populated on each side.
	max_slot = max(1, cols * rows)
	n_live = max(0, min(_resolve_num_people(live_in, 1), max_slot))
	n_ref  = max(0, min(_resolve_num_people(ref_in,  1), max_slot))
	num_pairs = min(n_live, n_ref)

	scores: list[float] = []
	hu_live_all: list[list[float]] = []
	hu_ref_all:  list[list[float]] = []

	for i in range(num_pairs):
		live_bin = _tile_bin(live_u8, max(cols, 1), max(rows, 1), canon, i, thresh_u8)
		ref_bin  = _tile_bin(ref_u8,  max(cols, 1), max(rows, 1), canon, i, thresh_u8)
		s, hu_l, hu_r = _hu_score_pair(live_bin, ref_bin)
		scores.append(s)
		hu_live_all.append(hu_l.tolist())
		hu_ref_all.append(hu_r.tolist())

	# Aggregate: mean of finite per-slot scores; INF if no valid pair.
	finite_scores = [s for s in scores if np.isfinite(s)]
	if finite_scores:
		aggregate = float(np.mean(finite_scores))
	else:
		aggregate = float('inf')

	interpret_score(aggregate)

	scriptOp.store('match_score', aggregate)
	scriptOp.store('match_scores', scores)
	scriptOp.store('num_pairs', num_pairs)
	scriptOp.store('hu_live', hu_live_all)
	scriptOp.store('hu_ref',  hu_ref_all)

	# Output an image (TD requires it). Toggle which one to view.
	if scriptOp.par.Passthroughref.eval():
		scriptOp.copyNumpyArray(ref)
	else:
		scriptOp.copyNumpyArray(live)
	return


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.

	Return one of the following:
		CookLevel.AUTOMATIC - inputs changed and output being used. 
							 TD default behavior.
		CookLevel.ON_CHANGE - inputs changed, output used or not.
		CookLevel.WHEN_USED - every frame when output is being used
		CookLevel.ALWAYS - every frame
	"""

	return CookLevel.AUTOMATIC

def interpret_score(score: float, *, always_print: bool = False) -> None:
	"""
	Print a human-readable interpretation of the Hu-moment match score.

	By default, only prints when the score crosses into a new band so the
	textport stays readable. Set always_print=True to print every cook.

	Score bands (lower = better match):
		< 0.5     MATCH     - essentially the same shape
		0.5 - 2   CLOSE     - same pose, minor differences
		2 - 5     SIMILAR   - similar but clearly different pose
		5 - 15    DIFFERENT - different pose / very different silhouette
		> 15      INVALID   - one side likely empty / noise / garbage
	"""
	global _last_band

	if score < 0.5:
		band, label = 'MATCH',     'Essentially the same shape'
	elif score < 2.0:
		band, label = 'CLOSE',     'Same pose, minor differences'
	elif score < 5.0:
		band, label = 'SIMILAR',   'Similar but clearly different pose'
	elif score < 15.0:
		band, label = 'DIFFERENT', 'Different pose / very different silhouette'
	else:
		band, label = 'INVALID',   'One side likely empty / noise / garbage'

	if not always_print and band == _last_band:
		return

	bars = {
		'MATCH':     '#####',
		'CLOSE':     '####.',
		'SIMILAR':   '###..',
		'DIFFERENT': '##...',
		'INVALID':   '.....',
	}[band]

	print(f"[{bars}] {band:<9} | score={score:6.2f} | {label}")
	_last_band = band
