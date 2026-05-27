"""
Script TOP Callbacks - Per-person YOLO bbox grid.

me - this DAT
scriptOp - the OP which is cooking

PURPOSE
-------
Sits between the black-and-white silhouette mask and `pose_preprocess.py`.
Its job is to **unfuse** people who are touching in the silhouette by
running a YOLO person detector, cropping each detected person out of the
silhouette using their bounding box, and pasting the crops into a uniform
grid separated by black gutters. `pose_preprocess.py` then sees N already-
separated white blobs (one per grid cell) instead of one big fused one,
and its existing connected-components / normalize pipeline produces a
clean canonical atlas downstream.

WHY THIS EXISTS
---------------
`pose_preprocess.py` already has logic to split fused person-blobs via
aspect / solidity / vertical-pinch heuristics and a distance-transform
watershed (see `_is_fused` / `_split_fused` in pose_preprocess.py). Those
heuristics are fragile: two people of different sizes touching at the
hips, or overlapping silhouettes from a top-down camera, can defeat them.
A real person detector is a far more robust upstream pre-split.

PIPELINE POSITION
-----------------
    camera -> threshold mask -> [THIS OP] -> pose_preprocess.py
        -> per-person atlas -> pose_match_iou.py

WHAT THIS OP IS NOT
-------------------
- It does NOT produce the canonical pose_preprocess atlas (no centroid
  alignment, no aspect normalization to a square canon).
- It does NOT publish `atlas_grid` / `num_people` / `slot_bboxes` /
  `slot_centroids` / `slot_areas`. Those keys are pose_preprocess.py's
  contract; we leave them alone so pose_match_iou.py keeps reading them
  from pose_preprocess exactly like today.

INPUT
-----
One TOP input: the black-and-white silhouette mask (white = subject on
black). Only the red channel is read, matching pose_preprocess.py's
convention. The mask is stacked to 3 identical channels before being
fed to YOLO so the standard RGB-trained detector accepts it.

OUTPUT
------
RGBA uint8 atlas. Layout:
    cols = ceil(sqrt(Maxpeople))
    rows = ceil(Maxpeople / cols)
    H    = rows * Cellsize + (rows + 1) * Gutterpx
    W    = cols * Cellsize + (cols + 1) * Gutterpx
Each populated cell holds one bbox-cropped silhouette, aspect-preserving
resized into Cellsize x Cellsize and letterboxed with black. RGB channels
all carry the same binarized mask, alpha is always 255. Empty cells stay
fully black RGB with alpha 255.

DIAGNOSTIC STORES (read via `op('pose_segmentation1').fetch(...)`)
------------------------------------------------------------------
    det_count           int, number of accepted detections this cook
    det_bboxes          [[x1, y1, x2, y2], ...] in source-mask px,
                        post bbox-pad, clamped to image bounds
    det_confidences     [float, ...] YOLO confidence per detection
    seg_grid            [cols, rows, cell, gutter] grid layout

NOTE on Gutterpx
----------------
pose_preprocess.py's morphological close kernel is sized as a percentage
of the image diagonal (default `Closepct` = 1.5%). On a typical grid
output (~1000 px diagonal) that's ~15 px, so the default Gutterpx here
is 32 to guarantee the close cannot bridge two cells. If you lower
Closepct downstream you can lower Gutterpx; if you raise it, raise the
gutter or your cells will fuse back together.

CAVEATS
-------
- YOLO is trained on RGB natural images. Stacked grayscale silhouettes
  work in practice for clean full-body standing people, but accuracy
  degrades on partial silhouettes or unusual poses. If detection is
  unreliable, the planned fallback is to run YOLO on the raw RGB feed
  (option 2 in the design discussion) and apply those bboxes to the
  silhouette - the cropping/packing logic below is unchanged.
- Overlapping bboxes mean a slice of the other person's silhouette will
  appear inside a crop. pose_preprocess's morphological close + min-area
  filter will usually clean small overlaps; if it doesn't, that's
  another reason to upgrade to RGB detection (or YOLO-seg) later.
"""

import math
import numpy as np
import cv2

try:
	from ultralytics import YOLO
	_ULTRALYTICS_AVAILABLE = True
except Exception as _e:
	# Keep the file importable even if ultralytics isn't installed in this
	# Python environment yet; the cook will surface a clear error.
	_ULTRALYTICS_AVAILABLE = False
	_ULTRALYTICS_ERR = _e


# ---------------------------------------------------------------------------
# Module-level singletons: load YOLO weights once, survive recooks.
# ---------------------------------------------------------------------------

_model = None
_model_name = None
_model_device = None


# ---------------------------------------------------------------------------
# Parameter page
# ---------------------------------------------------------------------------

def onSetupParameters(scriptOp: scriptTOP):
	"""
	Called to setup custom parameters for the Script TOP.

	Parameter reference:
	    Model         YOLO weights stem (no `.pt`). Default "yolov8n" -
	                  the smallest stock detector. "yolo11n" also works.
	                  Detection-only is enough; we just need bboxes.
	    Device        "cuda:0" for GPU, "cpu" for CPU inference.
	    Loadmodel     Pulse: (re)load the YOLO weights with the current
	                  Model + Device.
	    Confidence    YOLO `conf` threshold. Default is lower than YOLO's
	                  usual 0.25 because silhouette detection is harder
	                  than RGB detection.
	    Iou           YOLO NMS IoU threshold.
	    Imagesize     YOLO `imgsz` for inference.
	    Maxpeople     Hard cap on grid slots. Drives grid dimensions.
	    Cellsize      Per-cell side length in px on the output atlas.
	    Gutterpx      Black gutter between cells (and around the border).
	                  Must be wide enough that pose_preprocess.py's
	                  morphological close cannot bridge two cells.
	    Bboxpadpct    Expand YOLO bbox by this percent of its own width/
	                  height before cropping. Compensates for YOLO
	                  trimming limb tips.
	    Sortby        Slot ordering: left-to-right (centroid x), largest
	                  bbox first, or highest confidence first.
	"""
	page = scriptOp.appendCustomPage('Pose Segmentation')

	p = page.appendStr('Model', label='Model')
	p[0].default = 'yolov8n'

	m = page.appendMenu('Device', label='Device')
	m[0].menuNames = ['cuda:0', 'cpu']
	m[0].menuLabels = ['GPU (cuda:0)', 'CPU']
	m[0].default = 'cuda:0'

	page.appendPulse('Loadmodel', label='Load Model')

	p = page.appendFloat('Confidence', label='Confidence')
	p[0].default = 0.25
	p[0].normMin = 0.0
	p[0].normMax = 1.0

	p = page.appendFloat('Iou', label='NMS IoU')
	p[0].default = 0.7
	p[0].normMin = 0.0
	p[0].normMax = 1.0

	p = page.appendInt('Imagesize', label='YOLO Imgsz')
	p[0].default = 640
	p[0].min = 32
	p[0].clampMin = True
	p[0].normMin = 320
	p[0].normMax = 1280

	p = page.appendInt('Maxpeople', label='Max People')
	p[0].default = 4
	p[0].min = 1
	p[0].clampMin = True
	p[0].normMin = 1
	p[0].normMax = 8

	p = page.appendInt('Cellsize', label='Cell Size')
	p[0].default = 256
	p[0].min = 16
	p[0].clampMin = True
	p[0].normMin = 64
	p[0].normMax = 512

	p = page.appendInt('Gutterpx', label='Gutter px')
	p[0].default = 32
	p[0].min = 0
	p[0].clampMin = True
	p[0].normMin = 0
	p[0].normMax = 64

	p = page.appendFloat('Bboxpadpct', label='Bbox Pad %')
	p[0].default = 5.0
	p[0].normMin = 0.0
	p[0].normMax = 25.0

	m = page.appendMenu('Sortby', label='Slot Order')
	m[0].menuNames = ['xcent', 'area', 'conf']
	m[0].menuLabels = [
		'Left-to-Right (centroid x)',
		'Largest First (bbox area)',
		'Most Confident First',
	]
	m[0].default = 'xcent'

	return


def onPulse(par):
	"""
	Called when a custom pulse parameter is pushed. Only Loadmodel is
	pulsed today; other params are read at cook time.
	"""
	if par.name == 'Loadmodel':
		_load_model(str(par.owner.par.Model.eval()), str(par.owner.par.Device.eval()))
	return


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

DEFAULT_MODEL = 'yolov8n'


def _resolve_weights(model_name: str) -> str:
	"""Turn whatever the user typed in the Model field into a real weights
	argument that ultralytics.YOLO will accept.

	Rules (in order):
	    - Empty / whitespace only -> fall back to DEFAULT_MODEL (avoids
	      passing ".pt" to YOLO, which is the error this whole branch
	      exists to prevent).
	    - Already ends in .pt (case-insensitive) -> use as-is. Lets the
	      user paste an absolute or relative path to a local checkpoint.
	    - Otherwise -> append ".pt" so "yolov8n" becomes "yolov8n.pt",
	      which Ultralytics will then auto-download from its model zoo.
	"""
	name = (model_name or '').strip().strip('"').strip("'")
	if not name:
		name = DEFAULT_MODEL
	if name.lower().endswith('.pt'):
		return name
	return f'{name}.pt'


def _load_model(model_name: str, device: str) -> bool:
	"""(Re)load YOLO weights into the module-level singleton.

	Returns True on success, False otherwise. We intentionally don't
	raise here: a missing model file should be a soft failure that
	prints to the textport and lets the rest of the network keep cooking
	with an empty atlas.
	"""
	global _model, _model_name, _model_device

	if not _ULTRALYTICS_AVAILABLE:
		print(f'[pose_segmentation] ultralytics not importable: {_ULTRALYTICS_ERR}')
		_model = None
		return False

	weights = _resolve_weights(model_name)

	try:
		_model = YOLO(weights)
		_model_name = model_name
		_model_device = device
		print(f'[pose_segmentation] loaded {weights} on {device}')
		return True
	except Exception as e:
		import traceback
		print(f'[pose_segmentation] failed to load {weights!r}: {e}')
		print(traceback.format_exc())
		_model = None
		_model_name = None
		_model_device = None
		return False


def _ensure_model(model_name: str, device: str) -> bool:
	"""Lazy-load the model on first cook (or when params changed)."""
	global _model
	if _model is None or _model_name != model_name or _model_device != device:
		return _load_model(model_name, device)
	return True


# ---------------------------------------------------------------------------
# Atlas geometry helpers
# ---------------------------------------------------------------------------

def _grid_dims(maxpeople: int) -> tuple[int, int]:
	"""Most-square (cols, rows) layout that holds `maxpeople` cells.

	Same packing rule as pose_preprocess.py so the two grids look visually
	consistent when wired side-by-side for debugging.
	"""
	cols = max(1, int(math.ceil(math.sqrt(maxpeople))))
	rows = max(1, int(math.ceil(maxpeople / float(cols))))
	return cols, rows


def _atlas_shape(maxpeople: int, cell: int, gutter: int) -> tuple[int, int, int, int]:
	"""Return (H, W, cols, rows) for the grid atlas.

	Gutter is added on all sides (cells * gutter + 1 extra), so a single
	person produces a `cell + 2*gutter` square output rather than a bare
	`cell` square. Symmetric borders keep the result visually centered.
	"""
	cols, rows = _grid_dims(maxpeople)
	h = rows * cell + (rows + 1) * gutter
	w = cols * cell + (cols + 1) * gutter
	return h, w, cols, rows


def _empty_atlas(maxpeople: int, cell: int, gutter: int) -> np.ndarray:
	"""All-black RGBA atlas with alpha pre-filled to 255.

	Same opaque-empty convention as pose_preprocess._empty_atlas: empty
	slots are black RGB with full alpha, so downstream ops should treat
	the RGB channels (all zero) as "no person here", not the alpha.
	"""
	h, w, _, _ = _atlas_shape(maxpeople, cell, gutter)
	atlas = np.zeros((h, w, 4), dtype=np.uint8)
	atlas[..., 3] = 255
	return atlas


def _publish_empty(scriptOp, maxpeople: int, cell: int, gutter: int) -> None:
	"""Push an empty atlas + zeroed stores. Used by every fail/no-op branch.

	Keeping output dimensions stable across frames (rather than collapsing
	to 1x1 when there's nothing detected) makes the TOP wiring simpler -
	downstream ops don't have to handle a sudden resolution change.
	"""
	cols, rows = _grid_dims(maxpeople)
	scriptOp.copyNumpyArray(_empty_atlas(maxpeople, cell, gutter))
	scriptOp.store('det_count', 0)
	scriptOp.store('det_bboxes', [])
	scriptOp.store('det_confidences', [])
	scriptOp.store('seg_grid', [cols, rows, cell, gutter])


# ---------------------------------------------------------------------------
# Per-cell crop + fit
# ---------------------------------------------------------------------------

def _fit_cell(crop_u8: np.ndarray, cell: int) -> np.ndarray:
	"""Aspect-preserving resize of a bbox crop into a cell-sized tile.

	Steps:
	  1. Scale so the longer side hits `cell` exactly, shorter side <= cell.
	  2. Letterbox-pad the shorter axis with black, split evenly.
	  3. Re-threshold post-resize because INTER_AREA produces smoothed
	     gray edges; we want a hard 0/255 mask out the other side.

	pose_preprocess.py will normalize again from this tile, so we don't
	bother with centroid alignment here. The point of this op is purely
	to give pose_preprocess a non-fused input.
	"""
	if crop_u8.size == 0:
		return np.zeros((cell, cell), dtype=np.uint8)
	ch, cw = crop_u8.shape[:2]
	if ch == 0 or cw == 0:
		return np.zeros((cell, cell), dtype=np.uint8)
	scale = min(cell / float(cw), cell / float(ch))
	new_w = max(1, int(round(cw * scale)))
	new_h = max(1, int(round(ch * scale)))
	resized = cv2.resize(crop_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
	# Re-threshold after INTER_AREA (which smooths) so the cell stays
	# binary; pose_preprocess re-thresholds anyway but doing it here too
	# keeps the visual debug output crisp.
	_, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
	top = (cell - new_h) // 2
	bottom = cell - new_h - top
	left = (cell - new_w) // 2
	right = cell - new_w - left
	return cv2.copyMakeBorder(resized, top, bottom, left, right,
	                          cv2.BORDER_CONSTANT, value=0)


# ---------------------------------------------------------------------------
# Detection -> per-person records
# ---------------------------------------------------------------------------

def _pad_clamp_bbox(x1: float, y1: float, x2: float, y2: float,
                    pad_pct: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
	"""Expand a YOLO xyxy bbox by pad_pct of its own w/h, then clamp to
	the image rectangle. Returns ints because cropping uses slice indices.

	Padding compensates for YOLO trimming the silhouette right at the
	limb tips, which can lop off fingers/heads on tightly-detected
	people. 5% on each side is usually enough without grabbing the
	neighbor.
	"""
	w = x2 - x1
	h = y2 - y1
	if w <= 0 or h <= 0:
		return 0, 0, 0, 0
	pad = pad_pct / 100.0
	x1 = max(0, int(round(x1 - w * pad)))
	y1 = max(0, int(round(y1 - h * pad)))
	x2 = min(img_w, int(round(x2 + w * pad)))
	y2 = min(img_h, int(round(y2 + h * pad)))
	if x2 <= x1 or y2 <= y1:
		return 0, 0, 0, 0
	return x1, y1, x2, y2


def _detect_persons(sil_u8: np.ndarray, conf: float, iou: float, imgsz: int,
                    device: str, pad_pct: float):
	"""Run YOLO on the (3-channel-stacked) silhouette and return a list of
	person records.

	Each record is a dict with:
	    bbox    (x1, y1, x2, y2) in source-mask px, post-pad, post-clamp
	    conf    float YOLO confidence
	    xcent   float centroid x for sort ordering
	    area    int bbox area for sort ordering

	Returns [] if the model is missing, the input is empty, or YOLO
	produced no detections. Stacking the 1-channel silhouette to 3
	identical channels is the cheapest way to satisfy the standard
	RGB-trained detector's input shape without retraining.
	"""
	if _model is None or sil_u8.size == 0:
		return []

	img_h, img_w = sil_u8.shape[:2]
	# 3-channel stack: cv2.merge accepts a list of single-channel arrays.
	yolo_in = cv2.merge([sil_u8, sil_u8, sil_u8])

	try:
		# Detection-only; classes=[0] is COCO "person". We intentionally
		# don't use track() here because slot ordering is handled by
		# Sortby. If we add cross-frame stability later we'll switch to
		# model.track(persist=True) and add 'id' to the record.
		res = _model.predict(
			source=yolo_in,
			classes=[0],
			conf=float(conf),
			iou=float(iou),
			imgsz=int(imgsz),
			device=device,
			verbose=False,
		)
	except Exception as e:
		import traceback
		print(f'[pose_segmentation] YOLO inference failed: {e}')
		print(traceback.format_exc())
		return []

	if not res or res[0].boxes is None or len(res[0].boxes) == 0:
		return []

	boxes = res[0].boxes
	# `.cpu().numpy()` materializes the tensor regardless of device so
	# the rest of the code stays plain numpy.
	xyxy = boxes.xyxy.cpu().numpy()
	confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))

	persons = []
	for (x1, y1, x2, y2), c in zip(xyxy, confs):
		x1p, y1p, x2p, y2p = _pad_clamp_bbox(float(x1), float(y1), float(x2), float(y2),
		                                    pad_pct, img_w, img_h)
		if x2p <= x1p or y2p <= y1p:
			continue
		persons.append({
			'bbox': (x1p, y1p, x2p, y2p),
			'conf': float(c),
			'xcent': (x1p + x2p) / 2.0,
			'area': int((x2p - x1p) * (y2p - y1p)),
		})
	return persons


def _sort_persons(persons: list, mode: str) -> list:
	"""Order person records by the chosen sort mode.

	'xcent' keeps slot indices stable when subjects don't cross paths;
	'area' prioritizes the dominant subject when Maxpeople caps us off;
	'conf' surfaces the detector's most-trusted people first (useful
	for noisy silhouettes).
	"""
	if mode == 'area':
		return sorted(persons, key=lambda p: -p['area'])
	if mode == 'conf':
		return sorted(persons, key=lambda p: -p['conf'])
	return sorted(persons, key=lambda p: p['xcent'])


# ---------------------------------------------------------------------------
# Main cook
# ---------------------------------------------------------------------------

def onCook(scriptOp: scriptTOP):
	"""
	Called when the Script TOP needs to cook.

	Phases:
	    A. Defensive paths: no input wired, ultralytics missing, model
	       not loaded -> emit an empty atlas and zeroed stores.
	    B. Pull the silhouette (red channel, promoted to uint8).
	    C. Lazy-load YOLO weights if needed.
	    D. Detect persons on the stacked-3ch silhouette, pad+clamp bboxes.
	    E. Sort persons by Sortby and cap at Maxpeople.
	    F. For each kept person, crop the silhouette by bbox and fit
	       into a Cellsize x Cellsize tile.
	    G. Paste tiles into the gutter-separated grid atlas.
	    H. Publish atlas + light diagnostic stores.
	"""
	# Snapshot params up front so the cook is consistent even if a
	# slider wiggles mid-cook (same defensive pattern as pose_preprocess).
	maxpeople = max(1, int(scriptOp.par.Maxpeople.eval()))
	cell = max(16, int(scriptOp.par.Cellsize.eval()))
	gutter = max(0, int(scriptOp.par.Gutterpx.eval()))
	conf = float(scriptOp.par.Confidence.eval())
	iou = float(scriptOp.par.Iou.eval())
	imgsz = max(32, int(scriptOp.par.Imagesize.eval()))
	pad_pct = float(scriptOp.par.Bboxpadpct.eval())
	sort_mode = str(scriptOp.par.Sortby.eval())
	model_name = str(scriptOp.par.Model.eval()).strip()
	device = str(scriptOp.par.Device.eval())

	# (A) No input wired -> still publish a valid empty atlas so
	# downstream sizing stays stable.
	if len(scriptOp.inputs) < 1:
		_publish_empty(scriptOp, maxpeople, cell, gutter)
		return

	# (A cont.) ultralytics not importable -> empty + warning.
	if not _ULTRALYTICS_AVAILABLE:
		_publish_empty(scriptOp, maxpeople, cell, gutter)
		return

	# (B) Pull silhouette. TouchDesigner gives float32 RGBA in [0, 1];
	# OpenCV / YOLO want uint8. Only the red channel is read, matching
	# pose_preprocess.py's input convention so this op is wire-compatible
	# with whatever already feeds pose_preprocess today.
	src = scriptOp.inputs[0].numpyArray(delayed=False)
	if src is None or src.size == 0:
		_publish_empty(scriptOp, maxpeople, cell, gutter)
		return
	sil_u8 = (src[:, :, 0] * 255).astype(np.uint8)

	# (C) Lazy model load. Soft fail keeps the rest of the network
	# cooking; the user can fix the weights file and pulse Loadmodel.
	if not _ensure_model(model_name, device):
		_publish_empty(scriptOp, maxpeople, cell, gutter)
		return

	# (D) Detect.
	persons = _detect_persons(sil_u8, conf, iou, imgsz, device, pad_pct)

	# (E) Sort + cap.
	persons = _sort_persons(persons, sort_mode)
	persons = persons[:maxpeople]

	# (F/G) Build atlas and paste cropped, fit cells.
	atlas = _empty_atlas(maxpeople, cell, gutter)
	cols, rows = _grid_dims(maxpeople)
	for idx, p in enumerate(persons):
		x1, y1, x2, y2 = p['bbox']
		# Slice the silhouette; numpy slices are bounds-safe because we
		# already clamped the bbox in `_pad_clamp_bbox`.
		crop = sil_u8[y1:y2, x1:x2]
		tile = _fit_cell(crop, cell)
		r = idx // cols
		c = idx % cols
		y0 = gutter + r * (cell + gutter)
		x0 = gutter + c * (cell + gutter)
		# Replicate the binary tile into R/G/B; alpha was pre-filled to
		# 255 in _empty_atlas. Downstream consumers (pose_preprocess)
		# read the red channel, so any of the three channels would
		# suffice, but writing all three makes the TOP visually
		# inspectable in a regular Null TOP.
		atlas[y0:y0 + cell, x0:x0 + cell, 0] = tile
		atlas[y0:y0 + cell, x0:x0 + cell, 1] = tile
		atlas[y0:y0 + cell, x0:x0 + cell, 2] = tile

	# (H) Publish.
	scriptOp.copyNumpyArray(atlas)
	scriptOp.store('det_count', len(persons))
	scriptOp.store('det_bboxes', [list(p['bbox']) for p in persons])
	scriptOp.store('det_confidences', [p['conf'] for p in persons])
	scriptOp.store('seg_grid', [cols, rows, cell, gutter])
	return


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.
	"""
	return CookLevel.AUTOMATIC
