"""
Script TOP Callbacks — Pose matching via cv2.matchShapes()

me - this DAT
scriptOp - the OP which is cooking

Inputs:
	0 - Live thresholded silhouette (single-channel-ish; channel 0 is read)
	1 - Reference pose silhouette (same)

Output:
	Live or reference image passthrough (toggle on the OP).
	Match score is published via scriptOp.store('match_score', ...).
"""

import numpy as np
import cv2

# Tracks last interpretation band so we only print on changes
_last_band: str | None = None

# Maps the Match Method menu to the OpenCV constant
_METHOD_MAP = {
	'I1': cv2.CONTOURS_MATCH_I1,
	'I2': cv2.CONTOURS_MATCH_I2,
	'I3': cv2.CONTOURS_MATCH_I3,
}

# Band cutoffs per method. Lower score = better match.
# I2 is what the hand-rolled Hu script effectively computed, so its bands
# match the values you're already used to. I1 and I3 produce different
# magnitudes — these defaults are reasonable starting points; recalibrate
# by watching live values for your own setup.
_BANDS = {
	'I1': (0.05, 0.20, 0.50, 2.00),
	'I2': (0.50, 2.00, 5.00, 15.00),
	'I3': (0.05, 0.20, 0.50, 2.00),
}


# press 'Setup Parameters' in the OP to call this function to re-create
# the parameters.
def onSetupParameters(scriptOp: scriptTOP):
	"""
	Called to setup custom parameters for the Script TOP.
	"""
	page = scriptOp.appendCustomPage('Pose Match')

	m = page.appendMenu('Method', label='Match Method')
	m[0].menuNames  = ['I1', 'I2', 'I3']
	m[0].menuLabels = ['I1', 'I2', 'I3']
	m[0].default    = 'I2'

	page.appendToggle('Passthroughref', label='Show Reference')
	return


def onPulse(par: Par):
	"""
	Called when a custom pulse parameter is pushed.

	Args:
		par: The parameter that was pulsed
	"""
	return


def onCook(scriptOp: scriptTOP):
    """
    Called when the Script TOP needs to cook.
    """
    # Need both inputs to do a comparison
    if len(scriptOp.inputs) < 2:
        blank = np.zeros((2, 2, 4), dtype=np.uint8)
        scriptOp.copyNumpyArray(blank)
        return

    # numpyArray() returns float32 RGBA in 0..1
    live = scriptOp.inputs[0].numpyArray(delayed=False)
    ref  = scriptOp.inputs[1].numpyArray(delayed=False)

    # Network already binarized upstream — just pull channel 0 as uint8
    live_bin = (live[:, :, 0] * 255).astype(np.uint8)
    ref_bin  = (ref[:,  :, 0] * 255).astype(np.uint8)

    # Resolve the chosen comparison method
    method_name = str(scriptOp.par.Method.eval())
    method_const = _METHOD_MAP.get(method_name, cv2.CONTOURS_MATCH_I2)

    # Bail cleanly if either silhouette is empty (matchShapes would error)
    if cv2.countNonZero(live_bin) == 0 or cv2.countNonZero(ref_bin) == 0:
        score = float('inf')
    else:
        score = float(cv2.matchShapes(live_bin, ref_bin, method_const, 0.0))

    # interpret_score(score, method_name, always_print=True)
    print(f"Match: {1.0 / (1.0 + score):.3f}   (raw: {score:.4f})")
    # print(f"Method: {method_name}")

    # Stash for downstream ops to fetch()
    scriptOp.store('match_score', score)
    scriptOp.store('match_method', method_name)

    # Output an image (TD requires it). Toggle which one to view.
    if scriptOp.par.Passthroughref.eval():
        scriptOp.copyNumpyArray(ref)
    else:
        scriptOp.copyNumpyArray(live)
    return

    # scriptOp.copyNumpyArray(live_bin)


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


def interpret_score(score: float, method: str = 'I2', *, always_print: bool = False) -> None:
	"""
	Print a human-readable interpretation of a cv2.matchShapes() score.

	Only prints when the score crosses into a new band so the textport
	stays readable. Pass always_print=True to print every cook.

	Bands are method-specific (see _BANDS at top of file). Lower = better.
	"""
	global _last_band

	t1, t2, t3, t4 = _BANDS.get(method, _BANDS['I2'])

	if score < t1:
		band, label = 'MATCH',     'Essentially the same shape'
	elif score < t2:
		band, label = 'CLOSE',     'Same pose, minor differences'
	elif score < t3:
		band, label = 'SIMILAR',   'Similar but clearly different pose'
	elif score < t4:
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

	print(f"[{bars}] {band:<9} | method={method} | score={score:6.2f} | {label}")
	_last_band = band
