// Auto-crop a thresholded silhouette to the bounding box of its white pixels.
//
// Input 0  : single-channel-ish mask (channel 0 is read). White = subject.
// Output   : the bbox of the white region, fit into the output frame with
//            aspect ratio preserved (letterboxed in black).
//
// How it works
//   1. Scan the input on a SCAN_RES x SCAN_RES grid to find min/max UV of
//      pixels above kThreshold. This is the silhouette's bounding box.
//   2. Pad the bbox slightly so the crop isn't flush against the silhouette.
//   3. Fit the bbox into the output frame preserving aspect (no stretch),
//      then sample the input at the remapped UV. Anything outside the
//      letterboxed region is black.
//
// Performance
//   Cost is O(SCAN_RES^2) texture taps per output pixel. SCAN_RES = 64 is
//   ~4k taps per pixel, fine for static reference images that only re-cook
//   when the source changes. For a live, every-frame input drop SCAN_RES
//   to ~24-32, or — better — compute the bbox once with an analyzeTOP
//   ("Bounds Min" / "Bounds Max") and feed it in as uniforms (see the
//   commented uniform block at the bottom of this file).

#define SCAN_RES 64

const float kThreshold = 0.5;   // mask cutoff: pixels with .r above this count as "white"
const float kPadding   = 0.01;  // bbox padding as a fraction of bbox size per side

out vec4 fragColor;

void main()
{
	vec2 minUV = vec2( 2.0);
	vec2 maxUV = vec2(-1.0);

	for (int y = 0; y < SCAN_RES; ++y) {
		for (int x = 0; x < SCAN_RES; ++x) {
			vec2 uv = (vec2(x, y) + 0.5) / float(SCAN_RES);
			if (texture(sTD2DInputs[0], uv).r > kThreshold) {
				minUV = min(minUV, uv);
				maxUV = max(maxUV, uv);
			}
		}
	}

	// No white pixels found — emit black so downstream ops see an empty mask.
	if (maxUV.x < minUV.x) {
		fragColor = TDOutputSwizzle(vec4(0.0, 0.0, 0.0, 1.0));
		return;
	}

	// Pad and clamp the bbox.
	vec2 pad = (maxUV - minUV) * kPadding;
	minUV = clamp(minUV - pad, vec2(0.0), vec2(1.0));
	maxUV = clamp(maxUV + pad, vec2(0.0), vec2(1.0));

	// Bbox size in input pixels (so we can preserve aspect when fitting).
	vec2 inRes  = vec2(textureSize(sTD2DInputs[0], 0));
	vec2 outRes = uTDOutputInfo.res.zw;
	vec2 bboxPx = (maxUV - minUV) * inRes;

	// Uniform scale that fits the bbox inside the output frame (contain).
	float scale  = min(outRes.x / bboxPx.x, outRes.y / bboxPx.y);
	vec2  fitPx  = bboxPx * scale;
	vec2  margin = (outRes - fitPx) * 0.5;   // letterbox margins in output px

	// Map current output pixel into bbox-local UV [0,1], or mark out-of-frame.
	vec2 fragPx = vUV.st * outRes;
	vec2 local  = (fragPx - margin) / fitPx;

	if (any(lessThan(local, vec2(0.0))) || any(greaterThan(local, vec2(1.0)))) {
		fragColor = TDOutputSwizzle(vec4(0.0, 0.0, 0.0, 1.0));
		return;
	}

	vec2 cropUV = mix(minUV, maxUV, local);
	fragColor   = TDOutputSwizzle(texture(sTD2DInputs[0], cropUV));
}

// ---------------------------------------------------------------------------
// Faster alternative for live input: compute the bbox in the network with an
// analyzeTOP and pass it in here. Replace the scan loop with these uniforms:
//
//   uniform vec2 uBBoxMin;   // minUV from analyzeTOP "Bounds Min" (red/green)
//   uniform vec2 uBBoxMax;   // maxUV from analyzeTOP "Bounds Max"
//
// then delete the for-loops and use uBBoxMin/uBBoxMax in place of minUV/maxUV.
