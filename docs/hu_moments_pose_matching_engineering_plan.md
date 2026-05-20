# Engineering Plan: Pose-Matching Prototype Using Hu Moments

**Project:** Silhouette-based pose matching for live camera installation
**Method:** Hu Moments via `cv2.matchShapes`
**Hardware:** Orbbec Femto Bolt (depth + RGB)
**Status:** Prototype phase

---

## 1. Goal

Build a working prototype that determines whether a person standing in front of a depth camera is roughly matching the gross shape of a reference silhouette image. The system should produce a single numerical "match score" and a binary "matched / not matched" decision at video framerate (~30 fps).

The prototype is intentionally scoped to **general shape matching, not precise pose matching**. Exact limb angles do not need to align — only the overall silhouette envelope (e.g. "arms up in a Y", "one arm raised", "crouching with arms out").

---

## 2. Success Criteria

The prototype is successful when:

1. A live silhouette is extracted from the Femto Bolt depth stream at ≥25 fps with stable edges.
2. A Hu moments match score is computed every frame against a small library of reference silhouettes (3–5 poses).
3. Players intentionally posing in one of the reference shapes consistently score below the "match" threshold for that pose, and not for other poses.
4. Casual movement (walking past, standing neutral) does not trigger false matches.
5. The match score is stable enough to drive a UI state (no rapid flicker between matched/not matched).

---

## 3. System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Femto Bolt     │───▶│  Silhouette      │───▶│  Normalization  │
│  (depth stream) │    │  Extraction      │    │  & Cleanup      │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌────────▼────────┐
│  Reference      │───▶│  Hu Moments      │◀───│  Live Hu        │
│  Silhouettes    │    │  Comparison      │    │  Moments        │
│  (pre-computed) │    │  (matchShapes)   │    │  (per-frame)    │
└─────────────────┘    └────────┬─────────┘    └─────────────────┘
                                │
                       ┌────────▼─────────┐
                       │  Match Decision  │
                       │  + UI Feedback   │
                       └──────────────────┘
```

Five logical stages: depth capture, silhouette extraction, normalization, Hu moments comparison, and match decision/feedback. Each is independently testable.

---

## 4. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Fastest iteration for CV prototyping; OpenCV bindings are mature |
| Camera SDK | pyorbbecsdk (or k4a-python) | Official Femto Bolt access from Python |
| CV library | OpenCV 4.x | `cv2.matchShapes`, `cv2.moments`, `cv2.HuMoments` are built in |
| Numerical | NumPy | Array ops, normalization math |
| Display/UI | OpenCV `imshow` initially → upgrade later | Avoid framework overhead during algorithm tuning |
| Optional | TouchDesigner integration | Final installation runtime; defer until algorithm proves out |

Keep the prototype in pure Python with `cv2.imshow` for the first milestone. Integration into the installation runtime is a separate concern once the algorithm is validated.

---

## 5. Build Phases

### Phase 1 — Offline Reference Pipeline (Day 1)

Build the script that processes reference silhouette images (like the `people.jpg` example) into a precomputed library of Hu descriptors.

**Tasks:**
- Load reference PNG/JPG files from a `references/` directory.
- Threshold to binary masks (white silhouettes on black background → binary).
- Extract the largest contour per image (in case of noise or multiple figures, decide handling).
- For multi-figure references like `people.jpg`: treat each figure as a separate reference, OR composite as a single shape. **Decision needed early.**
- Compute `cv2.moments` → `cv2.HuMoments` for each.
- Save descriptors + reference metadata (name, threshold, etc.) to a JSON or pickle file.

**Deliverable:** `build_reference_library.py` produces `references.json` containing `{name, hu_moments, source_image_path}` entries.

**Validation:** Print Hu vectors for each reference; sanity-check that visually different poses produce visibly different vectors.

---

### Phase 2 — Live Silhouette Extraction (Day 2)

Capture from the Femto Bolt and produce a clean binary silhouette of the person.

**Tasks:**
- Initialize Femto Bolt depth stream via pyorbbecsdk.
- Apply depth thresholding: keep only pixels in the range `[min_depth, max_depth]` (e.g. 1.0m–3.5m) — this isolates the person from floor and background.
- Convert to binary mask.
- Apply morphological cleanup: opening (remove speckle), then closing (fill internal holes from depth noise).
- Find the largest connected component — discard everything else.
- Output: a single-channel binary image with the person as white pixels.

**Deliverable:** `silhouette_extractor.py` with a function `extract_silhouette(depth_frame) -> binary_mask`.

**Validation:** Display the live silhouette alongside the depth stream. Walk in/out of frame, raise arms, crouch. The silhouette should be clean, hole-free, and track the body without flicker.

**Known risks:**
- Depth shadows along body edges → tune morphological kernel sizes.
- Floor pixels included when person stands near camera → tighten min_depth or add a floor-plane filter.
- Two people in frame → for prototype, take only the largest blob and document the limitation.

---

### Phase 3 — Normalization (Day 2–3)

Even though Hu moments are translation/scale/rotation invariant by construction, normalization is still recommended for two reasons: it reduces numerical instability with very small or very large silhouettes, and it makes the silhouettes directly comparable for the IoU-based visual feedback layer added later.

**Tasks:**
- Compute bounding box of the live silhouette.
- Compute centroid (from `cv2.moments` — m10/m00, m01/m00).
- Crop to bounding box with a small padding (e.g. 5% on each side).
- Resize cropped silhouette to a canonical size (e.g. 256×256, preserving aspect ratio with letterboxing).
- Apply the same normalization to references at library-build time so both sides are consistent.

**Deliverable:** `normalize.py` with `normalize_silhouette(mask) -> normalized_mask`.

**Validation:** A person at different positions and distances from the camera should produce nearly identical normalized silhouettes when in the same pose.

---

### Phase 4 — Hu Moments Comparison (Day 3)

The core matching logic.

**Tasks:**
- Per frame, compute Hu moments of the normalized live silhouette.
- For each reference in the library, call `cv2.matchShapes(live_mask, ref_mask, cv2.CONTOURS_MATCH_I1, 0)`.
- Identify the reference with the lowest score (best match).
- Apply a threshold: if best score < `MATCH_THRESHOLD`, declare a match.

**Note:** `cv2.matchShapes` can take either contours or binary images; passing the masks directly is simplest. Test all three methods (`I1`, `I2`, `I3`) and pick the one that gives the cleanest separation between matching and non-matching poses on your reference set.

**Deliverable:** `pose_matcher.py` with `match_pose(live_mask, reference_library) -> (best_match_name, score, all_scores)`.

**Validation:** Pose intentionally in each reference shape; verify the correct reference scores lowest. Pose in nothing in particular; verify all scores are well above the threshold.

---

### Phase 5 — Temporal Smoothing & Match Decision (Day 4)

Raw per-frame scores will flicker. Smooth them before driving any UI state.

**Tasks:**
- Maintain a rolling window of the last N frames of scores per reference (N = 5–10).
- Use the median (not mean — more robust to single-frame outliers) as the smoothed score.
- Apply hysteresis on the match decision: require score to drop below `ENTER_THRESHOLD` to declare matched, and rise above a higher `EXIT_THRESHOLD` to declare unmatched. Prevents rapid toggling at the boundary.
- Optionally: require the match to be stable for K consecutive frames before triggering downstream events.

**Deliverable:** `match_state.py` encapsulating the smoothing and hysteresis logic.

---

### Phase 6 — Visualization & Tuning Harness (Day 4–5)

A debug UI for tuning thresholds and verifying behavior.

**Tasks:**
- OpenCV window showing: live silhouette, current best-match reference name, all scores as a bar chart, threshold lines, smoothed vs raw score.
- Keyboard shortcuts to cycle reference library, adjust thresholds live, save current frame as a new reference.
- FPS counter.

**Deliverable:** `debug_app.py` — single-file runnable prototype that ties all the above together.

---

## 6. Threshold Tuning Methodology

Hu moments scores are unbounded and depend on the matching method chosen. Empirical tuning required.

**Procedure:**
1. With the debug harness running, perform each reference pose deliberately for ~30 seconds. Log all scores.
2. Perform random, non-matching poses for ~30 seconds. Log all scores.
3. Plot histograms of "matching" vs "non-matching" scores per reference.
4. Choose `MATCH_THRESHOLD` at the valley between the two distributions, biased toward the non-matching side (false negatives are better than false positives in installation contexts — players will retry, but a false trigger breaks immersion).

Expected ranges based on prior experience with `matchShapes I1`: matches typically score <0.1, clearly non-matches >0.3. Your numbers will vary; trust the data.

---

## 7. Risks & Open Questions

| Risk | Likelihood | Mitigation |
|---|---|---|
| Two reference poses produce similar Hu signatures | Medium | At library-build time, compute pairwise `matchShapes` scores between all references. If any pair scores below 0.15, redesign that reference. |
| Mirror-image ambiguity (left arm up vs right arm up) | High | Hu moments are rotation-invariant, which includes reflection. **Verify this for your specific reference set.** If problematic, augment with a non-rotation-invariant secondary check (e.g. centroid-of-mass-relative-to-bbox-center on x-axis). |
| Multi-person handling (e.g. the 3-figure reference image) | High | **Decision needed:** does the live feed require multiple people simultaneously, or only one player at a time? This changes both the silhouette extraction and the reference design fundamentally. |
| Depth noise at body edges destabilizes Hu values | Medium | Aggressive morphological cleanup; temporal smoothing of the binary mask itself before computing moments. |
| Lighting changes affect Femto Bolt depth | Low | Depth is largely lighting-invariant (IR-based), but verify in the actual installation environment. |

---

## 8. Out of Scope for This Prototype

Explicitly deferred to follow-up work once the core algorithm is validated:

- IoU-based visual feedback overlay (player sees green/red silhouette overlap)
- Multi-person matching
- Fourier descriptors fallback (only if Hu moments prove insufficient)
- TouchDesigner integration
- Production UI / installation visuals
- Audio feedback
- Score logging / analytics

---

## 9. File Structure

```
pose_matching_prototype/
├── references/
│   ├── pose_y.png
│   ├── pose_arms_out.png
│   └── pose_crouch.png
├── lib/
│   ├── silhouette_extractor.py
│   ├── normalize.py
│   ├── pose_matcher.py
│   └── match_state.py
├── build_reference_library.py
├── debug_app.py
├── references.json          # generated
├── requirements.txt
└── README.md
```

---

## 10. Milestones

| Day | Deliverable | Definition of Done |
|---|---|---|
| 1 | Reference library builder | `references.json` contains valid Hu descriptors for 3+ poses |
| 2 | Live silhouette extraction | Stable binary silhouette displayed from Femto Bolt at ≥25 fps |
| 3 | End-to-end matching loop | Score printed every frame for each reference; correct reference scores lowest when posing |
| 4 | Smoothed match decisions | UI displays a stable "matched / not matched" state |
| 5 | Tuning harness + threshold calibration | Documented thresholds; demo of all reference poses triggering correctly |

---

## 11. Decisions Required Before Starting

1. **Single-person or multi-person matching?** Drives extractor design.
2. **How are reference silhouettes authored?** Photographed and traced, generated programmatically, or hand-drawn?
3. **What downstream system consumes the match signal?** OSC to TouchDesigner, direct rendering, MQTT, etc. — affects how Phase 5 outputs are exposed.
4. **Target install environment lighting / floor / camera placement** — affects depth thresholding defaults.
