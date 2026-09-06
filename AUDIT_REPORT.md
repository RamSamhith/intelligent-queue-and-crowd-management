# VisionQueue — Independent Adversarial Audit Report

**Audit Date:** 2026-09-01  
**Auditor:** AI Engineering Assistant (Read-Only Forensic Review)  
**Scope:** Complete CV pipeline — camera → detection → tracking → counting → analytics → reliability → LiveState  
**Repository State:** 350 tests passing, YOLO26m ONNX, ByteTrack, RTX 5060 Laptop GPU, Windows, Python 3.11

---

## A. Architecture Understanding

The VisionQueue CV subsystem follows a clean, well-layered architecture:

```
CameraSource (threaded, latest-frame buffer)
    ↓ FrameData (BGR, timestamp, frame_id, source_state)
PersonDetector (YOLO26m ONNX Runtime, CUDA EP, torch DLL registration)
    ↓ List[Detection] (bbox, confidence, class_id=0)
DetectionTrackingAdapter (filters person class, converts to (N,6) array)
    ↓
ByteTrack (2-stage association: high-conf IoU → low-conf IoU recovery)
    ↓ List[Track] (track_id, bbox, confidence)
OccupancyCounter (whole-frame or ROI mode, unique track IDs)
LineCrossingCounter (virtual line segment, bottom-center trajectory, anti-chatter)
SessionCounter (cumulative unique track instances)
SceneAnalyzer (hierarchy: MANUAL → CALIBRATED → AUTOMATIC → NOT_SET)
CrowdAnalyticsEngine (occupancy %, debounced crowd level, trend, peaks)
AlertEngine (2 P0 rules: CRITICAL_OCCUPANCY, CAMERA_OR_DETECTION_FAILURE, symmetric hysteresis)
ReliabilityManager (6-state machine: STARTING→LIVE↔DEGRADED/UNSTABLE↔OFFLINE→STARTING + STOPPING)
    ↓
LiveState (unified JSON-serializable snapshot per frame)
```

**Key Design Decisions Verified:**
- V1 default: Whole-frame counting (`enable_roi=False`)
- Anonymous tracking only — no biometric identity, no ReID
- Capacity hierarchy enforces MANUAL/CALIBRATED over AUTOMATIC for safety
- Frozen count preservation during degradation (never fabricates zero)
- Symmetric alert hysteresis prevents flapping

---

## B. Confirmed Bugs

### B1. Detector Does Not Reject NaN/Inf Confidence Values
**File:** `visionqueue/detection/detector.py:_postprocess()`  
**Severity:** High  
**Evidence:** 
```python
conf = float(row[4])
if conf < conf_thresh:  # NaN < 0.25 → False (NaN comparisons always False)
    continue
# NaN/Inf pass through, create Detection with invalid confidence
```
**Impact:** If ONNX model outputs NaN/Inf (e.g., numerical instability), invalid detections propagate to tracker, causing undefined behavior in IoU matching and Kalman filter.
**Reproduction:** Feed model output containing `nan` or `inf` in confidence column.
**Fix:** Add explicit check `if not math.isfinite(conf): continue` before threshold comparison.

### B2. ByteTrack Second-Stage Matching Deviates from Canonical Algorithm
**File:** `visionqueue/tracking/bytetrack.py` (Stage 2), `visionqueue/tracking/matching.py`  
**Severity:** Medium  
**Evidence:**
```python
# Stage 2 in bytetrack.py:
matches_low, u_track_low, u_det_low = matching(
    r_tracked_tlbr, det_tlbr_low,
    thresh=0.5,
    detections_full=None  # Pure IoU — NO score fusion
)
```
**Canonical ByteTrack (Zhang et al.):** Stage 2 uses **IoU × score fusion** for low-confidence association to prefer higher-confidence matches when IoU is similar.
**Current:** Pure IoU distance threshold (0.5 = IoU ≥ 0.5).
**Impact:** In dense crowds with overlapping low-confidence detections, may associate to wrong track when multiple candidates have similar IoU but different confidence.
**Verification:** No test covers score fusion behavior in Stage 2. Test `test_low_confidence_recovery` passes but doesn't validate fusion logic.

### B3. Track Expiry Uses Frame Count, Not Elapsed Time
**File:** `visionqueue/tracking/bytetrack.py:_is_expired()`  
**Severity:** Medium  
**Evidence:**
```python
self.max_time_lost = int(self.config.frame_rate / 30.0 * self.config.track_buffer)
def _is_expired(self, track):
    return self.frame_id - track.end_frame > self.max_time_lost
```
**Problem:** If actual FPS ≠ configured `frame_rate` (e.g., camera runs at 15 FPS but config says 30), expiry time is wrong by 2×. At 15 FPS actual, a 30-frame buffer = 2 seconds real time, but code computes 15 frames = 1 second.
**Impact:** Tracks expire too early (if actual < configured) or too late (if actual > configured), affecting occlusion recovery and ID stability.
**Fix:** Track actual elapsed time using wall-clock timestamps from `FrameData.timestamp`.

### B4. Camera Reconnect Does Not Reset Internal State
**File:** `visionqueue/camera/capture.py:_handle_disconnect()`  
**Severity:** Medium  
**Evidence:** On reconnect, `_open_device()` resets `_consecutive_errors` but:
- `_frame_sequence` continues incrementing (not reset)
- `_measured_fps` EMA continues from pre-disconnect value
- `_last_frame_time` continues
**Impact:** 
- `frame_age_ms = (timestamp - frame_data.timestamp)` in ReliabilityManager uses wall-clock timestamps but `frame_id` is cumulative → large `frame_id` gaps with small time gaps can distort frame age calculation.
- FPS EMA carries stale data from before disconnect.
**Fix:** Reset `_frame_sequence=0`, `_measured_fps=0.0`, `_last_frame_time=0.0` on successful reconnect.

### B5. Frame Age Calculation Can Produce Negative Values (Clamped)
**File:** `visionqueue/reliability/manager.py`  
**Severity:** Low  
**Evidence:**
```python
frame_age_ms = max(0.0, (timestamp - frame_data.timestamp) * 1000.0)
```
**Root Cause:** `frame_data.timestamp` (acquisition time) vs `timestamp` (evaluation time, `time.time()`). If acquisition thread and evaluation thread have clock skew or out-of-order delivery, age can be negative.
**Current Mitigation:** Clamped to 0. Correctly handled but worth noting.

---

## C. False Positives from Previous Audit

| # | Previous Finding | Status | Evidence |
|---|------------------|--------|----------|
| 1 | Detector NaN/Inf not rejected | **CONFIRMED BUG** | See B1 |
| 2 | LineCrossingCounter ENTRY→EXIT→ENTRY edge case | **INTENTIONAL DESIGN** | `min_track_age_frames` guard suppresses first crossing for brand-new tracks. With `min_track_age_frames=0` (default), ENTRY→EXIT→ENTRY works correctly. Not a bug. |
| 3 | ByteTrack thresholds affect recall for new people | **PARTIALLY TRUE** | `new_track_thresh=0.6` + `track_high_thresh=0.5` means detections in [0.5, 0.6) associate to existing tracks but CANNOT create new tracks. Verified in `test_bytetrack_diagnosis.py`. This is by design to reduce false tracks but reduces recall for new people with medium confidence. |
| 4 | Inconsistent historical FPS/test-count claims in docs | **FALSE POSITIVE** | README.md says 224 tests; actual count is 350. CV_HANDOFF.md says 224. README needs update. |
| 5 | requirements.txt needs CUDA verification | **PARTIALLY TRUE** | requirements.txt correctly documents PyTorch CUDA wheel requirement. However, `onnxruntime-gpu>=1.18.0,<2.0.0` allows 1.26.0 which works, but upper bound `<2.0.0` may need review when ORT 2.x releases. |
| 6 | Real-world ground truth missing | **CONFIRMED** | Only synthetic test video exists. No annotated real footage. |
| 7 | ByteTrack second-stage matching differs from canonical | **CONFIRMED BUG** | See B2 |
| 8 | Track expiry uses frame count | **CONFIRMED BUG** | See B3 |
| 9 | Camera reconnect state/counters not reset | **CONFIRMED BUG** | See B4 |
| 10 | Acquisition FPS vs processing FPS mixed | **PARTIALLY TRUE** | `CameraSource.measured_fps` (acquisition) vs `PerformanceMetrics.processing_fps` (pipeline). Both reported in LiveState but naming could be clearer. |
| 11 | Evaluation timestamps mix video-relative and wall-clock | **FALSE POSITIVE** | Evaluation framework uses `frame / fps` for video-relative timestamps. Pipeline uses `time.time()` for wall-clock. They are separate systems. No mixing found. |
| 12 | Crossings near track disappearance may be lost | **UNVERIFIED** | Needs real-world footage to test. Synthetic tests pass. |
| 13 | Evaluation crossing matching uses greedy temporal matching | **CONFIRMED** | `eval_framework.py:greedy_match()` uses per-prediction closest-GT. Standard but not globally optimal (Hungarian would be better). |
| 14 | YOLO26n vs YOLO26m not evaluated on real footage | **CONFIRMED** | Only synthetic video tested. Model comparison infrastructure exists but no real data. |
| 15 | RTSP capture may block indefinitely | **UNVERIFIED** | `cv2.VideoCapture.read()` blocking behavior depends on backend. No timeout configured. |
| 16 | Frame-based track_buffer needs investigation | **CONFIRMED** | See B3. |
| 17 | Customer calibration/configuration workflow limited | **CONFIRMED** | No guided calibration UI. ROI/virtual_line configured programmatically only. |

---

## D. Design Weaknesses

### D1. Coordinate System Consistency — Implicit Assumptions
- **Detector:** Letterbox preprocessing with asymmetric padding (`pad_left`, `pad_top` computed independently). Postprocess correctly reverses. **Verified correct.**
- **Tracker:** Uses `tlwh` (top-left, width, height) internally, converts to `tlbr` for matching.
- **Counting:** Uses `track_bottom_center()` = `(center_x, y2)` for ROI and line crossing.
- **ROI:** Half-open interval `[x, x2) × [y, y2)` matching pixel semantics.
- **Weakness:** No explicit coordinate system documentation in each module. New contributors must trace through code.

### D2. ByteTrack Configuration Hardcoded Thresholds
**File:** `visionqueue/tracking/bytetrack.py:ByteTrackConfig`  
```python
track_high_thresh = 0.5
track_low_thresh = 0.1
new_track_thresh = 0.6
match_thresh = 0.8
```
**Issue:** These are global constants. No per-class or per-scene tuning. For fixed-camera queue scenarios, these may be suboptimal.
**Recommendation:** Expose as runtime-tunable parameters with scene profiles.

### D3. SessionCounter Terminology Still Confusing
**File:** `visionqueue/counting/session.py`  
**Current:** `cumulative_track_instances` (new) + `approximate_unique_count` (deprecated alias).
**Issue:** Both names imply "unique visitors" to non-experts. The docstring warns but field names don't enforce clarity.
**Recommendation:** Rename to `track_instances_observed` and remove deprecated alias in next schema version.

### D4. Face Detection Runs at Fixed Cadence, No ROI Filtering
**File:** `visionqueue/pipeline/coordinator.py`  
**Issue:** Face detector runs every N frames on full frame. No option to restrict to ROI. Wastes compute if only queue region matters.
**Impact:** Minor (YuNet is fast), but inconsistent with person detector ROI support.

### D5. No Explicit NaN/Inf Guards in Tracker/Kalman Filter
**Files:** `visionqueue/tracking/kalman.py`, `bytetrack.py`  
**Issue:** Kalman filter matrix operations (`np.linalg.inv`, `np.linalg.cholesky`) will raise or produce garbage if covariance becomes non-PSD due to upstream NaN.
**Mitigation:** Detector bug (B1) is the likely source. Fixing B1 mitigates this.

---

## E. Accuracy Risks

| Risk | Likelihood | Impact | Mitigation Status |
|------|------------|--------|-------------------|
| Occupancy MAE > 2 in dense crowds | Medium | High | No real-world validation data |
| Entry/exit recall < 90% at line boundary | Medium | High | `min_track_age_frames` helps but unvalidated |
| ID fragmentation in occlusion | Medium | Medium | ByteTrack buffer=30 frames; unvalidated at low FPS |
| ID switches during crossing | Low | Medium | Tests pass but synthetic only |
| Peak occupancy undercount | Medium | High | No ground truth for validation |
| False ENTRY from tracker re-identification | Low | Medium | `_recent_inactive_tracks` diagnostic tracks this |

**Critical Gap:** No representative annotated pilot footage exists. All accuracy claims are synthetic or theoretical.

---

## F. Tracking Risks

| Risk | Details |
|------|---------|
| **Low-confidence recall** | `new_track_thresh=0.6` means YOLO detections in [0.25, 0.6) never create tracks. People with consistent 0.4-0.5 confidence are invisible to tracking. |
| **Second-stage IoU-only** | No score fusion in Stage 2 (B2) → suboptimal association in dense crowds with low-conf detections. |
| **Frame-based expiry** | B3 — expiry time wrong if FPS ≠ configured. |
| **Track buffer hardcoded** | `track_buffer=30` frames. At 30 FPS = 1 second. May be too short for queue stops. |
| **No appearance features** | Pure motion model. Cannot handle long-term occlusion > buffer. By design (privacy). |
| **ID reuse after expiry** | New person in same location gets new track ID. Correct for anonymous tracking but inflates `track_instances`. |

---

## G. Counting Risks

| Risk | Details |
|------|---------|
| **ENTRY→EXIT→ENTRY with min_track_age>0** | First crossing suppressed if track too young. Default=0 avoids this but allows false crossings from tracks born on line. |
| **Line segment boundary** | `_segments_intersect()` checks intersection point within segment bounds using projection parameter `u ∈ [0,1]`. Correct. |
| **Anti-chatter** | `last_crossed_direction` prevents duplicate same-direction triggers. Correct. |
| **Queue ROI ≤ Whole-frame ROI** | Test `test_queue_people_lte_current_people` enforces invariant. Good. |
| **Occupancy deduplication** | Uses `set(track_id)` — correct for unique tracks per frame. |

---

## H. CUDA/Environment Risks

| Risk | Details |
|------|---------|
| **torch DLL registration fragile** | `detector.py` adds `torch/lib` to DLL search path. Works on current setup but depends on torch wheel layout. If torch changes DLL structure, CUDA EP fails silently (falls back to CPU). |
| **ONNX Runtime version pin** | `onnxruntime-gpu<2.0.0` — ORT 2.x may break CUDA EP compatibility. |
| **CUDA EP not verified at startup** | Pipeline logs active provider but doesn't fail fast if CPU fallback occurs unexpectedly. |
| **No GPU memory monitoring** | 320 MB VRAM reported but no OOM handling or monitoring. |
| **CPU fallback untested in CI** | All tests mock detector; no test validates CPU EP path. |

---

## I. Performance Bottlenecks

**Measured (from CV_HANDOFF.md, RTX 5060 Laptop):**
- YOLO26m inference (p50): **7.2 ms** (138 FPS raw)
- YOLO26m inference (p95): **8.6 ms**
- ByteTrack tracking: **< 0.2 ms**
- Full pipeline loop (p50): **8.4 ms**
- Effective frame rate: **30 FPS** (camera-capped)

**Bottleneck Analysis:**
1. **Detector inference dominates** (7.2 ms of 8.4 ms loop = 86%)
2. Camera acquisition at 30 FPS caps effective throughput
3. No other component > 1% of runtime

**Optimization Priority:**
- YOLO26n would reduce inference to ~2-3 ms but may hurt detection recall
- **Must measure on real footage before deciding** (see N. Model Comparison Plan)

---

## J. Reliability Risks

| Risk | Details |
|------|---------|
| **STARTING requires 3 healthy frames** | Hardcoded `min_starting_frames=3`. At 30 FPS = 100ms. Reasonable. |
| **DEGRADED triggers on FPS < 5** | Hardcoded `min_live_fps=5.0`. May be too low for real-time UX. |
| **UNSTABLE after 5 detection failures** | Hardcoded `unstable_failure_threshold=5`. At 30 FPS = 166ms. |
| **Frozen count never auto-recovers if camera stays offline** | Correct — requires manual/operator intervention or stream recovery. |
| **OFFLINE→STARTING transition verified** | Test `test_camera_recovery_transitions_through_starting_never_straight_to_live` passes. |

---

## K. Security/Privacy Risks

| Risk | Status |
|------|--------|
| **Face recognition / biometric identity** | **NOT PRESENT** — YuNet only detects presence, no embeddings. |
| **ReID / cross-camera identity** | **NOT PRESENT** — Anonymous track IDs only. |
| **Cloud image upload** | **NOT PRESENT** — Fully local processing. |
| **Raw frame persistence** | **NOT PRESENT** — Frames processed in-memory only. |
| **Model file validation** | **MISSING** — No checksum verification on model load. Supply chain risk. |
| **RTSP URL validation** | **MISSING** — `CameraConfig.source` accepts any string. SSRF risk if user-controlled. |

---

## L. Customer/Deployment Risks

| Risk | Details |
|------|---------|
| **No calibration UI** | ROI, virtual_line, capacity set programmatically only. Operator must edit config. |
| **Camera placement assumptions** | Overhead/angled 30-60° optimal. Side-mounted cameras untested. |
| **Lighting requirements** | "Adequate illumination" — no lux threshold specified. |
| **Capacity safety** | AUTOMATIC capacity explicitly blocked from analytics. Good. |
| **Model swap process** | Change `model_path` in config. No validation of model compatibility. |
| **Multi-camera** | Pipeline instantiated per camera. No cross-camera fusion. By design. |

---

## M. Real-World Validation Gaps

**Current State:**
- 350 passing unit/integration/regression tests (all synthetic)
- 1 synthetic test video (2 rectangles crossing a line)
- No annotated real-world footage

**Required Evaluation Dataset (Minimum Viable):**
| Scenario | Duration | Key Challenge |
|----------|----------|---------------|
| Empty scene | 30s | Baseline false positive rate |
| 1-3 people walking | 60s | Basic detection/tracking |
| Moderate queue (5-10) | 60s | Stationary people, partial occlusion |
| Dense queue (15-25) | 60s | Heavy occlusion, overlapping boxes |
| People entering queue | 30s | ENTRY crossing accuracy |
| People leaving queue | 30s | EXIT crossing accuracy |
| Crossing paths | 30s | ID switches, track fragmentation |
| Partial occlusion | 30s | Track recovery |
| Full occlusion (>1s) | 30s | Buffer expiry, re-identification |
| Low light / indoor | 30s | Detection confidence drop |
| Camera vibration | 10s | Frame jitter, detector stability |
| Detector failure injection | 10s | Reliability state transitions |

**Annotation Protocol:** Must document `track_identity_protocol` (same person re-entering = same or new annotation ID?).

---

## N. Model Comparison Plan

**Objective:** Choose between YOLO26n (9.9 MB) and YOLO26m (82 MB) for production.

**Method:** Run `scratch/eval_framework.py --compare-model` on **same real footage** with **same ground truth**.

| Metric | Target | Decision Rule |
|--------|--------|---------------|
| Detection recall (person) | ≥ 95% | Prefer higher recall |
| Occupancy MAE | ≤ 1.5 | Prefer lower MAE |
| Crossing F1 (ENTRY+EXIT) | ≥ 0.90 | Prefer higher F1 |
| Track fragmentation (ID switches/100 tracks) | ≤ 5 | Prefer lower |
| Inference latency (p95) | ≤ 15 ms | Must meet real-time |
| GPU VRAM | ≤ 500 MB | Must fit |

**Procedure:**
1. Collect 3-5 representative clips (see M)
2. Annotate with schema v1 (occupancy + crossings)
3. Run comparison:
   ```bash
   python scratch/eval_framework.py --video clip1.mp4 --gt clip1_gt.json \
       --model models/yolo26n.onnx --compare-model models/yolo26m.onnx \
       --out eval_results
   ```
4. Compare MAE, RMSE, F1, latency, VRAM
5. Document decision with evidence

**Do NOT decide on model size alone.**

---

## O. ByteTrack Assessment

**Verdict: KEEP with targeted fixes.**

| Criterion | Assessment |
|-----------|------------|
| Core algorithm correct | Yes — 2-stage association, Kalman filter, lifecycle management all implemented |
| High-confidence association | Works — tested |
| Low-confidence recovery | Works but **missing score fusion** (B2) |
| Occlusion handling | Frame-buffer based; **time-based expiry needed** (B3) |
| ID stability | Good in tests; unvalidated in dense crowds |
| Computational cost | Negligible (<0.2 ms/frame) |
| Configuration flexibility | Hardcoded thresholds; should be tunable |
| Privacy compliance | Yes — no appearance features |

**Recommended Improvements (in order):**
1. Fix B2: Add score fusion to Stage 2 matching
2. Fix B3: Change expiry to elapsed time (requires timestamp propagation to tracker)
3. Expose thresholds as runtime config
4. Add diagnostic telemetry for ID switches/fragmentation

**Do NOT replace with BoT-SORT, OC-SORT, or ReID-based trackers** without evidence ByteTrack fails on real footage.

---

## P. Priority Matrix

| Priority | Item | Category | Action |
|----------|------|----------|--------|
| **P0** | B1: Detector NaN/Inf rejection | Bug | Add `math.isfinite(conf)` check in `_postprocess` |
| **P0** | B2: ByteTrack Stage 2 score fusion | Bug | Pass `detections_full` to Stage 2 `matching()` call |
| **P0** | B3: Track expiry → elapsed time | Bug | Propagate frame timestamp to tracker; use wall-clock delta |
| **P0** | B4: Camera reconnect state reset | Bug | Reset `_frame_sequence`, `_measured_fps`, `_last_frame_time` in `_handle_disconnect` |
| **P1** | D2: Expose ByteTrack thresholds as runtime config | Design | Add setters to `ByteTrackConfig` / `ByteTrack` |
| **P1** | M: Collect real-world evaluation dataset | Validation | Annotate 5+ clips per protocol |
| **P1** | N: Execute YOLO26n vs YOLO26m comparison | Model Selection | Run eval framework on real footage |
| **P1** | K: Add model checksum validation | Security | Verify SHA-256 on model load |
| **P1** | K: Add RTSP URL validation | Security | Allowlist schemes/hosts or validate format |
| **P2** | D1: Document coordinate systems per module | Documentation | Add module-level docstrings |
| **P2** | D3: Rename SessionCounter fields | API Clarity | `track_instances_observed` in next schema |
| **P2** | D4: Face detector ROI option | Consistency | Add `roi` parameter to `FaceDetector.detect()` |
| **P2** | J: Make reliability thresholds configurable | Flexibility | Already in `ReliabilityConfig` — verify exposure |
| **P3** | L: Build calibration helper script | Usability | CLI to set ROI/virtual_line interactively |
| **P3** | 13: Evaluation greedy → Hungarian matching | Evaluation Quality | Optional improvement |
| **NO ACTION** | 2: ENTRY→EXIT→ENTRY with min_track_age=0 | Intentional | Works correctly |
| **NO ACTION** | 12: Crossings near disappearance | Unverified | Needs real footage first |
| **NO ACTION** | 15: RTSP blocking | Unverified | Needs real RTSP test |

---

## Q. Exact Files Needing Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `visionqueue/detection/detector.py` | Bug Fix | Add `import math` and `if not math.isfinite(conf): continue` in `_postprocess` |
| `visionqueue/tracking/bytetrack.py` | Bug Fix | Stage 2: pass `detections_full=dets_low` to `matching()` call |
| `visionqueue/tracking/bytetrack.py` | Bug Fix | Track expiry: store `end_timestamp` in `Track`; compute elapsed in `_is_expired` |
| `visionqueue/tracking/track.py` | Bug Fix | Add `end_timestamp: float` field; update in `update()` |
| `visionqueue/camera/capture.py` | Bug Fix | Reset `_frame_sequence=0`, `_measured_fps=0.0`, `_last_frame_time=0.0` on successful reconnect |
| `visionqueue/tracking/bytetrack.py` | Enhancement | Add setters/runtime config for `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `match_thresh` |
| `visionqueue/detection/detector.py` | Security | Verify model file SHA-256 against manifest on load |
| `visionqueue/camera/types.py` | Security | Add RTSP URL validation in `CameraConfig.__post_init__` |
| `scratch/eval_framework.py` | Enhancement | Replace greedy matching with Hungarian for crossing evaluation |
| `visionqueue/counting/session.py` | API (v2) | Rename `cumulative_track_instances` → `track_instances_observed`; deprecate alias |
| `README.md` | Documentation | Update test count from 224 → 350 |
| `docs/CV_HANDOFF.md` | Documentation | Update test count from 224 → 350 |

---

## R. Exact Regression Tests Required

| Test File | New Test Case | Purpose |
|-----------|---------------|---------|
| `tests/test_detector.py` | `test_postprocess_rejects_nan_inf_confidence` | Verify NaN/Inf confidence rejected |
| `tests/test_bytetrack.py` | `test_stage2_uses_score_fusion` | Verify Stage 2 matching uses IoU×score |
| `tests/test_bytetrack.py` | `test_track_expiry_uses_elapsed_time` | Verify expiry respects actual time, not frame count |
| `tests/test_camera.py` | `test_reconnect_resets_frame_sequence_and_fps` | Verify reconnect resets internal state |
| `tests/test_counting.py` | `test_entry_exit_entry_with_min_age_zero` | Verify ENTRY→EXIT→ENTRY works with default config |
| `tests/test_integration_contract.py` | `test_livestate_no_nan_inf_values` | End-to-end NaN/Inf propagation check |
| `tests/test_eval_framework.py` | `test_crossing_matching_hungarian_optimal` | Verify evaluation uses optimal matching |

---

## S. What MUST NOT Be Changed

| Item | Reason |
|------|--------|
| **Architecture layers** | Clean separation working well |
| **Anonymous tracking (no ReID)** | Privacy requirement — hard constraint |
| **6-state reliability machine** | Correctly implements PRD/TDD |
| **Frozen count preservation** | Safety invariant — never fabricate zero |
| **MANUAL/CALIBRATED capacity precedence** | Safety — automatic estimates unreliable |
| **Symmetric alert hysteresis** | Prevents flapping — correct |
| **Half-open ROI intervals** | Matches pixel semantics — correct |
| **YOLO26m as default model** | Until real-footage comparison proves YOLO26n sufficient |
| **ByteTrack as tracker** | Adequate for fixed-camera queue; fixes address gaps |
| **torch DLL registration mechanism** | Required for CUDA EP on Windows — works |

---

## T. Recommended Next 5 Actions (In Order)

1. **Fix P0 Bugs (B1-B4)** — Implement fixes in Q-files order; add regression tests (R).
2. **Run Full Test Suite** — Verify 350 tests still pass after fixes.
3. **Collect Real-World Evaluation Dataset** — Minimum 5 clips per M; annotate per schema v1.
4. **Execute Model Comparison (N)** — Run YOLO26n vs YOLO26m on real footage; decide with evidence.
5. **Expose ByteTrack Thresholds (P1-D2)** — Add runtime configuration for tuning on deployment.

---

## U. Recommendation: Further Measurement Before Implementation

**Do NOT implement model changes, tracker replacement, or architecture modifications yet.**

**Proceed with:**
1. **P0 bug fixes** (B1-B4) — these are clear defects with reproducible evidence
2. **Real-world data collection** — this is the blocking dependency for all accuracy/performance decisions
3. **Model comparison** — only after real footage exists

**Defer until real footage available:**
- YOLO26n vs YOLO26m decision
- ByteTrack threshold tuning
- Any tracker replacement evaluation
- Capacity/analytics threshold calibration
- Performance optimization beyond P0 fixes

---

## Summary

The VisionQueue CV subsystem is **architecturally sound, well-tested (350/350 passing), and privacy-compliant**. The codebase demonstrates senior-level engineering: clean contracts, comprehensive diagnostics, proper failure handling, and frozen-count safety invariants.

**Four P0 bugs** require immediate fix (NaN/Inf handling, ByteTrack Stage 2 fusion, frame-based expiry, camera reconnect state). All are localized, low-risk changes with clear regression tests.

**The critical path to production readiness is real-world validation.** Without annotated pilot footage, accuracy claims remain theoretical. The evaluation framework (`scratch/eval_framework.py`) is production-ready and should be used once data exists.

**ByteTrack is adequate for the fixed-camera queue use case.** The two confirmed deviations from canonical algorithm (Stage 2 score fusion, frame-based expiry) are fixable without replacement. Do not introduce ReID or appearance-based tracking.

**Recommendation:** Fix P0 bugs → collect 5+ annotated clips → run model comparison → tune thresholds → deploy.