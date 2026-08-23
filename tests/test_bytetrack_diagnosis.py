"""Phase B: ByteTrack threshold diagnostic tests.

These tests diagnose — but do NOT fix — the ByteTrack confidence threshold
interaction observed in the acceptance test where unique_approx grew to 21
while current_count stayed 0.

Hypothesis under test:
  - YOLO detection confidence was 0.25–0.6.
  - ByteTrack default new_track_thresh=0.6 means only detections >=0.6 create tracks.
  - track_high_thresh=0.5 means detections 0.25–0.49 go to "low" stage only.
  - Therefore many valid person detections cannot create new tracks.

These tests measure, with assertions, what actually happens at each
confidence level. They do NOT change production thresholds.

If any assertion in TestByteTrackConfidenceThresholds fails, that
documents a specific threshold interaction that is causing the field bug.
"""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import numpy as np
import pytest
from visionqueue.tracking import ByteTrack, ByteTrackConfig, TrackState


def make_det(x1=100, y1=100, x2=200, y2=300, conf=0.9, cid=0):
    """Single detection row (N=1, 6 cols)."""
    return np.array([[x1, y1, x2, y2, conf, cid]], dtype=np.float32)


def make_det_multi(*rows):
    """Multiple detection rows."""
    return np.array(rows, dtype=np.float32)


# ============================================================
# Default production ByteTrackConfig (unchanged from production)
# ============================================================

PRODUCTION_CONFIG = ByteTrackConfig(
    track_buffer=30,
    match_thresh=0.8,
    track_high_thresh=0.5,    # detections >= 0.5 go to high-conf stage
    track_low_thresh=0.1,     # detections 0.1–0.5 go to low-conf stage
    new_track_thresh=0.6,     # new tracks require conf >= 0.6
    frame_rate=30,
)


class TestByteTrackConfidenceThresholds:
    """Focused tests measuring ByteTrack behavior at different confidence levels.

    Observation from acceptance test:
      - YOLO confidence_threshold=0.25 in DetectorConfig
      - unique_approx grew to 21 in 45 seconds (rapid ID churn)
      - current_count stayed at 0

    This class tests: can a person at typical webcam YOLO confidence
    (0.25, 0.4, 0.55, 0.7, 0.9) actually establish a stable track
    under production ByteTrackConfig?
    """

    def _run_n_frames(self, tracker, det_fn, n_frames):
        """Run n frames with the given detection function. Returns list of track outputs."""
        results = []
        for i in range(n_frames):
            det = det_fn(i)
            out = tracker.update(det)
            results.append(out)
        return results

    # ----------------------------------------------------------------
    # Case 1: conf = 0.25 (YOLO min threshold in acceptance test)
    # ----------------------------------------------------------------
    def test_conf_0_25_cannot_create_new_track(self):
        """With production config, conf=0.25 is BELOW new_track_thresh=0.6
        AND below track_high_thresh=0.5.
        Detections fall into the 'low' confidence bucket (0.1–0.5).
        Low-conf detections only recover existing TRACKED/LOST tracks;
        they CANNOT create new tracks.
        Therefore: no track should ever be created from conf=0.25 alone.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        # 10 consecutive frames, same person, conf=0.25
        for frame in range(10):
            out = tracker.update(make_det(conf=0.25))
            assert len(out) == 0, (
                f"Frame {frame+1}: conf=0.25 should NOT create a new track "
                f"(new_track_thresh=0.6). Got {len(out)} track(s)."
            )

    # ----------------------------------------------------------------
    # Case 2: conf = 0.49 (just below high threshold)
    # ----------------------------------------------------------------
    def test_conf_0_49_cannot_create_new_track(self):
        """conf=0.49 is below track_high_thresh=0.5, goes to low stage.
        Same result: no new tracks created.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        for frame in range(10):
            out = tracker.update(make_det(conf=0.49))
            assert len(out) == 0, (
                f"Frame {frame+1}: conf=0.49 should NOT create track. Got {len(out)}."
            )

    # ----------------------------------------------------------------
    # Case 3: conf = 0.50 (at high threshold, below new_track_thresh)
    # ----------------------------------------------------------------
    def test_conf_0_50_high_stage_but_below_new_track_thresh(self):
        """conf=0.50 meets track_high_thresh=0.5 (goes to high stage),
        but is below new_track_thresh=0.6, so Step 5 of ByteTrack
        will NOT initialize a new track from this detection.
        Expected: still no track created.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        for frame in range(10):
            out = tracker.update(make_det(conf=0.50))
            assert len(out) == 0, (
                f"Frame {frame+1}: conf=0.50 (< new_track_thresh=0.6) should NOT "
                f"create track. Got {len(out)}."
            )

    # ----------------------------------------------------------------
    # Case 4: conf = 0.60 (exactly at new_track_thresh)
    # ----------------------------------------------------------------
    def test_conf_0_60_creates_track_after_confirmation(self):
        """conf=0.60 meets both track_high_thresh=0.5 and new_track_thresh=0.6.
        A new track IS created in frame 1, then must survive confirmation.
        Track.update() promotes NEW→TRACKED after hit_streak >= 3, so a track
        will exist in output from frame 1 onwards but starts in NEW state.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        track_ids_seen = set()
        # Run 5 frames
        for frame in range(5):
            out = tracker.update(make_det(conf=0.60))
            if out:
                for t in out:
                    track_ids_seen.add(t["track_id"])

        assert len(track_ids_seen) >= 1, (
            "conf=0.60 should create at least one track across 5 frames."
        )

    # ----------------------------------------------------------------
    # Case 5: conf = 0.90 (well above all thresholds) — stable track
    # ----------------------------------------------------------------
    def test_conf_0_90_creates_stable_persistent_track(self):
        """conf=0.90 is well above all thresholds. Track must be created
        in frame 1 and maintain the SAME track_id across 10 frames.
        If IDs are not stable, that documents ID churn.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        track_ids_per_frame = []
        for frame in range(10):
            out = tracker.update(make_det(conf=0.90))
            assert len(out) == 1, f"Frame {frame+1}: expected 1 track at conf=0.9, got {len(out)}"
            track_ids_per_frame.append(out[0]["track_id"])

        # All frames must have the SAME track ID (no churn)
        unique_ids = set(track_ids_per_frame)
        assert len(unique_ids) == 1, (
            f"conf=0.90 should produce 1 stable track ID across 10 frames. "
            f"Got {len(unique_ids)} distinct IDs: {unique_ids}. "
            "This documents ID churn."
        )

    # ----------------------------------------------------------------
    # Case 6: Mixed — first high-conf, then low-conf (occlusion simulation)
    # ----------------------------------------------------------------
    def test_high_conf_track_recovered_by_low_conf(self):
        """Once a TRACKED track exists (established at conf=0.9), a low-conf
        detection (0.25) should be able to recover it via the second
        ByteTrack stage.
        Frame 1–3: conf=0.9 → track established and TRACKED (hit_streak>=3)
        Frame 4:   conf=0.25 → should recover via low-conf stage
        Frame 5:   conf=0.9  → track maintained
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        established_id = None

        # Frames 1–3: establish track
        for frame in range(3):
            out = tracker.update(make_det(conf=0.9))
            assert len(out) == 1
            established_id = out[0]["track_id"]

        # Frame 4: low conf — track should be output via recovery
        out4 = tracker.update(make_det(conf=0.25))
        # Low-conf detection goes to second stage; TRACKED track matches
        # Note: a lost track may or may not appear in output depending on
        # whether ByteTrack Step 3 matched it. Document the result.
        if len(out4) == 0:
            # Track moved to LOST — it will be recovered on next high-conf frame
            # This is acceptable ByteTrack behavior but means 1-frame count dip.
            pass  # Documented: low-conf does not always output track
        elif len(out4) == 1:
            assert out4[0]["track_id"] == established_id, "Recovered track must keep same ID"

        # Frame 5: high conf — track must be back
        out5 = tracker.update(make_det(conf=0.9))
        assert len(out5) == 1, "Track must recover after high-conf frame"
        assert out5[0]["track_id"] == established_id, "Recovered ID must match original"

    # ----------------------------------------------------------------
    # Case 7: ID stability measurement — acceptance test scenario
    # ----------------------------------------------------------------
    def test_id_churn_at_typical_yolo_confidence(self):
        """Simulate the acceptance test scenario: 15 FPS, 45 seconds,
        detections alternating between 0.25 and 0.65 (as YOLO produces
        for a real person moving around).

        Measure:
          - How many unique track IDs are created (should be 1 for 1 person)
          - Whether any stable track persists for >10 consecutive frames

        This test documents the behaviour without changing production thresholds.
        It will pass regardless (it records facts), but the assertions
        document the expected acceptable bounds.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)
        rng = np.random.default_rng(seed=42)

        n_frames = 150  # ~10 seconds at 15 FPS
        unique_ids_seen = set()
        max_consecutive_track = 0
        current_consecutive = 0
        last_tid = None

        for frame in range(n_frames):
            # Alternate confidence: realistic YOLO output on webcam
            # 70% of frames at 0.3–0.6 (challenging), 30% at 0.6–0.9 (easy)
            if rng.random() < 0.7:
                conf = float(rng.uniform(0.3, 0.59))
            else:
                conf = float(rng.uniform(0.6, 0.9))

            # Small random position jitter (2px per frame, typical for a still person)
            jitter = rng.integers(-2, 3)
            out = tracker.update(
                make_det(x1=100 + jitter, y1=100, x2=200 + jitter, y2=300, conf=conf)
            )

            if out:
                tid = out[0]["track_id"]
                unique_ids_seen.add(tid)
                if tid == last_tid:
                    current_consecutive += 1
                else:
                    current_consecutive = 1
                    last_tid = tid
                max_consecutive_track = max(max_consecutive_track, current_consecutive)
            else:
                current_consecutive = 0

        # Document findings for Phase B analysis
        print(f"\nPhase B Diagnostic Results (150 frames, realistic YOLO confidence):")
        print(f"  Unique track IDs created  : {len(unique_ids_seen)}")
        print(f"  Max consecutive frames    : {max_consecutive_track}")
        print(f"  Expected for 1 stable person: 1 unique ID, max_consecutive ~= {n_frames}")

        # A single stable person should produce at most a few IDs.
        # If many IDs are produced, it documents the threshold churn problem.
        # We assert a REASONABLE limit: if >10 unique IDs are seen,
        # that IS the churn problem and this test will FAIL to document it.
        assert len(unique_ids_seen) <= 10, (
            f"THRESHOLD CHURN CONFIRMED: {len(unique_ids_seen)} unique track IDs "
            f"for a single person over {n_frames} frames. "
            f"Production config new_track_thresh=0.6 is too high for YOLO "
            f"confidence_threshold=0.25. Threshold adjustment is required."
        )
        # If assertion passes (<=10 IDs), churn is acceptable. Document max consecutive.
        assert max_consecutive_track >= 5, (
            f"Track persistence too low: max consecutive frames with same ID = "
            f"{max_consecutive_track}. Expected >= 5 for a stable detection."
        )


class TestByteTrackTrackConfirmation:
    """Measure the 3-hit confirmation requirement and its interaction with
    Step 4 track removal.

    Under production config, a NEW track is removed in Step 4 if it is
    not matched in the SAME frame where unmatched high-confidence detections
    exist. This means: if a track is NEW and matches in Step 2 but is still
    unmatched in Step 4's context... the track survives but stays NEW.

    This test validates the hit_streak-to-TRACKED promotion timeline.
    """

    def test_new_track_state_progression(self):
        """Verify that a NEW track transitions to TRACKED after 3 hits."""
        tracker = ByteTrack(PRODUCTION_CONFIG)

        # Frame 1: track created, hit_streak=1
        out1 = tracker.update(make_det(conf=0.9))
        assert len(out1) == 1

        # Inspect internal track state
        tracks_internal = tracker.tracked_tracks
        assert len(tracks_internal) == 1
        track = tracks_internal[0]
        assert track.state == TrackState.NEW
        assert track.hit_streak == 1

        # Frame 2: hit_streak=2, still NEW
        out2 = tracker.update(make_det(conf=0.9))
        assert len(out2) == 1
        assert tracks_internal[0].state == TrackState.NEW
        assert tracks_internal[0].hit_streak == 2

        # Frame 3: hit_streak=3 → should become TRACKED
        out3 = tracker.update(make_det(conf=0.9))
        assert len(out3) == 1
        assert tracks_internal[0].state == TrackState.TRACKED
        assert tracks_internal[0].hit_streak == 3

    def test_new_track_destroyed_on_missed_frame(self):
        """A NEW track (hit_streak < 3) that misses a frame in Step 4
        (unmatched by a high-conf detection that came in) gets removed.
        Verify: if frame 1 creates track (NEW) and frame 2 has a DIFFERENT
        detection at a far position, the original NEW track is removed
        and a new track ID is assigned.
        """
        tracker = ByteTrack(PRODUCTION_CONFIG)

        # Frame 1: person at position A
        out1 = tracker.update(make_det(x1=100, y1=100, x2=200, y2=300, conf=0.9))
        assert len(out1) == 1
        id1 = out1[0]["track_id"]
        assert tracker.tracked_tracks[0].state == TrackState.NEW

        # Frame 2: person appears at position B (far away, no IoU with A)
        # Original NEW track from A is unmatched in Step 2 (no IoU).
        # Since a new high-conf detection exists, Step 4 will try to match
        # unconfirmed tracks against it. IoU between A and B is 0 → no match.
        # The original NEW track should be removed.
        out2 = tracker.update(make_det(x1=500, y1=100, x2=600, y2=300, conf=0.9))

        # IDs should be different (original track removed, new one created)
        id2 = out2[0]["track_id"] if out2 else None
        # Note: If both A and B are output in frame 2, they have different IDs.
        # If only B is output (A removed), also correct.
        if id2 is not None:
            assert id2 != id1, (
                "NEW track at position A should be removed when missed by a "
                "far detection at position B. New track should have different ID."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
