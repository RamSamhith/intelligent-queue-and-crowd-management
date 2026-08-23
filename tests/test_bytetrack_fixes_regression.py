"""Regression tests for ByteTrack fixes.

Verifies:
1. Low-confidence detection recovery in Stage 2 (no score-fusion rejection)
2. IoU 0.5–0.8 matching (correct distance threshold semantics)
3. Confidence oscillation handling
4. Temporary detection loss & recovery
5. LOST track recovery via low-confidence Stage 2
6. One persistent person maintaining one track ID
7. 500-frame synthetic confidence oscillation
8. No false ID recreation
"""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import numpy as np
import pytest
from visionqueue.tracking import ByteTrack, ByteTrackConfig, TrackState
from visionqueue.counting.session import SessionCounter
from visionqueue.tracking.adapter import Track as AdapterTrack


def make_det(x1=100.0, y1=100.0, x2=200.0, y2=300.0, conf=0.9, cid=0):
    """Create a single detection row."""
    return np.array([[x1, y1, x2, y2, conf, cid]], dtype=np.float32)


class TestByteTrackFixesRegression:
    """Comprehensive regression test suite for ByteTrack fixes."""

    @pytest.fixture
    def tracker(self):
        config = ByteTrackConfig(
            track_buffer=30,
            match_thresh=0.8,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            frame_rate=30,
        )
        return ByteTrack(config)

    # ----------------------------------------------------------------
    # 1. Low-confidence detection recovery in Stage 2
    # ----------------------------------------------------------------
    def test_low_confidence_detection_recovery(self, tracker):
        """Verify that after establishing a track (3 frames), low-confidence
        detections (0.20–0.45 < track_high_thresh) are successfully matched
        in Stage 2 via pure IoU distance and maintain the same track_id.
        """
        # Frames 1-3: Establish TRACKED state with high confidence
        for _ in range(3):
            out = tracker.update(make_det(100, 100, 200, 300, conf=0.90))
            assert len(out) == 1
            assert out[0]["track_id"] == 0

        assert tracker.tracked_tracks[0].state == TrackState.TRACKED

        # Frames 4-8: Low-confidence detections (0.20, 0.30, 0.40, 0.45, 0.35)
        low_confs = [0.20, 0.30, 0.40, 0.45, 0.35]
        for idx, conf in enumerate(low_confs):
            out = tracker.update(make_det(102, 102, 202, 302, conf=conf))
            assert len(out) == 1, f"Frame {idx+4} with conf={conf} should maintain track via Stage 2"
            assert out[0]["track_id"] == 0, f"Frame {idx+4} must keep original track_id 0"
            assert tracker.tracked_tracks[0].state == TrackState.TRACKED

    # ----------------------------------------------------------------
    # 2. IoU 0.5–0.8 matching (threshold semantics)
    # ----------------------------------------------------------------
    def test_iou_0_5_to_0_8_matching(self, tracker):
        """Verify that detections with IoU between 0.50 and 0.79 match
        properly in Stage 1 and Stage 2 without being rejected.
        """
        # Frame 1: Establish initial track [100, 100, 200, 300] (width=100, height=200, area=20000)
        tracker.update(make_det(100, 100, 200, 300, conf=0.90))
        tracker.update(make_det(100, 100, 200, 300, conf=0.90))
        tracker.update(make_det(100, 100, 200, 300, conf=0.90))

        # Shift box by dx=25, dy=0 -> overlap is 75 x 200 = 15000, union = 25000 -> IoU = 0.60
        out_stage1 = tracker.update(make_det(125, 100, 225, 300, conf=0.75))
        assert len(out_stage1) == 1
        assert out_stage1[0]["track_id"] == 0, "Stage 1 should match at IoU=0.60"

        # Shift box by another dx=20 with low confidence (0.40) -> IoU ~ 0.68
        out_stage2 = tracker.update(make_det(145, 100, 245, 300, conf=0.40))
        assert len(out_stage2) == 1
        assert out_stage2[0]["track_id"] == 0, "Stage 2 should match at IoU=0.68 with low conf"

    # ----------------------------------------------------------------
    # 3. Confidence oscillation
    # ----------------------------------------------------------------
    def test_confidence_oscillation(self, tracker):
        """Verify that rapid frame-to-frame confidence oscillations
        (high -> low -> high -> low) maintain exactly one track ID.
        """
        oscillating_confs = [0.85, 0.35, 0.78, 0.28, 0.92, 0.42, 0.65, 0.38, 0.88, 0.48]
        seen_ids = set()

        for f_idx, conf in enumerate(oscillating_confs):
            out = tracker.update(make_det(100 + f_idx, 100, 200 + f_idx, 300, conf=conf))
            assert len(out) == 1, f"Frame {f_idx+1} (conf={conf}) must output 1 track"
            seen_ids.add(out[0]["track_id"])

        assert seen_ids == {0}, f"Confidence oscillation produced multiple IDs: {seen_ids}"

    # ----------------------------------------------------------------
    # 4. Temporary detection loss & recovery
    # ----------------------------------------------------------------
    def test_temporary_detection_loss_and_recovery(self, tracker):
        """Verify that a track survives temporary occlusion (0 detections)
        and is cleanly recovered when detections resume.
        """
        # Frames 1-3: Establish track
        for _ in range(3):
            tracker.update(make_det(100, 100, 200, 300, conf=0.90))
        assert tracker.tracked_tracks[0].track_id == 0

        # Frames 4-6: Complete occlusion (3 frames of no detection)
        for _ in range(3):
            out = tracker.update(np.empty((0, 6), dtype=np.float32))
            assert len(out) == 0

        # Track is now in lost_tracks
        assert len(tracker.lost_tracks) == 1
        assert tracker.lost_tracks[0].track_id == 0
        assert tracker.lost_tracks[0].state == TrackState.LOST

        # Frame 7: Detection reappears at high confidence
        out_rec = tracker.update(make_det(105, 105, 205, 305, conf=0.80))
        assert len(out_rec) == 1
        assert out_rec[0]["track_id"] == 0, "Track should be recovered with original track_id 0"
        assert tracker.tracked_tracks[0].state == TrackState.TRACKED

    # ----------------------------------------------------------------
    # 5. LOST track recovery via low-confidence Stage 2
    # ----------------------------------------------------------------
    def test_lost_track_recovery_in_stage2(self, tracker):
        """Verify that a LOST track can be recovered directly by a
        low-confidence detection (0.1 < conf < 0.5) in Stage 2.
        """
        # Frames 1-3: Establish track
        for _ in range(3):
            tracker.update(make_det(100, 100, 200, 300, conf=0.90))

        # Frames 4-5: Occlusion -> moves to LOST
        for _ in range(2):
            tracker.update(np.empty((0, 6), dtype=np.float32))

        assert len(tracker.lost_tracks) == 1
        assert tracker.lost_tracks[0].state == TrackState.LOST

        # Frame 6: Detection returns with LOW confidence (0.35 < 0.50)
        out_low_rec = tracker.update(make_det(104, 104, 204, 304, conf=0.35))
        assert len(out_low_rec) == 1, "LOST track should be recovered by low-confidence Stage 2"
        assert out_low_rec[0]["track_id"] == 0, "Recovered track must preserve track_id 0"
        assert tracker.tracked_tracks[0].state == TrackState.TRACKED
        assert len(tracker.lost_tracks) == 0

    # ----------------------------------------------------------------
    # 6. One persistent person maintaining one track ID
    # ----------------------------------------------------------------
    def test_one_persistent_person_maintaining_one_track_id(self, tracker):
        """Verify a single person moving smoothly over 100 frames with varying
        confidence maintains exactly 1 unique track ID.
        """
        rng = np.random.default_rng(seed=123)
        session_counter = SessionCounter()
        x, y = 100.0, 100.0

        # Frames 1-3: Establish track at initial position
        for _ in range(3):
            out_init = tracker.update(make_det(x, y, x + 100, y + 200, conf=0.90))
            adapter_tracks = [AdapterTrack(t["track_id"], tuple(t["bbox"]), t["confidence"]) for t in out_init]
            session_counter.update(adapter_tracks)

        # Frames 4-103: 100 frames of movement and oscillating confidence [0.20, 0.95]
        for frame in range(100):
            x += float(rng.uniform(-2.0, 3.0))
            y += float(rng.uniform(-1.0, 1.0))
            conf = float(rng.uniform(0.20, 0.95))

            out = tracker.update(make_det(x, y, x + 100, y + 200, conf=conf))
            adapter_tracks = [AdapterTrack(t["track_id"], tuple(t["bbox"]), t["confidence"]) for t in out]
            session_counter.update(adapter_tracks)

            assert len(out) == 1
            assert out[0]["track_id"] == 0

        assert session_counter.approximate_unique_count == 1

    # ----------------------------------------------------------------
    # 7. 500-frame synthetic confidence oscillation
    # ----------------------------------------------------------------
    def test_500_frame_synthetic_confidence_oscillation(self, tracker):
        """Rigorous simulation of a 500-frame acceptance test run (~25s at 20 FPS).
        Includes:
        - 60% low confidence frames (0.25–0.49)
        - 40% high confidence frames (0.50–0.90)
        - Periodic 2-frame partial occlusions (0 detections)
        - Small positional motion / jitter
        """
        rng = np.random.default_rng(seed=42)
        session_counter = SessionCounter()
        x, y = 150.0, 100.0
        unique_ids = set()

        # Frames 1-3: Establish track initially
        for _ in range(3):
            out_init = tracker.update(make_det(x, y, x + 100, y + 200, conf=0.90))
            for t in out_init:
                unique_ids.add(t["track_id"])
            adapter_tracks = [AdapterTrack(t["track_id"], tuple(t["bbox"]), t["confidence"]) for t in out_init]
            session_counter.update(adapter_tracks)

        for frame in range(500):
            # 2-frame occlusion every 80 frames
            if 80 <= (frame % 100) < 82:
                det = np.empty((0, 6), dtype=np.float32)
            else:
                x += float(rng.uniform(-1.5, 2.0))
                # 60% low-conf, 40% high-conf
                if rng.random() < 0.60:
                    conf = float(rng.uniform(0.25, 0.49))
                else:
                    conf = float(rng.uniform(0.50, 0.90))
                det = make_det(x, y, x + 100, y + 200, conf=conf)

            out = tracker.update(det)
            for t in out:
                unique_ids.add(t["track_id"])
            
            adapter_tracks = [AdapterTrack(t["track_id"], tuple(t["bbox"]), t["confidence"]) for t in out]
            session_counter.update(adapter_tracks)

        print(f"\n500-Frame Simulation Result: unique_ids={unique_ids}, session_unique={session_counter.approximate_unique_count}")
        assert unique_ids == {0}, f"500-frame test produced multiple track IDs: {unique_ids}"
        assert session_counter.approximate_unique_count == 1, (
            f"Session counter reported {session_counter.approximate_unique_count} visitors for 1 person"
        )

    # ----------------------------------------------------------------
    # 8. No false ID recreation
    # ----------------------------------------------------------------
    def test_no_false_id_recreation(self, tracker):
        """Verify that when detection confidence drops and recovers, Step 5
        does NOT create a secondary duplicate track ID.
        """
        # Frame 1-3: High confidence -> ID 0 created
        for _ in range(3):
            tracker.update(make_det(100, 100, 200, 300, conf=0.90))

        # Frame 4: Drops to low conf (0.35) -> matched in Stage 2
        out4 = tracker.update(make_det(102, 102, 202, 302, conf=0.35))
        assert len(out4) == 1
        assert out4[0]["track_id"] == 0

        # Frame 5: High confidence (0.85) -> matches ID 0 in Stage 1, NO new ID created
        out5 = tracker.update(make_det(104, 104, 204, 304, conf=0.85))
        assert len(out5) == 1
        assert out5[0]["track_id"] == 0
        assert tracker.track_id_counter == 1, "Only 1 track ID should ever have been generated (counter=1)"
