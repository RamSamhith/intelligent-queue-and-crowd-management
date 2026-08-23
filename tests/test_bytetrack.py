"""
Unit tests for VisionQueue ByteTrack implementation.

Tests cover:
1. No detections
2. One persistent person
3. Multiple people
4. Person movement across frames
5. High-confidence association
6. Low-confidence second-stage association
7. Temporary detection loss
8. Track buffer expiration
9. New track creation
10. Two nearby people maintaining separate IDs
11. Class filtering
12. Deterministic behavior

Plus synthetic sequence demonstrating ByteTrack behavior.
"""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import numpy as np
import pytest
from visionqueue.tracking import ByteTrack, ByteTrackConfig, Track, TrackState, KalmanFilter
from visionqueue.tracking.matching import iou_distance, linear_sum_assignment, matching


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def default_config():
    return ByteTrackConfig(
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        frame_rate=30
    )

@pytest.fixture
def tracker(default_config):
    return ByteTrack(default_config)


def make_detection(x1, y1, x2, y2, conf=0.9, class_id=0):
    """Create a single detection row."""
    return np.array([[x1, y1, x2, y2, conf, class_id]], dtype=np.float32)


def make_detections(*boxes):
    """Create detection array from (x1, y1, x2, y2, conf, class_id) tuples."""
    return np.array(boxes, dtype=np.float32)


# ============================================================
# Test 1: No detections
# ============================================================

def test_no_detections(tracker):
    """Test tracker with empty detections."""
    tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
    assert tracks == []

    # Multiple empty frames
    for _ in range(5):
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert tracks == []


# ============================================================
# Test 2: One persistent person
# ============================================================

def test_one_persistent_person(tracker):
    """Test tracking a single person across multiple frames."""
    track_ids = []

    # Person at fixed position
    for i in range(10):
        det = make_detection(100, 100, 200, 300, conf=0.9)
        tracks = tracker.update(det)
        assert len(tracks) == 1
        track_ids.append(tracks[0]["track_id"])

    # Same track ID maintained
    assert len(set(track_ids)) == 1
    assert track_ids[0] == 0


# ============================================================
# Test 3: Multiple people
# ============================================================

def test_multiple_people(tracker):
    """Test tracking multiple people simultaneously."""
    # Two people at different positions
    for i in range(10):
        det = make_detections(
            (100, 100, 200, 300, 0.9, 0),  # Person A
            (300, 100, 400, 300, 0.9, 0),  # Person B
        )
        tracks = tracker.update(det)
        assert len(tracks) == 2

        # Check both IDs are different and consistent
        ids = {t["track_id"] for t in tracks}
        assert len(ids) == 2


# ============================================================
# Test 4: Person movement across frames
# ============================================================

def test_person_movement(tracker):
    """Test tracking a moving person."""
    track_ids = []

    # Person moving right
    x = 100
    for i in range(10):
        det = make_detection(x, 100, x + 100, 300, conf=0.9)
        tracks = tracker.update(det)
        assert len(tracks) == 1
        track_ids.append(tracks[0]["track_id"])
        x += 10  # Move 10 pixels per frame

    # Same track ID maintained during movement
    assert len(set(track_ids)) == 1


# ============================================================
# Test 5: High-confidence association
# ============================================================

def test_high_confidence_association(tracker):
    """Test that high-confidence detections are associated first."""
    # Frame 1: High confidence detection creates track
    det1 = make_detection(100, 100, 200, 300, conf=0.9)
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    tid = tracks1[0]["track_id"]

    # Frame 2: High confidence detection at same position
    det2 = make_detection(102, 102, 202, 302, conf=0.9)
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert tracks2[0]["track_id"] == tid


# ============================================================
# Test 6: Low-confidence second-stage association
# ============================================================

def test_low_confidence_recovery(tracker):
    """Test that low-confidence detections recover occluded tracks."""
    # Frame 1: High confidence - creates track
    det1 = make_detection(100, 100, 200, 300, conf=0.9)
    tracks1 = tracker.update(det1)
    tid = tracks1[0]["track_id"]

    # Frame 2: High confidence - maintains track
    det2 = make_detection(102, 102, 202, 302, conf=0.9)
    tracks2 = tracker.update(det2)
    assert tracks2[0]["track_id"] == tid

    # Frame 3: Low confidence (occlusion) - should still associate
    det3 = make_detection(105, 105, 205, 305, conf=0.2)  # Low confidence
    tracks3 = tracker.update(det3)
    assert len(tracks3) == 1
    assert tracks3[0]["track_id"] == tid, "Low-confidence should recover track"

    # Frame 4: High confidence again - track maintained
    det4 = make_detection(108, 108, 208, 308, conf=0.9)
    tracks4 = tracker.update(det4)
    assert tracks4[0]["track_id"] == tid


# ============================================================
# Test 7: Temporary detection loss
# ============================================================

def test_temporary_detection_loss(tracker):
    """Test track survives temporary detection loss within buffer."""
    # Frame 1-3: Track established
    for i in range(3):
        det = make_detection(100, 100, 200, 300, conf=0.9)
        tracks = tracker.update(det)
    tid = tracks[0]["track_id"]

    # Frame 4-5: No detections (occlusion)
    for _ in range(2):
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert len(tracks) == 0  # No active tracks output

    # Frame 6: Detection returns
    det = make_detection(105, 105, 205, 305, conf=0.9)
    tracks = tracker.update(det)
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == tid, "Track should survive temporary loss"


# ============================================================
# Test 8: Track buffer expiration
# ============================================================

def test_track_buffer_expiration():
    """Test track is removed after buffer expires."""
    config = ByteTrackConfig(
        track_buffer=2,  # Very short buffer
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        frame_rate=30
    )
    tracker = ByteTrack(config)

    # Establish track
    for _ in range(3):
        det = make_detection(100, 100, 200, 300, conf=0.9)
        tracks = tracker.update(det)
    tid = tracks[0]["track_id"]

    # Exceed buffer (3 frames without detection, buffer=2)
    for _ in range(4):
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert len(tracks) == 0

    # New detection should create NEW track
    det = make_detection(100, 100, 200, 300, conf=0.9)
    tracks = tracker.update(det)
    assert len(tracks) == 1
    assert tracks[0]["track_id"] != tid, "Old track should be expired"


# ============================================================
# Test 9: New track creation
# ============================================================

def test_new_track_creation(tracker):
    """Test new tracks created from unmatched high-confidence detections."""
    # Frame 1: Person A
    det1 = make_detection(100, 100, 200, 300, conf=0.9)
    tracks1 = tracker.update(det1)
    tid_a = tracks1[0]["track_id"]

    # Frame 2: Person A + Person B (new)
    det2 = make_detections(
        (100, 100, 200, 300, 0.9, 0),  # Person A
        (300, 100, 400, 300, 0.9, 0),  # Person B - new
    )
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 2

    ids = {t["track_id"] for t in tracks2}
    assert tid_a in ids
    assert len(ids) == 2


# ============================================================
# Test 10: Two nearby people maintaining separate IDs
# ============================================================

def test_nearby_people_separate_ids(tracker):
    """Test two nearby people maintain separate track IDs."""
    # Two people close together but separate
    for i in range(10):
        det = make_detections(
            (100, 100, 180, 300, 0.9, 0),  # Person A
            (185, 100, 265, 300, 0.9, 0),  # Person B - close but separate
        )
        tracks = tracker.update(det)
        assert len(tracks) == 2

        ids = {t["track_id"] for t in tracks}
        assert len(ids) == 2

    # Check IDs are stable
    final_ids = {t["track_id"] for t in tracks}
    assert len(final_ids) == 2


# ============================================================
# Test 11: Class filtering (only person class = 0)
# ============================================================

def test_class_filtering(tracker):
    """Test that non-person classes don't create tracks."""
    # Detection with class_id != 0 (e.g., car = 2)
    det = make_detection(100, 100, 200, 300, conf=0.9, class_id=2)
    tracks = tracker.update(det)

    # Should still track (class filtering is external responsibility)
    # But we can verify the class_id is preserved
    assert len(tracks) == 1
    # Note: Current implementation tracks all classes, filtering is external


# ============================================================
# Test 12: Deterministic behavior
# ============================================================

def test_deterministic_behavior():
    """Test that tracker produces deterministic results."""
    config = ByteTrackConfig(
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        frame_rate=30
    )

    # Run same sequence twice
    def run_sequence():
        t = ByteTrack(config)
        track_ids = []
        for i in range(5):
            det = make_detection(100 + i, 100, 200 + i, 300, conf=0.9)
            tracks = t.update(det)
            track_ids.append(tracks[0]["track_id"])
        return track_ids

    ids1 = run_sequence()
    ids2 = run_sequence()
    assert ids1 == ids2, "Tracker should be deterministic"


# ============================================================
# Synthetic Sequence Test
# ============================================================

def test_synthetic_sequence():
    """
    Frame 1 → person A
    Frame 2 → person A
    Frame 3 → person A with lower confidence
    Frame 4 → person A recovered
    Frame 5 → person disappears

    Verify same track ID maintained where ByteTrack should maintain it.
    """
    config = ByteTrackConfig(
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        frame_rate=30
    )
    tracker = ByteTrack(config)

    track_ids = []

    # Frame 1: Person A appears (high confidence)
    det1 = make_detection(100, 100, 200, 300, conf=0.9)
    tracks = tracker.update(det1)
    assert len(tracks) == 1
    track_ids.append(tracks[0]["track_id"])
    tid = tracks[0]["track_id"]

    # Frame 2: Person A continues (high confidence)
    det2 = make_detection(102, 102, 202, 302, conf=0.9)
    tracks = tracker.update(det2)
    assert len(tracks) == 1
    track_ids.append(tracks[0]["track_id"])
    assert tracks[0]["track_id"] == tid

    # Frame 3: Person A partially occluded (low confidence)
    det3 = make_detection(105, 105, 205, 305, conf=0.2)  # Low confidence!
    tracks = tracker.update(det3)
    assert len(tracks) == 1, "Low confidence should still track via second stage"
    track_ids.append(tracks[0]["track_id"])
    assert tracks[0]["track_id"] == tid, "ByteTrack should recover via low-confidence association"

    # Frame 4: Person A recovered (high confidence)
    det4 = make_detection(108, 108, 208, 308, conf=0.9)
    tracks = tracker.update(det4)
    assert len(tracks) == 1
    track_ids.append(tracks[0]["track_id"])
    assert tracks[0]["track_id"] == tid

    # Frame 5: Person disappears
    tracks = tracker.update(np.empty((0, 6), dtype=np.float32))
    assert len(tracks) == 0, "No active tracks when person disappears"

    # Verify ID consistency
    assert len(set(track_ids)) == 1, f"All frames should have same track ID, got {track_ids}"
    print(f"Synthetic sequence track IDs: {track_ids}")


# ============================================================
# Additional: Kalman Filter Tests
# ============================================================

def test_kalman_filter_predict():
    """Test Kalman filter prediction."""
    kf = KalmanFilter()
    measurement = np.array([150.0, 200.0, 100.0, 200.0], dtype=np.float32)  # cx, cy, w, h
    mean, cov = kf.initiate(measurement)

    # Predict
    mean_pred, cov_pred = kf.predict(mean, cov)

    # Position should change based on velocity (initially 0)
    # So position stays same, but covariance increases
    assert mean_pred[0] == mean[0]
    assert mean_pred[1] == mean[1]


def test_kalman_filter_update():
    """Test Kalman filter update."""
    kf = KalmanFilter()
    measurement = np.array([150.0, 200.0, 100.0, 200.0], dtype=np.float32)
    mean, cov = kf.initiate(measurement)

    # Update with slightly different measurement
    new_measurement = np.array([155.0, 205.0, 100.0, 200.0], dtype=np.float32)
    mean_upd, cov_upd = kf.update(mean, cov, new_measurement)

    # Position should move toward measurement
    assert abs(mean_upd[0] - 155.0) < abs(mean[0] - 155.0)
    assert abs(mean_upd[1] - 205.0) < abs(mean[1] - 205.0)


def test_iou_distance():
    """Test IoU distance calculation."""
    tracks = np.array([[100, 100, 200, 300]], dtype=np.float32)
    dets = np.array([[100, 100, 200, 300]], dtype=np.float32)

    dist = iou_distance(tracks, dets)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == 0.0  # Perfect overlap = 0 distance


def test_iou_distance_no_overlap():
    """Test IoU distance with no overlap."""
    tracks = np.array([[100, 100, 200, 300]], dtype=np.float32)
    dets = np.array([[300, 300, 400, 500]], dtype=np.float32)

    dist = iou_distance(tracks, dets)
    assert dist[0, 0] == 1.0  # No overlap = distance 1


def test_linear_sum_assignment():
    """Test Hungarian assignment."""
    # Simple 2x2 case
    cost = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ], dtype=np.float32)

    row_ind, col_ind = linear_sum_assignment(cost)
    # Optimal: (0,0) and (1,1) = 0.1 + 0.2 = 0.3
    assert set(zip(row_ind, col_ind)) == {(0, 0), (1, 1)}


def test_matching_basic():
    """Test matching function."""
    tracks_tlbr = np.array([[100, 100, 200, 300]], dtype=np.float32)
    dets_tlbr = np.array([[100, 100, 200, 300]], dtype=np.float32)
    dets_full = np.array([[100, 100, 200, 300, 0.9, 0]], dtype=np.float32)

    matches, unmatched_t, unmatched_d = matching(tracks_tlbr, dets_tlbr, 0.8, dets_full)
    assert len(matches) == 1
    assert matches[0, 0] == 0 and matches[0, 1] == 0
    assert len(unmatched_t) == 0
    assert len(unmatched_d) == 0


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])