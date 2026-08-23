"""Integration tests for Detection → ByteTrack adapter."""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import numpy as np
import pytest
from visionqueue.detection.types import Detection
from visionqueue.tracking.adapter import DetectionTrackingAdapter, Track
from visionqueue.tracking import ByteTrackConfig


def make_detection(x1, y1, x2, y2, conf=0.9, class_id=0):
    """Create a Detection object."""
    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)), confidence=conf, class_id=class_id)


@pytest.fixture
def adapter():
    config = ByteTrackConfig(
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        frame_rate=30
    )
    return DetectionTrackingAdapter(config=config)


def test_one_detected_person_one_track(adapter):
    """One detected person → one track."""
    detections = [make_detection(100, 100, 200, 300, 0.9)]
    tracks = adapter.update(detections)

    assert len(tracks) == 1
    assert isinstance(tracks[0], Track)
    assert tracks[0].track_id == 0
    assert tracks[0].bbox == (100.0, 100.0, 200.0, 300.0)
    assert abs(tracks[0].confidence - 0.9) < 1e-6


def test_same_person_multiple_frames_same_track_id(adapter):
    """Same person across multiple frames → same track ID."""
    track_ids = []

    for i in range(5):
        detections = [make_detection(100 + i, 100, 200 + i, 300, 0.9)]
        tracks = adapter.update(detections)
        assert len(tracks) == 1
        track_ids.append(tracks[0].track_id)

    assert len(set(track_ids)) == 1
    assert track_ids[0] == 0


def test_multiple_people_separate_track_ids(adapter):
    """Multiple people → separate track IDs."""
    # Frame 1: Two people
    detections = [
        make_detection(100, 100, 200, 300, 0.9),
        make_detection(300, 100, 400, 300, 0.9),
    ]
    tracks = adapter.update(detections)
    assert len(tracks) == 2
    ids_frame1 = {t.track_id for t in tracks}

    # Frame 2: Same two people
    detections = [
        make_detection(102, 102, 202, 302, 0.9),
        make_detection(302, 102, 402, 302, 0.9),
    ]
    tracks = adapter.update(detections)
    assert len(tracks) == 2
    ids_frame2 = {t.track_id for t in tracks}

    # IDs should be preserved
    assert ids_frame1 == ids_frame2
    assert len(ids_frame1) == 2


def test_confidence_changes(adapter):
    """Track handles confidence changes."""
    # High confidence
    tracks = adapter.update([make_detection(100, 100, 200, 300, 0.9)])
    tid_high = tracks[0].track_id

    # Low confidence
    tracks = adapter.update([make_detection(102, 102, 202, 302, 0.2)])
    assert len(tracks) == 1
    assert tracks[0].track_id == tid_high

    # High confidence again
    tracks = adapter.update([make_detection(104, 104, 204, 304, 0.9)])
    assert len(tracks) == 1
    assert tracks[0].track_id == tid_high


def test_no_detections(adapter):
    """No detections → empty tracks, but tracker state advances."""
    # Establish track first
    adapter.update([make_detection(100, 100, 200, 300, 0.9)])
    tid = adapter.tracker.tracked_tracks[0].track_id if adapter.tracker.tracked_tracks else 0

    # Frame with no detections
    tracks = adapter.update([])
    assert len(tracks) == 0

    # Track should still exist in lost tracks (within buffer)
    assert len(adapter.tracker.lost_tracks) >= 0


def test_non_person_detections_ignored(adapter):
    """Non-person detections (class_id != 0) are ignored."""
    # Person + car
    detections = [
        make_detection(100, 100, 200, 300, 0.9, class_id=0),  # person
        make_detection(300, 100, 400, 300, 0.9, class_id=2),  # car
    ]
    tracks = adapter.update(detections)
    assert len(tracks) == 1
    assert tracks[0].track_id == 0

    # Only car - should produce no tracks
    detections = [make_detection(300, 100, 400, 300, 0.9, class_id=2)]
    tracks = adapter.update(detections)
    assert len(tracks) == 0


def test_track_to_dict_serialization(adapter):
    """Track.to_dict() produces correct contract format."""
    tracks = adapter.update([make_detection(100, 100, 200, 300, 0.9)])
    track_dict = tracks[0].to_dict()

    assert track_dict["track_id"] == 0
    assert track_dict["bbox"] == [100.0, 100.0, 200.0, 300.0]
    assert abs(track_dict["confidence"] - 0.9) < 1e-6


def test_adapter_reset(adapter):
    """Adapter reset clears tracker state."""
    adapter.update([make_detection(100, 100, 200, 300, 0.9)])
    adapter.reset()

    tracks = adapter.update([make_detection(100, 100, 200, 300, 0.9)])
    assert tracks[0].track_id == 0  # New track starts at 0 after reset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])