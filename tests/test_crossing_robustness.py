"""VisionQueue V1.1 — Entry/Exit Robustness Regression Tests.

Tests for the LineCrossingCounter under diverse geometric, motion,
and lifecycle conditions, including the new min_track_age_frames guard.
"""

import pytest
from typing import Tuple

from visionqueue.counting import (
    CrossingDirection,
    LineCrossingCounter,
    VirtualLine,
)
from visionqueue.tracking.adapter import Track


def make_track(
    track_id: int,
    bbox: Tuple[float, float, float, float] = (100.0, 100.0, 150.0, 200.0),
    confidence: float = 0.9,
) -> Track:
    return Track(track_id=track_id, bbox=bbox, confidence=confidence)


class TestMinTrackAgeGuard:
    """Tests for the min_track_age_frames robustness parameter."""

    def test_default_no_age_guard(self):
        """Default min_track_age_frames=0 preserves legacy behavior."""
        line = VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))
        counter = LineCrossingCounter(line)
        t1 = make_track(1, bbox=(180.0, 150.0, 220.0, 200.0))
        counter.update([t1], frame_id=1)
        t2 = make_track(1, bbox=(380.0, 150.0, 420.0, 200.0))
        events = counter.update([t2], frame_id=2)
        assert len(events) == 1
        assert events[0].direction == CrossingDirection.ENTRY

    def test_min_track_age_suppresses_first_frame_crossing(self):
        """A track that crosses on its very first movement step is suppressed
        when min_track_age_frames=1. The crossing is emitted only after the
        track has been observed for at least 1 frame."""
        line = VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))
        counter = LineCrossingCounter(line, min_track_age_frames=1)

        # Frame 1: track born on left
        t1 = make_track(1, bbox=(180.0, 150.0, 220.0, 200.0))
        counter.update([t1], frame_id=1)
        assert counter.entries == 0

        # Frame 2: track crosses left->right (first motion step)
        # Track age = 2-1 = 1 >= min_track_age_frames=1 -> eligible, crossing emitted
        t2 = make_track(1, bbox=(380.0, 150.0, 420.0, 200.0))
        events2 = counter.update([t2], frame_id=2)
        assert len(events2) == 1
        assert counter.entries == 1

        # Frame 3: track crosses back right->left
        t3 = make_track(1, bbox=(180.0, 150.0, 220.0, 200.0))
        events3 = counter.update([t3], frame_id=3)
        assert len(events3) == 1
        assert events3[0].direction == CrossingDirection.EXIT
        assert counter.exits == 1

    def test_min_track_age_two_frames(self):
        """min_track_age_frames=2: first movement is suppressed (age=1 < 2),
        second movement is emitted (age=2 >= 2)."""
        line = VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))
        counter = LineCrossingCounter(line, min_track_age_frames=2)

        # Frame 1: track born on left
        counter.update([make_track(1, bbox=(180.0, 150.0, 220.0, 200.0))], frame_id=1)
        # Frame 2: still on left, age=1 < 2, not eligible
        counter.update([make_track(1, bbox=(190.0, 150.0, 230.0, 200.0))], frame_id=2)
        # Frame 3: crosses, age=2 >= 2 -> emitted
        events = counter.update([make_track(1, bbox=(380.0, 150.0, 420.0, 200.0))], frame_id=3)
        assert len(events) == 1
        assert counter.entries == 1

    def test_invalid_min_track_age_raises(self):
        with pytest.raises(ValueError):
            LineCrossingCounter(
                VirtualLine(pt1=(0.0, 0.0), pt2=(10.0, 0.0)),
                min_track_age_frames=-1,
            )


class TestDiagonalLine:
    def test_diagonal_line_no_crash(self):
        """Diagonal virtual line processes without crashing."""
        line = VirtualLine(pt1=(0.0, 0.0), pt2=(500.0, 500.0))
        counter = LineCrossingCounter(line)
        t1 = make_track(1, bbox=(50.0, 50.0, 70.0, 70.0))
        t2 = make_track(1, bbox=(150.0, 150.0, 170.0, 170.0))
        counter.update([t1], frame_id=1)
        events = counter.update([t2], frame_id=2)
        assert isinstance(events, list)


class TestSimultaneousCrossings:
    def test_multiple_tracks_cross_same_frame(self):
        """3 tracks crossing simultaneously each emit an independent event."""
        line = VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))
        counter = LineCrossingCounter(line)
        t1_f1 = make_track(1, bbox=(180.0, 100.0, 220.0, 150.0))
        t2_f1 = make_track(2, bbox=(180.0, 200.0, 220.0, 250.0))
        t3_f1 = make_track(3, bbox=(180.0, 300.0, 220.0, 350.0))
        counter.update([t1_f1, t2_f1, t3_f1], frame_id=1)
        t1_f2 = make_track(1, bbox=(380.0, 100.0, 420.0, 150.0))
        t2_f2 = make_track(2, bbox=(380.0, 200.0, 420.0, 250.0))
        t3_f2 = make_track(3, bbox=(380.0, 300.0, 420.0, 350.0))
        events = counter.update([t1_f2, t2_f2, t3_f2], frame_id=2)
        assert len(events) == 3
        assert all(e.direction == CrossingDirection.ENTRY for e in events)
        assert counter.entries == 3


class TestSessionReset:
    def test_session_reset_clears_counts(self):
        """reset() clears entry/exit counts and event history."""
        line = VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))
        counter = LineCrossingCounter(line)
        t1 = make_track(1, bbox=(180.0, 150.0, 220.0, 200.0))
        t2 = make_track(1, bbox=(380.0, 150.0, 420.0, 200.0))
        counter.update([t1], frame_id=1)
        counter.update([t2], frame_id=2)
        assert counter.entries == 1

        counter.reset()
        assert counter.entries == 0
        assert counter.exits == 0
        assert len(counter.events) == 0

        counter.update([t1], frame_id=10)
        events = counter.update([t2], frame_id=11)
        assert len(events) == 1
        assert counter.entries == 1


class TestShortLivedTrack:
    def test_track_appears_and_disappears_quickly(self):
        """A track that appears briefly then disappears should not corrupt state."""
        line = VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))
        counter = LineCrossingCounter(line, track_max_age=5)
        counter.update([make_track(1, bbox=(180.0, 150.0, 220.0, 200.0))], frame_id=1)
        counter.update([make_track(1, bbox=(190.0, 150.0, 230.0, 200.0))], frame_id=2)
        for f in range(3, 13):
            counter.update([], frame_id=f)
        assert counter.entries == 0
        assert counter.exits == 0
