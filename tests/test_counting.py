"""Comprehensive unit tests for the VisionQueue counting subsystem."""

import pytest
from typing import List, Tuple

from visionqueue.counting import (
    CrossingDirection,
    CrossingEvent,
    LineCrossingCounts,
    LineCrossingCounter,
    OccupancyCounter,
    OccupancyState,
    SessionCounter,
    SessionCounts,
    VirtualLine,
)
from visionqueue.roi import ROIConfig, ROIFilter
from visionqueue.tracking.adapter import Track


def make_track(
    track_id: int,
    bbox: Tuple[float, float, float, float] = (100.0, 100.0, 150.0, 200.0),
    confidence: float = 0.9,
) -> Track:
    """Helper to construct a Track object."""
    return Track(track_id=track_id, bbox=bbox, confidence=confidence)


# ============================================================================
# 1. Occupancy Counter Tests
# ============================================================================

class TestOccupancyCounter:
    """Tests for in-ROI instantaneous current occupancy counting."""

    def test_init_with_roi_config(self):
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        assert counter.roi == roi
        assert counter.count([]) == 0

    def test_init_with_roi_filter(self):
        roi_filter = ROIFilter(ROIConfig(x=50, y=50, width=100, height=100), 640, 480)
        counter = OccupancyCounter(roi_filter)
        assert counter.roi == roi_filter.roi

    def test_init_invalid_type_raises(self):
        with pytest.raises(TypeError):
            OccupancyCounter("invalid")  # type: ignore

    def test_zero_tracks(self):
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        assert counter.count([]) == 0
        state = counter.update([], frame_id=1, timestamp=0.033)
        assert state.current_count == 0
        assert state.active_track_ids == []
        assert state.frame_id == 1
        assert state.timestamp == 0.033

    def test_one_track_inside(self):
        # ROI from [100, 100] to [300, 300]
        # Track bottom center: ((150+250)/2, 250) = (200, 250) -> inside
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        track = make_track(track_id=1, bbox=(150.0, 150.0, 250.0, 250.0))

        assert counter.count([track]) == 1
        state = counter.update([track], frame_id=1)
        assert state.current_count == 1
        assert state.active_track_ids == [1]

    def test_one_track_outside(self):
        # ROI from [100, 100] to [300, 300]
        # Track bottom center: ((400+500)/2, 500) = (450, 500) -> outside
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        track = make_track(track_id=1, bbox=(400.0, 400.0, 500.0, 500.0))

        assert counter.count([track]) == 0
        state = counter.update([track], frame_id=1)
        assert state.current_count == 0
        assert state.active_track_ids == []

    def test_multiple_tracks_inside(self):
        roi = ROIConfig(x=100, y=100, width=300, height=300)
        counter = OccupancyCounter(roi)
        t1 = make_track(track_id=10, bbox=(120.0, 120.0, 180.0, 220.0))  # (150, 220) -> inside
        t2 = make_track(track_id=20, bbox=(220.0, 150.0, 280.0, 350.0))  # (250, 350) -> inside
        t3 = make_track(track_id=5, bbox=(150.0, 150.0, 250.0, 250.0))   # (200, 250) -> inside

        assert counter.count([t1, t2, t3]) == 3
        state = counter.update([t1, t2, t3], frame_id=2)
        assert state.current_count == 3
        assert state.active_track_ids == [5, 10, 20]  # Sorted

    def test_mixed_tracks_inside_and_outside(self):
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        t_in1 = make_track(track_id=1, bbox=(120.0, 120.0, 180.0, 220.0))   # inside
        t_out = make_track(track_id=2, bbox=(500.0, 500.0, 550.0, 600.0))   # outside
        t_in2 = make_track(track_id=3, bbox=(140.0, 140.0, 200.0, 240.0))   # inside

        state = counter.update([t_in1, t_out, t_in2], frame_id=3)
        assert state.current_count == 2
        assert state.active_track_ids == [1, 3]

    def test_duplicate_track_id_deduplication(self):
        # Even if multiple detections share the same track_id, it counts only once
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        t1 = make_track(track_id=7, bbox=(120.0, 120.0, 180.0, 220.0))
        t2 = make_track(track_id=7, bbox=(140.0, 140.0, 200.0, 240.0))

        assert counter.count([t1, t2]) == 1
        state = counter.update([t1, t2], frame_id=4)
        assert state.current_count == 1
        assert state.active_track_ids == [7]

    def test_frame_to_frame_transitions(self):
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)

        # Frame 1: track 1 outside, track 2 inside
        t1_f1 = make_track(track_id=1, bbox=(50.0, 50.0, 80.0, 80.0))      # (65, 80) outside
        t2_f1 = make_track(track_id=2, bbox=(150.0, 150.0, 250.0, 250.0))  # (200, 250) inside
        s1 = counter.update([t1_f1, t2_f1], frame_id=1)
        assert s1.current_count == 1
        assert s1.active_track_ids == [2]

        # Frame 2: track 1 moves inside, track 2 stays inside
        t1_f2 = make_track(track_id=1, bbox=(120.0, 120.0, 180.0, 180.0))  # (150, 180) inside
        t2_f2 = make_track(track_id=2, bbox=(160.0, 160.0, 240.0, 240.0))  # (200, 240) inside
        s2 = counter.update([t1_f2, t2_f2], frame_id=2)
        assert s2.current_count == 2
        assert s2.active_track_ids == [1, 2]

        # Frame 3: track 2 leaves ROI, track 1 stays
        t1_f3 = make_track(track_id=1, bbox=(120.0, 120.0, 180.0, 180.0))
        t2_f3 = make_track(track_id=2, bbox=(400.0, 400.0, 450.0, 450.0))  # outside
        s3 = counter.update([t1_f3, t2_f3], frame_id=3)
        assert s3.current_count == 1
        assert s3.active_track_ids == [1]

    def test_update_roi(self):
        roi1 = ROIConfig(x=100, y=100, width=100, height=100)
        roi2 = ROIConfig(x=300, y=300, width=100, height=100)
        counter = OccupancyCounter(roi1)

        track = make_track(track_id=1, bbox=(320.0, 320.0, 360.0, 360.0))  # (340, 360)
        assert counter.count([track]) == 0

        counter.update_roi(roi2)
        assert counter.count([track]) == 1

    def test_reset(self):
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi)
        track = make_track(track_id=1, bbox=(150.0, 150.0, 250.0, 250.0))
        counter.update([track], frame_id=1)
        assert counter.last_state.current_count == 1

        counter.reset()
        assert counter.last_state.current_count == 0
        assert counter.last_state.active_track_ids == []

    def test_occupancy_state_to_dict(self):
        state = OccupancyState(current_count=2, active_track_ids=[1, 4], frame_id=10, timestamp=1.5)
        d = state.to_dict()
        assert d == {
            "current_count": 2,
            "active_track_ids": [1, 4],
            "frame_id": 10,
            "timestamp": 1.5,
        }

    def test_init_without_roi_defaults_to_whole_frame(self):
        counter = OccupancyCounter()
        assert counter.is_whole_frame is True
        assert counter.enable_roi is False
        assert counter.roi is None

        # Tracks placed in distant corners of FOV
        t1 = make_track(track_id=1, bbox=(10.0, 10.0, 50.0, 80.0))
        t2 = make_track(track_id=2, bbox=(550.0, 400.0, 620.0, 470.0))
        assert counter.count([t1, t2]) == 2

        state = counter.update([t1, t2], frame_id=1, timestamp=0.033)
        assert state.current_count == 2
        assert state.active_track_ids == [1, 2]

    def test_enable_roi_flag_switching(self):
        roi = ROIConfig(x=100, y=100, width=200, height=200)
        counter = OccupancyCounter(roi, enable_roi=False)
        assert counter.is_whole_frame is True
        assert counter.enable_roi is False

        # Track 1 inside ROI [100, 100, 300, 300], Track 2 outside ROI at (500, 400)
        t1 = make_track(track_id=1, bbox=(150.0, 150.0, 250.0, 250.0))
        t2 = make_track(track_id=2, bbox=(450.0, 350.0, 550.0, 450.0))

        # In whole-frame mode: both are counted
        assert counter.count([t1, t2]) == 2
        s1 = counter.update([t1, t2], frame_id=1)
        assert s1.current_count == 2
        assert s1.active_track_ids == [1, 2]

        # Enable ROI mode: only t1 is counted
        counter.enable_roi = True
        assert counter.is_whole_frame is False
        assert counter.count([t1, t2]) == 1
        s2 = counter.update([t1, t2], frame_id=2)
        assert s2.current_count == 1
        assert s2.active_track_ids == [1]

        # Disable ROI mode: back to whole frame
        counter.enable_roi = False
        assert counter.count([t1, t2]) == 2



# ============================================================================
# 2. Line Crossing Counter Tests
# ============================================================================

class TestLineCrossingCounter:
    """Tests for virtual line crossing entry/exit detection."""

    @pytest.fixture
    def vertical_line(self):
        # Vertical line from top (300, 0) to bottom (300, 500)
        # Vector is (0, 500).
        # Left of line is x < 300 (cross product > 0).
        # Right of line is x > 300 (cross product < 0).
        # Default: Left-to-Right (x < 300 to x > 300) is ENTRY.
        # Right-to-Left (x > 300 to x < 300) is EXIT.
        return VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0))

    @pytest.fixture
    def horizontal_line(self):
        # Horizontal line from left (0, 300) to right (500, 300)
        # Vector is (500, 0).
        # Left side of line (y < 300) -> cross product > 0.
        # Right side of line (y > 300) -> cross product < 0.
        # Default: y < 300 to y > 300 is ENTRY.
        return VirtualLine(pt1=(0.0, 300.0), pt2=(500.0, 300.0))

    def test_virtual_line_invalid_points_raises(self):
        with pytest.raises(ValueError):
            VirtualLine(pt1=(100.0, 100.0), pt2=(100.0, 100.0))

    def test_virtual_line_properties(self):
        line = VirtualLine(pt1=(0.0, 0.0), pt2=(300.0, 400.0))
        assert line.x1 == 0.0
        assert line.y1 == 0.0
        assert line.x2 == 300.0
        assert line.y2 == 400.0
        assert line.length == 500.0

    def test_clear_entry_crossing(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Frame 1: Track at x=200 (left side) -> bottom-center (200, 200)
        t_f1 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))
        events_f1 = counter.update([t_f1], frame_id=1)
        assert len(events_f1) == 0
        assert counter.entries == 0
        assert counter.exits == 0

        # Frame 2: Track moves across line to x=400 (right side) -> bottom-center (400, 200)
        t_f2 = make_track(track_id=1, bbox=(380.0, 150.0, 420.0, 200.0))
        events_f2 = counter.update([t_f2], frame_id=2, timestamp=0.066)

        assert len(events_f2) == 1
        ev = events_f2[0]
        assert ev.track_id == 1
        assert ev.direction == CrossingDirection.ENTRY
        assert ev.frame_id == 2
        assert ev.timestamp == 0.066
        assert pytest.approx(ev.crossing_point[0], 0.1) == 300.0
        assert pytest.approx(ev.crossing_point[1], 0.1) == 200.0

        assert counter.entries == 1
        assert counter.exits == 0
        assert counter.net_count == 1

    def test_clear_exit_crossing(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Frame 1: Track at x=400 (right side) -> bottom-center (400, 200)
        t_f1 = make_track(track_id=2, bbox=(380.0, 150.0, 420.0, 200.0))
        counter.update([t_f1], frame_id=1)

        # Frame 2: Track moves across line to x=200 (left side) -> bottom-center (200, 200)
        t_f2 = make_track(track_id=2, bbox=(180.0, 150.0, 220.0, 200.0))
        events = counter.update([t_f2], frame_id=2)

        assert len(events) == 1
        assert events[0].track_id == 2
        assert events[0].direction == CrossingDirection.EXIT
        assert counter.entries == 0
        assert counter.exits == 1
        assert counter.net_count == -1

    def test_no_crossing_movement_on_same_side(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Track stays on left side (x=100 -> x=200)
        t_f1 = make_track(track_id=1, bbox=(80.0, 150.0, 120.0, 200.0))
        t_f2 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))

        counter.update([t_f1], frame_id=1)
        events = counter.update([t_f2], frame_id=2)

        assert len(events) == 0
        assert counter.entries == 0
        assert counter.exits == 0

    def test_movement_parallel_to_line(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Track moves parallel to vertical line at x=200, y: 100 -> 400
        t_f1 = make_track(track_id=1, bbox=(180.0, 50.0, 220.0, 100.0))
        t_f2 = make_track(track_id=1, bbox=(180.0, 350.0, 220.0, 400.0))

        counter.update([t_f1], frame_id=1)
        events = counter.update([t_f2], frame_id=2)

        assert len(events) == 0
        assert counter.entries == 0

    def test_trajectory_passes_outside_line_segment_bounds(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Line is (300, 0) -> (300, 500).
        # Track crosses infinite x=300 line at y=700 (outside segment range)
        t_f1 = make_track(track_id=1, bbox=(180.0, 650.0, 220.0, 700.0))  # (200, 700)
        t_f2 = make_track(track_id=1, bbox=(380.0, 650.0, 420.0, 700.0))  # (400, 700)

        counter.update([t_f1], frame_id=1)
        events = counter.update([t_f2], frame_id=2)

        assert len(events) == 0
        assert counter.entries == 0

    def test_anti_chatter_stationary_near_or_on_line(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # 1. Track starts on left side (x=200)
        t1 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))
        counter.update([t1], frame_id=1)

        # 2. Track crosses to right side (x=350) -> triggers ENTRY
        t2 = make_track(track_id=1, bbox=(330.0, 150.0, 370.0, 200.0))
        events2 = counter.update([t2], frame_id=2)
        assert len(events2) == 1
        assert events2[0].direction == CrossingDirection.ENTRY
        assert counter.entries == 1

        # 3. Track stays hovering on right side around x=350 for multiple frames
        t3 = make_track(track_id=1, bbox=(332.0, 150.0, 368.0, 200.0))
        t4 = make_track(track_id=1, bbox=(328.0, 150.0, 372.0, 200.0))
        events3 = counter.update([t3], frame_id=3)
        events4 = counter.update([t4], frame_id=4)

        assert len(events3) == 0
        assert len(events4) == 0
        assert counter.entries == 1  # No repeated counts!

    def test_re_crossing_in_opposite_direction(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Left -> Right (ENTRY)
        t1 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))  # x=200
        t2 = make_track(track_id=1, bbox=(380.0, 150.0, 420.0, 200.0))  # x=400
        counter.update([t1], frame_id=1)
        ev1 = counter.update([t2], frame_id=2)
        assert len(ev1) == 1
        assert ev1[0].direction == CrossingDirection.ENTRY

        # Right -> Left (EXIT)
        t3 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))  # x=200
        ev2 = counter.update([t3], frame_id=3)
        assert len(ev2) == 1
        assert ev2[0].direction == CrossingDirection.EXIT

        assert counter.entries == 1
        assert counter.exits == 1
        assert counter.net_count == 0

    def test_multiple_simultaneous_crossings(self, vertical_line):
        counter = LineCrossingCounter(vertical_line)

        # Track 1 on left, Track 2 on right
        t1_f1 = make_track(track_id=1, bbox=(180.0, 100.0, 220.0, 150.0))  # (200, 150)
        t2_f1 = make_track(track_id=2, bbox=(380.0, 200.0, 420.0, 250.0))  # (400, 250)
        counter.update([t1_f1, t2_f1], frame_id=1)

        # Track 1 moves left->right (ENTRY), Track 2 moves right->left (EXIT)
        t1_f2 = make_track(track_id=1, bbox=(380.0, 100.0, 420.0, 150.0))  # (400, 150)
        t2_f2 = make_track(track_id=2, bbox=(180.0, 200.0, 220.0, 250.0))  # (200, 250)
        events = counter.update([t1_f2, t2_f2], frame_id=2)

        assert len(events) == 2
        dir_map = {e.track_id: e.direction for e in events}
        assert dir_map[1] == CrossingDirection.ENTRY
        assert dir_map[2] == CrossingDirection.EXIT
        assert counter.entries == 1
        assert counter.exits == 1

    def test_horizontal_line_crossing(self, horizontal_line):
        counter = LineCrossingCounter(horizontal_line)

        # Line from (0, 300) to (500, 300) in OpenCV coords (+y down).
        # Bottom-to-Top: y=400 -> y=200 (Left-to-Right in directed frame, ENTRY)
        t1 = make_track(track_id=1, bbox=(200.0, 350.0, 300.0, 400.0))  # bottom_y = 400
        t2 = make_track(track_id=1, bbox=(200.0, 150.0, 300.0, 200.0))  # bottom_y = 200

        counter.update([t1], frame_id=1)
        events = counter.update([t2], frame_id=2)

        assert len(events) == 1
        assert events[0].direction == CrossingDirection.ENTRY
        assert counter.entries == 1

        # Top-to-Bottom: y=200 -> y=400 (EXIT)
        t3 = make_track(track_id=2, bbox=(200.0, 150.0, 300.0, 200.0))  # bottom_y = 200
        t4 = make_track(track_id=2, bbox=(200.0, 350.0, 300.0, 400.0))  # bottom_y = 400
        counter.update([t3], frame_id=3)
        events2 = counter.update([t4], frame_id=4)

        assert len(events2) == 1
        assert events2[0].direction == CrossingDirection.EXIT
        assert counter.exits == 1

    def test_track_expiration_and_stale_cleanup(self, vertical_line):
        counter = LineCrossingCounter(vertical_line, track_max_age=5)

        # Track 1 appears at frame 1
        t1 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))
        counter.update([t1], frame_id=1)
        assert 1 in counter._track_states

        # Track 1 disappears for 10 frames (frame 10)
        t2 = make_track(track_id=2, bbox=(180.0, 150.0, 220.0, 200.0))
        counter.update([t2], frame_id=10)

        # Track 1 state should be purged
        assert 1 not in counter._track_states
        assert 2 in counter._track_states

    def test_reset_and_update_line(self, vertical_line, horizontal_line):
        counter = LineCrossingCounter(vertical_line)
        t1 = make_track(track_id=1, bbox=(180.0, 150.0, 220.0, 200.0))
        t2 = make_track(track_id=1, bbox=(380.0, 150.0, 420.0, 200.0))
        counter.update([t1], frame_id=1)
        counter.update([t2], frame_id=2)
        assert counter.entries == 1

        counter.reset()
        assert counter.entries == 0
        assert counter.exits == 0
        assert counter.net_count == 0
        assert len(counter.events) == 0

        counter.update_line(horizontal_line)
        assert counter.line == horizontal_line

    def test_crossing_event_to_dict(self):
        ev = CrossingEvent(
            track_id=3,
            direction=CrossingDirection.ENTRY,
            frame_id=12,
            timestamp=0.4,
            crossing_point=(300.0, 200.0),
            bbox=(280.0, 150.0, 320.0, 200.0),
        )
        d = ev.to_dict()
        assert d["track_id"] == 3
        assert d["direction"] == "ENTRY"
        assert d["crossing_point"] == [300.0, 200.0]

    def test_line_crossing_counts_to_dict(self):
        counts = LineCrossingCounts(entries=5, exits=2)
        assert counts.net_count == 3
        assert counts.to_dict() == {"entries": 5, "exits": 2, "net_count": 3}


# ============================================================================
# 3. Session Counter Tests
# ============================================================================

class TestSessionCounter:
    """Tests for approximate session-unique visitor counting."""

    def test_initial_state(self):
        counter = SessionCounter()
        assert counter.approximate_unique_count == 0
        assert counter.seen_track_ids == set()

    def test_first_track_counted(self):
        counter = SessionCounter()
        t1 = make_track(track_id=1)
        counts = counter.update([t1])

        assert counts.approximate_unique_count == 1
        assert counts.currently_active_count == 1
        assert counts.active_track_ids == [1]

    def test_same_track_repeated_does_not_increment(self):
        counter = SessionCounter()
        t1 = make_track(track_id=1)

        # 10 frames with the same track ID
        for _ in range(10):
            counts = counter.update([t1])

        assert counts.approximate_unique_count == 1
        assert counts.currently_active_count == 1
        assert counter.approximate_unique_count == 1

    def test_multiple_tracks_across_frames(self):
        counter = SessionCounter()

        # Frame 1: tracks 1 and 2
        t1 = make_track(track_id=1)
        t2 = make_track(track_id=2)
        c1 = counter.update([t1, t2])
        assert c1.approximate_unique_count == 2
        assert c1.currently_active_count == 2
        assert c1.active_track_ids == [1, 2]

        # Frame 2: track 2 and new track 3
        t3 = make_track(track_id=3)
        c2 = counter.update([t2, t3])
        assert c2.approximate_unique_count == 3
        assert c2.currently_active_count == 2
        assert c2.active_track_ids == [2, 3]

        # Frame 3: no tracks active
        c3 = counter.update([])
        assert c3.approximate_unique_count == 3
        assert c3.currently_active_count == 0
        assert c3.active_track_ids == []

    def test_new_session_resets_seen_ids(self):
        counter = SessionCounter()
        t1 = make_track(track_id=1)
        t2 = make_track(track_id=2)
        counter.update([t1, t2])
        assert counter.approximate_unique_count == 2

        counter.new_session()
        assert counter.approximate_unique_count == 0
        assert counter.seen_track_ids == set()

        # Track 1 in new session is counted from zero
        c = counter.update([t1])
        assert c.approximate_unique_count == 1

    def test_session_counts_to_dict(self):
        counts = SessionCounts(
            cumulative_track_instances=5,
            approximate_unique_count=5,
            currently_active_count=2,
            active_track_ids=[3, 7],
        )
        d = counts.to_dict()
        assert d["approximate_unique_count"] == 5
        assert d["cumulative_track_instances"] == 5
        assert d["currently_active_count"] == 2
        assert d["active_track_ids"] == [3, 7]
