"""Unit tests for VisionQueue ROI module."""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import pytest
from visionqueue.roi import (
    ROIConfig,
    TrackROIStatus,
    track_bottom_center,
    track_in_roi,
    filter_tracks_by_roi,
    get_track_roi_status,
    validate_roi,
    ROIFilter,
)
from visionqueue.tracking.adapter import Track


# ============================================================
# Fixtures
# ============================================================

def make_track(track_id: int, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Track:
    """Create a Track object for testing."""
    return Track(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        confidence=conf,
    )


def make_roi(x: int, y: int, width: int, height: int) -> ROIConfig:
    """Create an ROIConfig for testing."""
    return ROIConfig(x=x, y=y, width=width, height=height)


# ============================================================
# Tests: track_bottom_center
# ============================================================

def test_track_bottom_center_basic():
    """Test bottom-center calculation for a standard box."""
    bbox = (100.0, 100.0, 200.0, 300.0)
    cx, bottom_y = track_bottom_center(bbox)
    assert cx == 150.0
    assert bottom_y == 300.0


def test_track_bottom_center_zero_width():
    """Test bottom-center with zero-width box."""
    bbox = (100.0, 100.0, 100.0, 300.0)
    cx, bottom_y = track_bottom_center(bbox)
    assert cx == 100.0
    assert bottom_y == 300.0


def test_track_bottom_center_negative_coords():
    """Test bottom-center with negative coordinates."""
    bbox = (-50.0, -20.0, 50.0, 100.0)
    cx, bottom_y = track_bottom_center(bbox)
    assert cx == 0.0
    assert bottom_y == 100.0


def test_track_bottom_center_fractional():
    """Test bottom-center with fractional coordinates."""
    bbox = (100.5, 100.5, 201.5, 301.5)
    cx, bottom_y = track_bottom_center(bbox)
    assert cx == 151.0
    assert bottom_y == 301.5


# ============================================================
# Tests: ROIConfig
# ============================================================

def test_roi_config_valid():
    """Test valid ROI creation."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    assert roi.x == 100
    assert roi.y == 100
    assert roi.width == 200
    assert roi.height == 150
    assert roi.x2 == 300
    assert roi.y2 == 250
    assert roi.bounds == (100, 100, 300, 250)


def test_roi_config_invalid_width():
    """Test ROI with invalid width raises ValueError."""
    with pytest.raises(ValueError):
        ROIConfig(x=100, y=100, width=0, height=150)
    with pytest.raises(ValueError):
        ROIConfig(x=100, y=100, width=-10, height=150)


def test_roi_config_invalid_height():
    """Test ROI with invalid height raises ValueError."""
    with pytest.raises(ValueError):
        ROIConfig(x=100, y=100, width=200, height=0)
    with pytest.raises(ValueError):
        ROIConfig(x=100, y=100, width=200, height=-10)


def test_roi_contains_point_inside():
    """Test point inside ROI."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    assert roi.contains_point(150, 150) is True  # center
    assert roi.contains_point(100, 100) is True  # top-left edge
    assert roi.contains_point(299.9, 249.9) is True  # just inside bottom-right


def test_roi_contains_point_outside():
    """Test point outside ROI."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    assert roi.contains_point(99, 150) is False  # left
    assert roi.contains_point(150, 99) is False  # top
    assert roi.contains_point(300, 150) is False  # right edge (exclusive)
    assert roi.contains_point(150, 250) is False  # bottom edge (exclusive)


def test_roi_contains_point_boundary():
    """Test point exactly on ROI boundaries."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    # Half-open interval [x, x2) × [y, y2)
    assert roi.contains_point(100, 100) is True   # top-left corner (inclusive)
    assert roi.contains_point(300, 100) is False  # top-right corner (exclusive)
    assert roi.contains_point(100, 250) is False  # bottom-left corner (exclusive)
    assert roi.contains_point(300, 250) is False  # bottom-right corner (exclusive)


def test_roi_area():
    """Test ROI area calculation."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    assert roi.area() == 30000


def test_roi_intersection_area():
    """Test ROI intersection area."""
    roi1 = ROIConfig(x=100, y=100, width=200, height=150)
    roi2 = ROIConfig(x=150, y=150, width=200, height=150)
    assert roi1.intersection_area(roi2) == 150 * 100  # 150x100 overlap


def test_roi_intersection_none():
    """Test ROI intersection with no overlap."""
    roi1 = ROIConfig(x=100, y=100, width=200, height=150)
    roi2 = ROIConfig(x=500, y=500, width=100, height=100)
    assert roi1.intersection_area(roi2) == 0


def test_roi_iou():
    """Test ROI IoU calculation."""
    roi1 = ROIConfig(x=100, y=100, width=200, height=150)
    roi2 = ROIConfig(x=150, y=150, width=200, height=150)
    inter = 150 * 100  # 15000
    union = 30000 + 30000 - 15000  # 45000
    expected_iou = inter / union
    assert abs(roi1.iou(roi2) - expected_iou) < 1e-6


def test_roi_clamp_to_frame():
    """Test ROI clamping to frame boundaries."""
    roi = ROIConfig(x=-50, y=-20, width=200, height=150)
    clamped = roi.clamp_to_frame(640, 480)
    assert clamped.x == 0
    assert clamped.y == 0
    assert clamped.width == 150  # 200 - 50
    assert clamped.height == 130  # 150 - 20


def test_roi_clamp_oversized():
    """Test ROI clamping when oversized."""
    roi = ROIConfig(x=500, y=400, width=200, height=150)
    clamped = roi.clamp_to_frame(640, 480)
    assert clamped.x == 500
    assert clamped.y == 400
    assert clamped.width == 140  # 640 - 500
    assert clamped.height == 80   # 480 - 400


def test_roi_is_valid_for_frame():
    """Test ROI validity check against frame."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    assert roi.is_valid_for_frame(640, 480) is True
    assert roi.is_valid_for_frame(50, 50) is False  # completely outside


# ============================================================
# Tests: validate_roi
# ============================================================

def test_validate_roi_valid():
    """Test validating a valid ROI."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    validated = validate_roi(roi, 640, 480)
    assert validated == roi


def test_validate_roi_outside_clamped():
    """Test ROI completely outside frame gets clamped."""
    roi = ROIConfig(x=-100, y=-100, width=50, height=50)
    validated = validate_roi(roi, 640, 480)
    assert validated.x == 0
    assert validated.y == 0
    assert validated.width == 1
    assert validated.height == 1


def test_validate_roi_partial_clamped():
    """Test ROI partially outside frame gets clamped."""
    roi = ROIConfig(x=600, y=400, width=100, height=100)
    validated = validate_roi(roi, 640, 480)
    assert validated.x == 600
    assert validated.y == 400
    assert validated.width == 40  # 640 - 600
    assert validated.height == 80  # 480 - 400


def test_validate_roi_invalid_frame():
    """Test validate_roi with invalid frame dimensions."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    with pytest.raises(ValueError):
        validate_roi(roi, 0, 480)
    with pytest.raises(ValueError):
        validate_roi(roi, 640, 0)
    with pytest.raises(ValueError):
        validate_roi(roi, -1, 480)


# ============================================================
# Tests: track_in_roi
# ============================================================

def test_track_in_roi_inside():
    """Test track clearly inside ROI."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    track = make_track(0, 150, 120, 250, 220)  # bottom-center: (200, 220)
    assert track_in_roi(track, roi) is True


def test_track_in_roi_outside():
    """Test track clearly outside ROI."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    # Track at x=99 (clearly left of ROI which starts at x=100)
    track = make_track(0, 49, 50, 149, 150)  # bottom-center: (99, 150)
    assert track_in_roi(track, roi) is False
    
    # Track above ROI
    track = make_track(0, 150, 50, 250, 99)  # bottom-center: (200, 99)
    assert track_in_roi(track, roi) is False
    
    # Track right of ROI
    track = make_track(0, 301, 150, 401, 250)  # bottom-center: (351, 250)
    assert track_in_roi(track, roi) is False
    
    # Track below ROI
    track = make_track(0, 150, 251, 250, 351)  # bottom-center: (200, 351)
    assert track_in_roi(track, roi) is False


def test_track_in_roi_on_boundary():
    """Test track exactly on ROI boundary."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    # Bottom-center exactly on left edge (x=100) - should be INSIDE (inclusive)
    track = make_track(0, 100, 100, 200, 200)  # bottom-center: (150, 200)
    assert track_in_roi(track, roi) is True

    # Bottom-center exactly on right edge (x=300) - should be OUTSIDE (exclusive)
    track = make_track(0, 200, 100, 400, 200)  # bottom-center: (300, 200)
    assert track_in_roi(track, roi) is False

    # Bottom-center exactly on top edge (y=100) - should be INSIDE (inclusive)
    track = make_track(0, 150, 50, 250, 100)  # bottom-center: (200, 100)
    assert track_in_roi(track, roi) is True

    # Bottom-center exactly on bottom edge (y=250) - should be OUTSIDE (exclusive)
    track = make_track(0, 150, 100, 250, 250)  # bottom-center: (200, 250)
    assert track_in_roi(track, roi) is False


# ============================================================
# Tests: filter_tracks_by_roi
# ============================================================

def test_filter_tracks_by_roi_single_inside():
    """Test filtering with one track inside."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    tracks = [make_track(0, 150, 120, 250, 220)]
    filtered = filter_tracks_by_roi(tracks, roi)
    assert len(filtered) == 1
    assert filtered[0].track_id == 0


def test_filter_tracks_by_roi_single_outside():
    """Test filtering with one track outside."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    # Track clearly outside (bottom-center at x=99)
    tracks = [make_track(0, 49, 50, 149, 150)]
    filtered = filter_tracks_by_roi(tracks, roi)
    assert len(filtered) == 0


def test_filter_tracks_by_roi_multiple():
    """Test filtering with multiple tracks."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    tracks = [
        make_track(0, 150, 120, 250, 220),  # inside: bottom-center (200, 220)
        make_track(1, 49, 50, 149, 150),    # outside: bottom-center (99, 150)
        make_track(2, 200, 150, 300, 240),  # inside: bottom-center (250, 240)
    ]
    filtered = filter_tracks_by_roi(tracks, roi)
    assert len(filtered) == 2
    assert {t.track_id for t in filtered} == {0, 2}


def test_filter_tracks_by_roi_empty():
    """Test filtering with empty track list."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filtered = filter_tracks_by_roi([], roi)
    assert filtered == []


# ============================================================
# Tests: get_track_roi_status
# ============================================================

def test_get_track_roi_status():
    """Test detailed ROI status for tracks."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    tracks = [
        make_track(0, 150, 120, 250, 220),  # inside: bottom-center (200, 220)
        make_track(1, 49, 50, 149, 150),    # outside: bottom-center (99, 150)
    ]
    statuses = get_track_roi_status(tracks, roi)
    assert len(statuses) == 2

    assert statuses[0].track_id == 0
    assert statuses[0].in_roi is True
    assert statuses[0].bottom_center == (200.0, 220.0)
    assert statuses[0].bbox == (150.0, 120.0, 250.0, 220.0)

    assert statuses[1].track_id == 1
    assert statuses[1].in_roi is False
    assert statuses[1].bottom_center == (99.0, 150.0)


# ============================================================
# Tests: ROIFilter
# ============================================================

def test_roi_filter_init():
    """Test ROIFilter initialization."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filter = ROIFilter(roi, 640, 480)
    assert filter.roi == roi
    assert filter.frame_width == 640
    assert filter.frame_height == 480


def test_roi_filter_init_clamps():
    """Test ROIFilter clamps ROI on init."""
    roi = ROIConfig(x=-50, y=-20, width=200, height=150)
    filter = ROIFilter(roi, 640, 480)
    assert filter.roi.x == 0
    assert filter.roi.y == 0
    assert filter.roi.width == 150
    assert filter.roi.height == 130


def test_roi_filter_update_roi():
    """Test ROIFilter.update_roi with validation."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filter = ROIFilter(roi, 640, 480)

    new_roi = ROIConfig(x=200, y=200, width=300, height=200)
    filter.update_roi(new_roi)
    assert filter.roi == new_roi

    # Update with out-of-bounds ROI
    filter.update_roi(ROIConfig(x=-10, y=-10, width=100, height=100))
    assert filter.roi.x == 0
    assert filter.roi.y == 0


def test_roi_filter_filter():
    """Test ROIFilter.filter method."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filter = ROIFilter(roi, 640, 480)
    tracks = [
        make_track(0, 150, 120, 250, 220),  # inside
        make_track(1, 49, 50, 149, 150),    # outside
    ]
    filtered = filter.filter(tracks)
    assert len(filtered) == 1
    assert filtered[0].track_id == 0


def test_roi_filter_get_statuses():
    """Test ROIFilter.get_statuses method."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filter = ROIFilter(roi, 640, 480)
    tracks = [make_track(0, 150, 120, 250, 220)]
    statuses = filter.get_statuses(tracks)
    assert len(statuses) == 1
    assert statuses[0].in_roi is True


def test_roi_filter_count_inside():
    """Test ROIFilter.count_inside method."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filter = ROIFilter(roi, 640, 480)
    tracks = [
        make_track(0, 150, 120, 250, 220),  # inside
        make_track(1, 49, 50, 149, 150),    # outside
        make_track(2, 200, 150, 300, 240),  # inside
    ]
    count = filter.count_inside(tracks)
    assert count == 2


# ============================================================
# Tests: Different frame resolutions
# ============================================================

@pytest.mark.parametrize("frame_w,frame_h", [
    (640, 480),   # VGA
    (1280, 720),  # 720p
    (1920, 1080), # 1080p
    (3840, 2160), # 4K
    (320, 240),   # QVGA
])
def test_roi_different_resolutions(frame_w, frame_h):
    """Test ROI works across different frame resolutions."""
    roi = ROIConfig(x=100, y=100, width=200, height=150)
    filter = ROIFilter(roi, frame_w, frame_h)

    # Track at same relative position
    track = make_track(0, 150, 120, 250, 220)
    assert filter.filter([track]) == [track]

    # Track outside
    track_out = make_track(1, 49, 50, 149, 150)
    assert filter.filter([track_out]) == []


# ============================================================
# Tests: Invalid ROI
# ============================================================

def test_invalid_roi_zero_width():
    """Test ROI with zero width is rejected."""
    with pytest.raises(ValueError):
        ROIConfig(x=100, y=100, width=0, height=100)


def test_invalid_roi_negative_width():
    """Test ROI with negative width is rejected."""
    with pytest.raises(ValueError):
        ROIConfig(x=100, y=100, width=-10, height=100)


def test_roi_outside_frame():
    """Test ROI completely outside frame gets clamped to minimal."""
    roi = ROIConfig(x=1000, y=1000, width=100, height=100)
    clamped = roi.clamp_to_frame(640, 480)
    assert clamped.x == 639
    assert clamped.y == 479
    assert clamped.width == 1
    assert clamped.height == 1


# ============================================================
# Tests: Track crossing into ROI (temporal sequence)
# ============================================================

def test_track_crossing_into_roi():
    """Test track moving from outside to inside ROI across frames."""
    roi = ROIConfig(x=200, y=100, width=200, height=200)
    filter = ROIFilter(roi, 640, 480)

    # Frame 1: Outside (left)
    track1 = make_track(0, 50, 150, 150, 250)
    assert filter.filter([track1]) == []

    # Frame 2: Crossing boundary
    track2 = make_track(0, 180, 150, 280, 250)  # bottom-center: (230, 250)
    assert len(filter.filter([track2])) == 1

    # Frame 3: Fully inside
    track3 = make_track(0, 250, 150, 350, 250)  # bottom-center: (300, 250)
    assert len(filter.filter([track3])) == 1

    # Frame 4: Crossing out (right)
    track4 = make_track(0, 380, 150, 480, 250)  # bottom-center: (430, 250)
    assert filter.filter([track4]) == []


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])