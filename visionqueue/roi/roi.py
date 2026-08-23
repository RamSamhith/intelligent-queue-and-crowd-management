"""ROI filtering logic for VisionQueue.

Provides utilities to:
- Validate ROI against frame dimensions
- Compute track bottom-center points
- Filter tracks by ROI membership
"""

from __future__ import annotations
from typing import List, Tuple

from visionqueue.roi.types import ROIConfig, TrackROIStatus
from visionqueue.tracking.adapter import Track


def track_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Compute bottom-center point of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] in frame coordinates.

    Returns:
        (center_x, bottom_y) where:
        - center_x = (x1 + x2) / 2
        - bottom_y = y2
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    bottom_y = y2
    return (center_x, bottom_y)


def track_in_roi(track: Track, roi: ROIConfig) -> bool:
    """Check if a track's bottom-center point is inside the ROI.

    Args:
        track: Track with bbox attribute [x1, y1, x2, y2].
        roi: ROIConfig defining the region.

    Returns:
        True if track's bottom-center is inside ROI.
    """
    cx, bottom_y = track_bottom_center(track.bbox)
    return roi.contains_point(cx, bottom_y)


def filter_tracks_by_roi(tracks: List[Track], roi: ROIConfig) -> List[Track]:
    """Return only tracks whose bottom-center point is inside the ROI.

    Args:
        tracks: List of Track objects.
        roi: ROIConfig defining the region.

    Returns:
        List of tracks inside ROI.
    """
    return [t for t in tracks if track_in_roi(t, roi)]


def get_track_roi_status(tracks: List[Track], roi: ROIConfig) -> List[TrackROIStatus]:
    """Get detailed ROI status for each track.

    Args:
        tracks: List of Track objects.
        roi: ROIConfig defining the region.

    Returns:
        List of TrackROIStatus with in_roi flag and bottom-center coords.
    """
    statuses = []
    for t in tracks:
        cx, bottom_y = track_bottom_center(t.bbox)
        in_roi = roi.contains_point(cx, bottom_y)
        statuses.append(TrackROIStatus(
            track_id=t.track_id,
            in_roi=in_roi,
            bottom_center=(cx, bottom_y),
            bbox=t.bbox,
        ))
    return statuses


def validate_roi(roi: ROIConfig, frame_width: int, frame_height: int) -> ROIConfig:
    """Validate and clamp ROI to frame boundaries.

    Args:
        roi: ROIConfig to validate.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Clamped ROIConfig guaranteed to overlap with frame.

    Raises:
        ValueError: If frame dimensions are invalid.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Invalid frame dimensions: {frame_width}x{frame_height}")

    if not roi.is_valid_for_frame(frame_width, frame_height):
        # ROI completely outside frame - clamp to minimal valid
        return roi.clamp_to_frame(frame_width, frame_height)

    # ROI partially overlaps - clamp to frame boundaries
    return roi.clamp_to_frame(frame_width, frame_height)


class ROIFilter:
    """Stateful ROI filter for processing track sequences.

    Maintains an ROI config and provides filtering methods.
    """

    def __init__(self, roi: ROIConfig, frame_width: int, frame_height: int):
        """Initialize with ROI and frame dimensions.

        Args:
            roi: ROIConfig defining the region of interest.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Raises:
            ValueError: If ROI is invalid or frame dimensions <= 0.
        """
        self._roi = validate_roi(roi, frame_width, frame_height)
        self._frame_width = frame_width
        self._frame_height = frame_height

    @property
    def roi(self) -> ROIConfig:
        return self._roi

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    def update_roi(self, roi: ROIConfig) -> None:
        """Update ROI config with validation."""
        self._roi = validate_roi(roi, self._frame_width, self._frame_height)

    def filter(self, tracks: List[Track]) -> List[Track]:
        """Filter tracks to only those inside ROI."""
        return filter_tracks_by_roi(tracks, self._roi)

    def get_statuses(self, tracks: List[Track]) -> List[TrackROIStatus]:
        """Get detailed ROI status for each track."""
        return get_track_roi_status(tracks, self._roi)

    def count_inside(self, tracks: List[Track]) -> int:
        """Count tracks inside ROI."""
        return sum(1 for t in tracks if track_in_roi(t, self._roi))