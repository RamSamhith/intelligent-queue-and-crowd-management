"""VisionQueue ROI - Monitoring Region of Interest module.

Usage:
    from visionqueue.roi import ROIConfig, ROIFilter, track_bottom_center

    # Define ROI: x, y, width, height (top-left origin)
    roi = ROIConfig(x=200, y=100, width=400, height=300)

    # Create filter with frame dimensions
    filter = ROIFilter(roi, frame_width=1280, frame_height=720)

    # Filter tracks
    inside_tracks = filter.filter(tracks)
    count = filter.count_inside(tracks)
"""

from .types import ROIConfig, TrackROIStatus
from .roi import (
    track_bottom_center,
    track_in_roi,
    filter_tracks_by_roi,
    get_track_roi_status,
    validate_roi,
    ROIFilter,
)

__all__ = [
    "ROIConfig",
    "TrackROIStatus",
    "track_bottom_center",
    "track_in_roi",
    "filter_tracks_by_roi",
    "get_track_roi_status",
    "validate_roi",
    "ROIFilter",
]

__version__ = "1.0.0"