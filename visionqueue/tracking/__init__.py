"""
VisionQueue ByteTrack - Independent ByteTrack implementation for person tracking.

Usage:
    from visionqueue.tracking import ByteTrack, ByteTrackConfig

    config = ByteTrackConfig(
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6
    )
    tracker = ByteTrack(config)

    # Each frame:
    detections = np.array([[x1, y1, x2, y2, conf, class_id], ...], dtype=np.float32)
    tracks = tracker.update(detections)
    # tracks = [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
"""

from .bytetrack import ByteTrack, ByteTrackConfig
from .track import Track, TrackState
from .kalman import KalmanFilter
from .matching import iou_distance, matching, linear_sum_assignment

__all__ = [
    "ByteTrack",
    "ByteTrackConfig",
    "Track",
    "TrackState",
    "KalmanFilter",
    "iou_distance",
    "matching",
    "linear_sum_assignment",
]

__version__ = "1.0.0"