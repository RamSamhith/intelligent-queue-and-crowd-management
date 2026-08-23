"""
Track class for ByteTrack - manages individual track state and lifecycle.
"""

from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from .kalman import KalmanFilter


class TrackState(IntEnum):
    """Track lifecycle states."""
    NEW = 0       # Just created, not yet confirmed
    TRACKED = 1   # Confirmed and actively tracked
    LOST = 2      # Lost but within buffer
    REMOVED = 3   # Expired, will be deleted


@dataclass
class Track:
    """Individual track with Kalman filter state."""
    track_id: int
    tlwh: NDArray[np.float32]      # [x, y, w, h] top-left format
    score: float                   # Detection confidence
    class_id: int                  # Class ID (0 for person)

    # Kalman filter state
    mean: NDArray[np.float32] = field(init=False)
    covariance: NDArray[np.float32] = field(init=False)

    # Lifecycle
    state: TrackState = field(default=TrackState.NEW, init=False)
    frame_id: int = field(default=0, init=False)
    start_frame: int = field(default=0, init=False)
    end_frame: int = field(default=0, init=False)
    age: int = field(default=0, init=False)  # frames since last update
    hit_streak: int = field(default=0, init=False)  # consecutive updates

    _kf: KalmanFilter = field(default_factory=KalmanFilter, init=False, repr=False)

    def __post_init__(self):
        """Initialize Kalman filter from detection."""
        xywh = self._kf.tlwh_to_xywh(self.tlwh)
        self.mean, self.covariance = self._kf.initiate(xywh)
        self.age = 0
        self.hit_streak = 1

    @property
    def tlbr(self) -> NDArray[np.float32]:
        """Get box in [x1, y1, x2, y2] format."""
        return self._kf.tlwh_to_tlbr(self.tlwh)

    @property
    def xywh(self) -> NDArray[np.float32]:
        """Get box in [cx, cy, w, h] format."""
        return self._kf.tlwh_to_xywh(self.tlwh)

    def predict(self) -> None:
        """Predict next state with Kalman filter."""
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.tlwh = self._kf.xywh_to_tlwh(self.mean[:4])
        self.age += 1

    def update(self, tlwh: NDArray[np.float32], score: float, frame_id: int) -> None:
        """Update track with new detection."""
        xywh = self._kf.tlwh_to_xywh(tlwh)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, xywh)
        self.tlwh = self._kf.xywh_to_tlwh(self.mean[:4])
        self.score = score
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.age = 0
        self.hit_streak += 1

        if self.state == TrackState.NEW and self.hit_streak >= 3:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED

    def re_activate(self, tlwh: NDArray[np.float32], score: float, frame_id: int, new_id: bool = False) -> None:
        """Reactivate a lost track."""
        xywh = self._kf.tlwh_to_xywh(tlwh)
        self.mean, self.covariance = self._kf.update(self.mean, self.covariance, xywh)
        self.tlwh = self._kf.xywh_to_tlwh(self.mean[:4])
        self.score = score
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.age = 0
        self.hit_streak = 1
        self.state = TrackState.TRACKED
        if new_id:
            # This shouldn't happen in ByteTrack but kept for compatibility
            pass

    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Mark track as removed."""
        self.state = TrackState.REMOVED

    @property
    def is_activated(self) -> bool:
        return self.state == TrackState.TRACKED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    def to_dict(self) -> dict:
        """Export track as dictionary matching output contract."""
        return {
            "track_id": self.track_id,
            "bbox": self.tlbr.tolist(),
            "confidence": float(self.score)
        }