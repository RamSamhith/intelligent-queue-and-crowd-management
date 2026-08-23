"""Thin integration adapter: Detection → ByteTrack.

This adapter bridges the VisionQueue Detection Contract (from PersonDetector)
to the ByteTrack input format, and converts ByteTrack output to the
VisionQueue Track Contract.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from visionqueue.detection.types import Detection
from visionqueue.tracking import ByteTrack, ByteTrackConfig


@dataclass(frozen=True)
class Track:
    """VisionQueue Track Contract - output of the tracking adapter.

    Attributes:
        track_id: Persistent unique identifier for this track.
        bbox: Bounding box as [x1, y1, x2, y2] in frame coordinates.
        confidence: Track confidence score in [0.0, 1.0].
    """
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float

    def to_dict(self) -> dict:
        """Serialize to the VisionQueue track contract dict."""
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }


class DetectionTrackingAdapter:
    """Adapter between PersonDetector and ByteTrack.

    Responsibilities:
    - Filter detections to person class only (class_id == 0)
    - Convert List[Detection] → numpy array for ByteTrack
    - Call ByteTrack.update()
    - Convert ByteTrack output dict → Track contract
    - Preserve track IDs across frames

    Does NOT:
    - Perform inference
    - Perform ROI/counting/entry-exit logic
    - Access database/backend/frontend
    - Duplicate tracking logic
    """

    def __init__(
        self,
        tracker: ByteTrack | None = None,
        config: ByteTrackConfig | None = None,
        person_class_id: int = 0,
    ) -> None:
        """Initialize the adapter.

        Args:
            tracker: Optional pre-configured ByteTrack instance. If None,
                creates a new one with the given config.
            config: ByteTrackConfig if creating a new tracker.
            person_class_id: COCO class ID for person (default: 0).
        """
        self._person_class_id = person_class_id
        self._tracker = tracker or ByteTrack(config or ByteTrackConfig())
        self._frame_id = 0

    @property
    def tracker(self) -> ByteTrack:
        """Access underlying ByteTrack instance for inspection/reset."""
        return self._tracker

    @property
    def last_diagnostics(self) -> dict:
        """Access the latest tracking diagnostics."""
        return self._tracker.last_diagnostics

    def update(self, detections: List[Detection]) -> List[Track]:
        """Process one frame of detections and return active tracks.

        Args:
            detections: List of Detection objects from PersonDetector.

        Returns:
            List of Track objects matching the VisionQueue Track Contract.
        """
        self._frame_id += 1

        # Filter to person class only
        person_dets = [d for d in detections if d.class_id == self._person_class_id]

        if not person_dets:
            # No person detections - still call update to advance tracker state
            self._tracker.update(np.empty((0, 6), dtype=np.float32))
            return []

        # Convert to ByteTrack input format: (N, 6) [x1, y1, x2, y2, conf, class_id]
        det_array = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
            for d in person_dets
        ], dtype=np.float32)

        # Update tracker
        tracker_output = self._tracker.update(det_array)

        # Convert to VisionQueue Track contract
        tracks = [
            Track(
                track_id=t["track_id"],
                bbox=tuple(t["bbox"]),
                confidence=t["confidence"],
            )
            for t in tracker_output
        ]

        return tracks

    def reset(self) -> None:
        """Reset tracker state and frame counter."""
        self._tracker.reset()
        self._frame_id = 0