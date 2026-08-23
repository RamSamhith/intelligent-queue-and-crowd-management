"""Occupancy counter for VisionQueue.

Computes instantaneous person occupancy from active tracks using either whole-frame
counting (default V1 mode) or a defined Region of Interest (configurable / future mode).
"""

from __future__ import annotations
from typing import List, Optional, Set, Union

from visionqueue.roi import ROIConfig, ROIFilter, track_in_roi
from visionqueue.tracking.adapter import Track
from visionqueue.counting.types import OccupancyState


class OccupancyCounter:
    """Calculates instantaneous person occupancy for the whole frame or a defined ROI.

    Modes:
    - Whole-Frame Mode (default V1): Counts all unique active tracks in the camera FOV.
    - ROI Mode (configurable / future): Filters active tracks using ROI boundary checks
      (track bottom-center point).

    Responsibilities:
    - Counts unique active tracks per frame (enforces 1 count per track ID).
    - Supports dynamic switching between whole-frame mode and ROI-filtered mode.
    - Produces deterministic OccupancyState snapshots.

    Does NOT:
    - Alter camera frames or YOLO inference.
    - Track line crossings or entry/exit events (handled by LineCrossingCounter).
    - Perform biometric identity matching.
    """

    def __init__(
        self,
        roi_or_filter: Optional[Union[ROIConfig, ROIFilter]] = None,
        enable_roi: Optional[bool] = None,
    ) -> None:
        """Initialize the occupancy counter.

        Args:
            roi_or_filter: Optional ROIConfig instance or pre-configured ROIFilter.
            enable_roi: Explicitly enable/disable ROI filtering. If None:
                - True if roi_or_filter is provided.
                - False if roi_or_filter is None (whole-frame mode).
        """
        if roi_or_filter is None:
            self._roi_filter: Optional[ROIFilter] = None
            self._roi: Optional[ROIConfig] = None
            self._enable_roi: bool = bool(enable_roi) if enable_roi is not None else False
        elif isinstance(roi_or_filter, ROIFilter):
            self._roi_filter = roi_or_filter
            self._roi = roi_or_filter.roi
            self._enable_roi = enable_roi if enable_roi is not None else True
        elif isinstance(roi_or_filter, ROIConfig):
            self._roi_filter = None
            self._roi = roi_or_filter
            self._enable_roi = enable_roi if enable_roi is not None else True
        else:
            raise TypeError(f"Expected ROIConfig, ROIFilter, or None, got {type(roi_or_filter)}")

        self._last_state = OccupancyState(current_count=0, active_track_ids=[])

    @property
    def enable_roi(self) -> bool:
        """Whether ROI filtering is active."""
        return self._enable_roi

    @enable_roi.setter
    def enable_roi(self, value: bool) -> None:
        """Set whether ROI filtering is active."""
        self._enable_roi = bool(value)

    @property
    def is_whole_frame(self) -> bool:
        """Whether whole-frame counting mode is active."""
        return not self._enable_roi or self.roi is None

    @property
    def roi(self) -> Optional[ROIConfig]:
        """Active ROI configuration if configured."""
        if self._roi_filter is not None:
            return self._roi_filter.roi
        return self._roi

    def update_roi(self, new_roi: ROIConfig) -> None:
        """Update active ROI configuration."""
        if self._roi_filter is not None:
            self._roi_filter.update_roi(new_roi)
            self._roi = self._roi_filter.roi
        else:
            self._roi = new_roi

    def count(self, tracks: List[Track]) -> int:
        """Compute instantaneous count of unique tracks (whole-frame or in-ROI).

        Args:
            tracks: List of Track objects from the tracking adapter.

        Returns:
            Integer count of unique tracks.
        """
        if not self._enable_roi or self.roi is None:
            seen_ids: Set[int] = {t.track_id for t in tracks}
            return len(seen_ids)

        seen_inside_ids: Set[int] = set()
        active_roi = self.roi
        for track in tracks:
            if track.track_id not in seen_inside_ids:
                if track_in_roi(track, active_roi):
                    seen_inside_ids.add(track.track_id)
        return len(seen_inside_ids)

    def update(
        self,
        tracks: List[Track],
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> OccupancyState:
        """Process tracks for a single frame and return detailed occupancy state.

        Args:
            tracks: List of Track objects from the tracking adapter.
            frame_id: Frame sequence number.
            timestamp: Frame acquisition / processing timestamp.

        Returns:
            OccupancyState containing count, active track IDs, and frame metadata.
        """
        if not self._enable_roi or self.roi is None:
            seen_ids: Set[int] = {t.track_id for t in tracks}
            sorted_ids = sorted(list(seen_ids))
            self._last_state = OccupancyState(
                current_count=len(sorted_ids),
                active_track_ids=sorted_ids,
                frame_id=frame_id,
                timestamp=timestamp,
            )
            return self._last_state

        seen_inside_ids: Set[int] = set()
        active_roi = self.roi
        for track in tracks:
            if track.track_id not in seen_inside_ids:
                if track_in_roi(track, active_roi):
                    seen_inside_ids.add(track.track_id)

        sorted_ids = sorted(list(seen_inside_ids))
        self._last_state = OccupancyState(
            current_count=len(sorted_ids),
            active_track_ids=sorted_ids,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        return self._last_state

    @property
    def last_state(self) -> OccupancyState:
        """Get the most recently computed OccupancyState."""
        return self._last_state

    def reset(self) -> None:
        """Reset internal state."""
        self._last_state = OccupancyState(current_count=0, active_track_ids=[])

