"""Occupancy and Crowd Analytics Engine for VisionQueue.

Processes raw in-ROI counts into:
1. Safe occupancy percentage (capacity-zero safe).
2. Debounced discrete crowd levels (LOW, MODERATE, HIGH, CRITICAL).
3. Core+ rolling crowd trend (INCREASING, STABLE, DECREASING).
4. Core+ session peak metrics.
"""

from __future__ import annotations
from collections import deque
from typing import Deque, Optional, Tuple

from visionqueue.analytics.types import (
    AnalyticsConfig,
    CapacityState,
    CrowdAnalyticsState,
    CrowdLevel,
    CrowdThresholds,
    CrowdTrend,
)


class CrowdAnalyticsEngine:
    """Evaluates occupancy percentage, debounced crowd level, and trend analytics.

    Design Principles:
    - Raw current count updates immediately per frame (never debounced).
    - Occupancy percentage safely handles capacity <= 0 or None without throwing or fabricating values.
    - Crowd level label applies temporal debouncing to prevent UI flickering.
    - Core+ metrics (trend, peak count, peak occupancy) are calculated deterministically.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None) -> None:
        """Initialize the analytics engine.

        Args:
            config: Optional AnalyticsConfig. Uses default parameters if omitted.
        """
        self._config: AnalyticsConfig = config or AnalyticsConfig()
        self._current_debounced_level: CrowdLevel = CrowdLevel.LOW
        self._candidate_level: Optional[CrowdLevel] = None
        self._candidate_streak: int = 0

        # Core+ history and peak tracking
        self._history_window: Deque[int] = deque(maxlen=self._config.trend_window_size)
        self._peak_count: int = 0
        self._peak_occupancy_percent: Optional[int] = None
        self._peak_timestamp: Optional[float] = None
        self._last_state: Optional[CrowdAnalyticsState] = None

    @property
    def config(self) -> AnalyticsConfig:
        """Current analytics configuration."""
        return self._config

    @property
    def capacity(self) -> Optional[int]:
        """Configured capacity."""
        return self._config.capacity

    @property
    def thresholds(self) -> CrowdThresholds:
        """Configured crowd thresholds."""
        return self._config.thresholds

    def update_config(self, new_config: AnalyticsConfig) -> None:
        """Update analytics configuration and re-align window buffers if size changed."""
        old_window_size = self._config.trend_window_size
        self._config = new_config
        if new_config.trend_window_size != old_window_size:
            existing = list(self._history_window)
            self._history_window = deque(
                existing[-new_config.trend_window_size:],
                maxlen=new_config.trend_window_size,
            )

    def update_capacity(self, capacity: Optional[int]) -> None:
        """Update the operator-configured capacity."""
        validated_capacity = capacity if (capacity is not None and capacity > 0) else None
        self._config = AnalyticsConfig(
            capacity=validated_capacity,
            thresholds=self._config.thresholds,
            debounce_frames=self._config.debounce_frames,
            trend_window_size=self._config.trend_window_size,
            trend_min_delta=self._config.trend_min_delta,
        )

    def update_thresholds(self, thresholds: CrowdThresholds) -> None:
        """Update crowd level thresholds."""
        self._config = AnalyticsConfig(
            capacity=self._config.capacity,
            thresholds=thresholds,
            debounce_frames=self._config.debounce_frames,
            trend_window_size=self._config.trend_window_size,
            trend_min_delta=self._config.trend_min_delta,
        )

    def calculate_occupancy(self, current_count: int) -> Tuple[Optional[int], str]:
        """Calculate whole percentage occupancy against configured capacity.

        Args:
            current_count: Number of persons currently within the ROI.

        Returns:
            Tuple of (occupancy_percent, capacity_state).
            If capacity is None or <= 0: (None, "NOT_SET").
            Otherwise: (round((current_count / capacity) * 100), "SET").
        """
        if self._config.capacity is None or self._config.capacity <= 0:
            return None, CapacityState.NOT_SET.value

        percent = round((current_count / self._config.capacity) * 100)
        return percent, CapacityState.SET.value

    def _debounce_crowd_level(self, raw_level: CrowdLevel) -> CrowdLevel:
        """Update debounce streak and return the active stable crowd level."""
        if raw_level == self._current_debounced_level:
            # Matches current stable level - reset candidate tracking
            self._candidate_level = None
            self._candidate_streak = 0
        elif raw_level == self._candidate_level:
            # Continuing candidate level streak
            self._candidate_streak += 1
            if self._candidate_streak >= self._config.debounce_frames:
                self._current_debounced_level = raw_level
                self._candidate_level = None
                self._candidate_streak = 0
        else:
            # New candidate level encountered
            self._candidate_level = raw_level
            self._candidate_streak = 1
            if self._candidate_streak >= self._config.debounce_frames:
                self._current_debounced_level = raw_level
                self._candidate_level = None
                self._candidate_streak = 0

        return self._current_debounced_level

    def _calculate_trend(self) -> CrowdTrend:
        """Compute crowd trend across rolling window."""
        if len(self._history_window) < 2:
            return CrowdTrend.STABLE

        # Compare latest count with start of rolling window
        delta = self._history_window[-1] - self._history_window[0]
        if delta >= self._config.trend_min_delta:
            return CrowdTrend.INCREASING
        elif delta <= -self._config.trend_min_delta:
            return CrowdTrend.DECREASING
        else:
            return CrowdTrend.STABLE

    def update(
        self,
        current_count: int,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> CrowdAnalyticsState:
        """Process a single frame count through the analytics pipeline.

        Args:
            current_count: Instantaneous headcount within the ROI.
            frame_id: Current frame sequence index.
            timestamp: Frame acquisition / processing timestamp.

        Returns:
            CrowdAnalyticsState with immediate count, safe occupancy %, debounced crowd level,
            trend, and session peaks.
        """
        # 1. Occupancy calculation
        occupancy_percent, capacity_state = self.calculate_occupancy(current_count)

        # 2. Raw crowd classification
        raw_level = self._config.thresholds.classify(
            float(occupancy_percent) if occupancy_percent is not None else None
        )

        # 3. Debounced crowd classification
        debounced_level = self._debounce_crowd_level(raw_level)

        # 4. Rolling trend history
        self._history_window.append(current_count)
        trend = self._calculate_trend()

        # 5. Session peaks
        if current_count > self._peak_count:
            self._peak_count = current_count
            self._peak_timestamp = timestamp
        elif self._peak_count == 0 and current_count == 0 and self._peak_timestamp is None:
            self._peak_timestamp = timestamp

        if occupancy_percent is not None:
            if self._peak_occupancy_percent is None or occupancy_percent > self._peak_occupancy_percent:
                self._peak_occupancy_percent = occupancy_percent

        self._last_state = CrowdAnalyticsState(
            current_count=current_count,
            capacity=self._config.capacity,
            capacity_state=capacity_state,
            occupancy_percent=occupancy_percent,
            raw_crowd_level=raw_level,
            crowd_level=debounced_level,
            trend=trend,
            peak_count=self._peak_count,
            peak_occupancy_percent=self._peak_occupancy_percent,
            peak_timestamp=self._peak_timestamp,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        return self._last_state

    @property
    def last_state(self) -> Optional[CrowdAnalyticsState]:
        """Most recent analytics state."""
        return self._last_state

    def reset(self) -> None:
        """Reset session peaks, rolling history, and debounce state."""
        self._current_debounced_level = CrowdLevel.LOW
        self._candidate_level = None
        self._candidate_streak = 0
        self._history_window.clear()
        self._peak_count = 0
        self._peak_occupancy_percent = None
        self._peak_timestamp = None
        self._last_state = None

    def new_session(self) -> None:
        """Alias for reset() to demarcate session boundaries."""
        self.reset()
