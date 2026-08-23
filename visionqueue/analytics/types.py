"""Type definitions and contracts for the VisionQueue analytics subsystem."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CrowdLevel(str, Enum):
    """Discrete crowd density classification."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CrowdTrend(str, Enum):
    """Directional trend of crowd count over a rolling historical window."""
    INCREASING = "INCREASING"
    STABLE = "STABLE"
    DECREASING = "DECREASING"


class CapacityState(str, Enum):
    """Status of operator-configured capacity."""
    SET = "SET"
    NOT_SET = "NOT_SET"


@dataclass(frozen=True)
class CrowdThresholds:
    """Configurable occupancy percentage thresholds for crowd classification.

    Threshold intervals (half-open / inclusive):
    - LOW:      [0, moderate_threshold)
    - MODERATE: [moderate_threshold, high_threshold)
    - HIGH:     [high_threshold, critical_threshold)
    - CRITICAL: [critical_threshold, inf)

    Attributes:
        moderate_threshold: Percentage to transition from LOW to MODERATE.
        high_threshold: Percentage to transition from MODERATE to HIGH.
        critical_threshold: Percentage to transition from HIGH to CRITICAL.
    """
    moderate_threshold: float = 40.0
    high_threshold: float = 70.0
    critical_threshold: float = 90.0

    def __post_init__(self):
        if not (0.0 <= self.moderate_threshold < self.high_threshold < self.critical_threshold):
            raise ValueError(
                f"CrowdThresholds must be strictly increasing with moderate >= 0: "
                f"got moderate={self.moderate_threshold}, high={self.high_threshold}, "
                f"critical={self.critical_threshold}"
            )

    def classify(self, occupancy_percent: Optional[float]) -> CrowdLevel:
        """Classify occupancy percentage into a CrowdLevel."""
        if occupancy_percent is None:
            return CrowdLevel.LOW

        if occupancy_percent >= self.critical_threshold:
            return CrowdLevel.CRITICAL
        elif occupancy_percent >= self.high_threshold:
            return CrowdLevel.HIGH
        elif occupancy_percent >= self.moderate_threshold:
            return CrowdLevel.MODERATE
        else:
            return CrowdLevel.LOW


@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for occupancy and crowd analytics.

    Attributes:
        capacity: Operator-defined numeric capacity (must be > 0 to calculate occupancy %).
        thresholds: Configurable crowd level thresholds.
        debounce_frames: Number of consecutive frames a new crowd level must persist
            before changing the reported crowd level (anti-flicker).
        trend_window_size: Number of rolling samples used to compute crowd trend (Core+).
        trend_min_delta: Minimum difference across window to trigger INCREASING/DECREASING.
    """
    capacity: Optional[int] = None
    thresholds: CrowdThresholds = field(default_factory=CrowdThresholds)
    debounce_frames: int = 5
    trend_window_size: int = 10
    trend_min_delta: int = 2

    def __post_init__(self):
        if self.capacity is not None and self.capacity <= 0:
            # Standardize invalid / zero capacity to None
            object.__setattr__(self, "capacity", None)
        if self.debounce_frames < 1:
            raise ValueError(f"debounce_frames must be >= 1, got {self.debounce_frames}")
        if self.trend_window_size < 2:
            raise ValueError(f"trend_window_size must be >= 2, got {self.trend_window_size}")


@dataclass(frozen=True)
class CrowdAnalyticsState:
    """Output state of the occupancy and crowd analytics layer for a single frame.

    Attributes:
        current_count: Instantaneous raw person count within ROI (immediate, not debounced).
        capacity: Configured capacity or None if unset/invalid.
        capacity_state: "SET" if capacity > 0, otherwise "NOT_SET".
        occupancy_percent: Calculated whole percentage (0-100+) or None if capacity is NOT_SET.
        raw_crowd_level: Undebounced crowd classification for this instantaneous frame.
        crowd_level: Debounced crowd classification (safe for UI display without flickering).
        trend: Rolling crowd trend (INCREASING | STABLE | DECREASING).
        peak_count: Maximum observed current count in the active session.
        peak_occupancy_percent: Maximum observed occupancy percentage in the active session.
        peak_timestamp: Timestamp when the maximum peak was first observed.
        frame_id: Frame sequence number.
        timestamp: Frame acquisition / processing timestamp.
    """
    current_count: int
    capacity: Optional[int]
    capacity_state: str
    occupancy_percent: Optional[int]
    raw_crowd_level: CrowdLevel
    crowd_level: CrowdLevel
    trend: CrowdTrend = CrowdTrend.STABLE
    peak_count: int = 0
    peak_occupancy_percent: Optional[int] = None
    peak_timestamp: Optional[float] = None
    frame_id: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize analytics state to standard application dictionary."""
        return {
            "current_count": self.current_count,
            "capacity": self.capacity,
            "capacity_state": self.capacity_state,
            "occupancy_percent": self.occupancy_percent,
            "raw_crowd_level": self.raw_crowd_level.value,
            "crowd_level": self.crowd_level.value,
            "trend": self.trend.value,
            "peak_count": self.peak_count,
            "peak_occupancy_percent": self.peak_occupancy_percent,
            "peak_timestamp": self.peak_timestamp,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }
