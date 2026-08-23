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


class CapacitySource(str, Enum):
    """Source provenance of the active capacity setting."""
    MANUAL = "MANUAL"          # Explicitly set by operator / manual capacity
    CALIBRATED = "CALIBRATED"  # Derived from physical usable area and target density
    NOT_SET = "NOT_SET"        # No capacity configured


@dataclass(frozen=True)
class SceneProfile:
    """Optional physical scene and camera installation profile.

    Enables intelligent derivation of effective capacity and deployment-specific
    crowd density thresholds without modifying the underlying detector or tracker.

    Attributes:
        name: Human-readable identifier for the scene / deployment profile.
        manual_capacity: Explicit operator-defined capacity ceiling.
        usable_area_m2: Estimated walkable/usable surface area in square meters.
        target_density_persons_per_m2: Desired spatial comfort density (default: 1.0 person/m²).
        camera_height_m: Camera mounting height in meters (for documentation/future calibration).
        camera_tilt_deg: Camera mounting tilt angle in degrees from horizontal.
        horizontal_fov_deg: Camera lens horizontal field of view in degrees.
        thresholds: Custom crowd level thresholds, or None to use defaults.
    """
    name: str = "default"
    manual_capacity: Optional[int] = None
    usable_area_m2: Optional[float] = None
    target_density_persons_per_m2: float = 1.0
    camera_height_m: Optional[float] = None
    camera_tilt_deg: Optional[float] = None
    horizontal_fov_deg: Optional[float] = None
    thresholds: Optional[CrowdThresholds] = None

    def derive_effective_capacity(self) -> Tuple[Optional[int], CapacitySource]:
        """Derive authoritative capacity from manual override or calibrated area.

        Rules:
        1. Manual capacity takes precedence if provided and > 0.
        2. Calibrated capacity is calculated as floor(usable_area_m2 * target_density)
           if usable_area_m2 > 0 and target_density > 0.
        3. Returns (None, CapacitySource.NOT_SET) if insufficient data exists.
        4. Never divides by zero or fabricates estimates.
        """
        if self.manual_capacity is not None and self.manual_capacity > 0:
            return self.manual_capacity, CapacitySource.MANUAL

        if (
            self.usable_area_m2 is not None
            and self.usable_area_m2 > 0
            and self.target_density_persons_per_m2 > 0
        ):
            import math
            calc_cap = int(math.floor(self.usable_area_m2 * self.target_density_persons_per_m2))
            if calc_cap > 0:
                return calc_cap, CapacitySource.CALIBRATED

        return None, CapacitySource.NOT_SET

    def to_analytics_config(
        self,
        debounce_frames: int = 5,
        trend_window_size: int = 10,
        trend_min_delta: int = 2,
    ) -> AnalyticsConfig:
        """Convert scene profile directly to an AnalyticsConfig."""
        cap, _ = self.derive_effective_capacity()
        thresh = self.thresholds if self.thresholds is not None else CrowdThresholds()
        return AnalyticsConfig(
            capacity=cap,
            thresholds=thresh,
            debounce_frames=debounce_frames,
            trend_window_size=trend_window_size,
            trend_min_delta=trend_min_delta,
        )

    def to_dict(self) -> dict:
        """Serialize scene profile for telemetry / configuration export."""
        cap, cap_source = self.derive_effective_capacity()
        return {
            "name": self.name,
            "manual_capacity": self.manual_capacity,
            "usable_area_m2": self.usable_area_m2,
            "target_density_persons_per_m2": self.target_density_persons_per_m2,
            "effective_capacity": cap,
            "capacity_source": cap_source.value,
            "camera_height_m": self.camera_height_m,
            "camera_tilt_deg": self.camera_tilt_deg,
            "horizontal_fov_deg": self.horizontal_fov_deg,
        }


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
