"""VisionQueue Scene Intelligence & Automatic Capacity Analysis Module.

Provides robust, perspective-aware scene analysis, ground-plane footprint
estimation, and temporally stabilized automatic capacity estimation for
monocular camera feeds.

Hierarchy:
1. MANUAL: Explicit operator-configured capacity.
2. CALIBRATED: Derived from known physical usable area (m²) and target density.
3. AUTOMATIC: Dynamically estimated from perspective cues, ground-plane footprint,
   and camera geometric priors when sufficient observations exist.
4. NOT_SET: Insufficient observations or calibration data (safe fallback).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from visionqueue.analytics.types import (
    CapacitySource,
    CrowdThresholds,
    SceneProfile,
)


@dataclass(frozen=True)
class SceneAnalysisState:
    """Output state of automatic scene intelligence and capacity estimation.

    Attributes:
        effective_capacity: Authoritative numeric capacity (None if NOT_SET).
        capacity_source: Provenance of capacity (MANUAL, CALIBRATED, AUTOMATIC, NOT_SET).
        confidence: Confidence score of the estimate [0.0, 1.0].
        visible_area_m2_approx: Estimated observable floor/ground area in square meters.
        calibration_required: True if manual/physical calibration is needed for high precision.
        scene_quality: Qualitative rating ("EXCELLENT", "GOOD", "FAIR", "UNCONFIRMED").
        geometry_quality: Methodology used ("MANUAL", "CALIBRATED", "PERSPECTIVE_ESTIMATED", "INSUFFICIENT_DATA").
        reason: Human-readable explanation of the current capacity determination.
        observed_samples_count: Total valid person ground samples collected.
        timestamp: Timestamp of the analysis.
    """
    effective_capacity: Optional[int]
    capacity_source: CapacitySource
    confidence: float
    visible_area_m2_approx: Optional[float]
    calibration_required: bool
    scene_quality: str
    geometry_quality: str
    reason: str
    observed_samples_count: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize state to standard dictionary."""
        return {
            "effective_capacity": self.effective_capacity,
            "capacity_source": self.capacity_source.value,
            "confidence": round(self.confidence, 3),
            "visible_area_m2_approx": round(self.visible_area_m2_approx, 2) if self.visible_area_m2_approx is not None else None,
            "calibration_required": self.calibration_required,
            "scene_quality": self.scene_quality,
            "geometry_quality": self.geometry_quality,
            "reason": self.reason,
            "observed_samples_count": self.observed_samples_count,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SceneAnalyzerConfig:
    """Configuration parameters for the SceneAnalyzer engine.

    Attributes:
        enabled: Whether automatic scene analysis is active.
        min_samples_for_estimate: Minimum number of observed person ground anchors
            required before computing an automatic estimate.
        max_history_samples: Size of the rolling observation buffer.
        target_density_persons_per_m2: Spatial comfort density standard (persons/m²).
        nominal_camera_height_m: Default ceiling mount height prior in meters.
        nominal_hfov_deg: Default horizontal field-of-view prior in degrees.
        temporal_smoothing_alpha: Exponential moving average smoothing factor for capacity.
        capacity_hysteresis_threshold: Integer deadband to prevent integer capacity flickering.
        cadence_frames: Execution cadence (every N frames) to preserve pipeline FPS.
    """
    enabled: bool = True
    min_samples_for_estimate: int = 8
    max_history_samples: int = 120
    target_density_persons_per_m2: float = 0.8  # ~1.25 m² per person (comfortable standing)
    nominal_camera_height_m: float = 2.8
    nominal_hfov_deg: float = 75.0
    temporal_smoothing_alpha: float = 0.15
    capacity_hysteresis_threshold: int = 2
    cadence_frames: int = 5


class SceneAnalyzer:
    """Analyzes scene geometry, ground anchors, and perspective cues to produce

    a stable, defensible effective capacity and scene intelligence report.

    Design Principles:
    - Never invents capacity out of thin air.
    - Follows strict hierarchy: MANUAL -> CALIBRATED -> AUTOMATIC -> NOT_SET.
    - Employs temporal hysteresis so capacity remains rock-solid stable over time.
    - Operates on a low-overhead cadence to ensure 0% impact on real-time YOLO/ByteTrack.
    """

    def __init__(
        self,
        config: Optional[SceneAnalyzerConfig] = None,
        profile: Optional[SceneProfile] = None,
    ) -> None:
        self._config: SceneAnalyzerConfig = config or SceneAnalyzerConfig()
        self._profile: Optional[SceneProfile] = profile

        # Rolling buffer of observed ground contact points (x_norm, y_norm, bbox_h_norm)
        self._observations: Deque[Tuple[float, float, float]] = deque(
            maxlen=self._config.max_history_samples
        )

        # Temporal smoothing state
        self._smoothed_capacity_float: Optional[float] = None
        self._active_stable_capacity: Optional[int] = None
        self._last_state: Optional[SceneAnalysisState] = None
        self._frame_counter: int = 0

    @property
    def config(self) -> SceneAnalyzerConfig:
        return self._config

    @property
    def profile(self) -> Optional[SceneProfile]:
        return self._profile

    def update_profile(self, profile: Optional[SceneProfile]) -> None:
        """Update or replace the active scene profile."""
        self._profile = profile
        self.reset_history()

    def reset_history(self) -> None:
        """Reset historical observations and temporal smoothing buffers."""
        self._observations.clear()
        self._smoothed_capacity_float = None
        self._active_stable_capacity = None
        self._last_state = None
        self._frame_counter = 0

    def add_observation(self, bbox: Tuple[float, float, float, float], frame_w: int, frame_h: int) -> None:
        """Add a person detection bounding box observation to the rolling geometry buffer.

        Args:
            bbox: (x1, y1, x2, y2) in pixel coordinates.
            frame_w: Frame pixel width.
            frame_h: Frame pixel height.
        """
        if frame_w <= 0 or frame_h <= 0:
            return
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Aspect ratio check for typical upright human (~1.8 to 4.0 h/w ratio)
        if bh / bw < 1.0 or bh > frame_h:
            return

        # Ground contact point (center bottom) normalized [0.0, 1.0]
        cx_norm = float(np_clamp(((x1 + x2) / 2.0) / frame_w, 0.0, 1.0))
        y2_norm = float(np_clamp(y2 / frame_h, 0.0, 1.0))
        bh_norm = float(np_clamp(bh / frame_h, 0.01, 1.0))

        self._observations.append((cx_norm, y2_norm, bh_norm))

    def analyze(
        self,
        active_tracks: List[dict],
        frame_w: int,
        frame_h: int,
        timestamp: float = 0.0,
        force_eval: bool = False,
    ) -> SceneAnalysisState:
        """Evaluate scene geometry and return the authoritative SceneAnalysisState.

        Args:
            active_tracks: List of active track dictionaries with "bbox" field.
            frame_w: Width of the current camera frame.
            frame_h: Height of the current camera frame.
            timestamp: Frame timestamp.
            force_eval: When True, ignores frame cadence and computes immediately.
        """
        self._frame_counter += 1

        # 1. Ingest new track observations
        for trk in active_tracks:
            bbox = trk.get("bbox")
            if bbox is not None and len(bbox) == 4:
                self.add_observation((bbox[0], bbox[1], bbox[2], bbox[3]), frame_w, frame_h)

        # Check execution cadence (re-use previous state on skip frames if available)
        if not force_eval and self._last_state is not None and (self._frame_counter % self._config.cadence_frames != 0):
            return self._last_state

        # ============================================================
        # Hierarchy Step 1: MANUAL CAPACITY OVERRIDE
        # ============================================================
        if self._profile is not None and self._profile.manual_capacity is not None and self._profile.manual_capacity > 0:
            self._last_state = SceneAnalysisState(
                effective_capacity=self._profile.manual_capacity,
                capacity_source=CapacitySource.MANUAL,
                confidence=1.0,
                visible_area_m2_approx=self._profile.usable_area_m2,
                calibration_required=False,
                scene_quality="EXCELLENT",
                geometry_quality="MANUAL",
                reason=f"Operator manual capacity configured ({self._profile.manual_capacity} persons).",
                observed_samples_count=len(self._observations),
                timestamp=timestamp,
            )
            return self._last_state

        # ============================================================
        # Hierarchy Step 2: CALIBRATED PHYSICAL SCENE AREA
        # ============================================================
        if (
            self._profile is not None
            and self._profile.usable_area_m2 is not None
            and self._profile.usable_area_m2 > 0
            and self._profile.target_density_persons_per_m2 > 0
        ):
            calc_cap = int(math.floor(self._profile.usable_area_m2 * self._profile.target_density_persons_per_m2))
            calc_cap = max(1, calc_cap)
            self._last_state = SceneAnalysisState(
                effective_capacity=calc_cap,
                capacity_source=CapacitySource.CALIBRATED,
                confidence=1.0,
                visible_area_m2_approx=self._profile.usable_area_m2,
                calibration_required=False,
                scene_quality="EXCELLENT",
                geometry_quality="CALIBRATED",
                reason=f"Calibrated physical area ({self._profile.usable_area_m2:.1f} m² @ {self._profile.target_density_persons_per_m2:.2f} p/m²).",
                observed_samples_count=len(self._observations),
                timestamp=timestamp,
            )
            return self._last_state

        # ============================================================
        # Hierarchy Step 3: AUTOMATIC PERSPECTIVE SCENE ESTIMATION
        # ============================================================
        if self._config.enabled:
            auto_state = self._estimate_automatic_capacity(frame_w, frame_h, timestamp)
            if auto_state is not None:
                self._last_state = auto_state
                return self._last_state

        # ============================================================
        # Hierarchy Step 4: NOT_SET (Safe Fallback)
        # ============================================================
        self._last_state = SceneAnalysisState(
            effective_capacity=None,
            capacity_source=CapacitySource.NOT_SET,
            confidence=0.0,
            visible_area_m2_approx=None,
            calibration_required=True,
            scene_quality="UNCONFIRMED",
            geometry_quality="INSUFFICIENT_DATA",
            reason="Insufficient calibration data or observations to establish ground scale.",
            observed_samples_count=len(self._observations),
            timestamp=timestamp,
        )
        return self._last_state

    def _estimate_automatic_capacity(
        self,
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> Optional[SceneAnalysisState]:
        """Estimate visible ground area and capacity from accumulated perspective cues."""
        n_samples = len(self._observations)
        if n_samples < self._config.min_samples_for_estimate:
            return SceneAnalysisState(
                effective_capacity=None,
                capacity_source=CapacitySource.NOT_SET,
                confidence=float(n_samples / max(1, self._config.min_samples_for_estimate) * 0.4),
                visible_area_m2_approx=None,
                calibration_required=True,
                scene_quality="UNCONFIRMED",
                geometry_quality="INSUFFICIENT_DATA",
                reason=f"Accumulating scene observations ({n_samples}/{self._config.min_samples_for_estimate} samples).",
                observed_samples_count=n_samples,
                timestamp=timestamp,
            )

        # Extract observations
        xs = [obs[0] for obs in self._observations]
        ys = [obs[1] for obs in self._observations]
        bhs = [obs[2] for obs in self._observations]

        # Geometry Priors
        cam_h = (
            self._profile.camera_height_m
            if self._profile and self._profile.camera_height_m and self._profile.camera_height_m > 0
            else self._config.nominal_camera_height_m
        )
        hfov_deg = (
            self._profile.horizontal_fov_deg
            if self._profile and self._profile.horizontal_fov_deg and self._profile.horizontal_fov_deg > 0
            else self._config.nominal_hfov_deg
        )
        target_density = (
            self._profile.target_density_persons_per_m2
            if self._profile and self._profile.target_density_persons_per_m2 > 0
            else self._config.target_density_persons_per_m2
        )

        # 1. Perspective Scale Gradient Analysis:
        # Standard human height prior: ~1.7 meters.
        # Person normalized height h_norm = (1.7 * fy) / Z.
        # Median observed bounding box height gives representative scale at median ground position:
        med_bh = float(sorted(bhs)[len(bhs) // 2])
        med_y = float(sorted(ys)[len(ys) // 2])

        # Focal length in normalized units (from horizontal FOV)
        f_norm = 0.5 / math.tan(math.radians(hfov_deg / 2.0))

        # Approximate distance to ground center:
        # Z_approx = 1.7 * f_norm / med_bh
        # Clamp Z to reasonable indoor/outdoor surveillance ranges (1.5m to 25.0m)
        z_est = (1.7 * f_norm) / max(0.04, med_bh)
        z_est = max(1.5, min(25.0, z_est))

        # 2. Estimate Ground Footprint Extent:
        # Spread of observed ground contact positions
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        span_x_norm = max(0.25, x_max - x_min)
        span_y_norm = max(0.20, y_max - y_min)

        # Ground width W_ground = span_x * (Z / f_norm)
        w_ground = span_x_norm * (z_est / f_norm)
        # Ground depth D_ground = span_y * (Z / f_norm) * perspective_foreshortening_factor
        # Assuming nominal downward angle ~35-45 deg -> 1 / sin(40 deg) ~ 1.55
        d_ground = span_y_norm * (z_est / f_norm) * 1.5

        # Bounded floor area (m²)
        raw_area_m2 = max(2.0, w_ground * d_ground)

        # Raw capacity = area * target_density
        raw_cap_float = raw_area_m2 * target_density

        # 3. Temporal Stabilization with Hysteresis:
        if self._smoothed_capacity_float is None:
            self._smoothed_capacity_float = raw_cap_float
            self._active_stable_capacity = max(1, int(round(raw_cap_float)))
        else:
            # Exponential Moving Average smoothing
            alpha = self._config.temporal_smoothing_alpha
            self._smoothed_capacity_float = (alpha * raw_cap_float) + ((1.0 - alpha) * self._smoothed_capacity_float)

            candidate_cap = max(1, int(round(self._smoothed_capacity_float)))
            if self._active_stable_capacity is None:
                self._active_stable_capacity = candidate_cap
            else:
                # Apply deadband hysteresis: only step if change exceeds threshold
                if abs(candidate_cap - self._active_stable_capacity) >= self._config.capacity_hysteresis_threshold:
                    self._active_stable_capacity = candidate_cap

        # Confidence calculation based on sample count and spatial spread
        sample_conf = min(1.0, n_samples / (self._config.min_samples_for_estimate * 2.5))
        spread_conf = min(1.0, (span_x_norm * span_y_norm) / 0.15)
        overall_conf = float(0.55 * sample_conf + 0.45 * spread_conf)
        overall_conf = max(0.40, min(0.88, overall_conf))  # Capped at 0.88 to reflect monocular estimation uncertainty

        quality = "GOOD" if overall_conf >= 0.70 else "FAIR"

        return SceneAnalysisState(
            effective_capacity=self._active_stable_capacity,
            capacity_source=CapacitySource.AUTOMATIC,
            confidence=overall_conf,
            visible_area_m2_approx=raw_area_m2,
            calibration_required=True,
            scene_quality=quality,
            geometry_quality="PERSPECTIVE_ESTIMATED",
            reason=f"Automatic perspective estimate (~{raw_area_m2:.1f} m² visible ground footprint, confidence {overall_conf:.2f}).",
            observed_samples_count=n_samples,
            timestamp=timestamp,
        )


def np_clamp(val: float, min_val: float, max_val: float) -> float:
    """Helper clamping function."""
    return max(min_val, min(max_val, val))
