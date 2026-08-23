"""Comprehensive unit tests for the VisionQueue occupancy and crowd analytics subsystem."""

import pytest
from visionqueue.analytics import (
    AnalyticsConfig,
    CapacitySource,
    CapacityState,
    CrowdAnalyticsEngine,
    CrowdAnalyticsState,
    CrowdLevel,
    CrowdThresholds,
    CrowdTrend,
    SceneProfile,
)


class TestOccupancyCalculation:
    """Tests for capacity-safe occupancy percentage calculations."""

    def test_capacity_unset_returns_none(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=None))
        pct, state = engine.calculate_occupancy(current_count=5)
        assert pct is None
        assert state == CapacityState.NOT_SET.value

        res = engine.update(current_count=5, frame_id=1)
        assert res.occupancy_percent is None
        assert res.capacity is None
        assert res.capacity_state == "NOT_SET"

    def test_capacity_zero_returns_none(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=0))
        pct, state = engine.calculate_occupancy(current_count=10)
        assert pct is None
        assert state == CapacityState.NOT_SET.value

    def test_capacity_negative_returns_none(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=-5))
        pct, state = engine.calculate_occupancy(current_count=10)
        assert pct is None
        assert state == CapacityState.NOT_SET.value

    def test_normal_occupancy_calculation(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10))
        pct, state = engine.calculate_occupancy(current_count=7)
        assert pct == 70
        assert state == CapacityState.SET.value

        res = engine.update(current_count=7, frame_id=1)
        assert res.occupancy_percent == 70
        assert res.capacity == 10
        assert res.capacity_state == "SET"

    def test_occupancy_rounding(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=3))
        # 1 / 3 = 33.333% -> 33
        pct1, _ = engine.calculate_occupancy(current_count=1)
        assert pct1 == 33

        # 2 / 3 = 66.666% -> 67
        pct2, _ = engine.calculate_occupancy(current_count=2)
        assert pct2 == 67

        # 3 / 3 = 100% -> 100
        pct3, _ = engine.calculate_occupancy(current_count=3)
        assert pct3 == 100

    def test_occupancy_over_capacity(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10))
        pct, state = engine.calculate_occupancy(current_count=15)
        assert pct == 150
        assert state == CapacityState.SET.value

    def test_update_capacity_dynamically(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10))
        assert engine.calculate_occupancy(5)[0] == 50

        engine.update_capacity(20)
        assert engine.capacity == 20
        assert engine.calculate_occupancy(5)[0] == 25

        engine.update_capacity(0)
        assert engine.capacity is None
        assert engine.calculate_occupancy(5)[0] is None


class TestCrowdLevelClassification:
    """Tests for discrete crowd level mapping and thresholds."""

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            CrowdThresholds(moderate_threshold=50.0, high_threshold=40.0, critical_threshold=90.0)

        with pytest.raises(ValueError):
            CrowdThresholds(moderate_threshold=-1.0, high_threshold=50.0, critical_threshold=90.0)

    def test_default_threshold_classification(self):
        # Default: moderate=40.0, high=70.0, critical=90.0
        thresh = CrowdThresholds()
        assert thresh.classify(0.0) == CrowdLevel.LOW
        assert thresh.classify(39.9) == CrowdLevel.LOW
        assert thresh.classify(40.0) == CrowdLevel.MODERATE
        assert thresh.classify(69.9) == CrowdLevel.MODERATE
        assert thresh.classify(70.0) == CrowdLevel.HIGH
        assert thresh.classify(89.9) == CrowdLevel.HIGH
        assert thresh.classify(90.0) == CrowdLevel.CRITICAL
        assert thresh.classify(150.0) == CrowdLevel.CRITICAL
        assert thresh.classify(None) == CrowdLevel.LOW

    def test_custom_threshold_classification(self):
        custom = CrowdThresholds(moderate_threshold=25.0, high_threshold=50.0, critical_threshold=75.0)
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=100, thresholds=custom, debounce_frames=1))

        assert engine.update(10).raw_crowd_level == CrowdLevel.LOW
        assert engine.update(25).raw_crowd_level == CrowdLevel.MODERATE
        assert engine.update(50).raw_crowd_level == CrowdLevel.HIGH
        assert engine.update(75).raw_crowd_level == CrowdLevel.CRITICAL


class TestCrowdLevelDebouncing:
    """Tests for temporal debounce mechanism of derived crowd levels."""

    def test_raw_count_remains_immediate(self):
        # Even with high debounce frames, raw count and occupancy update immediately
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10, debounce_frames=5))

        s1 = engine.update(current_count=2, frame_id=1)
        assert s1.current_count == 2
        assert s1.occupancy_percent == 20

        s2 = engine.update(current_count=8, frame_id=2)
        assert s2.current_count == 8
        assert s2.occupancy_percent == 80
        assert s2.raw_crowd_level == CrowdLevel.HIGH
        # But debounced crowd level is still LOW because it hasn't held for 5 frames
        assert s2.crowd_level == CrowdLevel.LOW

    def test_debounce_transition_after_required_frames(self):
        # 3-frame debounce window
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10, debounce_frames=3))

        # Initial state: LOW
        engine.update(current_count=2, frame_id=1)
        assert engine.last_state.crowd_level == CrowdLevel.LOW

        # Frame 2: jumps to HIGH (count=8) -> streak 1
        s2 = engine.update(current_count=8, frame_id=2)
        assert s2.raw_crowd_level == CrowdLevel.HIGH
        assert s2.crowd_level == CrowdLevel.LOW

        # Frame 3: stays at HIGH (count=8) -> streak 2
        s3 = engine.update(current_count=8, frame_id=3)
        assert s3.crowd_level == CrowdLevel.LOW

        # Frame 4: stays at HIGH (count=8) -> streak 3 -> transition to HIGH!
        s4 = engine.update(current_count=8, frame_id=4)
        assert s4.crowd_level == CrowdLevel.HIGH

    def test_debounce_suppresses_flicker_at_threshold_boundary(self):
        # Capacity=100. High threshold = 70.
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=100, debounce_frames=3))

        # Start at MODERATE (count=60)
        for f in range(1, 5):
            engine.update(current_count=60, frame_id=f)
        assert engine.last_state.crowd_level == CrowdLevel.MODERATE

        # Frame 5: momentary spike to 71% (HIGH)
        s5 = engine.update(current_count=71, frame_id=5)
        assert s5.raw_crowd_level == CrowdLevel.HIGH
        assert s5.crowd_level == CrowdLevel.MODERATE  # debounced!

        # Frame 6: dips back to 69% (MODERATE)
        s6 = engine.update(current_count=69, frame_id=6)
        assert s6.raw_crowd_level == CrowdLevel.MODERATE
        assert s6.crowd_level == CrowdLevel.MODERATE

        # Frame 7: momentary spike to 70% (HIGH)
        s7 = engine.update(current_count=70, frame_id=7)
        assert s7.crowd_level == CrowdLevel.MODERATE

        # Never changed to HIGH during boundary oscillations
        assert engine.last_state.crowd_level == CrowdLevel.MODERATE


class TestCorePlusAnalytics:
    """Tests for Core+ rolling trend analysis and session peak tracking."""

    def test_increasing_trend(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(
            capacity=10,
            trend_window_size=5,
            trend_min_delta=2,
            debounce_frames=1,
        ))

        # Count sequence increasing: 1, 2, 3, 4, 5
        counts = [1, 2, 3, 4, 5]
        for idx, cnt in enumerate(counts):
            state = engine.update(current_count=cnt, frame_id=idx + 1)

        assert state.trend == CrowdTrend.INCREASING

    def test_decreasing_trend(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(
            capacity=10,
            trend_window_size=5,
            trend_min_delta=2,
            debounce_frames=1,
        ))

        # Count sequence decreasing: 8, 7, 5, 4, 2
        counts = [8, 7, 5, 4, 2]
        for idx, cnt in enumerate(counts):
            state = engine.update(current_count=cnt, frame_id=idx + 1)

        assert state.trend == CrowdTrend.DECREASING

    def test_stable_trend(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(
            capacity=10,
            trend_window_size=5,
            trend_min_delta=3,
            debounce_frames=1,
        ))

        # Small fluctuations: 5, 6, 5, 6, 5 (delta is <= 1 < 3)
        counts = [5, 6, 5, 6, 5]
        for idx, cnt in enumerate(counts):
            state = engine.update(current_count=cnt, frame_id=idx + 1)

        assert state.trend == CrowdTrend.STABLE

    def test_peak_metrics_tracking(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10))

        # Frame 1: count=3 (30%) at t=1.0
        s1 = engine.update(current_count=3, frame_id=1, timestamp=1.0)
        assert s1.peak_count == 3
        assert s1.peak_occupancy_percent == 30
        assert s1.peak_timestamp == 1.0

        # Frame 2: count=8 (80%) at t=2.5 -> new peak
        s2 = engine.update(current_count=8, frame_id=2, timestamp=2.5)
        assert s2.peak_count == 8
        assert s2.peak_occupancy_percent == 80
        assert s2.peak_timestamp == 2.5

        # Frame 3: count=4 (40%) at t=3.0 -> peak remains 8
        s3 = engine.update(current_count=4, frame_id=3, timestamp=3.0)
        assert s3.peak_count == 8
        assert s3.peak_occupancy_percent == 80
        assert s3.peak_timestamp == 2.5

    def test_session_reset(self):
        engine = CrowdAnalyticsEngine(AnalyticsConfig(capacity=10))
        engine.update(current_count=9, frame_id=1, timestamp=5.0)
        assert engine.last_state.peak_count == 9

        engine.new_session()
        assert engine.last_state is None

        # Next update in new session starts from fresh peak
        s = engine.update(current_count=2, frame_id=1, timestamp=10.0)
        assert s.peak_count == 2
        assert s.peak_occupancy_percent == 20
        assert s.peak_timestamp == 10.0

    def test_state_serialization_to_dict(self):
        state = CrowdAnalyticsState(
            current_count=5,
            capacity=10,
            capacity_state="SET",
            occupancy_percent=50,
            raw_crowd_level=CrowdLevel.MODERATE,
            crowd_level=CrowdLevel.MODERATE,
            trend=CrowdTrend.STABLE,
            peak_count=7,
            peak_occupancy_percent=70,
            peak_timestamp=2.0,
            frame_id=15,
            timestamp=3.5,
        )
        d = state.to_dict()
        assert d == {
            "current_count": 5,
            "capacity": 10,
            "capacity_state": "SET",
            "occupancy_percent": 50,
            "raw_crowd_level": "MODERATE",
            "crowd_level": "MODERATE",
            "trend": "STABLE",
            "peak_count": 7,
            "peak_occupancy_percent": 70,
            "peak_timestamp": 2.0,
            "frame_id": 15,
            "timestamp": 3.5,
        }


class TestSceneIntelligenceAndCalibration:
    """Tests for optional SceneProfile calibration and effective capacity derivation."""

    def test_manual_capacity_precedence(self):
        profile = SceneProfile(
            name="corridor_main",
            manual_capacity=15,
            usable_area_m2=50.0,
            target_density_persons_per_m2=0.5,
        )
        cap, source = profile.derive_effective_capacity()
        assert cap == 15
        assert source == CapacitySource.MANUAL

    def test_calibrated_capacity_from_area_and_density(self):
        # 35.5 m^2 * 0.8 persons/m^2 = 28.4 -> floor(28.4) = 28
        profile = SceneProfile(
            name="checkout_zone",
            manual_capacity=None,
            usable_area_m2=35.5,
            target_density_persons_per_m2=0.8,
        )
        cap, source = profile.derive_effective_capacity()
        assert cap == 28
        assert source == CapacitySource.CALIBRATED

    def test_insufficient_calibration_returns_not_set(self):
        profile = SceneProfile(
            name="uncalibrated",
            manual_capacity=None,
            usable_area_m2=None,
        )
        cap, source = profile.derive_effective_capacity()
        assert cap is None
        assert source == CapacitySource.NOT_SET

    def test_invalid_area_or_density_returns_not_set(self):
        profile_zero = SceneProfile(usable_area_m2=0.0)
        assert profile_zero.derive_effective_capacity() == (None, CapacitySource.NOT_SET)

        profile_neg = SceneProfile(usable_area_m2=-10.0)
        assert profile_neg.derive_effective_capacity() == (None, CapacitySource.NOT_SET)

        profile_density_zero = SceneProfile(usable_area_m2=20.0, target_density_persons_per_m2=0.0)
        assert profile_density_zero.derive_effective_capacity() == (None, CapacitySource.NOT_SET)

    def test_to_analytics_config_generation(self):
        custom_thresh = CrowdThresholds(moderate_threshold=30.0, high_threshold=60.0, critical_threshold=85.0)
        profile = SceneProfile(
            usable_area_m2=20.0,
            target_density_persons_per_m2=1.0,  # 20 capacity
            thresholds=custom_thresh,
        )
        cfg = profile.to_analytics_config(debounce_frames=4, trend_window_size=8)
        assert cfg.capacity == 20
        assert cfg.thresholds.moderate_threshold == 30.0
        assert cfg.debounce_frames == 4
        assert cfg.trend_window_size == 8

    def test_scene_profile_serialization(self):
        profile = SceneProfile(
            name="entrance_cam_01",
            manual_capacity=25,
            usable_area_m2=40.0,
            camera_height_m=3.2,
            camera_tilt_deg=45.0,
            horizontal_fov_deg=85.0,
        )
        d = profile.to_dict()
        assert d["name"] == "entrance_cam_01"
        assert d["manual_capacity"] == 25
        assert d["effective_capacity"] == 25
        assert d["capacity_source"] == "MANUAL"
        assert d["camera_height_m"] == 3.2
        assert d["camera_tilt_deg"] == 45.0
        assert d["horizontal_fov_deg"] == 85.0
