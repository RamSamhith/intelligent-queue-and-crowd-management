"""Comprehensive unit tests for the Scene Intelligence & Automatic Capacity Analysis subsystem."""

import pytest
from unittest.mock import MagicMock

from visionqueue.analytics.types import (
    CapacitySource,
    CrowdThresholds,
    SceneProfile,
)
from visionqueue.analytics.scene import (
    SceneAnalysisState,
    SceneAnalyzer,
    SceneAnalyzerConfig,
)
from visionqueue.detection.types import Detection
from visionqueue.camera.types import FrameData, SourceState
from visionqueue.pipeline import CVPipeline, CVPipelineConfig
import numpy as np


class TestSceneAnalyzerHierarchy:
    """Tests for the strict capacity derivation hierarchy (MANUAL -> CALIBRATED -> AUTOMATIC -> NOT_SET)."""

    def test_manual_capacity_precedence(self):
        profile = SceneProfile(
            name="lobby",
            manual_capacity=20,
            usable_area_m2=100.0,
            target_density_persons_per_m2=1.0,
        )
        analyzer = SceneAnalyzer(profile=profile)
        tracks = [{"track_id": 1, "bbox": [100.0, 100.0, 200.0, 400.0]}]
        state = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=1.0)

        assert state.effective_capacity == 20
        assert state.capacity_source == CapacitySource.MANUAL
        assert state.confidence == 1.0
        assert not state.calibration_required
        assert state.scene_quality == "EXCELLENT"

    def test_calibrated_area_precedence(self):
        # 50 m^2 @ 0.8 persons/m^2 = 40 capacity
        profile = SceneProfile(
            name="corridor",
            manual_capacity=None,
            usable_area_m2=50.0,
            target_density_persons_per_m2=0.8,
        )
        analyzer = SceneAnalyzer(profile=profile)
        tracks = [{"track_id": 1, "bbox": [100.0, 100.0, 200.0, 400.0]}]
        state = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=1.0)

        assert state.effective_capacity == 40
        assert state.capacity_source == CapacitySource.CALIBRATED
        assert state.confidence == 1.0
        assert not state.calibration_required

    def test_automatic_estimation_insufficient_samples_returns_not_set(self):
        # Need min 8 samples by default
        analyzer = SceneAnalyzer(config=SceneAnalyzerConfig(min_samples_for_estimate=8))

        # Feed 3 samples
        for i in range(3):
            tracks = [{"track_id": i + 1, "bbox": [100.0 + i * 20, 100.0, 200.0 + i * 20, 400.0]}]
            state = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=float(i), force_eval=True)

        assert state.effective_capacity is None
        assert state.capacity_source == CapacitySource.NOT_SET
        assert state.calibration_required is True
        assert state.observed_samples_count == 3
        assert "Accumulating scene observations" in state.reason

    def test_automatic_estimation_sufficient_samples_yields_valid_capacity(self):
        analyzer = SceneAnalyzer(
            config=SceneAnalyzerConfig(
                min_samples_for_estimate=8,
                target_density_persons_per_m2=0.8,
            )
        )

        # Feed 10 observations with realistic perspective spread
        for i in range(10):
            # Ground anchors spread across frame
            x1 = 100.0 + (i % 5) * 80.0
            y1 = 150.0 + (i // 5) * 80.0
            x2 = x1 + 60.0
            y2 = y1 + 180.0  # height 180
            tracks = [{"track_id": i + 1, "bbox": [x1, y1, x2, y2]}]
            state = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=float(i), force_eval=True)

        assert state.effective_capacity is not None
        assert state.effective_capacity > 0
        assert state.capacity_source == CapacitySource.AUTOMATIC
        assert state.confidence >= 0.40
        assert state.visible_area_m2_approx is not None
        assert state.visible_area_m2_approx > 0
        assert state.geometry_quality == "PERSPECTIVE_ESTIMATED"

    def test_automatic_capacity_temporal_stabilization_no_frame_flicker(self):
        """Capacity must not oscillate frame-by-frame with small noise."""
        analyzer = SceneAnalyzer(
            config=SceneAnalyzerConfig(
                min_samples_for_estimate=5,
                temporal_smoothing_alpha=0.1,
                capacity_hysteresis_threshold=2,
            )
        )

        # Feed initial stable cluster
        for i in range(6):
            tracks = [{"track_id": 1, "bbox": [200.0, 150.0, 280.0, 380.0]}]
            s = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=float(i), force_eval=True)

        initial_cap = s.effective_capacity
        assert initial_cap is not None

        # Slight frame variations / bounding box jitter should NOT change stable capacity
        for i in range(15):
            jitter = (i % 3) * 2.0
            tracks = [{"track_id": 1, "bbox": [200.0 + jitter, 150.0 - jitter, 280.0 + jitter, 380.0 + jitter]}]
            s_jitter = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=float(i + 10), force_eval=True)
            assert s_jitter.effective_capacity == initial_cap

    def test_disabled_analyzer_returns_not_set(self):
        analyzer = SceneAnalyzer(config=SceneAnalyzerConfig(enabled=False))
        tracks = [{"track_id": 1, "bbox": [100.0, 100.0, 200.0, 400.0]}]
        state = analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=1.0)
        assert state.effective_capacity is None
        assert state.capacity_source == CapacitySource.NOT_SET

    def test_reset_history_clears_observations(self):
        analyzer = SceneAnalyzer(config=SceneAnalyzerConfig(min_samples_for_estimate=5))
        for i in range(8):
            tracks = [{"track_id": 1, "bbox": [100.0, 100.0, 200.0, 400.0]}]
            analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=float(i), force_eval=True)

        assert len(analyzer._observations) == 8
        analyzer.reset_history()
        assert len(analyzer._observations) == 0

    def test_serialization_to_dict(self):
        state = SceneAnalysisState(
            effective_capacity=12,
            capacity_source=CapacitySource.AUTOMATIC,
            confidence=0.75,
            visible_area_m2_approx=15.2,
            calibration_required=True,
            scene_quality="GOOD",
            geometry_quality="PERSPECTIVE_ESTIMATED",
            reason="Estimated from perspective cues",
            observed_samples_count=20,
            timestamp=5.0,
        )
        d = state.to_dict()
        assert d["effective_capacity"] == 12
        assert d["capacity_source"] == "AUTOMATIC"
        assert d["confidence"] == 0.75
        assert d["visible_area_m2_approx"] == 15.2
        assert d["calibration_required"] is True
        assert d["scene_quality"] == "GOOD"
        assert d["observed_samples_count"] == 20


class TestPipelineAutomaticCapacityIntegration:
    """Tests CVPipeline integration with automatic scene capacity."""

    def test_pipeline_diagnostics_contain_scene_analysis(self):
        mock_detector = MagicMock()
        det = Detection(bbox=(100.0, 100.0, 200.0, 400.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det], 10.0)

        pipeline = CVPipeline(detector=mock_detector)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_data = FrameData(
            frame_id=1,
            timestamp=1.0,
            frame=dummy_frame,
            width=640,
            height=480,
            source_state=SourceState.RUNNING,
        )

        pipeline.process_frame(frame_data, timestamp=1.01)
        diag = pipeline.last_diagnostics

        assert "scene_analysis" in diag
        assert "capacity_source" in diag["scene_analysis"]
        assert "confidence" in diag["scene_analysis"]
        assert "reason" in diag["scene_analysis"]

    def test_invalid_aspect_ratio_observations_rejected(self):
        analyzer = SceneAnalyzer(config=SceneAnalyzerConfig(min_samples_for_estimate=5))
        # Add non-human wide box (width 400, height 50 -> bh/bw = 0.125 < 1.0)
        tracks = [{"track_id": 1, "bbox": [50.0, 100.0, 450.0, 150.0]}]
        analyzer.analyze(tracks, frame_w=640, frame_h=480, timestamp=1.0, force_eval=True)
        assert len(analyzer._observations) == 0

        # Add valid upright human box (width 80, height 240 -> bh/bw = 3.0)
        tracks_valid = [{"track_id": 1, "bbox": [100.0, 100.0, 180.0, 340.0]}]
        analyzer.analyze(tracks_valid, frame_w=640, frame_h=480, timestamp=2.0, force_eval=True)
        assert len(analyzer._observations) == 1

    def test_zero_frame_dimensions_handled_safely(self):
        analyzer = SceneAnalyzer()
        tracks = [{"track_id": 1, "bbox": [10.0, 10.0, 50.0, 100.0]}]
        state = analyzer.analyze(tracks, frame_w=0, frame_h=0, timestamp=1.0, force_eval=True)
        assert len(analyzer._observations) == 0
        assert state.effective_capacity is None
        assert state.capacity_source == CapacitySource.NOT_SET
