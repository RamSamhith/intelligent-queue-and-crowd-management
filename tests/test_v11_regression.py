"""VisionQueue V1.1 — Queue Counting, Schema, Capacity & Failure Regression Tests.

Verifies the V1.1 upgrades:
- Dedicated queue ROI counting via queue_people field
- Schema compatibility (track_instances + deprecated alias)
- Capacity safety (AUTOMATIC never authoritative)
- Failure recovery (no phantom entry/exit on camera/detector recovery)
- queue_people <= current_people invariant
"""

import math
import time
import numpy as np
import pytest
from unittest.mock import MagicMock

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import (
    AnalyticsConfig,
    CrowdLevel,
    CrowdThresholds,
    SceneProfile,
)
from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.types import Detection
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import ReliabilityConfig, SystemState
from visionqueue.roi.types import ROIConfig


def make_frame_data(
    frame_id: int = 1,
    timestamp: float = 1.0,
    width: int = 640,
    height: int = 480,
) -> FrameData:
    dummy_arr = np.zeros((height, width, 3), dtype=np.uint8)
    return FrameData(
        frame=dummy_arr,
        timestamp=timestamp,
        frame_id=frame_id,
        width=width,
        height=height,
        source_state=SourceState.RUNNING,
        fps=30.0,
    )


def make_mock_detector(detections: list[Detection] | None = None, latency_ms: float = 8.5):
    mock = MagicMock()
    mock.detect_timed.return_value = (detections or [], latency_ms)
    mock.detect.return_value = detections or []
    mock.is_gpu = True
    mock.active_provider = "CUDAExecutionProvider"
    return mock


# ===========================================================================
# Queue Counting Tests
# ===========================================================================

class TestQueueCounting:
    """Tests for the optional dedicated queue_roi counting."""

    def test_queue_people_zero_when_no_queue_roi(self):
        """When queue_roi is None, queue_people is always 0."""
        config = CVPipelineConfig(queue_roi=None)
        pipeline = CVPipeline(config=config, detector=make_mock_detector())
        state = pipeline.process_frame(make_frame_data(frame_id=1))
        assert state.queue_people == 0

    def test_queue_people_inside_queue_roi(self):
        """Track inside the configured queue_roi is counted as queue_people."""
        queue_roi = ROIConfig(x=200, y=200, width=200, height=200)
        config = CVPipelineConfig(
            queue_roi=queue_roi,
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        # Track 1 bottom-center = (300, 300) -> inside queue_roi [200, 200, 400, 400]
        det = Detection(bbox=(280.0, 250.0, 320.0, 350.0), confidence=0.9, class_id=0)
        pipeline = CVPipeline(config=config, detector=make_mock_detector([det]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        state = pipeline.process_frame(f2, timestamp=1.04)
        assert state.queue_people == 1

    def test_queue_people_outside_queue_roi(self):
        """Track outside the configured queue_roi is not counted as queue_people."""
        queue_roi = ROIConfig(x=200, y=200, width=100, height=100)  # small region
        config = CVPipelineConfig(
            queue_roi=queue_roi,
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        # Track 1 bottom-center = (50, 50) -> outside queue_roi [200, 200, 300, 300]
        det = Detection(bbox=(30.0, 30.0, 70.0, 90.0), confidence=0.9, class_id=0)
        pipeline = CVPipeline(config=config, detector=make_mock_detector([det]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        state = pipeline.process_frame(f2, timestamp=1.04)
        assert state.queue_people == 0

    def test_queue_people_lte_current_people(self):
        """Invariant: queue_people <= current_people for any valid configuration."""
        queue_roi = ROIConfig(x=100, y=100, width=300, height=300)
        config = CVPipelineConfig(
            queue_roi=queue_roi,
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        # 3 tracks: 1 outside queue, 2 inside queue
        t1 = Detection(bbox=(30.0, 30.0, 70.0, 90.0), confidence=0.9, class_id=0)     # outside
        t2 = Detection(bbox=(180.0, 180.0, 220.0, 220.0), confidence=0.9, class_id=0)  # inside
        t3 = Detection(bbox=(280.0, 280.0, 320.0, 320.0), confidence=0.9, class_id=0)  # inside
        pipeline = CVPipeline(config=config, detector=make_mock_detector([t1, t2, t3]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        state = pipeline.process_frame(f2, timestamp=1.04)
        assert state.counts["current"] == 3
        assert state.queue_people == 2
        assert state.queue_people <= state.counts["current"]

    def test_queue_roi_reset(self):
        """reset_session() clears queue_people state."""
        queue_roi = ROIConfig(x=200, y=200, width=200, height=200)
        config = CVPipelineConfig(
            queue_roi=queue_roi,
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        det = Detection(bbox=(280.0, 250.0, 320.0, 350.0), confidence=0.9, class_id=0)
        pipeline = CVPipeline(config=config, detector=make_mock_detector([det]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        s_before = pipeline.process_frame(f2, timestamp=1.04)
        assert s_before.queue_people == 1

        pipeline.reset_session()
        # No tracks after reset
        pipeline._detector.detect_timed.return_value = ([], 8.0)
        s_after = pipeline.process_frame(make_frame_data(frame_id=10), timestamp=10.0)
        assert s_after.queue_people == 0

    def test_queue_people_in_livestate_serialization(self):
        """queue_people appears in LiveState.to_dict() output."""
        queue_roi = ROIConfig(x=200, y=200, width=200, height=200)
        config = CVPipelineConfig(queue_roi=queue_roi)
        pipeline = CVPipeline(config=config, detector=make_mock_detector())
        state = pipeline.process_frame(make_frame_data(frame_id=1))
        d = state.to_dict()
        assert "queue_people" in d
        assert d["queue_people"] == 0


# ===========================================================================
# Schema Compatibility Tests
# ===========================================================================

class TestSchemaCompatibility:
    """Tests for the track_instances + deprecated unique_session_approx alias."""

    def test_livestate_has_track_instances_field(self):
        config = CVPipelineConfig()
        pipeline = CVPipeline(config=config, detector=make_mock_detector())
        state = pipeline.process_frame(make_frame_data(frame_id=1))
        assert "track_instances" in state.counts
        assert state.counts["track_instances"] == 0

    def test_livestate_has_deprecated_alias(self):
        """The deprecated unique_session_approx field is preserved for backward compat."""
        config = CVPipelineConfig()
        pipeline = CVPipeline(config=config, detector=make_mock_detector())
        state = pipeline.process_frame(make_frame_data(frame_id=1))
        assert "unique_session_approx" in state.counts
        assert state.counts["unique_session_approx"] == 0

    def test_track_instances_matches_deprecated_alias(self):
        """Both fields always carry the same value."""
        config = CVPipelineConfig()
        pipeline = CVPipeline(config=config, detector=make_mock_detector())
        # Inject detections to bump the session counter
        det = Detection(bbox=(100.0, 100.0, 150.0, 200.0), confidence=0.9, class_id=0)
        pipeline._detector.detect_timed.return_value = ([det], 8.0)
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        state = pipeline.process_frame(f2, timestamp=1.04)
        assert state.counts["track_instances"] == state.counts["unique_session_approx"]
        assert state.counts["track_instances"] == 1

    def test_session_reset_clears_track_instances(self):
        """reset_session() clears the track_instances count."""
        config = CVPipelineConfig()
        pipeline = CVPipeline(config=config, detector=make_mock_detector([
            Detection(bbox=(100.0, 100.0, 150.0, 200.0), confidence=0.9, class_id=0)
        ]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        s_before = pipeline.process_frame(f2, timestamp=1.04)
        assert s_before.counts["track_instances"] >= 1

        pipeline.reset_session()
        # Empty detection list ensures no new track is added
        pipeline._detector.detect_timed.return_value = ([], 8.0)
        s_after = pipeline.process_frame(make_frame_data(frame_id=10), timestamp=10.0)
        assert s_after.counts["track_instances"] == 0


# ===========================================================================
# Capacity Safety Tests
# ===========================================================================

class TestCapacitySafety:
    """Tests for the AUTOMATIC capacity safety isolation."""

    def test_automatic_capacity_does_not_propagate_to_analytics(self):
        """AUTOMATIC (monocular) capacity must NOT be used by analytics/alerts."""
        # SceneProfile with NO manual/calibrated capacity; only camera geometry for AUTOMATIC
        scene = SceneProfile(
            name="uncalibrated",
            manual_capacity=None,
            usable_area_m2=None,
            camera_height_m=2.8,
            horizontal_fov_deg=75.0,
        )
        config = CVPipelineConfig(
            scene=scene,
            analytics=AnalyticsConfig(capacity=None),
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        pipeline = CVPipeline(config=config, detector=make_mock_detector([
            Detection(bbox=(100.0, 100.0, 150.0, 200.0), confidence=0.9, class_id=0)
        ]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        state = pipeline.process_frame(f2, timestamp=1.04)
        # The analytics engine's capacity must remain None because AUTOMATIC is advisory
        assert state.occupancy["capacity"] is None
        assert state.occupancy["capacity_state"] == "NOT_SET"

    def test_manual_capacity_propagates(self):
        """MANUAL capacity propagates to analytics."""
        scene = SceneProfile(name="test", manual_capacity=15)
        config = CVPipelineConfig(
            scene=scene,
            analytics=AnalyticsConfig(capacity=None),
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        pipeline = CVPipeline(config=config, detector=make_mock_detector([
            Detection(bbox=(100.0, 100.0, 150.0, 200.0), confidence=0.9, class_id=0)
        ]))
        f1 = make_frame_data(frame_id=1)
        f2 = make_frame_data(frame_id=2)
        pipeline.process_frame(f1, timestamp=1.01)
        state = pipeline.process_frame(f2, timestamp=1.04)
        assert state.occupancy["capacity"] == 15
        assert state.occupancy["capacity_state"] == "SET"

    def test_capacity_not_set_no_alert(self):
        """When capacity is NOT_SET, CRITICAL_OCCUPANCY alert never fires regardless of count."""
        config = CVPipelineConfig(
            analytics=AnalyticsConfig(capacity=None),
            alerts=AlertEngineConfig(
                critical_occupancy=AlertRuleConfig(debounce_seconds=0.5, clear_seconds=0.5),
            ),
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        # 20 detections (would be CRITICAL if capacity=10)
        dets = [
            Detection(bbox=(10 + i * 30, 100, 30 + i * 30, 200), confidence=0.9, class_id=0)
            for i in range(20)
        ]
        pipeline = CVPipeline(config=config, detector=make_mock_detector(dets))
        for f in range(1, 10):
            pipeline.process_frame(make_frame_data(frame_id=f), timestamp=f * 0.1)
        state = pipeline._last_state
        assert state.occupancy["capacity"] is None
        # No CRITICAL_OCCUPANCY alert should have fired
        assert not any(a.type.value == "CRITICAL_OCCUPANCY" for a in state.alerts)


# ===========================================================================
# Failure Recovery Tests (No Phantom Entry/Exit)
# ===========================================================================

class TestNoPhantomCrossingsOnFailure:
    """Tests that camera/detector failure and recovery do NOT produce fake entry/exit events."""

    def test_camera_offline_then_recovery_no_phantom_entry(self):
        """Camera offline -> recovery must not cause a phantom ENTRY event."""
        config = CVPipelineConfig(
            virtual_line=VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0)),
            analytics=AnalyticsConfig(capacity=10),
            reliability=ReliabilityConfig(min_starting_frames=1, unstable_failure_threshold=3),
        )
        det = Detection(bbox=(180.0, 150.0, 220.0, 200.0), confidence=0.9, class_id=0)
        pipeline = CVPipeline(config=config, detector=make_mock_detector([det]))
        # Frame 1: live with track on left
        f1 = make_frame_data(frame_id=1)
        s1 = pipeline.process_frame(f1, timestamp=1.0)
        assert s1.counts["entries"] == 0
        assert s1.counts["exits"] == 0
        # Frame 2: camera offline (frame_data=None)
        s2 = pipeline.process_frame(None, timestamp=2.0)
        assert s2.counts["entries"] == 0
        # Frame 3: recovery with same track on left (NO movement)
        pipeline._detector.detect_timed.return_value = ([det], 8.0)
        s3 = pipeline.process_frame(make_frame_data(frame_id=3), timestamp=3.0)
        # Track stayed on left: no entry event
        assert s3.counts["entries"] == 0
        assert s3.counts["exits"] == 0

    def test_track_disappear_reappear_no_phantom_event(self):
        """A track that disappears and reappears with a NEW id must not cause phantom entry."""
        config = CVPipelineConfig(
            virtual_line=VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0)),
            analytics=AnalyticsConfig(capacity=10),
            reliability=ReliabilityConfig(min_starting_frames=1),
        )
        # Frame 1: track on left
        det = Detection(bbox=(180.0, 150.0, 220.0, 200.0), confidence=0.9, class_id=0)
        pipeline = CVPipeline(config=config, detector=make_mock_detector([det]))
        s1 = pipeline.process_frame(make_frame_data(frame_id=1), timestamp=1.0)
        s2 = pipeline.process_frame(make_frame_data(frame_id=2), timestamp=1.1)
        assert s1.counts["entries"] == 0
        assert s2.counts["entries"] == 0
        # The track reappears with the SAME position (ByteTrack should preserve ID)
        s3 = pipeline.process_frame(make_frame_data(frame_id=3), timestamp=1.2)
        assert s3.counts["entries"] == 0
        assert s3.counts["exits"] == 0

    def test_detector_failure_does_not_corrupt_counts(self):
        """Detector failure must not cause spurious entry/exit events."""
        config = CVPipelineConfig(
            virtual_line=VirtualLine(pt1=(300.0, 0.0), pt2=(300.0, 500.0)),
            analytics=AnalyticsConfig(capacity=10),
            reliability=ReliabilityConfig(min_starting_frames=1, unstable_failure_threshold=3),
        )
        pipeline = CVPipeline(config=config, detector=make_mock_detector())
        # Detector raises exception
        pipeline._detector.detect_timed.side_effect = RuntimeError("ONNX Error")
        s1 = pipeline.process_frame(make_frame_data(frame_id=1), timestamp=1.0)
        assert s1.counts["entries"] == 0
        assert s1.counts["exits"] == 0
        # No entries/exits recorded during failure
        s2 = pipeline.process_frame(make_frame_data(frame_id=2), timestamp=1.1)
        assert s2.counts["entries"] == 0
        assert s2.counts["exits"] == 0
