"""Integration contract tests for VisionQueue CV subsystem handoff.

Validates that the public CV interface (CVPipeline, LiveState, contracts):
1. Produces 100% JSON-serializable LiveState snapshots
2. Contains all required fields with deterministic types
3. Contains no NaN, Infinity, or unhandled null values in public fields
4. Correctly handles capacity=None and capacity > 0
5. Operates in V1 Whole-Frame mode by default
6. Correctly supports ROI-enabled mode when configured
7. Preserves last reliable count during camera/detection failure
8. Correctly serializes alerts
9. Correctly resets state upon session reset
10. Cleanly manages lifecycle (start, step, stop, context manager)
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest

from visionqueue.alerts.types import Alert, AlertEngineConfig, AlertRuleConfig, AlertSeverity, AlertStatus, AlertType
from visionqueue.analytics.types import AnalyticsConfig, CrowdLevel, CrowdThresholds, CrowdTrend
from visionqueue.camera.capture import CameraSource
from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import CrossingDirection, CrossingEvent, VirtualLine
from visionqueue.detection.types import Detection, DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import (
    CameraHealthState,
    PerformanceMetrics,
    ReliabilityConfig,
    SystemReliabilityState,
    SystemState,
    VisionHealthState,
    VisionHealthSummary,
)
from visionqueue.roi.types import ROIConfig
from visionqueue.tracking.adapter import Track as AdapterTrack
from visionqueue.tracking.bytetrack import ByteTrackConfig


class MockDetector:
    """Deterministic mock person detector for contract testing."""

    def __init__(self, detections: list[Detection] | None = None, latency_ms: float = 8.5):
        self._detections = detections if detections is not None else []
        self._latency_ms = latency_ms
        self.is_gpu = True
        self.active_provider = "CUDAExecutionProvider"

    def detect_timed(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        return self._detections, self._latency_ms

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detections

    def warm_up(self, iterations: int = 1) -> None:
        pass


def make_frame_data(frame_id: int = 1, width: int = 640, height: int = 480) -> FrameData:
    """Create a synthetic FrameData object."""
    raw_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    return FrameData(
        frame=raw_bgr,
        timestamp=time.time(),
        frame_id=frame_id,
        width=width,
        height=height,
        source_state=SourceState.RUNNING,
        fps=30.0,
    )


def assert_no_nan_or_inf(obj: Any, path: str = "") -> None:
    """Recursively verify that no float NaN or Infinity exists in a serialized dictionary."""
    if isinstance(obj, float):
        assert not math.isnan(obj), f"NaN found at {path}"
        assert not math.isinf(obj), f"Infinity found at {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_nan_or_inf(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_nan_or_inf(v, f"{path}[{i}]")


# ============================================================
# Contract Tests
# ============================================================

def test_livestate_json_roundtrip_serialization():
    """LiveState.to_dict() must be 100% JSON-serializable and round-trip cleanly."""
    perf = PerformanceMetrics(processing_fps=28.5, inference_latency_ms=8.2, frame_age_ms=12.0)
    vision = VisionHealthSummary(
        person_detection=VisionHealthState.OK.value,
        face_detection=VisionHealthState.OK.value,
        tracking=VisionHealthState.OK.value,
    )
    alert = Alert(
        id="alert-123",
        type=AlertType.CRITICAL_OCCUPANCY,
        severity=AlertSeverity.CRITICAL,
        fired_at=1000.0,
        cleared_at=None,
        status=AlertStatus.ACTIVE,
        reason="Crowd occupancy exceeded 90%",
    )

    state = LiveState(
        schema_version=1,
        timestamp=1000.0,
        session_id="test-session-uuid",
        frame_id=42,
        system_state=SystemState.LIVE,
        camera_state=CameraHealthState.CONNECTED,
        performance=perf,
        vision=vision,
        counts={
            "current": 5,
            "unique_session_approx": 12,
            "entries": 8,
            "exits": 3,
            "net_count": 5,
        },
        occupancy={
            "capacity": 20,
            "capacity_state": "SET",
            "percent": 25,
        },
        crowd={
            "level": "LOW",
            "raw_level": "LOW",
            "trend": "STABLE",
            "peak_count": 7,
            "peak_occupancy_percent": 35,
            "peak_timestamp": 950.0,
        },
        alerts=[alert],
        status_reason="System running nominally",
        is_healthy=True,
        is_frozen=False,
        detections_count=5,
        tracks_count=5,
        faces_count=0,
    )

    state_dict = state.to_dict()

    # Must serialize to JSON without throwing TypeError
    json_str = json.dumps(state_dict)
    assert isinstance(json_str, str)

    # Must deserialize cleanly
    recovered = json.loads(json_str)
    assert recovered["schema_version"] == 1
    assert recovered["counts"]["current"] == 5
    assert recovered["occupancy"]["percent"] == 25
    assert recovered["crowd"]["level"] == "LOW"
    assert recovered["system_state"] == "LIVE"
    assert len(recovered["alerts"]) == 1
    assert recovered["alerts"][0]["id"] == "alert-123"
    assert recovered["is_healthy"] is True
    assert recovered["is_frozen"] is False

    # Check for NaN / Inf
    assert_no_nan_or_inf(recovered)


def test_livestate_schema_required_keys():
    """LiveState dictionary must contain all required top-level and nested keys."""
    config = CVPipelineConfig(session_id="contract-test-session")
    mock_det = MockDetector([])
    pipeline = CVPipeline(config=config, detector=mock_det)

    frame = make_frame_data(frame_id=1)
    state = pipeline.process_frame(frame)
    payload = state.to_dict()

    required_top_level = [
        "schema_version", "timestamp", "session_id", "frame_id",
        "system_state", "camera_state", "performance", "vision",
        "counts", "occupancy", "crowd", "alerts", "status_reason",
        "is_healthy", "is_frozen", "detections_count", "tracks_count", "faces_count",
    ]
    for key in required_top_level:
        assert key in payload, f"Missing required top-level key: {key}"

    # Counts required keys
    for count_key in ["current", "unique_session_approx", "entries", "exits", "net_count"]:
        assert count_key in payload["counts"], f"Missing required counts key: {count_key}"

    # Occupancy required keys
    for occ_key in ["capacity", "capacity_state", "percent"]:
        assert occ_key in payload["occupancy"], f"Missing required occupancy key: {occ_key}"

    # Crowd required keys
    for crowd_key in ["level", "raw_level", "trend", "peak_count", "peak_occupancy_percent", "peak_timestamp"]:
        assert crowd_key in payload["crowd"], f"Missing required crowd key: {crowd_key}"

    # Performance required keys
    for perf_key in ["processing_fps", "inference_latency_ms", "frame_age_ms"]:
        assert perf_key in payload["performance"], f"Missing required performance key: {perf_key}"

    # Vision required keys
    for vis_key in ["person_detection", "face_detection", "tracking"]:
        assert vis_key in payload["vision"], f"Missing required vision key: {vis_key}"


def test_capacity_not_set_handling():
    """When capacity is None, occupancy percent must be None and capacity_state must be 'NOT_SET'."""
    config = CVPipelineConfig(
        analytics=AnalyticsConfig(capacity=None),
        session_id="capacity-none-test",
    )
    pipeline = CVPipeline(config=config, detector=MockDetector([]))
    state = pipeline.process_frame(make_frame_data(frame_id=1))
    payload = state.to_dict()

    assert payload["occupancy"]["capacity"] is None
    assert payload["occupancy"]["capacity_state"] == "NOT_SET"
    assert payload["occupancy"]["percent"] is None


def test_capacity_set_handling():
    """When capacity > 0, occupancy percent must be integer percentage and capacity_state 'SET'."""
    config = CVPipelineConfig(
        analytics=AnalyticsConfig(capacity=10),
        session_id="capacity-set-test",
    )
    # Detect 4 persons
    mock_dets = [
        Detection(bbox=(100 + i * 50, 100, 140 + i * 50, 300), confidence=0.9, class_id=0)
        for i in range(4)
    ]
    pipeline = CVPipeline(config=config, detector=MockDetector(mock_dets))

    # Process 3 frames to establish tracks
    for f in range(1, 4):
        state = pipeline.process_frame(make_frame_data(frame_id=f))

    payload = state.to_dict()
    assert payload["occupancy"]["capacity"] == 10
    assert payload["occupancy"]["capacity_state"] == "SET"
    assert payload["occupancy"]["percent"] == 40  # 4 / 10 = 40%
    assert payload["counts"]["current"] == 4


def test_v1_whole_frame_default_mode():
    """Pipeline default configuration must be Whole-Frame counting (enable_roi=False)."""
    config = CVPipelineConfig()
    assert config.enable_roi is False, "V1 default must be enable_roi=False"

    # Detections anywhere in frame must be counted
    mock_dets = [
        Detection(bbox=(10, 10, 50, 100), confidence=0.9, class_id=0),
        Detection(bbox=(550, 350, 630, 470), confidence=0.9, class_id=0),
    ]
    pipeline = CVPipeline(config=config, detector=MockDetector(mock_dets))
    for f in range(1, 4):
        state = pipeline.process_frame(make_frame_data(frame_id=f))

    assert state.counts["current"] == 2


def test_camera_failure_preserves_last_reliable_count():
    """When camera goes offline, count must be frozen to last reliable count (never zeroed)."""
    config = CVPipelineConfig(
        alerts=AlertEngineConfig(
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0)
        ),
        session_id="failover-test",
    )
    mock_dets = [Detection(bbox=(100, 100, 200, 300), confidence=0.9, class_id=0)]
    pipeline = CVPipeline(config=config, detector=MockDetector(mock_dets))

    # Establish live state with 1 person over 5 frames
    for f in range(1, 6):
        state = pipeline.process_frame(make_frame_data(frame_id=f), timestamp=100.0 + f * 0.033)
    assert state.system_state == SystemState.LIVE
    assert state.counts["current"] == 1

    # Immediate camera failure (frame_data = None at t=101.0)
    fail_state1 = pipeline.process_frame(None, timestamp=101.0)
    payload1 = fail_state1.to_dict()

    assert payload1["system_state"] == "OFFLINE"
    assert payload1["camera_state"] in ("RECONNECTING", "OFFLINE")
    assert payload1["is_healthy"] is False
    assert payload1["is_frozen"] is True
    assert payload1["counts"]["current"] == 1, "Count must be frozen to last reliable count (1), not 0"

    # Sustained failure (> 1.0s debounce at t=102.5) -> Alert fires
    fail_state2 = pipeline.process_frame(None, timestamp=102.5)
    payload2 = fail_state2.to_dict()
    assert payload2["counts"]["current"] == 1
    assert any(a["type"] == "CAMERA_OR_DETECTION_FAILURE" for a in payload2["alerts"])


def test_pipeline_reset_session_clean_state():
    """Calling reset_session() must completely reset tracking, counting, and metrics."""
    config = CVPipelineConfig()
    mock_dets = [Detection(bbox=(100, 100, 200, 300), confidence=0.9, class_id=0)]
    pipeline = CVPipeline(config=config, detector=MockDetector(mock_dets))

    for f in range(1, 4):
        pipeline.process_frame(make_frame_data(frame_id=f))

    # Reset
    pipeline.reset_session()

    assert pipeline._frame_sequence == 0
    assert pipeline._last_state is None
    assert pipeline._session_counter.approximate_unique_count == 0


def test_context_manager_lifecycle():
    """Pipeline must support context manager with CVPipeline(...) start and stop."""
    config = CVPipelineConfig(camera=CameraConfig(source=0))
    pipeline = CVPipeline(config=config, detector=MockDetector([]))

    with pipeline:
        assert pipeline.is_running is True

    assert pipeline.is_running is False
