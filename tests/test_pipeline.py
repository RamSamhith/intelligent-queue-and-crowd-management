"""Comprehensive integration tests for the unified VisionQueue CV Pipeline Coordinator."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig, AlertType
from visionqueue.analytics.types import AnalyticsConfig, CrowdLevel
from visionqueue.camera.types import FrameData, SourceState
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.types import Detection
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import ReliabilityConfig, SystemState
from visionqueue.roi.types import ROIConfig


def make_frame_data(
    frame_id: int = 1,
    timestamp: float = 1.0,
    fps: float = 30.0,
    source_state: SourceState = SourceState.RUNNING,
) -> FrameData:
    """Helper to create dummy FrameData."""
    dummy_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameData(
        frame=dummy_arr,
        timestamp=timestamp,
        frame_id=frame_id,
        width=640,
        height=480,
        source_state=source_state,
        fps=fps,
    )


class TestCVPipelineIntegration:
    """End-to-end integration tests for CVPipeline."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock PersonDetector."""
        mock = MagicMock()
        mock.detect_timed.return_value = ([], 10.0)
        mock.detect.return_value = []
        return mock

    @pytest.fixture
    def pipeline_config(self):
        """Standard pipeline test configuration."""
        return CVPipelineConfig(
            roi=ROIConfig(x=100, y=100, width=400, height=300),
            virtual_line=VirtualLine(pt1=(300.0, 100.0), pt2=(300.0, 400.0)),
            analytics=AnalyticsConfig(capacity=10, debounce_frames=2),
            alerts=AlertEngineConfig(
                critical_occupancy=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
                camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
            ),
            reliability=ReliabilityConfig(min_starting_frames=2),
            enable_face_detection=False,
        )

    def test_normal_frame_empty_scene(self, pipeline_config, mock_detector):
        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        s1 = pipeline.process_frame(f1, timestamp=1.01)

        assert isinstance(s1, LiveState)
        assert s1.frame_id == 1
        assert s1.system_state == SystemState.STARTING
        assert s1.counts["current"] == 0
        assert s1.counts["unique_session_approx"] == 0
        assert s1.occupancy["percent"] == 0
        assert s1.crowd["level"] == "LOW"

    def test_default_whole_frame_counting(self, pipeline_config, mock_detector):
        # 3 people across the camera FOV (one at top-left, one in middle, one at bottom-right)
        det1 = Detection(bbox=(180.0, 200.0, 220.0, 250.0), confidence=0.9, class_id=0)
        det2 = Detection(bbox=(330.0, 250.0, 370.0, 300.0), confidence=0.85, class_id=0)
        det3 = Detection(bbox=(30.0, 20.0, 70.0, 50.0), confidence=0.95, class_id=0)

        mock_detector.detect_timed.return_value = ([det1, det2, det3], 12.5)

        # Default config has enable_roi=False (whole-frame mode)
        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        f2 = make_frame_data(frame_id=2, timestamp=1.033)

        pipeline.process_frame(f1, timestamp=1.01)
        s2 = pipeline.process_frame(f2, timestamp=1.04)

        assert s2.system_state == SystemState.LIVE
        assert s2.is_healthy is True
        assert s2.detections_count == 3
        assert s2.tracks_count == 3
        # In whole-frame mode: ALL 3 people are counted
        assert s2.counts["current"] == 3
        assert s2.occupancy["percent"] == 30  # 3 / 10 = 30%
        assert s2.crowd["level"] == "LOW"

    def test_multiple_people_and_roi_filtering(self, pipeline_config, mock_detector):
        # Enable ROI mode explicitly
        pipeline_config.enable_roi = True

        # ROI: [100, 100, 500, 400]
        # Person 1: inside ROI -> bottom-center (200, 250)
        # Person 2: inside ROI -> bottom-center (350, 300)
        # Person 3: outside ROI -> bottom-center (50, 50)
        det1 = Detection(bbox=(180.0, 200.0, 220.0, 250.0), confidence=0.9, class_id=0)
        det2 = Detection(bbox=(330.0, 250.0, 370.0, 300.0), confidence=0.85, class_id=0)
        det3 = Detection(bbox=(30.0, 20.0, 70.0, 50.0), confidence=0.95, class_id=0)

        mock_detector.detect_timed.return_value = ([det1, det2, det3], 12.5)

        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        # Run 2 frames to reach LIVE
        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        f2 = make_frame_data(frame_id=2, timestamp=1.033)

        pipeline.process_frame(f1, timestamp=1.01)
        s2 = pipeline.process_frame(f2, timestamp=1.04)

        assert s2.system_state == SystemState.LIVE
        assert s2.is_healthy is True
        assert s2.detections_count == 3
        assert s2.tracks_count == 3
        # In-ROI count: only 2 people are inside ROI!
        assert s2.counts["current"] == 2
        assert s2.occupancy["percent"] == 20  # 2 / 10 = 20%
        assert s2.crowd["level"] == "LOW"

    def test_entry_exit_line_crossing(self, pipeline_config, mock_detector):
        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        # Virtual line is at x=300 (from y=100 to y=400).
        # Box width = 100.
        # Frame 1: Person at x=280 (bbox 230..330, bottom-center = (280, 250), left of x=300)
        det_f1 = Detection(bbox=(230.0, 200.0, 330.0, 250.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det_f1], 10.0)
        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        s1 = pipeline.process_frame(f1, timestamp=1.01)
        assert s1.counts["entries"] == 0

        # Frame 2: Person at x=290 (bbox 240..340, IoU > 0.8, bottom-center = (290, 250))
        det_f2 = Detection(bbox=(240.0, 200.0, 340.0, 250.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det_f2], 10.0)
        f2 = make_frame_data(frame_id=2, timestamp=1.033)
        pipeline.process_frame(f2, timestamp=1.04)

        # Frame 3: Person at x=310 (bbox 260..360, crosses x=300, bottom-center = (310, 250))
        det_f3 = Detection(bbox=(260.0, 200.0, 360.0, 250.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det_f3], 10.0)
        f3 = make_frame_data(frame_id=3, timestamp=1.066)
        s3 = pipeline.process_frame(f3, timestamp=1.07)

        assert s3.counts["entries"] == 1
        assert s3.counts["exits"] == 0
        assert s3.counts["net_count"] == 1

    def test_critical_occupancy_alert_trigger(self, pipeline_config, mock_detector):
        # Capacity = 10. High crowd >= 90% (9 people)
        dets = [
            Detection(bbox=(150.0 + i * 20, 150.0, 170.0 + i * 20, 250.0), confidence=0.9, class_id=0)
            for i in range(9)
        ]
        mock_detector.detect_timed.return_value = (dets, 15.0)

        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        # Run frames with critical crowd
        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        pipeline.process_frame(f1, timestamp=1.0)

        f2 = make_frame_data(frame_id=2, timestamp=1.5)
        pipeline.process_frame(f2, timestamp=1.5)

        # At t=2.5 (> 1.0s debounce past t=1.5 when CRITICAL held)
        f3 = make_frame_data(frame_id=3, timestamp=2.5)
        s3 = pipeline.process_frame(f3, timestamp=2.5)

        assert s3.crowd["level"] == "CRITICAL"
        assert len(s3.alerts) == 1
        assert s3.alerts[0].type == AlertType.CRITICAL_OCCUPANCY
        assert s3.alerts[0].status.value == "ACTIVE"

    def test_camera_failure_freezes_count_and_triggers_alert(self, pipeline_config, mock_detector):
        det = Detection(bbox=(200.0, 200.0, 250.0, 250.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det], 10.0)

        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        # 1. Establish LIVE with count = 1
        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        f2 = make_frame_data(frame_id=2, timestamp=1.033)
        pipeline.process_frame(f1, timestamp=1.01)
        s_live = pipeline.process_frame(f2, timestamp=1.04)
        assert s_live.system_state == SystemState.LIVE
        assert s_live.counts["current"] == 1

        # 2. Camera feed lost (frame_data is None)
        s_fail1 = pipeline.process_frame(frame_data=None, timestamp=2.0)
        assert s_fail1.system_state == SystemState.OFFLINE
        assert s_fail1.is_frozen is True
        assert s_fail1.counts["current"] == 1  # Preserved! Never 0!

        # 3. Sustained failure for 1.0s -> Alert fires
        s_fail2 = pipeline.process_frame(frame_data=None, timestamp=3.1)
        assert s_fail2.system_state == SystemState.OFFLINE
        assert s_fail2.counts["current"] == 1
        fail_alerts = [a for a in s_fail2.alerts if a.type == AlertType.CAMERA_OR_DETECTION_FAILURE]
        assert len(fail_alerts) == 1

    def test_camera_recovery_lifecycle(self, pipeline_config, mock_detector):
        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        # 1. LIVE
        f1 = make_dummy_frame_pipeline(1, 1.0)
        f2 = make_dummy_frame_pipeline(2, 1.033)
        pipeline.process_frame(f1, timestamp=1.01)
        pipeline.process_frame(f2, timestamp=1.04)

        # 2. OFFLINE
        pipeline.process_frame(None, timestamp=2.0)
        assert pipeline._last_state.system_state == SystemState.OFFLINE

        # 3. Stream recovered on frame 3 -> Must be STARTING
        f3 = make_dummy_frame_pipeline(3, 3.0)
        s3 = pipeline.process_frame(f3, timestamp=3.01)
        assert s3.system_state == SystemState.STARTING
        assert not s3.is_healthy

        # 4. Second healthy frame -> reaches LIVE
        f4 = make_dummy_frame_pipeline(4, 3.033)
        s4 = pipeline.process_frame(f4, timestamp=3.04)
        assert s4.system_state == SystemState.LIVE
        assert s4.is_healthy

    def test_detector_failure_isolation(self, pipeline_config, mock_detector):
        # Inject detector exception
        mock_detector.detect_timed.side_effect = RuntimeError("ONNX Execution Error")

        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)

        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        # Should not raise exception; handles gracefully via reliability manager
        s1 = pipeline.process_frame(f1, timestamp=1.01)

        assert s1.system_state == SystemState.DEGRADED
        assert s1.vision.person_detection == "ERROR"
        assert not s1.is_healthy

    def test_face_detector_failure_does_not_break_person_counting(self, pipeline_config, mock_detector):
        # Enable face detection with failing face detector
        mock_face = MagicMock()
        mock_face.detect.side_effect = RuntimeError("YuNet face error")

        det = Detection(bbox=(200.0, 200.0, 250.0, 250.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det], 10.0)

        config = CVPipelineConfig(
            roi=pipeline_config.roi,
            analytics=pipeline_config.analytics,
            enable_face_detection=True,
            face_cadence=1,
        )
        pipeline = CVPipeline(config=config, detector=mock_detector, face_detector=mock_face)

        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        s1 = pipeline.process_frame(f1, timestamp=1.01)

        # Person counting still succeeded!
        assert s1.counts["current"] == 1
        assert s1.vision.face_detection == "ERROR"
        assert s1.vision.person_detection == "OK"

    def test_livestate_serialization_roundtrip(self, pipeline_config, mock_detector):
        pipeline = CVPipeline(config=pipeline_config, detector=mock_detector)
        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        state = pipeline.process_frame(f1, timestamp=1.01)

        d = state.to_dict()
        assert d["schema_version"] == 1
        assert "counts" in d
        assert "occupancy" in d
        assert "crowd" in d
        assert "performance" in d
        assert "vision" in d
        assert "alerts" in d
        assert d["system_state"] in ("STARTING", "LIVE", "DEGRADED", "UNSTABLE", "OFFLINE", "STOPPING")

    def test_scene_profile_capacity_integration(self, mock_detector):
        from visionqueue.analytics import SceneProfile
        det = Detection(bbox=(100.0, 100.0, 200.0, 300.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det], 10.0)

        # Usable area = 50.0 m^2, density = 0.4 persons/m^2 -> effective capacity = 20
        scene = SceneProfile(
            name="atrium",
            usable_area_m2=50.0,
            target_density_persons_per_m2=0.4,
        )
        config = CVPipelineConfig(scene=scene)
        pipeline = CVPipeline(config=config, detector=mock_detector)

        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        s1 = pipeline.process_frame(f1, timestamp=1.01)

        assert s1.occupancy["capacity"] == 20
        assert s1.occupancy["capacity_state"] == "SET"
        assert s1.occupancy["percent"] == 5  # 1 person / 20 capacity = 5%


def make_dummy_frame_pipeline(frame_id: int, timestamp: float) -> FrameData:
    return make_frame_data(frame_id=frame_id, timestamp=timestamp)


# ============================================================
# Tests: ROI frame-dimension fix (Phase A regression)
# ============================================================

class TestROIFrameDimensions:
    """Verify ROIFilter receives actual frame dimensions, not roi.x2/roi.y2.

    Root cause that was fixed: coordinator previously used roi.x2 and roi.y2
    as the frame_width/frame_height arguments to ROIFilter, so a track whose
    bottom-center sat between roi.y2 and the real frame height (e.g. person
    with feet near y=475 in a 480px frame when roi.y2=440) was incorrectly
    excluded from the in-ROI count.
    """

    @pytest.fixture
    def mock_detector(self):
        mock = MagicMock()
        mock.detect_timed.return_value = ([], 10.0)
        return mock

    def test_bottom_center_at_frame_edge_is_counted(self, mock_detector):
        """A track whose bottom-center is near the frame bottom must be counted
        when the ROI covers that area, regardless of whether the bottom is
        'beyond' roi.y2 in the old (broken) scheme.

        ROI  : x=0, y=0, width=640, height=480  → covers full 640×480 frame.
        Track: bbox (200, 50, 400, 475)          → bottom-center = (300, 475).
        Expected: track IS inside ROI → current_count == 1.
        """
        roi = ROIConfig(x=0, y=0, width=640, height=480)
        # Detection with high confidence whose bbox bottom is at y=475
        det = Detection(bbox=(200.0, 50.0, 400.0, 475.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det], 10.0)

        config = CVPipelineConfig(
            enable_roi=True,
            roi=roi,
            analytics=AnalyticsConfig(capacity=10, debounce_frames=2),
            reliability=ReliabilityConfig(min_starting_frames=2),
            enable_face_detection=False,
        )
        pipeline = CVPipeline(config=config, detector=mock_detector)

        # Use a 640×480 FrameData so the pipeline knows actual dimensions.
        f1 = make_frame_data(frame_id=1, timestamp=1.0)   # 640×480 from helper
        f2 = make_frame_data(frame_id=2, timestamp=1.033)
        pipeline.process_frame(f1, timestamp=1.01)
        s2 = pipeline.process_frame(f2, timestamp=1.04)

        assert s2.counts["current"] == 1, (
            "Track with bottom-center at y=475 must be inside a full-frame ROI "
            f"(height=480). Got current={s2.counts['current']}."
        )

    def test_roi_smaller_than_frame_clips_bottom_correctly(self, mock_detector):
        """ROI that does not cover the bottom of the frame should correctly
        exclude a track whose bottom-center falls below roi.y2.

        ROI  : x=0, y=0, width=640, height=440  → y2=440.
        Track: bbox (200, 50, 400, 475)          → bottom-center = (300, 475).
        Expected: track IS NOT inside ROI → current_count == 0.
        """
        roi = ROIConfig(x=0, y=0, width=640, height=440)  # y2 = 440
        det = Detection(bbox=(200.0, 50.0, 400.0, 475.0), confidence=0.9, class_id=0)
        mock_detector.detect_timed.return_value = ([det], 10.0)

        config = CVPipelineConfig(
            enable_roi=True,
            roi=roi,
            analytics=AnalyticsConfig(capacity=10, debounce_frames=2),
            reliability=ReliabilityConfig(min_starting_frames=2),
            enable_face_detection=False,
        )
        pipeline = CVPipeline(config=config, detector=mock_detector)

        # Real frame is 640×480 — ROI only covers y=[0,440), bottom at y=475 is outside.
        f1 = make_frame_data(frame_id=1, timestamp=1.0)
        f2 = make_frame_data(frame_id=2, timestamp=1.033)
        pipeline.process_frame(f1, timestamp=1.01)
        s2 = pipeline.process_frame(f2, timestamp=1.04)

        assert s2.counts["current"] == 0, (
            "Track with bottom-center at y=475 must be OUTSIDE roi.y2=440. "
            f"Got current={s2.counts['current']}."
        )

    def test_frame_dims_confirmed_from_framedata(self, mock_detector):
        """When CameraConfig does not specify width/height, the pipeline must
        update ROIFilter with the actual frame dimensions from the first
        FrameData. The _frame_dims_confirmed flag must be True after that.
        """
        roi = ROIConfig(x=0, y=0, width=640, height=480)
        config = CVPipelineConfig(
            # No explicit width/height in CameraConfig
            roi=roi,
            reliability=ReliabilityConfig(min_starting_frames=1),
            enable_face_detection=False,
        )
        pipeline = CVPipeline(config=config, detector=mock_detector)

        # Before any frame: dims not yet confirmed from a real FrameData
        # (CameraConfig.width is None → _frame_dims_confirmed starts False)
        assert pipeline._frame_dims_confirmed is False

        f1 = make_frame_data(frame_id=1, timestamp=1.0)  # 640×480
        pipeline.process_frame(f1, timestamp=1.01)

        # After first real frame, dims must be confirmed
        assert pipeline._frame_dims_confirmed is True
        assert pipeline._roi_filter.frame_width == 640
        assert pipeline._roi_filter.frame_height == 480

    def test_camera_config_dims_pre_confirm(self, mock_detector):
        """When CameraConfig explicitly provides width/height, the pipeline
        must mark dims as already confirmed without waiting for a FrameData.
        """
        from visionqueue.camera.types import CameraConfig
        roi = ROIConfig(x=0, y=0, width=1280, height=720)
        config = CVPipelineConfig(
            camera=CameraConfig(source=0, width=1280, height=720),
            roi=roi,
            reliability=ReliabilityConfig(min_starting_frames=1),
            enable_face_detection=False,
        )
        pipeline = CVPipeline(config=config, detector=mock_detector)

        # CameraConfig provides explicit dimensions → confirmed immediately
        assert pipeline._frame_dims_confirmed is True
        assert pipeline._roi_filter.frame_width == 1280
        assert pipeline._roi_filter.frame_height == 720

