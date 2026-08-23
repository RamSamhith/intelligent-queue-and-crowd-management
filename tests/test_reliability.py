"""Comprehensive unit tests for the VisionQueue reliability and system state subsystem."""

import pytest
import numpy as np

from visionqueue.camera.types import FrameData, SourceState
from visionqueue.reliability import (
    CameraHealthState,
    PerformanceMetrics,
    ReliabilityConfig,
    ReliabilityManager,
    SystemReliabilityState,
    SystemState,
    VisionHealthState,
    VisionHealthSummary,
)


def make_dummy_frame(
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


class TestReliabilityManager:
    """Unit tests for the ReliabilityManager state machine and health evaluator."""

    def test_initial_state_is_starting(self):
        manager = ReliabilityManager()
        assert manager.system_state == SystemState.STARTING
        assert manager.camera_state == CameraHealthState.CONNECTED
        assert manager.last_reliable_count == 0

    def test_startup_requires_min_healthy_frames_before_live(self):
        # min_starting_frames = 3
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=3))

        # Frame 1
        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        s1 = manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=4, timestamp=1.01)
        assert s1.system_state == SystemState.STARTING
        assert not s1.is_healthy
        assert "Verifying initial readings (1/3" in s1.status_reason

        # Frame 2
        f2 = make_dummy_frame(frame_id=2, timestamp=1.04)
        s2 = manager.update(source_state=SourceState.RUNNING, frame_data=f2, current_count=4, timestamp=1.05)
        assert s2.system_state == SystemState.STARTING
        assert "Verifying initial readings (2/3" in s2.status_reason

        # Frame 3 -> Enters LIVE!
        f3 = make_dummy_frame(frame_id=3, timestamp=1.07)
        s3 = manager.update(source_state=SourceState.RUNNING, frame_data=f3, current_count=4, timestamp=1.08)
        assert s3.system_state == SystemState.LIVE
        assert s3.is_healthy
        assert not s3.is_frozen
        assert s3.last_reliable_count == 4
        assert s3.last_reliable_timestamp == 1.08

    def test_camera_offline_preserves_last_reliable_count_never_zeroes(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1))

        # Reach LIVE with count = 9
        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=9, timestamp=1.0)
        assert manager.system_state == SystemState.LIVE
        assert manager.last_reliable_count == 9

        # Camera disconnects!
        s_off = manager.update(
            source_state=SourceState.DISCONNECTED,
            frame_data=None,
            current_count=0,  # Even if 0 is passed during failure
            timestamp=2.0,
        )

        assert s_off.system_state == SystemState.OFFLINE
        assert s_off.camera_state == CameraHealthState.RECONNECTING
        assert not s_off.is_healthy
        assert s_off.is_frozen
        assert s_off.last_reliable_count == 9  # Preserved!
        assert "Camera feed lost" in s_off.status_reason

    def test_camera_recovery_transitions_through_starting_never_straight_to_live(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=2))

        # 1. LIVE
        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=5, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=5, timestamp=1.03)
        assert manager.system_state == SystemState.LIVE

        # 2. OFFLINE
        manager.update(source_state=SourceState.DISCONNECTED, timestamp=2.0)
        assert manager.system_state == SystemState.OFFLINE

        # 3. Stream recovered -> Must enter STARTING (not straight to LIVE!)
        f_rec1 = make_dummy_frame(frame_id=10, timestamp=3.0)
        s_rec1 = manager.update(source_state=SourceState.RUNNING, frame_data=f_rec1, timestamp=3.02)
        assert s_rec1.system_state == SystemState.STARTING
        assert not s_rec1.is_healthy

        # 4. Second healthy frame -> reaches LIVE
        f_rec2 = make_dummy_frame(frame_id=11, timestamp=3.04)
        s_rec2 = manager.update(source_state=SourceState.RUNNING, frame_data=f_rec2, current_count=6, timestamp=3.06)
        assert s_rec2.system_state == SystemState.LIVE
        assert s_rec2.is_healthy
        assert s_rec2.last_reliable_count == 6

    def test_transient_detection_failure_triggers_degraded(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1, unstable_failure_threshold=3))

        # In LIVE state with count=7
        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=7, timestamp=1.0)
        assert manager.system_state == SystemState.LIVE

        # 1st detection failure -> DEGRADED
        s_fail1 = manager.update(
            source_state=SourceState.RUNNING,
            frame_data=f1,
            detection_success=False,
            timestamp=1.05,
        )
        assert s_fail1.system_state == SystemState.DEGRADED
        assert s_fail1.vision.person_detection == VisionHealthState.ERROR.value
        assert s_fail1.last_reliable_count == 7
        assert s_fail1.is_frozen

    def test_repeated_consecutive_detection_failures_trigger_unstable(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1, unstable_failure_threshold=3))

        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=7, timestamp=1.0)

        # Failures 1 and 2: DEGRADED
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, detection_success=False, timestamp=1.1)
        s2 = manager.update(source_state=SourceState.RUNNING, frame_data=f1, detection_success=False, timestamp=1.2)
        assert s2.system_state == SystemState.DEGRADED

        # Failure 3 (reaches threshold 3) -> UNSTABLE!
        s3 = manager.update(source_state=SourceState.RUNNING, frame_data=f1, detection_success=False, timestamp=1.3)
        assert s3.system_state == SystemState.UNSTABLE
        assert "Detection reliability collapsed (3 consecutive failures)" in s3.status_reason
        assert s3.last_reliable_count == 7
        assert s3.is_frozen

    def test_detection_recovery_from_unstable(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1, unstable_failure_threshold=2))

        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=7, timestamp=1.0)

        # 2 failures -> UNSTABLE
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, detection_success=False, timestamp=1.1)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, detection_success=False, timestamp=1.2)
        assert manager.system_state == SystemState.UNSTABLE

        # Recovery on next frame
        f2 = make_dummy_frame(frame_id=4, timestamp=1.3)
        s_rec = manager.update(source_state=SourceState.RUNNING, frame_data=f2, detection_success=True, current_count=8, timestamp=1.32)
        assert s_rec.system_state == SystemState.LIVE
        assert s_rec.is_healthy
        assert s_rec.last_reliable_count == 8

    def test_performance_degradation_high_frame_age(self):
        # max_frame_age_ms = 500ms
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1, max_frame_age_ms=500.0))

        # 1. Reach LIVE
        f_init = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f_init, current_count=3, timestamp=1.01)
        assert manager.system_state == SystemState.LIVE

        # 2. Acquisition at t=2.0, processing at t=2.8 (frame age 800ms > 500ms) -> DEGRADED
        f_stale = make_dummy_frame(frame_id=2, timestamp=2.0)
        s = manager.update(source_state=SourceState.RUNNING, frame_data=f_stale, current_count=3, timestamp=2.8)

        assert s.system_state == SystemState.DEGRADED
        assert "Frame age (800.0ms > 500.0ms)" in s.status_reason
        assert pytest.approx(s.performance.frame_age_ms, 0.1) == 800.0
        assert s.last_reliable_count == 3
        assert s.is_frozen

    def test_performance_degradation_high_inference_latency(self):
        # max_inference_latency_ms = 150ms
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1, max_inference_latency_ms=150.0))

        # 1. Reach LIVE
        f_init = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f_init, current_count=4, timestamp=1.01)
        assert manager.system_state == SystemState.LIVE

        # 2. Injected slow inference latency 220.5ms -> DEGRADED
        f2 = make_dummy_frame(frame_id=2, timestamp=2.0)
        s = manager.update(
            source_state=SourceState.RUNNING,
            frame_data=f2,
            inference_latency_ms=220.5,
            timestamp=2.02,
        )

        assert s.system_state == SystemState.DEGRADED
        assert "Inference latency (220.5ms > 150.0ms)" in s.status_reason
        assert s.is_frozen

    def test_performance_degradation_low_fps(self):
        # min_live_fps = 10.0
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1, min_live_fps=10.0))

        # 1. Reach LIVE with healthy FPS (30.0)
        f_init = make_dummy_frame(frame_id=1, timestamp=1.0, fps=30.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f_init, current_count=5, timestamp=1.01)
        assert manager.system_state == SystemState.LIVE

        # 2. FPS drops to 4.5 -> DEGRADED
        f_slow = make_dummy_frame(frame_id=2, timestamp=2.0, fps=4.5)
        s = manager.update(source_state=SourceState.RUNNING, frame_data=f_slow, timestamp=2.02)

        assert s.system_state == SystemState.DEGRADED
        assert "FPS (4.5 < 10.0)" in s.status_reason
        assert s.is_frozen

    def test_operator_stopping_state(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1))

        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=3, timestamp=1.0)
        assert manager.system_state == SystemState.LIVE

        s_stop = manager.update(operator_stop=True, timestamp=2.0)
        assert s_stop.system_state == SystemState.STOPPING
        assert s_stop.camera_state == CameraHealthState.OFFLINE
        assert "Operator initiated monitoring shutdown" in s_stop.status_reason

    def test_serialization_to_dict(self):
        metrics = PerformanceMetrics(processing_fps=29.8, inference_latency_ms=12.5, frame_age_ms=33.2)
        vision = VisionHealthSummary(person_detection="OK", face_detection="OK", tracking="OK")
        state = SystemReliabilityState(
            system_state=SystemState.LIVE,
            camera_state=CameraHealthState.CONNECTED,
            vision=vision,
            performance=metrics,
            is_healthy=True,
            is_frozen=False,
            last_reliable_count=5,
            last_reliable_timestamp=10.0,
            status_reason="System healthy and live",
            timestamp=10.033,
        )

        d = state.to_dict()
        assert d["system_state"] == "LIVE"
        assert d["camera_state"] == "CONNECTED"
        assert d["is_healthy"] is True
        assert d["is_frozen"] is False
        assert d["last_reliable_count"] == 5
        assert d["performance"]["processing_fps"] == 29.8

    def test_reset_and_new_session(self):
        manager = ReliabilityManager(ReliabilityConfig(min_starting_frames=1))
        f1 = make_dummy_frame(frame_id=1, timestamp=1.0)
        manager.update(source_state=SourceState.RUNNING, frame_data=f1, current_count=10, timestamp=1.0)
        assert manager.last_reliable_count == 10

        manager.new_session()
        assert manager.system_state == SystemState.STARTING
        assert manager.last_reliable_count == 0
        assert manager.last_state is None
