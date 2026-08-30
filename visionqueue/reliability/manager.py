"""Reliability and System State Manager for VisionQueue.

Aggregates camera connectivity, detection status, tracking health, and real-time
telemetry into the authoritative 6-state reliability machine:
STARTING -> LIVE <-> DEGRADED / UNSTABLE <-> OFFLINE -> STARTING (+ STOPPING).
"""

from __future__ import annotations
from typing import Optional, Union

from visionqueue.camera.types import FrameData, SourceState
from visionqueue.reliability.types import (
    CameraHealthState,
    PerformanceMetrics,
    ReliabilityConfig,
    SystemReliabilityState,
    SystemState,
    VisionHealthState,
    VisionHealthSummary,
)


class ReliabilityManager:
    """Manages system health state transitions and preserves reliable counts.

    Guarantees:
    - Enforces the 6 PRD/TDD states: STARTING, LIVE, DEGRADED, UNSTABLE, OFFLINE, STOPPING.
    - Preserves last reliable count during failures (never fabricates zero).
    - Transitions OFFLINE -> STARTING on stream recovery (never straight to LIVE).
    - Provides explicit explanation for why the system is not in a healthy state.
    """

    def __init__(self, config: Optional[ReliabilityConfig] = None) -> None:
        """Initialize the reliability manager.

        Args:
            config: Optional ReliabilityConfig with custom degradation thresholds.
        """
        self._config: ReliabilityConfig = config or ReliabilityConfig()

        self._system_state: SystemState = SystemState.STARTING
        self._camera_state: CameraHealthState = CameraHealthState.CONNECTED
        self._consecutive_healthy_frames: int = 0
        self._consecutive_detection_errors: int = 0

        self._last_reliable_count: int = 0
        self._last_reliable_timestamp: Optional[float] = None
        self._last_state: Optional[SystemReliabilityState] = None

    @property
    def config(self) -> ReliabilityConfig:
        return self._config

    @property
    def system_state(self) -> SystemState:
        return self._system_state

    @property
    def camera_state(self) -> CameraHealthState:
        return self._camera_state

    @property
    def last_reliable_count(self) -> int:
        return self._last_reliable_count

    def update(
        self,
        source_state: Union[SourceState, str] = SourceState.RUNNING,
        frame_data: Optional[FrameData] = None,
        detection_success: bool = True,
        tracking_success: bool = True,
        current_count: Optional[int] = None,
        inference_latency_ms: float = 0.0,
        face_detection_success: bool = True,
        operator_stop: bool = False,
        timestamp: float = 0.0,
    ) -> SystemReliabilityState:
        """Process a frame/cycle status and calculate the authoritative reliability state.

        Args:
            source_state: CameraSource state (e.g. RUNNING, DISCONNECTED, ERROR).
            frame_data: FrameData instance if a frame was acquired, or None.
            detection_success: True if person detection executed without error.
            tracking_success: True if tracking update executed without error.
            current_count: Current in-ROI count from counting module (if healthy).
            inference_latency_ms: Latency of person detection inference in milliseconds.
            face_detection_success: True if face detection executed without error.
            operator_stop: True if operator initiated a clean shutdown.
            timestamp: Snapshot evaluation timestamp.

        Returns:
            SystemReliabilityState containing state, metrics, and explanatory reason.
        """
        # Normalize source state string
        s_state_str = source_state.value if isinstance(source_state, SourceState) else str(source_state).upper()

        # 1. Evaluate Operator Stop
        if operator_stop:
            self._system_state = SystemState.STOPPING
            self._camera_state = CameraHealthState.OFFLINE
            return self._build_state(
                reason="Operator initiated monitoring shutdown",
                fps=0.0,
                latency_ms=0.0,
                frame_age_ms=0.0,
                p_det_ok=detection_success,
                face_ok=face_detection_success,
                track_ok=tracking_success,
                timestamp=timestamp,
            )

        # 2. Evaluate Camera Connectivity
        camera_offline = s_state_str in (SourceState.DISCONNECTED.value, SourceState.ERROR.value, SourceState.STOPPED.value)
        camera_reconnecting = s_state_str == SourceState.DISCONNECTED.value

        if camera_offline:
            prev_state = self._system_state
            self._camera_state = CameraHealthState.RECONNECTING if camera_reconnecting else CameraHealthState.OFFLINE
            self._system_state = SystemState.OFFLINE
            self._consecutive_healthy_frames = 0
            reason = "Camera feed lost / disconnected" if camera_reconnecting else f"Camera unavailable ({s_state_str})"

            return self._build_state(
                reason=reason,
                fps=0.0,
                latency_ms=0.0,
                frame_age_ms=0.0,
                p_det_ok=detection_success,
                face_ok=face_detection_success,
                track_ok=tracking_success,
                timestamp=timestamp,
            )

        # Camera is connected
        self._camera_state = CameraHealthState.CONNECTED

        # If recovering from OFFLINE, strictly transition to STARTING (never straight to LIVE)
        if self._system_state == SystemState.OFFLINE:
            self._system_state = SystemState.STARTING
            self._consecutive_healthy_frames = 0

        # 3. Calculate Performance Metrics
        fps = frame_data.fps if frame_data is not None else 0.0
        frame_age_ms = 0.0
        if frame_data is not None and frame_data.timestamp > 0:
            # Both frame_data.timestamp and timestamp MUST be wall-clock seconds (time.time())
            # so subtraction is meaningful. We clamp to >= 0 because out-of-order acquisition
            # can occasionally produce a slightly negative age.
            frame_age_ms = max(0.0, (timestamp - frame_data.timestamp) * 1000.0)

        # 4. Evaluate Vision & Detection Health
        if not detection_success:
            self._consecutive_detection_errors += 1
            self._consecutive_healthy_frames = 0

            if self._consecutive_detection_errors >= self._config.unstable_failure_threshold:
                self._system_state = SystemState.UNSTABLE
                reason = f"Detection reliability collapsed ({self._consecutive_detection_errors} consecutive failures)"
            else:
                self._system_state = SystemState.DEGRADED
                reason = f"Detection execution error ({self._consecutive_detection_errors} failures)"

            return self._build_state(
                reason=reason,
                fps=fps,
                latency_ms=inference_latency_ms,
                frame_age_ms=frame_age_ms,
                p_det_ok=False,
                face_ok=face_detection_success,
                track_ok=tracking_success,
                timestamp=timestamp,
            )

        # Detection succeeded
        self._consecutive_detection_errors = 0

        # Check for Tracking Failures
        if not tracking_success:
            self._system_state = SystemState.UNSTABLE
            return self._build_state(
                reason="Tracking internal update failure",
                fps=fps,
                latency_ms=inference_latency_ms,
                frame_age_ms=frame_age_ms,
                p_det_ok=True,
                face_ok=face_detection_success,
                track_ok=False,
                timestamp=timestamp,
            )

        # 5. Evaluate Performance Degradation (Stale frames, low FPS, high latency)
        degraded_reasons = []
        if frame_age_ms > self._config.max_frame_age_ms:
            degraded_reasons.append(f"Frame age ({frame_age_ms:.1f}ms > {self._config.max_frame_age_ms:.1f}ms)")
        if inference_latency_ms > self._config.max_inference_latency_ms:
            degraded_reasons.append(f"Inference latency ({inference_latency_ms:.1f}ms > {self._config.max_inference_latency_ms:.1f}ms)")
        if 0.0 < fps < self._config.min_live_fps:
            degraded_reasons.append(f"FPS ({fps:.1f} < {self._config.min_live_fps:.1f})")

        is_degraded = len(degraded_reasons) > 0

        # 6. Handle STARTING vs LIVE Transitions
        if self._system_state == SystemState.STARTING:
            if not is_degraded:
                self._consecutive_healthy_frames += 1
                if self._consecutive_healthy_frames >= self._config.min_starting_frames:
                    self._system_state = SystemState.LIVE
                    reason = "System healthy and live"
                else:
                    reason = f"Verifying initial readings ({self._consecutive_healthy_frames}/{self._config.min_starting_frames} frames)"
            else:
                self._consecutive_healthy_frames = 0
                reason = "STARTING with degraded performance: " + "; ".join(degraded_reasons)

        elif is_degraded:
            self._system_state = SystemState.DEGRADED
            reason = "Degraded performance: " + "; ".join(degraded_reasons)

        else:
            self._system_state = SystemState.LIVE
            reason = "System healthy and live"

        # 7. Update Last Reliable Count
        if self._system_state == SystemState.LIVE and current_count is not None:
            self._last_reliable_count = current_count
            self._last_reliable_timestamp = timestamp

        return self._build_state(
            reason=reason,
            fps=fps,
            latency_ms=inference_latency_ms,
            frame_age_ms=frame_age_ms,
            p_det_ok=True,
            face_ok=face_detection_success,
            track_ok=True,
            timestamp=timestamp,
        )

    def _build_state(
        self,
        reason: str,
        fps: float,
        latency_ms: float,
        frame_age_ms: float,
        p_det_ok: bool,
        face_ok: bool,
        track_ok: bool,
        timestamp: float,
    ) -> SystemReliabilityState:
        """Helper to construct the immutable SystemReliabilityState."""
        is_healthy = self._system_state == SystemState.LIVE
        is_frozen = self._system_state in (SystemState.DEGRADED, SystemState.UNSTABLE, SystemState.OFFLINE)

        vision_summary = VisionHealthSummary(
            person_detection=VisionHealthState.OK.value if p_det_ok else VisionHealthState.ERROR.value,
            face_detection=VisionHealthState.OK.value if face_ok else VisionHealthState.ERROR.value,
            tracking=VisionHealthState.OK.value if track_ok else VisionHealthState.ERROR.value,
        )

        metrics = PerformanceMetrics(
            processing_fps=fps,
            inference_latency_ms=latency_ms,
            frame_age_ms=frame_age_ms,
        )

        self._last_state = SystemReliabilityState(
            system_state=self._system_state,
            camera_state=self._camera_state,
            vision=vision_summary,
            performance=metrics,
            is_healthy=is_healthy,
            is_frozen=is_frozen,
            last_reliable_count=self._last_reliable_count,
            last_reliable_timestamp=self._last_reliable_timestamp,
            status_reason=reason,
            timestamp=timestamp,
        )
        return self._last_state

    @property
    def last_state(self) -> Optional[SystemReliabilityState]:
        return self._last_state

    def reset(self) -> None:
        """Reset state machine to initial STARTING state."""
        self._system_state = SystemState.STARTING
        self._camera_state = CameraHealthState.CONNECTED
        self._consecutive_healthy_frames = 0
        self._consecutive_detection_errors = 0
        self._last_reliable_count = 0
        self._last_reliable_timestamp = None
        self._last_state = None

    def new_session(self) -> None:
        """Alias for reset() to demarcate session boundaries."""
        self.reset()
