"""Type definitions and contracts for the VisionQueue reliability and health subsystem."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SystemState(str, Enum):
    """Authoritative operational system states defined in PRD and TDD."""
    STARTING = "STARTING"    # Initializing / recovering; awaiting first confident readings
    LIVE = "LIVE"            # Normal operation; camera connected, detector confident, performance healthy
    DEGRADED = "DEGRADED"    # Running, but FPS or confidence has dropped below usable thresholds
    UNSTABLE = "UNSTABLE"    # Detection or tracking reliability has broken down (counts frozen)
    OFFLINE = "OFFLINE"      # Camera feed lost entirely (last reliable count preserved)
    STOPPING = "STOPPING"    # Operator-initiated shutdown


class CameraHealthState(str, Enum):
    """Normalized camera connectivity states."""
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"


class VisionHealthState(str, Enum):
    """Normalized vision subsystem health states."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PerformanceMetrics:
    """Real-time performance and latency telemetry."""
    processing_fps: float = 0.0
    inference_latency_ms: float = 0.0
    frame_age_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "processing_fps": round(self.processing_fps, 1),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "frame_age_ms": round(self.frame_age_ms, 2),
        }


@dataclass(frozen=True)
class VisionHealthSummary:
    """Summary of vision component health."""
    person_detection: str = VisionHealthState.OK.value
    face_detection: str = VisionHealthState.OK.value
    tracking: str = VisionHealthState.OK.value

    def to_dict(self) -> dict:
        return {
            "person_detection": self.person_detection,
            "face_detection": self.face_detection,
            "tracking": self.tracking,
        }


@dataclass(frozen=True)
class ReliabilityConfig:
    """Configuration thresholds for reliability state transitions.

    Attributes:
        min_live_fps: Minimum FPS to maintain LIVE state without degrading.
        max_frame_age_ms: Maximum frame age before flagging stale/degraded.
        max_inference_latency_ms: Maximum inference latency before flagging degraded.
        unstable_failure_threshold: Number of consecutive failures before marking UNSTABLE.
        min_starting_frames: Consecutive healthy frames required in STARTING before entering LIVE.
    """
    min_live_fps: float = 5.0
    max_frame_age_ms: float = 1000.0
    max_inference_latency_ms: float = 200.0
    unstable_failure_threshold: int = 5
    min_starting_frames: int = 3

    def __post_init__(self):
        if self.min_live_fps <= 0:
            raise ValueError(f"min_live_fps must be > 0, got {self.min_live_fps}")
        if self.max_frame_age_ms <= 0:
            raise ValueError(f"max_frame_age_ms must be > 0, got {self.max_frame_age_ms}")
        if self.unstable_failure_threshold < 1:
            raise ValueError(f"unstable_failure_threshold must be >= 1, got {self.unstable_failure_threshold}")
        if self.min_starting_frames < 1:
            raise ValueError(f"min_starting_frames must be >= 1, got {self.min_starting_frames}")


@dataclass(frozen=True)
class SystemReliabilityState:
    """Authoritative aggregated health state of the VisionQueue system.

    Attributes:
        system_state: High-level state (STARTING/LIVE/DEGRADED/UNSTABLE/OFFLINE/STOPPING).
        camera_state: CONNECTED, RECONNECTING, or OFFLINE.
        vision: Summary of person/face/tracking health.
        performance: Telemetry including FPS, inference latency, and frame age.
        is_healthy: True if and only if system_state is LIVE.
        is_frozen: True if counts are frozen due to degradation or failure.
        last_reliable_count: Last trusted count (never fabricated zero during failure).
        last_reliable_timestamp: Timestamp when the last reliable count was captured.
        status_reason: Human-readable explanation of the current system status.
        timestamp: Snapshot timestamp.
    """
    system_state: SystemState
    camera_state: CameraHealthState
    vision: VisionHealthSummary
    performance: PerformanceMetrics
    is_healthy: bool
    is_frozen: bool
    last_reliable_count: int
    last_reliable_timestamp: Optional[float]
    status_reason: str
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize state snapshot to application dictionary."""
        return {
            "system_state": self.system_state.value,
            "camera_state": self.camera_state.value,
            "vision": self.vision.to_dict(),
            "performance": self.performance.to_dict(),
            "is_healthy": self.is_healthy,
            "is_frozen": self.is_frozen,
            "last_reliable_count": self.last_reliable_count,
            "last_reliable_timestamp": self.last_reliable_timestamp,
            "status_reason": self.status_reason,
            "timestamp": self.timestamp,
        }
