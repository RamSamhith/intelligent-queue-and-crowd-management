"""Type definitions and contracts for the VisionQueue CV pipeline coordinator."""

from __future__ import annotations
from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, List, Optional

from visionqueue.alerts.types import Alert, AlertEngineConfig
from visionqueue.analytics.types import AnalyticsConfig, CrowdAnalyticsState, SceneProfile
from visionqueue.analytics.scene import SceneAnalyzerConfig
from visionqueue.camera.types import CameraConfig
from visionqueue.counting.types import LineCrossingCounts, SessionCounts, VirtualLine
from visionqueue.detection.types import DetectorConfig
from visionqueue.detection.face_detector import FaceDetectorConfig
from visionqueue.reliability.types import (
    CameraHealthState,
    PerformanceMetrics,
    ReliabilityConfig,
    SystemReliabilityState,
    SystemState,
    VisionHealthSummary,
)
from visionqueue.roi.types import ROIConfig
from visionqueue.tracking.bytetrack import ByteTrackConfig


@dataclass(frozen=True)
class LiveState:
    """Authoritative unified application-level LiveState produced by the CV pipeline.

    Matches the PRD & TDD live state specification for backend & dashboard consumption.
    """
    schema_version: int
    timestamp: float
    session_id: str
    frame_id: int
    system_state: SystemState
    camera_state: CameraHealthState
    performance: PerformanceMetrics
    vision: VisionHealthSummary
    counts: Dict[str, Any]
    occupancy: Dict[str, Any]
    crowd: Dict[str, Any]
    alerts: List[Alert]
    status_reason: str
    is_healthy: bool
    is_frozen: bool
    detections_count: int = 0
    tracks_count: int = 0
    faces_count: int = 0

    def to_dict(self) -> dict:
        """Serialize complete LiveState to JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "system_state": self.system_state.value,
            "camera_state": self.camera_state.value,
            "performance": self.performance.to_dict(),
            "vision": self.vision.to_dict(),
            "counts": dict(self.counts),
            "occupancy": dict(self.occupancy),
            "crowd": dict(self.crowd),
            "alerts": [a.to_dict() for a in self.alerts],
            "status_reason": self.status_reason,
            "is_healthy": self.is_healthy,
            "is_frozen": self.is_frozen,
            "detections_count": self.detections_count,
            "tracks_count": self.tracks_count,
            "faces_count": self.faces_count,
        }


@dataclass
class CVPipelineConfig:
    """Comprehensive configuration for the unified VisionQueue CV pipeline.

    Attributes:
        camera: Video capture source configuration.
        detector: YOLO26n person detector configuration.
        tracker: ByteTrack tracker configuration.
        enable_roi: When True, restricts person counting to ROI. Default: False (V1 Whole-Frame Counting).
        roi: Region of Interest configuration (used when enable_roi is True).
        virtual_line: Optional virtual line for entry/exit counting.
        analytics: Occupancy and crowd analytics configuration.
        scene: Optional scene and deployment profile for effective capacity derivation.
        alerts: P0 alert engine timing configuration.
        reliability: System health and degradation thresholds.
        enable_face_detection: When True, runs YuNet face presence detection.
        face_config: YuNet detector configuration.
        face_cadence: Frequency (every N frames) to run face detection.
        session_id: Active session UUID identifier.
    """
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    enable_roi: bool = False
    roi: Optional[ROIConfig] = field(default_factory=lambda: ROIConfig(x=0, y=0, width=640, height=480))
    virtual_line: Optional[VirtualLine] = None
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    scene: Optional[SceneProfile] = None
    scene_analyzer: SceneAnalyzerConfig = field(default_factory=SceneAnalyzerConfig)
    alerts: AlertEngineConfig = field(default_factory=AlertEngineConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    enable_face_detection: bool = False
    face_config: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    face_cadence: int = 5
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
