"""Pydantic response schemas for the VisionQueue REST API.

These models exist only at the API boundary for documentation and response
validation. They mirror the fields produced by LiveState.to_dict() and do not
duplicate any CV or business logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Nested component schemas
# ------------------------------------------------------------------

class PerformanceSchema(BaseModel):
    """Performance telemetry."""
    processing_fps: float
    inference_latency_ms: float
    frame_age_ms: float


class VisionSchema(BaseModel):
    """Vision subsystem health summary."""
    person_detection: str
    face_detection: str
    tracking: str


# ------------------------------------------------------------------
# Endpoint response schemas
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /health — API process health."""
    status: str = Field(examples=["ok"])


class StatusResponse(BaseModel):
    """GET /api/v1/status — pipeline and system status."""
    system_state: str
    camera_state: str
    is_healthy: bool
    is_frozen: bool
    status_reason: str
    performance: PerformanceSchema
    vision: VisionSchema
    active_model: Optional[str] = None
    execution_provider: Optional[str] = None
    is_gpu: Optional[bool] = None
    persistence_healthy: Optional[bool] = None
    pipeline_running: Optional[bool] = None
    last_frame_timestamp: Optional[float] = None


class ReadinessResponse(BaseModel):
    """GET /health/ready or /ready — pipeline readiness status."""
    status: str = Field(examples=["ready", "not_ready"])
    ready: bool
    detail: str
    pipeline_running: bool
    has_state: bool
    active_model: Optional[str] = None
    execution_provider: Optional[str] = None
    is_gpu: Optional[bool] = None
    persistence_healthy: Optional[bool] = None


class CountsResponse(BaseModel):
    """GET /api/v1/counts — current counting data."""
    current: int
    track_instances: int
    unique_session_approx: int
    entries: int
    exits: int
    net_count: int


class OccupancyResponse(BaseModel):
    """GET /api/v1/occupancy — occupancy data."""
    capacity: Optional[int] = None
    capacity_state: str
    percent: Optional[float] = None


class CrowdResponse(BaseModel):
    """GET /api/v1/crowd — crowd analytics."""
    level: str
    raw_level: str
    trend: str
    peak_count: int
    peak_occupancy_percent: Optional[float] = None
    peak_timestamp: Optional[float] = None


class AlertResponse(BaseModel):
    """Single alert record."""
    id: str
    type: str
    severity: str
    fired_at: float
    cleared_at: Optional[float] = None
    status: str
    reason: str


class TrackResponse(BaseModel):
    """Single active track."""
    track_id: int
    bbox: List[float]
    confidence: float


class LiveStateResponse(BaseModel):
    """GET /api/v1/live — complete LiveState snapshot."""
    schema_version: int
    timestamp: float
    session_id: str
    frame_id: int
    system_state: str
    camera_state: str
    performance: PerformanceSchema
    vision: VisionSchema
    counts: CountsResponse
    occupancy: OccupancyResponse
    crowd: CrowdResponse
    alerts: List[AlertResponse]
    status_reason: str
    is_healthy: bool
    is_frozen: bool
    detections_count: int
    tracks_count: int
    faces_count: int
    queue_people: int


class PipelineActionResponse(BaseModel):
    """POST /api/v1/pipeline/* — lifecycle action result."""
    status: str = Field(examples=["ok"])
    message: str


class ErrorResponse(BaseModel):
    """Standard error response body."""
    detail: str


# ------------------------------------------------------------------
# Historical API Response Models
# ------------------------------------------------------------------

class SessionResponse(BaseModel):
    id: str
    camera_id: int
    started_at: str
    ended_at: Optional[str] = None
    status: str


class MeasurementHistoryResponse(BaseModel):
    id: int
    session_id: str
    recorded_at: str
    current_count: int
    unique_count: int
    entries: int
    exits: int
    net_count: int
    occupancy_percent: Optional[float] = None
    crowd_level: Optional[str] = None
    crowd_trend: Optional[str] = None
    peak_count: Optional[int] = None
    peak_occupancy_percent: Optional[float] = None
    peak_timestamp: Optional[str] = None
    processing_latency_ms: Optional[float] = None
    system_state: Optional[str] = None
    camera_state: Optional[str] = None
    queue_people: Optional[int] = 0


class EventResponse(BaseModel):
    id: int
    session_id: str
    event_type: str
    occurred_at: str
    value: Optional[int] = None
    zone: Optional[str] = None
    metadata_json: Optional[str] = None


class AlertHistoryResponse(BaseModel):
    id: str
    session_id: str
    type: str
    severity: str
    fired_at: str
    cleared_at: Optional[str] = None
    status: str
    reason: Optional[str] = None
