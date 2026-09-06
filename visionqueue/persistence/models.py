from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CameraRecord:
    id: Optional[int]
    name: str
    source: str
    location: Optional[str]
    capacity: Optional[int]
    roi_x: Optional[int]
    roi_y: Optional[int]
    roi_width: Optional[int]
    roi_height: Optional[int]
    created_at: str

@dataclass(frozen=True)
class SessionCreateRecord:
    id: str
    camera_id: int
    started_at: str
    status: str = "RUNNING"
    @property
    def record_key(self) -> str: return f"session_{self.id}"

@dataclass(frozen=True)
class SessionCloseRecord:
    session_id: str
    ended_at: str
    status: str
    @property
    def record_key(self) -> str: return f"session_close_{self.session_id}"

@dataclass(frozen=True)
class MeasurementRecord:
    session_id: str
    recorded_at: str
    current_count: int
    unique_count: int
    entries: int
    exits: int
    net_count: int
    occupancy_percent: Optional[float]
    crowd_level: Optional[str]
    crowd_trend: Optional[str]
    peak_count: Optional[int]
    peak_occupancy_percent: Optional[float]
    peak_timestamp: Optional[str]
    processing_latency_ms: Optional[float]
    system_state: Optional[str]
    camera_state: Optional[str]
    queue_people: Optional[int] = 0

@dataclass(frozen=True)
class EventRecord:
    session_id: str
    event_type: str
    occurred_at: str
    value: Optional[int]
    zone: Optional[str]
    metadata_json: Optional[str]
    dedup_key: str
    @property
    def record_key(self) -> str: return self.dedup_key

@dataclass(frozen=True)
class AlertRecord:
    id: str
    session_id: str
    type: str
    severity: str
    fired_at: str
    status: str
    reason: Optional[str]
    @property
    def record_key(self) -> str: return f"{self.id}:ACTIVE"

@dataclass(frozen=True)
class AlertClearRecord:
    alert_id: str
    cleared_at: str
    @property
    def record_key(self) -> str: return f"{self.alert_id}:CLEARED"

@dataclass(frozen=True)
class Ack:
    record: object
    success: bool
    error: Optional[str] = None
