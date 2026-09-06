import sqlite3
import logging
from typing import Optional, List, Dict, Any
from .models import (CameraRecord, SessionCreateRecord, SessionCloseRecord, 
                     MeasurementRecord, EventRecord, AlertRecord, AlertClearRecord)
from .database import Database

logger = logging.getLogger(__name__)

class PersistenceRepository:
    def __init__(self, database: Database):
        self._db = database

    def execute(self, record, conn: sqlite3.Connection = None) -> None:
        manage_conn = False
        if conn is None:
            conn = self._db.get_connection()
            manage_conn = True
            
        try:
            if isinstance(record, SessionCreateRecord):
                conn.execute(
                    """INSERT INTO sessions (id, camera_id, started_at, status) VALUES (?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           started_at = excluded.started_at,
                           ended_at = NULL,
                           status = excluded.status""",
                    (record.id, record.camera_id, record.started_at, record.status)
                )
            elif isinstance(record, SessionCloseRecord):
                conn.execute(
                    "UPDATE sessions SET ended_at = ?, status = ? WHERE id = ? AND status = 'RUNNING'",
                    (record.ended_at, record.status, record.session_id)
                )
            elif isinstance(record, MeasurementRecord):
                conn.execute(
                    """INSERT INTO measurements (
                        session_id, recorded_at, current_count, unique_count, entries, exits, net_count,
                        occupancy_percent, crowd_level, crowd_trend, peak_count, peak_occupancy_percent,
                        peak_timestamp, processing_latency_ms, system_state, camera_state, queue_people
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record.session_id, record.recorded_at, record.current_count, record.unique_count,
                     record.entries, record.exits, record.net_count, record.occupancy_percent,
                     record.crowd_level, record.crowd_trend, record.peak_count, record.peak_occupancy_percent,
                     record.peak_timestamp, record.processing_latency_ms, record.system_state, record.camera_state,
                     record.queue_people)
                )
            elif isinstance(record, EventRecord):
                conn.execute(
                    """INSERT OR IGNORE INTO events (
                        session_id, event_type, occurred_at, value, zone, metadata_json, dedup_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (record.session_id, record.event_type, record.occurred_at, record.value,
                     record.zone, record.metadata_json, record.dedup_key)
                )
            elif isinstance(record, AlertRecord):
                conn.execute(
                    """INSERT OR IGNORE INTO alerts (
                        id, session_id, type, severity, fired_at, status, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (record.id, record.session_id, record.type, record.severity, record.fired_at,
                     record.status, record.reason)
                )
            elif isinstance(record, AlertClearRecord):
                conn.execute(
                    "UPDATE alerts SET status = 'CLEARED', cleared_at = ? WHERE id = ? AND status = 'ACTIVE'",
                    (record.cleared_at, record.alert_id)
                )
            else:
                raise ValueError(f"Unknown record type {type(record)}")
                
            if manage_conn:
                conn.commit()
                
        except Exception:
            if manage_conn:
                conn.rollback()
            raise
        finally:
            if manage_conn:
                conn.close()

    def get_or_create_camera(self, camera: CameraRecord) -> int:
        conn = self._db.get_connection()
        try:
            cur = conn.execute("SELECT id FROM cameras WHERE source = ? AND name = ?", (camera.source, camera.name))
            row = cur.fetchone()
            if row:
                return row["id"]
                
            cur = conn.execute(
                """INSERT INTO cameras (name, source, location, capacity, roi_x, roi_y, roi_width, roi_height, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (camera.name, camera.source, camera.location, camera.capacity, camera.roi_x, camera.roi_y,
                 camera.roi_width, camera.roi_height, camera.created_at)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
            
    def get_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._db.get_connection()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()]
        finally:
            conn.close()
            
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._db.get_connection()
        try:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
            
    def get_history(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._db.get_connection()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM measurements WHERE session_id = ? ORDER BY recorded_at DESC LIMIT ? OFFSET ?", (session_id, limit, offset)).fetchall()]
        finally:
            conn.close()
            
    def get_events(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._db.get_connection()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM events WHERE session_id = ? ORDER BY occurred_at DESC LIMIT ? OFFSET ?", (session_id, limit, offset)).fetchall()]
        finally:
            conn.close()
            
    def get_alerts(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._db.get_connection()
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM alerts WHERE session_id = ? ORDER BY fired_at DESC LIMIT ? OFFSET ?", (session_id, limit, offset)).fetchall()]
        finally:
            conn.close()
