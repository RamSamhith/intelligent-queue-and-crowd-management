"""CVService — thread-safe bridge between the FastAPI REST API and CVPipeline.

Follows the integration pattern from docs/CV_HANDOFF.md:
- Runs CVPipeline.step() in a dedicated daemon thread
- Caches latest_state_dict (from LiveState.to_dict()) for non-blocking API reads
- Caches latest_tracks for safe track exposure without leaking internal diagnostics keys
- Uses explicit synchronization for all lifecycle and state operations

Concurrency design:
- _state_lock: protects latest_state_dict and latest_tracks reads/writes
- _lifecycle_lock: serializes start/stop/reset against the worker loop
- _stop_event: clean daemon thread shutdown without polling a bare bool
"""

from __future__ import annotations

import logging
import threading
import time
import json
import datetime
import queue
from typing import Dict, List, Optional, Set, Any

from visionqueue.pipeline import CVPipeline, CVPipelineConfig
from visionqueue.persistence.config import PersistenceConfig
from visionqueue.persistence.database import Database
from visionqueue.persistence.repository import PersistenceRepository
from visionqueue.persistence.writer import PersistenceWriter
from visionqueue.persistence.models import (
    CameraRecord, SessionCreateRecord, SessionCloseRecord, MeasurementRecord,
    EventRecord, AlertRecord, AlertClearRecord, Ack
)

logger = logging.getLogger(__name__)


class CVService:
    def __init__(
        self,
        pipeline: Optional[CVPipeline] = None,
        config: Optional[CVPipelineConfig] = None,
        persistence_config: Optional[PersistenceConfig] = None,
    ) -> None:
        self._pipeline: CVPipeline = pipeline or CVPipeline(config=config or CVPipelineConfig())
        self._persistence_config = persistence_config or PersistenceConfig(enabled=False)
        self._thread: Optional[threading.Thread] = None

        self._latest_state_dict: Dict = {}
        self._latest_tracks: List[Dict] = []

        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False
        
        # Persistence state
        self._writer: Optional[PersistenceWriter] = None
        self._repo: Optional[PersistenceRepository] = None
        self._active_session_id: Optional[str] = None
        self._camera_id: Optional[int] = None
        self._last_measurement_time: float = 0.0
        self._persistence_healthy: bool = True
        
        self._pending_critical: Dict[str, object] = {}
        self._in_flight_critical: Set[str] = set()
        
        self._known_active_alerts: Dict[str, str] = {}
        self._prev_persist_state: Dict[str, Any] = self._initial_prev_state()

    def _initial_prev_state(self) -> Dict[str, Any]:
        return {
            "entries": 0, "exits": 0, "crowd_level": None,
            "system_state": None, "camera_state": None,
        }

    def _now_utc(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pipeline(self) -> CVPipeline:
        return self._pipeline

    def get_latest_state(self) -> Dict:
        with self._state_lock:
            return dict(self._latest_state_dict)

    def get_latest_tracks(self) -> List[Dict]:
        with self._state_lock:
            return list(self._latest_tracks)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return genuine health and runtime diagnostics."""
        det = getattr(self._pipeline, "_detector", None)
        active_model = None
        execution_provider = None
        is_gpu = False
        if det is not None:
            if hasattr(det, "config") and det.config and getattr(det.config, "model_path", None):
                active_model = det.config.model_path.replace("\\", "/").split("/")[-1]
            execution_provider = getattr(det, "active_provider", None)
            is_gpu = getattr(det, "is_gpu", False)
        return {
            "pipeline_running": self._running,
            "active_model": active_model,
            "execution_provider": execution_provider,
            "is_gpu": is_gpu,
            "persistence_healthy": self._persistence_healthy,
            "persistence_enabled": self._persistence_config.enabled,
        }

    def _init_persistence(self) -> None:
        if not self._writer:
            db = Database(self._persistence_config.database_path)
            db.initialize()
            self._repo = PersistenceRepository(db)
            self._writer = PersistenceWriter(self._repo, self._persistence_config.max_queue_size)

    def _ensure_camera_record(self) -> int:
        cfg = self._pipeline.config.camera
        acfg = self._pipeline.config.analytics
        rcfg = self._pipeline.config.roi
        name = f"Camera {cfg.source}"
        
        rec = CameraRecord(
            id=None,
            name=name,
            source=str(cfg.source),
            location=None,
            capacity=acfg.capacity if acfg else None,
            roi_x=rcfg.x if self._pipeline.config.enable_roi and rcfg else None,
            roi_y=rcfg.y if self._pipeline.config.enable_roi and rcfg else None,
            roi_width=rcfg.width if self._pipeline.config.enable_roi and rcfg else None,
            roi_height=rcfg.height if self._pipeline.config.enable_roi and rcfg else None,
            created_at=self._now_utc()
        )
        return self._repo.get_or_create_camera(rec)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                logger.info("CVService.start() called but already running — no-op.")
                return

            self._stop_event.clear()
            
            if self._persistence_config.enabled:
                try:
                    self._init_persistence()
                    self._camera_id = self._ensure_camera_record()
                    session_id = self._pipeline.config.session_id
                    success = self._writer.write_sync(SessionCreateRecord(
                        id=session_id, camera_id=self._camera_id, started_at=self._now_utc()
                    ))
                    if not success:
                        raise RuntimeError("Failed to durably create persistence session")
                        
                    self._active_session_id = session_id
                    self._writer.start_async_worker()
                except Exception as e:
                    logger.critical("Persistence init failed. Aborting CV start.")
                    self._persistence_healthy = False
                    raise e
                    
            try:
                self._pipeline.start()
            except Exception as e:
                logger.exception("Pipeline failed to start. Closing session as FAILED.")
                if self._persistence_config.enabled and self._active_session_id:
                    self._writer.write_sync(SessionCloseRecord(
                        session_id=self._active_session_id, status="FAILED", ended_at=self._now_utc()
                    ))
                    self._writer.stop_async_worker()
                raise e

            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, name="CVService-Worker", daemon=True)
            self._thread.start()
            logger.info("CVService started.")

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        with self._lifecycle_lock:
            self._pipeline.stop()
            if self._pipeline.last_state is not None:
                with self._state_lock:
                    self._latest_state_dict = self._pipeline.last_state.to_dict()
                    self._latest_tracks = []
            logger.info("CVService stopped.")

            if self._persistence_config.enabled and self._active_session_id:
                self._writer.flush_all(timeout=5.0)
                success = self._writer.write_sync(SessionCloseRecord(
                    session_id=self._active_session_id, status="STOPPED", ended_at=self._now_utc()
                ))
                if not success:
                    logger.critical("SessionClose write failed! Session %s may remain RUNNING.", self._active_session_id)
                    self._persistence_healthy = False
                self._writer.stop_async_worker()
                self._active_session_id = None

    def reset_session(self) -> None:
        with self._lifecycle_lock:
            if self._persistence_config.enabled and self._active_session_id and self._writer:
                self._writer.flush_all(timeout=5.0)
                success = self._writer.write_sync(SessionCloseRecord(
                    session_id=self._active_session_id, status="STOPPED", ended_at=self._now_utc()
                ))
                if not success:
                    self._persistence_healthy = False
                self._active_session_id = None

            self._pipeline.reset_session()

            if self._running and self._persistence_config.enabled and self._writer:
                new_id = self._pipeline.config.session_id
                success = self._writer.write_sync(SessionCreateRecord(
                    id=new_id, camera_id=self._camera_id, started_at=self._now_utc()
                ))
                if success:
                    self._active_session_id = new_id
                else:
                    self._persistence_healthy = False
                self._prev_persist_state = self._initial_prev_state()
                self._known_active_alerts.clear()
                self._pending_critical.clear()
                self._in_flight_critical.clear()
                
            with self._state_lock:
                self._latest_state_dict = {}
                self._latest_tracks = []
            logger.info("CVService session reset.")

    def _worker_loop(self) -> None:
        logger.info("CVService worker loop started.")
        while not self._stop_event.is_set():
            try:
                if self._persistence_config.enabled:
                    self._process_persistence_acks()

                with self._lifecycle_lock:
                    if self._stop_event.is_set():
                        break
                    state = self._pipeline.step(timeout=1.0)

                state_dict = state.to_dict()
                tracks = self._extract_tracks()

                with self._state_lock:
                    self._latest_state_dict = state_dict
                    self._latest_tracks = tracks

                if self._persistence_config.enabled and self._active_session_id:
                    self._persist_state(state_dict)

            except Exception:
                logger.exception("CVService worker loop error")
                if self._persistence_config.enabled and self._active_session_id:
                    self._writer.write_sync(SessionCloseRecord(
                        session_id=self._active_session_id, status="FAILED", ended_at=self._now_utc()
                    ))
                self._stop_event.wait(0.1)
                raise
        logger.info("CVService worker loop exited.")

    def _extract_tracks(self) -> List[Dict]:
        diag = self._pipeline.last_diagnostics
        raw_tracks = diag.get("active_tracks", [])
        return [dict(t) for t in raw_tracks]

    def _enqueue_critical(self, record: Any) -> None:
        key = record.record_key
        self._pending_critical[key] = record
        try:
            self._writer.enqueue_critical(record)
            self._in_flight_critical.add(key)
        except queue.Full:
            self._persistence_healthy = False
            logger.error("Critical queue full. Record buffered for later: %s", key)

    def _process_persistence_acks(self) -> None:
        # Flush pending that aren't in flight
        for key, record in list(self._pending_critical.items()):
            if key not in self._in_flight_critical:
                try:
                    self._writer.enqueue_critical(record)
                    self._in_flight_critical.add(key)
                except queue.Full:
                    pass

        # Drain ACKs
        while True:
            ack = self._writer.get_ack_nowait()
            if not ack:
                break
            
            record_key = ack.record.record_key
            if ack.success:
                self._pending_critical.pop(record_key, None)
                self._in_flight_critical.discard(record_key)
                self._advance_state(ack.record)
            else:
                self._in_flight_critical.discard(record_key)
                self._persistence_healthy = False
                logger.error("Critical record DB write failed after retries. Buffered for recovery: %s", record_key)

    def _advance_state(self, record: Any) -> None:
        if isinstance(record, EventRecord):
            if record.event_type == "ENTRY":
                self._prev_persist_state["entries"] += (record.value or 0)
            elif record.event_type == "EXIT":
                self._prev_persist_state["exits"] += (record.value or 0)
            elif record.event_type == "CROWD_LEVEL_CHANGED":
                if record.metadata_json:
                    self._prev_persist_state["crowd_level"] = json.loads(record.metadata_json)["to"]
            elif record.event_type == "SYSTEM_STATE_CHANGED":
                if record.metadata_json:
                    self._prev_persist_state["system_state"] = json.loads(record.metadata_json)["to"]
            elif record.event_type == "CAMERA_STATE_CHANGED":
                if record.metadata_json:
                    self._prev_persist_state["camera_state"] = json.loads(record.metadata_json)["to"]
        elif isinstance(record, AlertRecord):
            self._known_active_alerts[record.id] = record.type
        elif isinstance(record, AlertClearRecord):
            self._known_active_alerts.pop(record.alert_id, None)

    def _persist_state(self, state_dict: dict) -> None:
        now_utc = self._now_utc()
        session_id = self._active_session_id
        if not session_id:
            return
        counts = state_dict["counts"]
        prev = self._prev_persist_state

        # Measurements
        if time.monotonic() - self._last_measurement_time >= self._persistence_config.measurement_interval_seconds:
            self._last_measurement_time = time.monotonic()
            peak_ts = state_dict["crowd"].get("peak_timestamp")
            m = MeasurementRecord(
                session_id=session_id,
                recorded_at=now_utc,
                current_count=counts["current"],
                unique_count=counts["unique_session_approx"],
                entries=counts["entries"],
                exits=counts["exits"],
                net_count=counts["net_count"],
                occupancy_percent=state_dict["occupancy"].get("percent"),
                crowd_level=state_dict["crowd"]["level"],
                crowd_trend=state_dict["crowd"]["trend"],
                peak_count=state_dict["crowd"]["peak_count"],
                peak_occupancy_percent=state_dict["crowd"].get("peak_occupancy_percent"),
                peak_timestamp=datetime.datetime.fromtimestamp(peak_ts, datetime.timezone.utc).isoformat() if peak_ts else None,
                processing_latency_ms=state_dict["performance"]["inference_latency_ms"],
                system_state=state_dict["system_state"],
                camera_state=state_dict["camera_state"],
                queue_people=state_dict.get("queue_people", 0)
            )
            self._writer.enqueue_measurement(m)

        # Events
        curr_entries = counts["entries"]
        if curr_entries > prev["entries"]:
            delta = curr_entries - prev["entries"]
            dedup_key = f"{session_id}:ENTRY:{curr_entries}"
            if dedup_key not in self._pending_critical:
                self._enqueue_critical(EventRecord(session_id, "ENTRY", now_utc, delta, None, None, dedup_key))

        curr_exits = counts["exits"]
        if curr_exits > prev["exits"]:
            delta = curr_exits - prev["exits"]
            dedup_key = f"{session_id}:EXIT:{curr_exits}"
            if dedup_key not in self._pending_critical:
                self._enqueue_critical(EventRecord(session_id, "EXIT", now_utc, delta, None, None, dedup_key))

        curr_crowd = state_dict["crowd"]["level"]
        if prev["crowd_level"] is not None and curr_crowd != prev["crowd_level"]:
            dedup_key = f"{session_id}:CROWD:{prev['crowd_level']}:{curr_crowd}"
            if dedup_key not in self._pending_critical:
                self._enqueue_critical(EventRecord(session_id, "CROWD_LEVEL_CHANGED", now_utc, None, None, json.dumps({"from": prev["crowd_level"], "to": curr_crowd}), dedup_key))
        elif prev["crowd_level"] is None:
            prev["crowd_level"] = curr_crowd

        curr_sys = state_dict["system_state"]
        if prev["system_state"] is not None and curr_sys != prev["system_state"]:
            dedup_key = f"{session_id}:SYS:{prev['system_state']}:{curr_sys}"
            if dedup_key not in self._pending_critical:
                self._enqueue_critical(EventRecord(session_id, "SYSTEM_STATE_CHANGED", now_utc, None, None, json.dumps({"from": prev["system_state"], "to": curr_sys}), dedup_key))
        elif prev["system_state"] is None:
            prev["system_state"] = curr_sys

        curr_cam = state_dict["camera_state"]
        if prev["camera_state"] is not None and curr_cam != prev["camera_state"]:
            dedup_key = f"{session_id}:CAM:{prev['camera_state']}:{curr_cam}"
            if dedup_key not in self._pending_critical:
                self._enqueue_critical(EventRecord(session_id, "CAMERA_STATE_CHANGED", now_utc, None, None, json.dumps({"from": prev["camera_state"], "to": curr_cam}), dedup_key))
        elif prev["camera_state"] is None:
            prev["camera_state"] = curr_cam

        # Alerts
        current_alert_ids = set()
        for alert_dict in state_dict["alerts"]:
            alert_id = alert_dict["id"]
            current_alert_ids.add(alert_id)
            if alert_id not in self._known_active_alerts:
                record_key = f"{alert_id}:ACTIVE"
                if record_key not in self._pending_critical:
                    self._enqueue_critical(AlertRecord(
                        id=alert_id, session_id=session_id, type=alert_dict["type"],
                        severity=alert_dict["severity"],
                        fired_at=datetime.datetime.fromtimestamp(alert_dict["fired_at"], datetime.timezone.utc).isoformat(),
                        status="ACTIVE", reason=alert_dict.get("reason")
                    ))

        cleared_ids = set(self._known_active_alerts.keys()) - current_alert_ids
        for alert_id in cleared_ids:
            record_key = f"{alert_id}:CLEARED"
            if record_key not in self._pending_critical:
                self._enqueue_critical(AlertClearRecord(alert_id, now_utc))
