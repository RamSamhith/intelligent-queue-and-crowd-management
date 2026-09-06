import pytest
import sqlite3
import time
import os
import queue
from unittest.mock import MagicMock

from visionqueue.persistence.models import (
    CameraRecord, SessionCreateRecord, SessionCloseRecord,
    MeasurementRecord, EventRecord, AlertRecord, AlertClearRecord
)
from visionqueue.persistence.database import Database
from visionqueue.persistence.repository import PersistenceRepository
from visionqueue.persistence.writer import PersistenceWriter


@pytest.fixture
def test_db_path(tmp_path):
    return str(tmp_path / "test_visionqueue.db")


@pytest.fixture
def writer(test_db_path):
    db = Database(test_db_path)
    db.initialize()
    repo = PersistenceRepository(db)
    writer = PersistenceWriter(repo, max_queue_size=10)
    yield writer
    writer.stop_async_worker()


def test_database_initialization(test_db_path):
    db = Database(test_db_path)
    db.initialize()
    conn = db.get_connection()
    try:
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "cameras" in tables
        assert "sessions" in tables
        assert "measurements" in tables
        assert "events" in tables
        assert "alerts" in tables
    finally:
        conn.close()


def test_repository_camera_and_session(test_db_path):
    db = Database(test_db_path)
    db.initialize()
    repo = PersistenceRepository(db)

    cam = CameraRecord(
        id=None,
        name="Main Entrance",
        source="0",
        location="Front Gate",
        capacity=100,
        roi_x=10,
        roi_y=20,
        roi_width=300,
        roi_height=400,
        created_at="2026-09-06T12:00:00Z"
    )
    cam_id = repo.get_or_create_camera(cam)
    assert cam_id is not None

    # Idempotent camera creation
    cam_id_2 = repo.get_or_create_camera(cam)
    assert cam_id_2 == cam_id

    # Create session
    session_rec = SessionCreateRecord(
        id="sess_001",
        camera_id=cam_id,
        started_at="2026-09-06T12:00:00Z",
        status="RUNNING"
    )
    repo.execute(session_rec)

    sessions = repo.get_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == "sess_001"
    assert sessions[0]["status"] == "RUNNING"

    # Close session
    close_rec = SessionCloseRecord(
        session_id="sess_001",
        ended_at="2026-09-06T12:30:00Z",
        status="STOPPED"
    )
    repo.execute(close_rec)

    session = repo.get_session("sess_001")
    assert session is not None
    assert session["status"] == "STOPPED"
    assert session["ended_at"] == "2026-09-06T12:30:00Z"


def test_repository_measurements_events_alerts(test_db_path):
    db = Database(test_db_path)
    db.initialize()
    repo = PersistenceRepository(db)

    cam_id = repo.get_or_create_camera(
        CameraRecord(None, "Cam1", "0", None, 50, None, None, None, None, "2026-09-06T12:00:00Z")
    )
    repo.execute(SessionCreateRecord(id="s1", camera_id=cam_id, started_at="2026-09-06T12:00:00Z"))

    # Measurement
    m = MeasurementRecord(
        session_id="s1",
        recorded_at="2026-09-06T12:01:00Z",
        current_count=5,
        unique_count=12,
        entries=8,
        exits=3,
        net_count=5,
        occupancy_percent=10.0,
        crowd_level="LOW",
        crowd_trend="STABLE",
        peak_count=7,
        peak_occupancy_percent=14.0,
        peak_timestamp="2026-09-06T12:00:30Z",
        processing_latency_ms=12.5,
        system_state="LIVE",
        camera_state="ONLINE",
        queue_people=2
    )
    repo.execute(m)

    history = repo.get_history("s1")
    assert len(history) == 1
    assert history[0]["current_count"] == 5
    assert history[0]["queue_people"] == 2

    # Event
    evt = EventRecord(
        session_id="s1",
        event_type="ENTRY",
        occurred_at="2026-09-06T12:01:05Z",
        value=1,
        zone="entrance",
        metadata_json='{"track_id": 4}',
        dedup_key="s1:ENTRY:1"
    )
    repo.execute(evt)
    events = repo.get_events("s1")
    assert len(events) == 1
    assert events[0]["event_type"] == "ENTRY"

    # Alert active then cleared
    alrt = AlertRecord(
        id="alert_1",
        session_id="s1",
        type="HIGH_CROWD_DENSITY",
        severity="WARNING",
        fired_at="2026-09-06T12:02:00Z",
        status="ACTIVE",
        reason="Crowd level exceeded threshold"
    )
    repo.execute(alrt)
    alerts = repo.get_alerts("s1")
    assert len(alerts) == 1
    assert alerts[0]["status"] == "ACTIVE"

    clear = AlertClearRecord(alert_id="alert_1", cleared_at="2026-09-06T12:03:00Z")
    repo.execute(clear)
    alerts = repo.get_alerts("s1")
    assert len(alerts) == 1
    assert alerts[0]["status"] == "CLEARED"


def test_critical_queue_full_raises():
    db = Database(":memory:")
    repo = PersistenceRepository(db)
    writer = PersistenceWriter(repo, max_queue_size=1)

    writer.enqueue_critical(EventRecord("s", "E", "2026", 1, None, None, "1"))
    with pytest.raises(queue.Full):
        writer.enqueue_critical(EventRecord("s", "E", "2026", 1, None, None, "2"))


def test_writer_loop_processes_critical(writer):
    writer.start_async_worker()

    cid = writer._repo.get_or_create_camera(
        CameraRecord(None, "test", "0", None, None, None, None, None, None, "2026")
    )
    assert writer.write_sync(SessionCreateRecord(id="s1", camera_id=cid, started_at="2026"))

    rec = EventRecord("s1", "ENTRY", "2026", 1, None, None, "d1")
    writer.enqueue_critical(rec)

    ack = None
    for _ in range(50):
        ack = writer.get_ack_nowait()
        if ack:
            break
        time.sleep(0.01)

    assert ack is not None
    assert ack.success is True
    assert ack.record == rec


def test_retry_happens_in_writer_thread(writer):
    original_execute = writer._repo.execute
    call_count = 0

    def fake_execute(record):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_execute(record)

    writer._repo.execute = fake_execute
    writer.start_async_worker()

    cid = writer._repo.get_or_create_camera(
        CameraRecord(None, "test", "0", None, None, None, None, None, None, "2026")
    )
    assert writer.write_sync(SessionCreateRecord(id="s2", camera_id=cid, started_at="2026"))

    rec = EventRecord("s2", "ENTRY", "2026", 1, None, None, "d2")
    writer.enqueue_critical(rec)

    ack = None
    for _ in range(100):
        ack = writer.get_ack_nowait()
        if ack:
            break
        time.sleep(0.05)

    assert ack is not None
    assert ack.success is True
    assert call_count >= 2
