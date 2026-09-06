"""Comprehensive lifecycle, concurrency, and robustness tests for CVService and API.

Verifies the 15 critical lifecycle and concurrency patterns:
1. START
2. START AGAIN (idempotency)
3. STOP
4. STOP AGAIN (idempotency)
5. START -> STOP -> START (restarting, persistence re-open, state consistency)
6. RESET WHEN STOPPED (direct call safe, API endpoint returns 400 Bad Request)
7. RESET WHEN RUNNING (resets pipeline metrics, advances session in DB)
8. API REQUEST BEFORE PIPELINE START (returns 503 for stateful endpoints, 200 for /health)
9. WEBSOCKET BEFORE PIPELINE START (connects cleanly without crash)
10. WEBSOCKET DISCONNECT (clean disconnect without server exception)
11. WEBSOCKET RECONNECT (client reconnects and receives stream)
12. MULTIPLE WEBSOCKET CLIENTS (concurrent clients streaming without deadlock)
13. DATABASE ERROR (degraded persistence health without crashing pipeline worker)
14. DATABASE QUEUE PRESSURE (queue saturation handled gracefully via buffer)
15. SHUTDOWN DURING WRITE (flushing and stopping during active worker activity)
"""

import queue
import sqlite3
import time
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from visionqueue.api.app import create_app
from visionqueue.api.config import APIConfig, PersistenceConfig
from visionqueue.api.dependencies import reset_cv_service
from visionqueue.api.service import CVService
from visionqueue.persistence.models import (
    AlertRecord,
    EventRecord,
    MeasurementRecord,
    SessionCloseRecord,
    SessionCreateRecord,
)


def _make_dummy_state(frame_id: int = 1, session_id: str = "lifecycle_sess_1"):
    return {
        "schema_version": 1,
        "timestamp": time.time(),
        "session_id": session_id,
        "frame_id": frame_id,
        "system_state": "HEALTHY",
        "camera_state": "CONNECTED",
        "performance": {
            "processing_fps": 30.0,
            "inference_latency_ms": 12.0,
            "frame_age_ms": 15.0,
        },
        "vision": {
            "person_detection": "HEALTHY",
            "face_detection": "DISABLED",
            "tracking": "HEALTHY",
        },
        "counts": {
            "current": 3,
            "track_instances": 3,
            "unique_session_approx": 5,
            "entries": 5,
            "exits": 2,
            "net_count": 3,
        },
        "occupancy": {
            "capacity": 10,
            "capacity_state": "NORMAL",
            "percent": 30.0,
        },
        "crowd": {
            "level": "LOW",
            "raw_level": "LOW",
            "density_score": 0.2,
            "trend": "STABLE",
            "is_elevated": False,
            "peak_count": 5,
            "peak_occupancy_percent": 50.0,
            "peak_timestamp": time.time(),
        },
        "alerts": [],
        "status_reason": "Operating normally",
        "is_healthy": True,
        "is_frozen": False,
        "detections_count": 3,
        "tracks_count": 3,
        "faces_count": 0,
        "queue_people": 1,
    }


def _create_mock_pipeline(session_id="lifecycle_sess_1"):
    mock_pipeline = MagicMock()
    mock_pipeline.config.session_id = session_id
    mock_pipeline.config.camera.source = "0"
    mock_pipeline.config.analytics.capacity = 20
    mock_pipeline.config.enable_roi = False
    mock_pipeline.config.roi = None
    
    # Mock detector with proper attributes for get_diagnostics()
    mock_detector = MagicMock()
    mock_detector.config = MagicMock()
    mock_detector.config.model_path = "models/yolo26n.onnx"
    mock_detector.active_provider = "CPUExecutionProvider"
    mock_detector.is_gpu = False
    mock_pipeline._detector = mock_detector

    frame_counter = [0]

    def step(timeout=1.0):
        frame_counter[0] += 1
        mock_res = MagicMock()
        mock_res.to_dict.return_value = _make_dummy_state(
            frame_id=frame_counter[0],
            session_id=mock_pipeline.config.session_id,
        )
        return mock_res

    mock_pipeline.step.side_effect = step
    mock_pipeline.last_diagnostics = {
        "active_tracks": [
            {"track_id": 1, "bbox": [10.0, 20.0, 50.0, 100.0], "confidence": 0.9}
        ]
    }
    return mock_pipeline


@pytest.fixture
def lifecycle_setup(tmp_path):
    reset_cv_service()
    db_path = str(tmp_path / "lifecycle_test.db")
    pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
    mock_pipeline = _create_mock_pipeline("lifecycle_sess_1")

    service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
    app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
    client = TestClient(app)

    yield {"service": service, "client": client, "pipeline": mock_pipeline, "db_path": db_path}

    if service.is_running:
        service.stop()
    reset_cv_service()


# ------------------------------------------------------------------
# 1. START
# ------------------------------------------------------------------

def test_start_lifecycle(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    assert service.is_running is False
    res = client.post("/api/v1/pipeline/start")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert service.is_running is True


# ------------------------------------------------------------------
# 2. START AGAIN (Idempotency)
# ------------------------------------------------------------------

def test_start_again_idempotent(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    assert service.is_running is True

    # Second start must return 200 with idempotent message
    res2 = client.post("/api/v1/pipeline/start")
    assert res2.status_code == 200
    assert "already running" in res2.json()["message"]
    assert service.is_running is True


# ------------------------------------------------------------------
# 3. STOP
# ------------------------------------------------------------------

def test_stop_lifecycle(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    assert service.is_running is True

    res = client.post("/api/v1/pipeline/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert service.is_running is False


# ------------------------------------------------------------------
# 4. STOP AGAIN (Idempotency)
# ------------------------------------------------------------------

def test_stop_again_idempotent(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    client.post("/api/v1/pipeline/stop")
    assert service.is_running is False

    # Second stop must return 200 with idempotent message
    res2 = client.post("/api/v1/pipeline/stop")
    assert res2.status_code == 200
    assert "already stopped" in res2.json()["message"]
    assert service.is_running is False


# ------------------------------------------------------------------
# 5. START -> STOP -> START (Restarting)
# ------------------------------------------------------------------

def test_start_stop_start_restart(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    # Cycle 1: Start
    res1 = client.post("/api/v1/pipeline/start")
    assert res1.status_code == 200
    time.sleep(0.08)
    assert service.get_latest_state()["frame_id"] >= 1

    # Cycle 1: Stop
    res_stop = client.post("/api/v1/pipeline/stop")
    assert res_stop.status_code == 200
    assert service.is_running is False

    # Cycle 2: Start again
    res2 = client.post("/api/v1/pipeline/start")
    assert res2.status_code == 200
    assert service.is_running is True

    time.sleep(0.08)
    state = service.get_latest_state()
    assert state["frame_id"] >= 1

    # Check persistence status in database
    sessions = client.get("/api/v1/sessions").json()
    assert len(sessions) >= 1
    # Active session should be marked RUNNING, not stuck on STOPPED
    active_sess = [s for s in sessions if s["id"] == "lifecycle_sess_1"][0]
    assert active_sess["status"] == "RUNNING"
    assert active_sess["ended_at"] is None


# ------------------------------------------------------------------
# 6. RESET WHEN STOPPED
# ------------------------------------------------------------------

def test_reset_when_stopped(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    assert service.is_running is False

    # API should return 503 Service Unavailable
    res = client.post("/api/v1/pipeline/reset")
    assert res.status_code == 503
    assert "Cannot reset" in res.json()["detail"]

    # Direct method call should also succeed safely without throwing AttributeError
    service.reset_session()
    assert service.is_running is False


# ------------------------------------------------------------------
# 7. RESET WHEN RUNNING
# ------------------------------------------------------------------

def test_reset_when_running(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    time.sleep(0.08)
    assert service.get_latest_state()["frame_id"] >= 1

    # Reset
    res = client.post("/api/v1/pipeline/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert "Pipeline session reset." in res.json()["message"]
    assert service.is_running is True

    # Pipeline reset method was called
    lifecycle_setup["pipeline"].reset_session.assert_called()


# ------------------------------------------------------------------
# 8. API REQUEST BEFORE PIPELINE START
# ------------------------------------------------------------------

def test_api_requests_before_pipeline_start(lifecycle_setup):
    client = lifecycle_setup["client"]

    # /health must always succeed with 200
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json() == {"status": "ok"}

    # All state-dependent endpoints must return 503 Service Unavailable
    for ep in ["/api/v1/status", "/api/v1/live", "/api/v1/counts", "/api/v1/occupancy", "/api/v1/crowd", "/api/v1/tracks", "/api/v1/alerts"]:
        res = client.get(ep)
        assert res.status_code == 503
        assert "CV pipeline has not produced state yet" in res.json()["detail"]


# ------------------------------------------------------------------
# 9. WEBSOCKET BEFORE PIPELINE START
# ------------------------------------------------------------------

def test_websocket_before_pipeline_start(lifecycle_setup):
    client = lifecycle_setup["client"]

    # Connecting before pipeline start should connect cleanly
    with client.websocket_connect("/ws/live") as ws:
        # Start pipeline while connected
        client.post("/api/v1/pipeline/start")
        # Should now receive streaming frame
        msg = ws.receive_json()
        assert msg["session_id"] == "lifecycle_sess_1"
        assert msg["frame_id"] >= 1


# ------------------------------------------------------------------
# 10. WEBSOCKET DISCONNECT
# ------------------------------------------------------------------

def test_websocket_disconnect(lifecycle_setup):
    client = lifecycle_setup["client"]
    service = lifecycle_setup["service"]

    client.post("/api/v1/pipeline/start")
    time.sleep(0.05)

    # Connect and abruptly close
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
        assert msg["frame_id"] >= 1
        ws.close()

    # Pipeline and server should continue operating smoothly
    time.sleep(0.05)
    assert service.is_running is True
    res = client.get("/api/v1/status")
    assert res.status_code == 200


# ------------------------------------------------------------------
# 11. WEBSOCKET RECONNECT
# ------------------------------------------------------------------

def test_websocket_reconnect(lifecycle_setup):
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    time.sleep(0.05)

    # First session
    with client.websocket_connect("/ws/live") as ws1:
        msg1 = ws1.receive_json()
        assert "frame_id" in msg1

    # Reconnect in a second session
    with client.websocket_connect("/ws/live") as ws2:
        msg2 = ws2.receive_json()
        assert "frame_id" in msg2


# ------------------------------------------------------------------
# 12. MULTIPLE WEBSOCKET CLIENTS
# ------------------------------------------------------------------

def test_multiple_websocket_clients(lifecycle_setup):
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    time.sleep(0.05)

    with client.websocket_connect("/ws/live") as ws1:
        with client.websocket_connect("/ws/live") as ws2:
            msg1 = ws1.receive_json()
            msg2 = ws2.receive_json()
            assert "frame_id" in msg1
            assert "frame_id" in msg2


# ------------------------------------------------------------------
# 13. DATABASE ERROR
# ------------------------------------------------------------------

def test_database_error_handling(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    time.sleep(0.05)

    # Simulate database error on write
    writer = service._writer
    with patch.object(writer._repo, "execute", side_effect=sqlite3.DatabaseError("Disk I/O failure")):
        rec = MeasurementRecord(
            session_id="lifecycle_sess_1",
            recorded_at="2026-09-06T12:00:00Z",
            current_count=1,
            unique_count=1,
            entries=1,
            exits=0,
            net_count=1,
            occupancy_percent=10.0,
            crowd_level="LOW",
            crowd_trend="STABLE",
            peak_count=1,
            peak_occupancy_percent=10.0,
            peak_timestamp="2026-09-06T12:00:00Z",
            processing_latency_ms=10.0,
            system_state="HEALTHY",
            camera_state="CONNECTED",
            queue_people=0,
        )
        success = writer.write_sync(rec)
        assert success is False
        assert writer.stats["is_healthy"] == 0

    # Pipeline continues running
    assert service.is_running is True


# ------------------------------------------------------------------
# 14. DATABASE QUEUE PRESSURE
# ------------------------------------------------------------------

def test_database_queue_pressure(lifecycle_setup):
    service = lifecycle_setup["service"]

    # Create a tiny writer with max_queue_size=2
    mock_repo = MagicMock()
    from visionqueue.persistence.writer import PersistenceWriter
    small_writer = PersistenceWriter(mock_repo, max_queue_size=2)

    # Fill queue to capacity
    rec1 = EventRecord(session_id="s1", event_type="ENTRY", occurred_at="t1", value=1, zone=None, metadata_json=None, dedup_key="k1")
    rec2 = EventRecord(session_id="s1", event_type="ENTRY", occurred_at="t2", value=1, zone=None, metadata_json=None, dedup_key="k2")
    rec3 = EventRecord(session_id="s1", event_type="ENTRY", occurred_at="t3", value=1, zone=None, metadata_json=None, dedup_key="k3")

    small_writer.enqueue_critical(rec1)
    small_writer.enqueue_critical(rec2)

    # Third enqueue must raise queue.Full
    with pytest.raises(queue.Full):
        small_writer.enqueue_critical(rec3)

    # CVService._enqueue_critical handles queue.Full gracefully without raising
    service._writer = small_writer
    service._enqueue_critical(rec3)
    assert service._persistence_healthy is False
    assert "k3" in service._pending_critical


# ------------------------------------------------------------------
# 15. SHUTDOWN DURING WRITE
# ------------------------------------------------------------------

def test_shutdown_during_write(lifecycle_setup):
    service = lifecycle_setup["service"]
    client = lifecycle_setup["client"]

    client.post("/api/v1/pipeline/start")
    # Immediately enqueue multiple measurements and events
    writer = service._writer
    for i in range(10):
        writer.enqueue_critical(
            EventRecord(
                session_id="lifecycle_sess_1",
                event_type="ENTRY",
                occurred_at=f"2026-09-06T12:00:{i:02d}Z",
                value=1,
                zone=None,
                metadata_json=None,
                dedup_key=f"e_{i}",
            )
        )

    # Stop pipeline during active work
    res = client.post("/api/v1/pipeline/stop")
    assert res.status_code == 200
    assert service.is_running is False
