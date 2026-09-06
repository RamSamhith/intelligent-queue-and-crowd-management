"""Tests for VisionQueue REST API and WebSocket endpoints."""

import json
import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from visionqueue.api.app import create_app
from visionqueue.api.config import APIConfig
from visionqueue.api.dependencies import init_cv_service, reset_cv_service
from visionqueue.api.service import CVService
from visionqueue.persistence.config import PersistenceConfig
from visionqueue.persistence.database import Database
from visionqueue.persistence.repository import PersistenceRepository
from visionqueue.persistence.models import CameraRecord, SessionCreateRecord, MeasurementRecord, EventRecord, AlertRecord


def make_dummy_state():
    return {
        "schema_version": 1,
        "timestamp": 1726000000.0,
        "session_id": "test_session_123",
        "frame_id": 42,
        "system_state": "LIVE",
        "camera_state": "ONLINE",
        "performance": {
            "processing_fps": 28.5,
            "inference_latency_ms": 14.2,
            "frame_age_ms": 16.0,
        },
        "vision": {
            "person_detection": "ACTIVE",
            "face_detection": "DISABLED",
            "tracking": "ACTIVE",
        },
        "counts": {
            "current": 4,
            "track_instances": 4,
            "unique_session_approx": 10,
            "entries": 8,
            "exits": 4,
            "net_count": 4,
        },
        "occupancy": {
            "capacity": 20,
            "capacity_state": "NORMAL",
            "percent": 20.0,
        },
        "crowd": {
            "level": "LOW",
            "raw_level": "LOW",
            "trend": "STABLE",
            "peak_count": 6,
            "peak_occupancy_percent": 30.0,
            "peak_timestamp": 1726000000.0,
        },
        "alerts": [
            {
                "id": "alert_01",
                "type": "HIGH_CROWD_DENSITY",
                "severity": "WARNING",
                "fired_at": 1726000000.0,
                "cleared_at": None,
                "status": "ACTIVE",
                "reason": "Density high",
            }
        ],
        "status_reason": "Operational",
        "is_healthy": True,
        "is_frozen": False,
        "detections_count": 4,
        "tracks_count": 4,
        "faces_count": 0,
        "queue_people": 2,
    }


@pytest.fixture
def mock_service(tmp_path):
    reset_cv_service()
    db_path = str(tmp_path / "api_test.db")
    pcfg = PersistenceConfig(enabled=True, database_path=db_path)
    
    mock_pipeline = MagicMock()
    mock_pipeline.config.session_id = "test_session_123"
    mock_pipeline.config.camera.source = "0"
    mock_pipeline.config.analytics.capacity = 50
    mock_pipeline.config.enable_roi = False
    mock_pipeline.config.roi = None
    
    # Mock detector with proper attributes for get_diagnostics()
    mock_detector = MagicMock()
    mock_detector.config = MagicMock()
    mock_detector.config.model_path = "models/yolo26n.onnx"
    mock_detector.active_provider = "CPUExecutionProvider"
    mock_detector.is_gpu = False
    mock_pipeline._detector = mock_detector
    
    mock_step_result = MagicMock()
    mock_step_result.to_dict.return_value = make_dummy_state()
    mock_pipeline.step.return_value = mock_step_result
    mock_pipeline.last_diagnostics = {
        "active_tracks": [
            {"track_id": 1, "bbox": [10.0, 20.0, 50.0, 100.0], "confidence": 0.88}
        ]
    }
    
    service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
    yield service
    if service.is_running:
        service.stop()
    reset_cv_service()


@pytest.fixture
def client(mock_service):
    app = create_app(APIConfig(debug=True, persistence=mock_service._persistence_config), cv_service=mock_service)
    return TestClient(app)


# ------------------------------------------------------------------
# Test Health, Docs, OpenAPI
# ------------------------------------------------------------------

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_docs_and_openapi(client):
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200

    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    spec = res_openapi.json()
    assert spec["info"]["title"] == "VisionQueue API"
    assert "/api/v1/live" in spec["paths"]
    assert "/api/v1/status" in spec["paths"]


# ------------------------------------------------------------------
# Test 503 when no state
# ------------------------------------------------------------------

def test_endpoints_503_when_no_state(client):
    assert client.get("/api/v1/status").status_code == 503
    assert client.get("/api/v1/live").status_code == 503
    assert client.get("/api/v1/counts").status_code == 503
    assert client.get("/api/v1/occupancy").status_code == 503
    assert client.get("/api/v1/crowd").status_code == 503
    assert client.get("/api/v1/alerts").status_code == 503
    assert client.get("/api/v1/tracks").status_code == 503


# ------------------------------------------------------------------
# Test endpoints with populated state
# ------------------------------------------------------------------

def test_endpoints_with_state(client, mock_service):
    dummy = make_dummy_state()
    with mock_service._state_lock:
        mock_service._latest_state_dict = dummy
        mock_service._latest_tracks = [{"track_id": 1, "bbox": [10.0, 20.0, 50.0, 100.0], "confidence": 0.88}]

    # /status
    res = client.get("/api/v1/status")
    assert res.status_code == 200
    data = res.json()
    assert data["system_state"] == "LIVE"
    assert data["camera_state"] == "ONLINE"
    assert data["is_healthy"] is True
    assert data["performance"]["processing_fps"] == 28.5

    # /live
    res = client.get("/api/v1/live")
    assert res.status_code == 200
    assert res.json()["frame_id"] == 42
    assert res.json()["queue_people"] == 2

    # /counts
    res = client.get("/api/v1/counts")
    assert res.status_code == 200
    assert res.json() == {
        "current": 4,
        "track_instances": 4,
        "unique_session_approx": 10,
        "entries": 8,
        "exits": 4,
        "net_count": 4,
    }

    # /occupancy
    res = client.get("/api/v1/occupancy")
    assert res.status_code == 200
    assert res.json() == {
        "capacity": 20,
        "capacity_state": "NORMAL",
        "percent": 20.0,
    }

    # /crowd
    res = client.get("/api/v1/crowd")
    assert res.status_code == 200
    assert res.json()["level"] == "LOW"
    assert res.json()["trend"] == "STABLE"

    # /alerts
    res = client.get("/api/v1/alerts")
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["id"] == "alert_01"

    # /tracks
    res = client.get("/api/v1/tracks")
    assert res.status_code == 200
    tracks = res.json()
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1


# ------------------------------------------------------------------
# Test Pipeline Lifecycle Actions
# ------------------------------------------------------------------

def test_pipeline_actions(client, mock_service):
    # Pipeline start
    res = client.post("/api/v1/pipeline/start")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert mock_service.is_running is True

    # Already running start
    res2 = client.post("/api/v1/pipeline/start")
    assert res2.status_code == 200
    assert "already running" in res2.json()["message"]

    # Reset
    res = client.post("/api/v1/pipeline/reset")
    assert res.status_code == 200
    assert res.json()["message"] == "Pipeline session reset."

    # Stop
    res = client.post("/api/v1/pipeline/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert mock_service.is_running is False


# ------------------------------------------------------------------
# Test History Endpoints
# ------------------------------------------------------------------

def test_history_endpoints(client, mock_service):
    # Initialize DB & Seed data
    mock_service._init_persistence()
    repo = mock_service._repo
    cam_id = repo.get_or_create_camera(
        CameraRecord(None, "Cam0", "0", None, 100, None, None, None, None, "2026-09-06T12:00:00Z")
    )
    repo.execute(SessionCreateRecord(id="sess_hist_1", camera_id=cam_id, started_at="2026-09-06T12:00:00Z"))
    repo.execute(MeasurementRecord(
        session_id="sess_hist_1", recorded_at="2026-09-06T12:01:00Z", current_count=3,
        unique_count=5, entries=4, exits=1, net_count=3, occupancy_percent=3.0,
        crowd_level="LOW", crowd_trend="STABLE", peak_count=3, peak_occupancy_percent=3.0,
        peak_timestamp=None, processing_latency_ms=10.0, system_state="LIVE", camera_state="ONLINE",
        queue_people=1
    ))
    repo.execute(EventRecord(
        session_id="sess_hist_1", event_type="ENTRY", occurred_at="2026-09-06T12:01:10Z",
        value=1, zone="z1", metadata_json=None, dedup_key="d1"
    ))
    repo.execute(AlertRecord(
        id="a1", session_id="sess_hist_1", type="CROWD", severity="WARNING",
        fired_at="2026-09-06T12:01:20Z", status="ACTIVE", reason="high"
    ))

    # Test GET /api/v1/sessions
    res = client.get("/api/v1/sessions")
    assert res.status_code == 200
    sessions = res.json()
    assert len(sessions) >= 1
    assert any(s["id"] == "sess_hist_1" for s in sessions)

    # Test GET /api/v1/sessions/{id}
    res = client.get("/api/v1/sessions/sess_hist_1")
    assert res.status_code == 200
    assert res.json()["id"] == "sess_hist_1"

    # Test 404
    res = client.get("/api/v1/sessions/non_existent")
    assert res.status_code == 404

    # Test GET /api/v1/history
    res = client.get("/api/v1/history?session_id=sess_hist_1")
    assert res.status_code == 200
    history = res.json()
    assert len(history) == 1
    assert history[0]["current_count"] == 3
    assert history[0]["queue_people"] == 1

    # Test GET /api/v1/history/events
    res = client.get("/api/v1/history/events?session_id=sess_hist_1")
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["event_type"] == "ENTRY"

    # Test GET /api/v1/history/alerts
    res = client.get("/api/v1/history/alerts?session_id=sess_hist_1")
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["id"] == "a1"


# ------------------------------------------------------------------
# Test WebSocket Live Streaming
# ------------------------------------------------------------------

def test_websocket_live_streaming(client, mock_service):
    dummy = make_dummy_state()
    with mock_service._state_lock:
        mock_service._latest_state_dict = dummy

    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
        assert msg["session_id"] == "test_session_123"
        assert msg["frame_id"] == 42
        assert msg["counts"]["current"] == 4
        assert msg["queue_people"] == 2

    # Also test /api/v1/ws/live
    with client.websocket_connect("/api/v1/ws/live") as ws:
        msg = ws.receive_json()
        assert msg["session_id"] == "test_session_123"
        assert msg["frame_id"] == 42
