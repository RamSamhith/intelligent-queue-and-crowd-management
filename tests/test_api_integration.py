"""End-to-end integration test for CVPipeline, CVService, FastAPI, and SQLite Persistence."""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock
from starlette.testclient import TestClient

from visionqueue.api.app import create_app
from visionqueue.api.config import APIConfig
from visionqueue.api.service import CVService
from visionqueue.camera.types import FrameData, SourceState
from visionqueue.detection.types import Detection
from visionqueue.persistence.config import PersistenceConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig


def make_test_frame(frame_id: int, timestamp: float) -> FrameData:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameData(
        frame=frame,
        timestamp=timestamp,
        frame_id=frame_id,
        width=640,
        height=480,
        source_state=SourceState.RUNNING,
        fps=30.0,
    )


@pytest.fixture
def integration_env(tmp_path):
    db_path = str(tmp_path / "integration_visionqueue.db")
    pcfg = PersistenceConfig(
        enabled=True,
        database_path=db_path,
        measurement_interval_seconds=0.05,
    )

    # Mock camera to supply continuous valid frames
    mock_camera = MagicMock()
    mock_camera.state = SourceState.RUNNING
    frame_counter = 0

    def fake_get_latest_frame(wait_new=True, timeout=1.0):
        nonlocal frame_counter
        frame_counter += 1
        return make_test_frame(frame_counter, time.time())

    mock_camera.get_latest_frame.side_effect = fake_get_latest_frame

    # Mock detector returning realistic detections
    mock_detector = MagicMock()
    mock_detector.detect_timed.return_value = (
        [
            Detection(bbox=(150.0, 150.0, 250.0, 350.0), confidence=0.92, class_id=0),
            Detection(bbox=(300.0, 150.0, 400.0, 350.0), confidence=0.88, class_id=0),
        ],
        8.5,
    )
    mock_detector.detect.return_value = [
        Detection(bbox=(150.0, 150.0, 250.0, 350.0), confidence=0.92, class_id=0),
        Detection(bbox=(300.0, 150.0, 400.0, 350.0), confidence=0.88, class_id=0),
    ]
    # Add attributes needed by CVService.get_diagnostics()
    mock_detector.config = MagicMock()
    mock_detector.config.model_path = "models/yolo26n.onnx"
    mock_detector.active_provider = "CPUExecutionProvider"
    mock_detector.is_gpu = False

    config = CVPipelineConfig(session_id="e2e_session_integration")
    pipeline = CVPipeline(
        config=config,
        camera=mock_camera,
        detector=mock_detector,
    )

    service = CVService(pipeline=pipeline, persistence_config=pcfg)
    app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
    client = TestClient(app)

    yield service, client, pcfg

    if service.is_running:
        service.stop()


def test_full_pipeline_service_api_persistence_e2e(integration_env):
    service, client, pcfg = integration_env

    # 1. Verify health before pipeline start
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    # 2. Start pipeline via API
    res = client.post("/api/v1/pipeline/start")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert service.is_running is True

    # 3. Wait briefly for worker loop to process several frames
    for _ in range(50):
        state = service.get_latest_state()
        if state and state.get("frame_id", 0) >= 3:
            break
        time.sleep(0.05)

    assert service.get_latest_state()["frame_id"] >= 3

    # 4. Check REST endpoints
    # Status
    status_res = client.get("/api/v1/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["camera_state"] in ("ONLINE", "CONNECTED")
    assert status_data["is_healthy"] is True

    # Live
    live_res = client.get("/api/v1/live")
    assert live_res.status_code == 200
    live_data = live_res.json()
    assert live_data["session_id"] == "e2e_session_integration"
    assert live_data["counts"]["current"] >= 1

    # Counts
    counts_res = client.get("/api/v1/counts")
    assert counts_res.status_code == 200
    assert counts_res.json()["current"] >= 1

    # Occupancy
    occ_res = client.get("/api/v1/occupancy")
    assert occ_res.status_code == 200
    assert occ_res.json()["capacity_state"] in ("NORMAL", "UNCAPPED", "NOT_SET")

    # Crowd
    crowd_res = client.get("/api/v1/crowd")
    assert crowd_res.status_code == 200
    assert "level" in crowd_res.json()

    # Active tracks
    tracks_res = client.get("/api/v1/tracks")
    assert tracks_res.status_code == 200
    assert isinstance(tracks_res.json(), list)

    # 5. WebSocket live stream
    with client.websocket_connect("/ws/live") as ws:
        frame_msg = ws.receive_json()
        assert frame_msg["session_id"] == "e2e_session_integration"
        assert "frame_id" in frame_msg
        assert "counts" in frame_msg

    # 6. Stop pipeline via API
    res_stop = client.post("/api/v1/pipeline/stop")
    assert res_stop.status_code == 200
    assert service.is_running is False

    # 7. Check persistence history
    sessions_res = client.get("/api/v1/sessions")
    assert sessions_res.status_code == 200
    sessions = sessions_res.json()
    assert len(sessions) >= 1
    assert any(s["id"] == "e2e_session_integration" for s in sessions)

    hist_res = client.get("/api/v1/history?session_id=e2e_session_integration")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) >= 1
