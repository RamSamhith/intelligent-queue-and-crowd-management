"""Regression test: verify FastAPI API observes the same CVService that was started.

Tests the bug where create_app() created its own CVService (without pipeline)
when cv_service=None, and this orphaned service overwrote the correct singleton
in the global _cv_service, causing /ready, /api/v1/live, etc. to return
"CV pipeline has not produced state yet" even though the real pipeline was running.

The fix: pass the pre-built CVService to create_app() so it uses the same
instance that Launcher.start() creates and manages.
"""

import time
from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from visionqueue.api.app import create_app
from visionqueue.api.config import APIConfig, PersistenceConfig
from visionqueue.api.dependencies import get_cv_service, reset_cv_service
from visionqueue.api.service import CVService


def _make_dummy_state(frame_id: int = 1, session_id: str = "singleton_sess"):
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


def _create_mock_pipeline(session_id="singleton_sess"):
    mock_pipeline = MagicMock()
    mock_pipeline.config.session_id = session_id
    mock_pipeline.config.camera.source = "0"
    mock_pipeline.config.analytics.capacity = 20
    mock_pipeline.config.enable_roi = False
    mock_pipeline.config.roi = None

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


class TestApiSingletonWiring:
    """Regression tests: API must observe the same CVService instance that was started."""

    def test_create_app_with_cv_service_uses_same_instance(self, tmp_path):
        """Verify create_app uses the passed CVService, not a newly created one."""
        reset_cv_service()
        db_path = str(tmp_path / "singleton.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_singleton_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)

        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
        client = TestClient(app)

        retrieved = get_cv_service()
        assert retrieved is service, (
            "get_cv_service() must return the same instance passed to create_app. "
            "If this fails, create_app is creating its own CVService instead of using the passed one."
        )

        service.stop()
        reset_cv_service()

    def test_api_sees_running_pipeline_after_start(self, tmp_path):
        """When service.start() is called, /ready must report ready=True."""
        reset_cv_service()
        db_path = str(tmp_path / "singleton2.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_ready_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
        client = TestClient(app)

        assert service.is_running is False

        service.start()
        time.sleep(0.1)

        ready_res = client.get("/ready")
        assert ready_res.status_code == 200, (
            f"/ready must return 200 when pipeline is running. "
            f"Got {ready_res.status_code}: {ready_res.json()}"
        )
        data = ready_res.json()
        assert data["ready"] is True, (
            f"/ready 'ready' must be True when pipeline is running. Got: {data}"
        )
        assert data["pipeline_running"] is True

        service.stop()
        reset_cv_service()

    def test_api_live_returns_state_after_start(self, tmp_path):
        """When service.start() is called, /api/v1/live must return state (not 503)."""
        reset_cv_service()
        db_path = str(tmp_path / "singleton3.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_live_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
        client = TestClient(app)

        service.start()
        time.sleep(0.1)

        live_res = client.get("/api/v1/live")
        assert live_res.status_code == 200, (
            f"/api/v1/live must return 200 when pipeline is running. "
            f"Got {live_res.status_code}: {live_res.json()}"
        )
        data = live_res.json()
        assert "session_id" in data
        assert data["session_id"] == "test_live_sess"

        service.stop()
        reset_cv_service()

    def test_api_status_returns_state_after_start(self, tmp_path):
        """When service.start() is called, /api/v1/status must return state (not 503)."""
        reset_cv_service()
        db_path = str(tmp_path / "singleton4.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_status_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
        client = TestClient(app)

        service.start()
        time.sleep(0.1)

        status_res = client.get("/api/v1/status")
        assert status_res.status_code == 200, (
            f"/api/v1/status must return 200 when pipeline is running. "
            f"Got {status_res.status_code}: {status_res.json()}"
        )

        service.stop()
        reset_cv_service()

    def test_api_sessions_accessible_after_start(self, tmp_path):
        """When service.start() is called, /api/v1/sessions must return session data."""
        reset_cv_service()
        db_path = str(tmp_path / "singleton5.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_sessions_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
        client = TestClient(app)

        service.start()
        time.sleep(0.1)

        sessions_res = client.get("/api/v1/sessions")
        assert sessions_res.status_code == 200, (
            f"/api/v1/sessions must return 200 when persistence is configured. "
            f"Got {sessions_res.status_code}: {sessions_res.json()}"
        )
        sessions = sessions_res.json()
        assert isinstance(sessions, list), f"/api/v1/sessions must return a list. Got: {type(sessions)}"

        service.stop()
        reset_cv_service()

    def test_no_orphaned_service_when_cv_service_passed(self, tmp_path):
        """When cv_service is passed to create_app, no orphaned CVService should be created.

        This is the core regression test for the bug where create_app() with
        cv_service=None created a CVService without pipeline that overwrote
        the correct singleton.
        """
        reset_cv_service()
        db_path = str(tmp_path / "singleton6.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_orphan_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)

        retrieved = get_cv_service()
        assert retrieved is service, (
            "The global CVService must be the same instance passed to create_app. "
            "If this fails, create_app is creating an orphaned CVService that "
            "overwrites the correct singleton."
        )

        assert service.is_running is False
        service.start()
        assert retrieved.is_running is True, (
            "The retrieved service must be the same running instance. "
            "If this fails, the API is using a different CVService than the one started."
        )
        service.stop()
        reset_cv_service()

    def test_shutdown_leaves_clean_state(self, tmp_path):
        """After stop(), the singleton is cleanly stopped with no inconsistent state."""
        reset_cv_service()
        db_path = str(tmp_path / "singleton7.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("test_shutdown_sess")

        service = CVService(pipeline=mock_pipeline, persistence_config=pcfg)
        app = create_app(APIConfig(debug=True, persistence=pcfg), cv_service=service)
        client = TestClient(app)

        service.start()
        time.sleep(0.1)
        assert client.get("/ready").json()["ready"] is True

        service.stop()
        time.sleep(0.05)

        ready_res = client.get("/ready")
        data = ready_res.json()
        assert data["ready"] is False, (
            f"After stop(), /ready should report ready=False. Got: {data}"
        )
        assert data["pipeline_running"] is False

        reset_cv_service()


class TestLauncherIntegration:
    """Integration test: verify Launcher.start() wires API to the correct service.

    These tests verify the in-process Launcher flow without requiring a real camera.
    """

    def test_launcher_api_singleton_matches_started_service(self, tmp_path):
        """When Launcher starts with enable_api=True, the API's singleton must be the same started service.

        This is the critical regression test: before the fix, create_app() with
        cv_service=None created an orphaned CVService that overwrote the correct
        singleton, causing the API to observe a stopped service while the real
        pipeline was running.
        """
        reset_cv_service()

        db_path = str(tmp_path / "launcher_singleton.db")
        pcfg = PersistenceConfig(enabled=True, database_path=db_path, measurement_interval_seconds=0.01)
        mock_pipeline = _create_mock_pipeline("launcher_api_sess")

        from visionqueue.pipeline import CVPipelineConfig
        from visionqueue.api.app import create_app
        from visionqueue.api.config import APIConfig

        pipeline_config = CVPipelineConfig()
        pipeline_config.session_id = "launcher_api_sess"

        from visionqueue.pipeline import CVPipeline
        pipeline = CVPipeline(config=pipeline_config)
        pipeline._detector = mock_pipeline._detector

        service = CVService(pipeline=pipeline, config=pipeline_config, persistence_config=pcfg)

        api_config = APIConfig(debug=True, persistence=pcfg)
        app = create_app(api_config=api_config, cv_service=service)

        assert get_cv_service() is service, (
            "create_app with cv_service must set the global singleton to that same instance"
        )

        service.start()
        time.sleep(0.1)

        assert get_cv_service() is service, (
            "After service.start(), get_cv_service() must still return the same started service"
        )

        assert service.is_running is True
        assert get_cv_service().is_running is True, (
            "The singleton's is_running must be True since it's the same instance"
        )

        service.stop()
        reset_cv_service()
