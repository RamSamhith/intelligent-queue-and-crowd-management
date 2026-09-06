"""run.py — One-command launcher for VisionQueue local live operation.

Starts the full production stack from a single command:

    python run.py

What this does (in order):
    1. Builds a CVPipelineConfig (camera + detector + analytics + reliability).
    2. Constructs the production CVService with SQLite persistence enabled,
       so all counts / events / alerts / measurements are durably recorded.
    3. Optionally starts the FastAPI REST + WebSocket backend (uvicorn) in a
       background thread so other tools and dashboards can attach.
    4. Starts CVService (camera + YOLO26m/Crowd inference + ByteTrack +
       counting + crowd analytics + reliability watchdog).
    5. Opens the local operator display (OpenCV window) reading from the
       same cached LiveState and active tracks the backend serves.
    6. Waits for Ctrl+C / SIGINT.
    7. Graceful shutdown in reverse order:
         a. Close the display window.
         b. CVService.stop() — drains the persistence writer, closes the
            session row in SQLite (status='STOPPED'), joins the worker
            thread, stops the camera.
         c. Stop the uvicorn server.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from types import FrameType
from typing import Optional

# Ensure project root is importable when this script is run from any cwd.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig  # noqa: E402
from visionqueue.analytics.types import AnalyticsConfig  # noqa: E402
from visionqueue.analytics.scene import SceneAnalyzerConfig  # noqa: E402
from visionqueue.api.app import create_app  # noqa: E402
from visionqueue.api.config import APIConfig  # noqa: E402
from visionqueue.api.dependencies import init_cv_service, reset_cv_service  # noqa: E402
from visionqueue.api.service import CVService  # noqa: E402
from visionqueue.camera.types import CameraConfig  # noqa: E402
from visionqueue.detection.types import DetectorConfig  # noqa: E402
from visionqueue.persistence.config import PersistenceConfig  # noqa: E402
from visionqueue.pipeline import CVPipeline, CVPipelineConfig  # noqa: E402
from visionqueue.reliability.types import ReliabilityConfig  # noqa: E402
from visionqueue.viewer import SimpleDisplay  # noqa: E402

logger = logging.getLogger("visionqueue.run")


# ---------------------------------------------------------------------------
# Lifecycle controller (testable in-process)
# ---------------------------------------------------------------------------

class Launcher:
    """One-shot orchestrator that boots and gracefully tears down the stack.

    Exposed as a class so unit tests can drive start()/stop() without spawning
    a subprocess. CLI main() instantiates Launcher and blocks on run().
    """

    def __init__(
        self,
        source: int | str = 0,
        model_path: str = "models/yolo26m_crowd.onnx",
        confidence: float = 0.25,
        capacity: Optional[int] = None,
        db_path: str = "data/visionqueue.db",
        enable_api: bool = True,
        api_host: str = "127.0.0.1",
        api_port: int = 8000,
        headless: bool = False,
        display_window: str = "VisionQueue Live",
        display_width: int = 1024,
        display_height: int = 768,
    ) -> None:
        self._source = source
        self._model_path = model_path
        self._confidence = confidence
        self._capacity = capacity
        self._db_path = db_path
        self._enable_api = enable_api
        self._api_host = api_host
        self._api_port = api_port
        self._headless = headless
        self._display_window = display_window
        self._display_width = display_width
        self._display_height = display_height

        # Filled in by start()
        self._service: Optional[CVService] = None
        self._display: Optional[SimpleDisplay] = None
        self._uvicorn_server = None
        self._uvicorn_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build CVService, optionally start uvicorn, start CV, open display."""
        with self._start_lock:
            if self._started:
                logger.info("Launcher.start() called but already started.")
                return

            pipeline_config = self._build_pipeline_config()
            persistence_config = PersistenceConfig(
                enabled=True,
                database_path=self._db_path,
            )
            api_config = APIConfig(
                host=self._api_host,
                port=self._api_port,
                persistence=persistence_config,
            )

            # Construct the pipeline + the CVService directly so we can hold
            # a strong reference (the FastAPI app also installs it as a
            # module-level singleton via init_cv_service).
            pipeline = CVPipeline(config=pipeline_config)
            service = CVService(
                pipeline=pipeline,
                config=pipeline_config,
                persistence_config=persistence_config,
            )

            # Track resources started so far for transactional cleanup on failure.
            # Order matters: teardown happens in reverse.
            started_resources: list[str] = []

            try:
                if self._enable_api:
                    # Initialize the singleton the API will use, then create_app.
                    init_cv_service(
                        pipeline=service.pipeline,
                        config=pipeline_config,
                        persistence_config=persistence_config,
                    )
                    started_resources.append("cv_service_singleton")

                    self._uvicorn_server = self._start_uvicorn(api_config, cv_service=service)
                    started_resources.append("uvicorn")

                # Start the camera + pipeline + persistence worker.
                service.start()
                started_resources.append("cv_service")

                # Open the operator display, reading from the same CVService.
                # Pass on_exit_callback so Q/ESC triggers full Launcher shutdown.
                self._display = SimpleDisplay(
                    service=service,
                    window_name=self._display_window,
                    headless=self._headless,
                    width=self._display_width,
                    height=self._display_height,
                    on_exit_callback=self.stop,
                )
                self._display.start()
                started_resources.append("display")

                # All resources started successfully; transfer ownership.
                self._service = service
                self._started = True

            except Exception:
                # Transactional rollback: clean up any resources that were
                # started before the failure.  _started is still False so
                # stop() is safe to call (it guards against _started=False).
                logger.exception("Launcher.start() failed — rolling back started resources: %s", started_resources)
                # Build a minimal display-less stop to clean up the service/singleton.
                # We cannot reuse self._display.stop() because self._display may be None
                # or in a bad state; instead do targeted teardown.
                if "display" in started_resources and self._display is not None:
                    try:
                        self._display.stop(timeout=2.0)
                    except Exception:
                        logger.debug("Display stop during rollback", exc_info=True)
                    self._display = None
                if "cv_service" in started_resources and service is not None:
                    try:
                        service.stop()
                    except Exception:
                        logger.debug("CVService stop during rollback", exc_info=True)
                if "uvicorn" in started_resources:
                    self._teardown_uvicorn()
                if "cv_service_singleton" in started_resources:
                    try:
                        reset_cv_service()
                    except Exception:
                        logger.debug("reset_cv_service during rollback", exc_info=True)
                raise

            logger.info(
                "VisionQueue started. API=%s | DB=%s | Display=%s",
                f"http://{self._api_host}:{self._api_port}" if self._enable_api else "disabled",
                self._db_path,
                "headless" if self._headless else "window",
            )

    def stop(self) -> None:
        """Reverse-order shutdown: display → CVService → uvicorn."""
        if not self._started:
            return
        self._stop_event.set()

        # 1. Close the operator window.
        if self._display is not None:
            try:
                self._display.stop(timeout=3.0)
            except Exception:
                logger.exception("Display stop failed")
            self._display = None

        # 2. Stop the CVService (drains persistence writer + closes DB session).
        if self._service is not None:
            try:
                self._service.stop()
            except Exception:
                logger.exception("CVService stop failed")
            self._service = None

        # 3. Stop the FastAPI server (uvicorn).
        self._teardown_uvicorn()

        # 4. Reset the DI singleton so a future run() call starts clean.
        try:
            reset_cv_service()
        except Exception:
            logger.debug("reset_cv_service() failed", exc_info=True)

        self._started = False
        logger.info("VisionQueue stopped cleanly.")

    def _teardown_uvicorn(self) -> None:
        """Stop the uvicorn server and join its thread."""
        if self._uvicorn_server is not None:
            try:
                self._uvicorn_server.should_exit = True
            except Exception:
                logger.exception("Uvicorn stop failed")
            if self._uvicorn_thread is not None:
                self._uvicorn_thread.join(timeout=3.0)
            self._uvicorn_server = None
            self._uvicorn_thread = None

    def wait(self) -> None:
        """Block until stop() is called (or Ctrl+C interrupts).

        On non-headless runs this pumps the OpenCV/Windows GUI event loop
        on the calling thread by delegating to display.wait(), which is
        required for the window to stay responsive and redraw on Windows.
        """
        try:
            if self._display is not None and not self._headless:
                self._display.wait()
            else:
                while not self._stop_event.is_set():
                    time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_pipeline_config(self) -> CVPipelineConfig:
        api_pref = cv2.CAP_DSHOW if isinstance(self._source, int) and sys.platform.startswith("win") else 0
        camera_config = CameraConfig(
            source=self._source,
            api_preference=api_pref,
        )
        detector_config = DetectorConfig(
            model_path=self._model_path,
            confidence_threshold=self._confidence,
        )
        analytics_config = AnalyticsConfig(
            capacity=self._capacity,
            debounce_frames=3,
        )
        alerts_config = AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.5, clear_seconds=1.5),
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.5, clear_seconds=1.5),
        )
        reliability_config = ReliabilityConfig(min_starting_frames=3)
        return CVPipelineConfig(
            camera=camera_config,
            detector=detector_config,
            enable_roi=False,
            analytics=analytics_config,
            scene_analyzer=SceneAnalyzerConfig(enabled=True),
            alerts=alerts_config,
            reliability=reliability_config,
            enable_face_detection=True,
        )

    def _start_uvicorn(self, api_config: APIConfig, cv_service):
        """Start the FastAPI app in a background thread; return the server."""
        import uvicorn

        # Defer imports so non-API launches don't pay for uvicorn.
        # Pass the pre-built CVService so the API uses the SAME instance
        # that Launcher.start() creates and manages (avoids creating an
        # orphaned CVService in create_app that would overwrite the singleton).
        app = create_app(api_config=api_config, cv_service=cv_service)

        config = uvicorn.Config(
            app=app,
            host=api_config.host,
            port=api_config.port,
            log_level="warning",
            lifespan="on",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="VisionQueue-UVicorn", daemon=True)
        thread.start()

        # Wait briefly for the server to bind, but don't block forever.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if server.started:
                break
            time.sleep(0.05)
        self._uvicorn_thread = thread
        return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _install_signal_handlers(launcher: Launcher) -> None:
    def _handler(signum: int, frame: Optional[FrameType]) -> None:
        logger.info("Signal %s received — initiating graceful shutdown.", signum)
        launcher.stop()
        # Force-exit so a stuck cv2.waitKey thread can't pin the process.
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python run.py",
        description="VisionQueue local live operation launcher.",
    )
    p.add_argument("--source", type=str, default="0",
                   help="Camera index (e.g. 0), video file path, or stream URL.")
    p.add_argument("--model", type=str, default="models/yolo26m_crowd.onnx",
                   help="ONNX model path.")
    p.add_argument("--confidence", type=float, default=0.25,
                   help="Person detection confidence threshold.")
    p.add_argument("--capacity", type=int, default=None,
                   help="Optional manual venue capacity for occupancy %.")
    p.add_argument("--db-path", type=str, default="data/visionqueue.db",
                   help="SQLite database path for persistence.")
    p.add_argument("--no-api", action="store_true",
                   help="Do not start the FastAPI backend (display only).")
    p.add_argument("--api-host", type=str, default="127.0.0.1",
                   help="FastAPI bind host.")
    p.add_argument("--api-port", type=int, default=8000,
                   help="FastAPI bind port.")
    p.add_argument("--headless", action="store_true",
                   help="Run without an OpenCV window (for CI / smoke tests).")
    p.add_argument("--window", type=str, default="VisionQueue Live",
                   help="OpenCV window title.")
    p.add_argument("--width", type=int, default=1024, help="Display width.")
    p.add_argument("--height", type=int, default=768, help="Display height.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    source: int | str = int(args.source) if str(args.source).isdigit() else args.source

    launcher = Launcher(
        source=source,
        model_path=args.model,
        confidence=args.confidence,
        capacity=args.capacity,
        db_path=args.db_path,
        enable_api=not args.no_api,
        api_host=args.api_host,
        api_port=args.api_port,
        headless=args.headless,
        display_window=args.window,
        display_width=args.width,
        display_height=args.height,
    )
    _install_signal_handlers(launcher)

    try:
        launcher.start()
        launcher.wait()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down.")
    finally:
        launcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
