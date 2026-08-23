"""VisionQueue CV Pipeline Coordinator.

Orchestrates the end-to-end computer vision and analytics pipeline:
CameraSource -> PersonDetector -> TrackingAdapter -> ROI/Counting -> Analytics -> Alerts -> Reliability -> LiveState.
"""

from __future__ import annotations
import logging
import time
from typing import Callable, List, Optional

import numpy as np

from visionqueue.alerts.engine import AlertEngine
from visionqueue.analytics.types import AnalyticsConfig
from visionqueue.analytics.crowd import CrowdAnalyticsEngine
from visionqueue.analytics.scene import SceneAnalysisState, SceneAnalyzer
from visionqueue.camera.capture import CameraSource
from visionqueue.camera.types import FrameData, SourceState
from visionqueue.counting.line_crossing import LineCrossingCounter
from visionqueue.counting.occupancy import OccupancyCounter
from visionqueue.counting.session import SessionCounter
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.face_detector import FaceDetector
from visionqueue.pipeline.types import CVPipelineConfig, LiveState
from visionqueue.reliability.manager import ReliabilityManager
from visionqueue.roi.roi import ROIFilter
from visionqueue.tracking.adapter import DetectionTrackingAdapter

logger = logging.getLogger(__name__)


class CVPipeline:
    """Executable pipeline coordinator for VisionQueue real-time video analytics.

    Responsibilities:
    - Coordinates frame acquisition from CameraSource.
    - Runs YOLO26n person detection and optional YuNet face detection.
    - Bridges detections into ByteTrack.
    - Computes in-ROI occupancy, line crossings, and session-unique counts.
    - Evaluates crowd analytics (percentages, tiers, debouncing, trend).
    - Manages P0 alerts with symmetric hysteresis.
    - Manages system reliability states and preserves last reliable counts.
    - Emits unified LiveState snapshots for backend / WebSocket publishing.
    """

    def __init__(
        self,
        config: Optional[CVPipelineConfig] = None,
        camera: Optional[CameraSource] = None,
        detector: Optional[PersonDetector] = None,
        face_detector: Optional[FaceDetector] = None,
    ) -> None:
        """Initialize the CV pipeline coordinator.

        Args:
            config: Optional CVPipelineConfig with full pipeline parameters.
            camera: Optional injected CameraSource (or creates one from config).
            detector: Optional injected PersonDetector.
            face_detector: Optional injected FaceDetector.
        """
        self._config: CVPipelineConfig = config or CVPipelineConfig()

        # 1. Camera
        self._camera: CameraSource = camera or CameraSource(self._config.camera)

        # 2. Vision Models
        self._detector: Optional[PersonDetector] = detector
        self._face_detector: Optional[FaceDetector] = face_detector

        # 3. Tracking & Spatial
        self._tracking_adapter = DetectionTrackingAdapter(config=self._config.tracker)

        init_frame_w = self._config.camera.width or 4096
        init_frame_h = self._config.camera.height or 4096
        self._roi_filter: Optional[ROIFilter] = None
        if self._config.roi is not None:
            self._roi_filter = ROIFilter(
                roi=self._config.roi,
                frame_width=init_frame_w,
                frame_height=init_frame_h,
            )

        self._frame_dims_confirmed: bool = bool(
            self._config.camera.width and self._config.camera.height
        )

        # 4. Counting (V1 default: Whole-frame counting; ROI mode: configurable when enable_roi=True)
        self._occupancy_counter = OccupancyCounter(
            roi_or_filter=self._roi_filter,
            enable_roi=self._config.enable_roi,
        )
        self._session_counter = SessionCounter()
        self._line_crossing_counter: Optional[LineCrossingCounter] = (
            LineCrossingCounter(self._config.virtual_line)
            if self._config.virtual_line is not None
            else None
        )

        # 5. Analytics, Scene Intelligence & Alerts
        analytics_cfg = self._config.analytics
        self._scene_analyzer = SceneAnalyzer(
            config=self._config.scene_analyzer,
            profile=self._config.scene,
        )
        if self._config.scene is not None and analytics_cfg.capacity is None:
            derived_cap, _ = self._config.scene.derive_effective_capacity()
            if derived_cap is not None:
                analytics_cfg = AnalyticsConfig(
                    capacity=derived_cap,
                    thresholds=self._config.scene.thresholds if self._config.scene.thresholds is not None else analytics_cfg.thresholds,
                    debounce_frames=analytics_cfg.debounce_frames,
                    trend_window_size=analytics_cfg.trend_window_size,
                    trend_min_delta=analytics_cfg.trend_min_delta,
                )
        self._analytics_engine = CrowdAnalyticsEngine(analytics_cfg)
        self._alert_engine = AlertEngine(self._config.alerts)

        # 6. Reliability & State
        self._reliability_manager = ReliabilityManager(self._config.reliability)

        self._frame_sequence: int = 0
        self._last_state: Optional[LiveState] = None
        self._is_running: bool = False
        self._prev_in_roi_ids: set[int] = set()
        self._last_diagnostics: dict = {}

    @property
    def config(self) -> CVPipelineConfig:
        return self._config

    @property
    def camera(self) -> CameraSource:
        return self._camera

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def last_diagnostics(self) -> dict:
        """Access detailed per-frame diagnostic telemetry."""
        return self._last_diagnostics

    def _ensure_models_loaded(self) -> None:
        """Lazy-load vision models if not injected."""
        if self._detector is None:
            self._detector = PersonDetector(self._config.detector)

        if self._config.enable_face_detection and self._face_detector is None:
            self._face_detector = FaceDetector(self._config.face_config)

    def start(self) -> None:
        """Start the camera acquisition thread and prepare models."""
        self._ensure_models_loaded()
        if not self._camera.is_running:
            self._camera.start()
        self._is_running = True
        logger.info("CVPipeline started for session %s.", self._config.session_id)

    def stop(self) -> None:
        """Stop camera acquisition and pipeline execution."""
        self._is_running = False
        if self._camera.is_running:
            self._camera.stop()
        # Notify reliability manager of operator stop
        self._reliability_manager.update(operator_stop=True, timestamp=time.time())
        logger.info("CVPipeline stopped.")

    def __enter__(self) -> "CVPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def process_frame(
        self,
        frame_data: Optional[FrameData],
        timestamp: Optional[float] = None,
    ) -> LiveState:
        """Process an acquired frame (or handle missing frame) through the complete pipeline.

        Args:
            frame_data: FrameData instance from CameraSource, or None if acquisition failed.
            timestamp: Evaluation timestamp (defaults to time.time()).

        Returns:
            Authoritative LiveState snapshot.
        """
        now = timestamp if timestamp is not None else time.time()
        self._frame_sequence += 1
        loop_start = time.perf_counter()

        # Handle Camera Offline / Disconnected
        if frame_data is None:
            source_state = self._camera.state
            if source_state not in (SourceState.DISCONNECTED, SourceState.ERROR, SourceState.STOPPED):
                source_state = SourceState.DISCONNECTED

            rel_state = self._reliability_manager.update(
                source_state=source_state,
                frame_data=None,
                detection_success=False,
                timestamp=now,
            )

            # Update alerts with camera failure
            self._alert_engine.update(
                crowd_level=self._analytics_engine.last_state.crowd_level
                if self._analytics_engine.last_state
                else "LOW",
                camera_offline=True,
                detection_failed=False,
                timestamp=now,
            )

            return self._build_live_state(
                frame_id=self._frame_sequence,
                timestamp=now,
                rel_state=rel_state,
                detections_count=0,
                tracks_count=0,
                faces_count=0,
                loop_latency_ms=(time.perf_counter() - loop_start) * 1000.0,
            )

        # Frame successfully acquired
        raw_bgr = frame_data.frame

        # On the first real frame (or whenever not yet confirmed), update the
        # ROIFilter's frame dimensions from the actual acquired FrameData.
        if not self._frame_dims_confirmed and self._config.roi is not None:
            self._roi_filter = ROIFilter(
                roi=self._config.roi,
                frame_width=frame_data.width,
                frame_height=frame_data.height,
            )
            self._occupancy_counter.update_roi(self._roi_filter.roi)
            self._frame_dims_confirmed = True
            logger.debug(
                "ROIFilter frame dimensions confirmed from FrameData: %dx%d",
                frame_data.width,
                frame_data.height,
            )
        elif self._roi_filter is not None and self._config.roi is not None:
            self._roi_filter.update_roi(self._config.roi)

        # 1. Person Detection
        self._ensure_models_loaded()
        assert self._detector is not None

        detection_ok = True
        detections = []
        infer_ms = 0.0
        try:
            detections, infer_ms = self._detector.detect_timed(raw_bgr)
        except Exception as e:
            logger.error("PersonDetector inference failed: %s", e)
            detection_ok = False

        # 2. Optional Face Detection (cadence controlled)
        faces_count = 0
        face_ok = True
        if self._config.enable_face_detection and self._face_detector is not None:
            if self._frame_sequence % self._config.face_cadence == 0:
                try:
                    face_dets = self._face_detector.detect(raw_bgr)
                    faces_count = len(face_dets)
                except Exception as e:
                    logger.error("FaceDetector inference failed: %s", e)
                    face_ok = False

        # 3. Tracking
        tracking_ok = True
        tracks = []
        try:
            tracks = self._tracking_adapter.update(detections)
        except Exception as e:
            logger.error("Tracking adapter update failed: %s", e)
            tracking_ok = False

        # 4. Counting
        current_occ = self._occupancy_counter.update(tracks, self._frame_sequence, now)
        session_occ = self._session_counter.update(tracks)
        crossing_events = []
        if self._line_crossing_counter is not None:
            crossing_events = self._line_crossing_counter.update(tracks, self._frame_sequence, now)

        # Compute ROI transition events
        current_in_roi = set(current_occ.active_track_ids)
        entered_roi_ids = sorted(list(current_in_roi - self._prev_in_roi_ids))
        left_roi_ids = sorted(list(self._prev_in_roi_ids - current_in_roi))
        self._prev_in_roi_ids = current_in_roi

        # 5. Scene Intelligence & Capacity Analysis
        scene_analysis_state = self._scene_analyzer.analyze(
            active_tracks=[t.to_dict() for t in tracks],
            frame_w=frame_data.width,
            frame_h=frame_data.height,
            timestamp=now,
        )
        if self._analytics_engine.capacity is None and scene_analysis_state.effective_capacity is not None:
            self._analytics_engine.update_capacity(scene_analysis_state.effective_capacity)

        # 6. Occupancy & Crowd Analytics
        analytics_state = self._analytics_engine.update(
            current_count=current_occ.current_count,
            frame_id=self._frame_sequence,
            timestamp=now,
        )

        # 7. P0 Alert Engine
        active_alerts_events = self._alert_engine.update(
            crowd_level=analytics_state.crowd_level,
            camera_offline=False,
            detection_failed=not detection_ok,
            timestamp=now,
        )

        # 8. System Reliability & Health State
        rel_state = self._reliability_manager.update(
            source_state=frame_data.source_state,
            frame_data=frame_data,
            detection_success=detection_ok,
            tracking_success=tracking_ok,
            current_count=current_occ.current_count,
            inference_latency_ms=infer_ms,
            face_detection_success=face_ok,
            timestamp=now,
        )

        loop_latency_ms = (time.perf_counter() - loop_start) * 1000.0

        # Assemble unified diagnostics
        tracker_diag = self._tracking_adapter.last_diagnostics
        self._last_diagnostics = {
            "frame_id": self._frame_sequence,
            "timestamp": now,
            "detections": [
                {
                    "bbox": [round(float(c), 2) for c in d.bbox],
                    "confidence": round(float(d.confidence), 4),
                    "class_id": int(d.class_id),
                }
                for d in detections
            ],
            "detections_count": len(detections),
            "active_tracks_count": len(tracks),
            "active_track_ids": [t.track_id for t in tracks],
            "active_tracks": [t.to_dict() for t in tracks],
            "track_ages": tracker_diag.get("track_ages", {}),
            "time_since_last_association": tracker_diag.get("time_since_last_association", {}),
            "tracks_created": tracker_diag.get("tracks_created", []),
            "tracks_lost": tracker_diag.get("tracks_lost", []),
            "id_replacements": tracker_diag.get("id_replacements", []),
            "association_ious": tracker_diag.get("association_ious", []),
            "matching_result": tracker_diag.get("matching_result", {}),
            "track_terminations": tracker_diag.get("track_terminations", []),
            "line_crossing_events": [
                {
                    "track_id": e.track_id,
                    "direction": e.direction.value,
                    "crossing_point": [round(float(c), 2) for c in e.crossing_point],
                    "bbox": [round(float(c), 2) for c in e.bbox],
                }
                for e in crossing_events
            ],
            "roi_events": {
                "entered_track_ids": entered_roi_ids,
                "left_track_ids": left_roi_ids,
                "in_roi_track_ids": sorted(list(current_in_roi)),
                "current_count": current_occ.current_count,
            },
            "scene_analysis": scene_analysis_state.to_dict(),
            "counts": {
                "current": current_occ.current_count,
                "entries": self._line_crossing_counter.entries if self._line_crossing_counter else 0,
                "exits": self._line_crossing_counter.exits if self._line_crossing_counter else 0,
                "net_count": self._line_crossing_counter.net_count if self._line_crossing_counter else 0,
                "unique_session_approx": session_occ.approximate_unique_count,
            },
        }

        return self._build_live_state(
            frame_id=self._frame_sequence,
            timestamp=now,
            rel_state=rel_state,
            detections_count=len(detections),
            tracks_count=len(tracks),
            faces_count=faces_count,
            loop_latency_ms=loop_latency_ms,
        )

    def step(self, timeout: float = 1.0) -> LiveState:
        """Acquire the newest frame from CameraSource and execute the pipeline.

        Returns:
            LiveState snapshot.
        """
        frame_data = self._camera.get_latest_frame(wait_new=True, timeout=timeout)
        return self.process_frame(frame_data)

    def _build_live_state(
        self,
        frame_id: int,
        timestamp: float,
        rel_state: SystemReliabilityState,
        detections_count: int,
        tracks_count: int,
        faces_count: int,
        loop_latency_ms: float,
    ) -> LiveState:
        """Assemble the authoritative LiveState snapshot."""
        # Counts
        line_counts = (
            self._line_crossing_counter.counts
            if self._line_crossing_counter is not None
            else None
        )
        current_headcount = (
            rel_state.last_reliable_count
            if rel_state.is_frozen
            else (self._occupancy_counter.last_state.current_count if self._occupancy_counter.last_state else 0)
        )

        counts_dict = {
            "current": current_headcount,
            "unique_session_approx": self._session_counter.approximate_unique_count,
            "entries": line_counts.entries if line_counts else 0,
            "exits": line_counts.exits if line_counts else 0,
            "net_count": line_counts.net_count if line_counts else 0,
        }

        # Occupancy
        occ_percent, cap_state = self._analytics_engine.calculate_occupancy(current_headcount)
        occupancy_dict = {
            "capacity": self._analytics_engine.capacity,
            "capacity_state": cap_state,
            "percent": occ_percent,
        }

        # Crowd
        ana_last = self._analytics_engine.last_state
        crowd_dict = {
            "level": ana_last.crowd_level.value if ana_last else "LOW",
            "raw_level": ana_last.raw_crowd_level.value if ana_last else "LOW",
            "trend": ana_last.trend.value if ana_last else "STABLE",
            "peak_count": ana_last.peak_count if ana_last else 0,
            "peak_occupancy_percent": ana_last.peak_occupancy_percent if ana_last else None,
            "peak_timestamp": ana_last.peak_timestamp if ana_last else None,
        }

        self._last_state = LiveState(
            schema_version=1,
            timestamp=timestamp,
            session_id=self._config.session_id,
            frame_id=frame_id,
            system_state=rel_state.system_state,
            camera_state=rel_state.camera_state,
            performance=rel_state.performance,
            vision=rel_state.vision,
            counts=counts_dict,
            occupancy=occupancy_dict,
            crowd=crowd_dict,
            alerts=self._alert_engine.active_alerts,
            status_reason=rel_state.status_reason,
            is_healthy=rel_state.is_healthy,
            is_frozen=rel_state.is_frozen,
            detections_count=detections_count,
            tracks_count=tracks_count,
            faces_count=faces_count,
        )
        return self._last_state

    def reset_session(self) -> None:
        """Reset tracking, counting, analytics, alerts, scene analysis, and reliability state."""
        self._tracking_adapter.reset()
        self._occupancy_counter.reset()
        self._session_counter.reset()
        if self._line_crossing_counter is not None:
            self._line_crossing_counter.reset()
        self._scene_analyzer.reset_history()
        self._analytics_engine.reset()
        self._alert_engine.reset()
        self._reliability_manager.reset()
        self._frame_sequence = 0
        self._last_state = None

    @property
    def scene_analyzer(self) -> SceneAnalyzer:
        """Access the underlying SceneAnalyzer engine."""
        return self._scene_analyzer
