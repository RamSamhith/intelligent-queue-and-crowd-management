"""Threaded real-time camera capture module with bounded latest-frame semantics.

Designed for high-throughput, low-latency computer vision pipelines. Ensures that
downstream CV processing always receives the freshest available frame, dropping stale
frames automatically when processing latency exceeds acquisition intervals.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

from visionqueue.camera.types import CameraConfig, FrameData, SourceState

logger = logging.getLogger(__name__)


class CameraSource:
    """Threaded OpenCV VideoCapture source with zero-copy/latest-frame buffer."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        """Initialize camera source with optional configuration.
        
        Args:
            config: CameraConfig instance. If None, default CameraConfig(source=0) is used.
        """
        self._config = config or CameraConfig()
        self._state = SourceState.INITIALIZING
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Thread synchronization & frame storage
        self._lock = threading.Lock()
        self._new_frame_condition = threading.Condition(self._lock)
        self._latest_frame: Optional[FrameData] = None
        self._last_served_frame_id: int = -1
        self._frame_sequence: int = 0
        
        # Background worker control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance & telemetry
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._reported_fps: float = 0.0
        self._measured_fps: float = 0.0
        self._last_frame_time: float = 0.0
        self._fps_alpha: float = 0.1  # Exponential moving average smoothing
        self._consecutive_errors: int = 0
        self._reconnect_count: int = 0

    @property
    def config(self) -> CameraConfig:
        """Return the active camera configuration."""
        return self._config

    @property
    def state(self) -> SourceState:
        """Return the current source lifecycle state."""
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Return True if the source is actively capturing frames."""
        with self._lock:
            return self._state == SourceState.RUNNING and not self._stop_event.is_set()

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Return (width, height) of the captured stream."""
        return self._actual_width, self._actual_height

    @property
    def measured_fps(self) -> float:
        """Return real-time measured acquisition FPS."""
        with self._lock:
            return self._measured_fps

    @property
    def reported_fps(self) -> float:
        """Return the FPS reported by OpenCV hardware properties."""
        return self._reported_fps

    def start(self) -> CameraSource:
        """Open the video capture source and start the background acquisition thread.
        
        Returns:
            self for chaining.
            
        Raises:
            RuntimeError: If the capture source cannot be opened and no reconnect is possible.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("CameraSource is already started and running.")
            return self

        self._stop_event.clear()
        
        # Attempt initial hardware connection
        if not self._open_device():
            with self._lock:
                self._state = SourceState.ERROR
            raise RuntimeError(
                f"Failed to open video source: '{self._config.source}' "
                f"(api_preference={self._config.api_preference})"
            )

        with self._lock:
            self._state = SourceState.RUNNING

        # Spawn background capture worker
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"VisionQueue-CaptureThread-{self._config.source}",
            daemon=True
        )
        self._thread.start()
        logger.info(
            "CameraSource started on source '%s' (%dx%d @ ~%.1f FPS)",
            self._config.source,
            self._actual_width,
            self._actual_height,
            self._reported_fps
        )
        return self

    def stop(self, timeout: float = 3.0) -> None:
        """Stop frame acquisition and release hardware resources.
        
        Args:
            timeout: Maximum seconds to wait for background thread termination.
        """
        self._stop_event.set()
        
        with self._lock:
            self._state = SourceState.STOPPED
            self._new_frame_condition.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Capture thread did not exit cleanly within timeout.")
            self._thread = None

        self._close_device()
        with self._lock:
            self._state = SourceState.STOPPED
        logger.info("CameraSource stopped and released.")

    def release(self) -> None:
        """Alias for stop()."""
        self.stop()

    def get_latest_frame(self, wait_new: bool = True, timeout: Optional[float] = 1.0) -> Optional[FrameData]:
        """Retrieve the newest available frame.
        
        If downstream processing is slower than camera acquisition, stale frames are
        automatically dropped, ensuring zero pipeline latency lag.
        
        Args:
            wait_new: If True, blocks until a frame newer than the last retrieved one arrives.
                      If False, returns current latest frame immediately without blocking.
            timeout: Maximum seconds to wait when wait_new is True (None for infinite).
            
        Returns:
            FrameData container if available, None if source is stopped or timeout expires.
        """
        with self._new_frame_condition:
            if not wait_new:
                return self._latest_frame

            start_time = time.perf_counter()
            while not self._stop_event.is_set():
                if self._latest_frame is not None and self._latest_frame.frame_id > self._last_served_frame_id:
                    self._last_served_frame_id = self._latest_frame.frame_id
                    return self._latest_frame

                if self._state in (SourceState.STOPPED, SourceState.ERROR):
                    return None

                elapsed = time.perf_counter() - start_time
                if timeout is not None:
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return None
                    self._new_frame_condition.wait(timeout=remaining)
                else:
                    self._new_frame_condition.wait()

            return None

    def read(self, timeout: Optional[float] = 1.0) -> Tuple[bool, Optional[FrameData]]:
        """OpenCV-style read method returning (success, FrameData)."""
        frame_data = self.get_latest_frame(wait_new=True, timeout=timeout)
        return (frame_data is not None), frame_data

    def frames(self, timeout: Optional[float] = 1.0) -> Iterator[FrameData]:
        """Generator yielding latest frames continuously until stopped."""
        while self.is_running:
            frame_data = self.get_latest_frame(wait_new=True, timeout=timeout)
            if frame_data is None:
                if self.state in (SourceState.STOPPED, SourceState.ERROR):
                    break
                continue
            yield frame_data

    def __enter__(self) -> CameraSource:
        """Context manager support."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def _open_device(self) -> bool:
        """Open and configure OpenCV VideoCapture instance."""
        self._close_device()

        source = self._config.source
        api = self._config.api_preference

        # Auto-detect Windows DirectShow for integer webcams if not explicitly set
        if isinstance(source, int) and api == 0 and os.name == "nt":
            api = cv2.CAP_DSHOW

        try:
            if api != 0:
                self._cap = cv2.VideoCapture(source, api)
            else:
                self._cap = cv2.VideoCapture(source)
        except Exception as exc:
            logger.error("Exception opening VideoCapture(%s): %s", source, exc)
            return False

        if self._cap is None or not self._cap.isOpened():
            logger.error("VideoCapture could not open source: %s", source)
            return False

        # Apply buffer size if supported by backend
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._config.buffer_size)
        except Exception:
            pass

        # Apply requested dimensions
        if self._config.width is not None and self._config.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        if self._config.height is not None and self._config.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        if self._config.fps is not None and self._config.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, float(self._config.fps))

        # Query actual operational properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self._reported_fps <= 0 or np.isnan(self._reported_fps):
            self._reported_fps = 30.0  # Fallback assumption

        self._consecutive_errors = 0
        return True

    def _close_device(self) -> None:
        """Safely release VideoCapture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.debug("Exception releasing VideoCapture: %s", exc)
            self._cap = None

    def _is_file_source(self) -> bool:
        """Return True if source is a local video file path."""
        if isinstance(self._config.source, str):
            if os.path.isfile(self._config.source):
                return True
            exts = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm")
            return self._config.source.lower().endswith(exts)
        return False

    def _capture_loop(self) -> None:
        """Background thread loop continuously capturing frames into single-slot buffer."""
        is_file = self._is_file_source()

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._handle_disconnect():
                    break
                continue

            grabbed, frame = self._cap.read()
            now = time.time()

            if not grabbed or frame is None:
                self._consecutive_errors += 1
                
                # If reading from a video file: handle loop or EOF
                if is_file:
                    if self._config.loop_video:
                        logger.debug("Looping video source at EOF.")
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._consecutive_errors = 0
                        continue
                    else:
                        logger.info("Reached EOF for video source: %s", self._config.source)
                        with self._lock:
                            self._state = SourceState.STOPPED
                            self._new_frame_condition.notify_all()
                        break

                # For live streams / webcams: check error threshold
                if self._consecutive_errors >= self._config.max_consecutive_errors:
                    logger.warning(
                        "Source '%s' exceeded max consecutive frame errors (%d).",
                        self._config.source,
                        self._consecutive_errors
                    )
                    with self._lock:
                        self._state = SourceState.DISCONNECTED
                    
                    if not self._handle_disconnect():
                        break
                else:
                    time.sleep(0.005)
                continue

            # Frame read succeeded
            self._consecutive_errors = 0

            # Calculate measured FPS (exponential moving average)
            if self._last_frame_time > 0.0:
                dt = now - self._last_frame_time
                if dt > 0.0:
                    instantaneous_fps = 1.0 / dt
                    if self._measured_fps == 0.0:
                        self._measured_fps = instantaneous_fps
                    else:
                        self._measured_fps = (self._fps_alpha * instantaneous_fps) + (
                            (1.0 - self._fps_alpha) * self._measured_fps
                        )
            self._last_frame_time = now

            h, w = frame.shape[:2]
            self._actual_width = w
            self._actual_height = h

            frame_data = FrameData(
                frame=frame,
                timestamp=now,
                frame_id=self._frame_sequence,
                width=w,
                height=h,
                source_state=SourceState.RUNNING,
                fps=self._measured_fps
            )
            self._frame_sequence += 1

            # Atomic update of single-slot latest frame
            with self._new_frame_condition:
                if not self._stop_event.is_set():
                    self._latest_frame = frame_data
                    self._state = SourceState.RUNNING
                    self._new_frame_condition.notify_all()

    def _handle_disconnect(self) -> bool:
        """Attempt reconnection if enabled in configuration."""
        if self._config.max_reconnect_attempts <= 0:
            with self._lock:
                self._state = SourceState.ERROR
                self._new_frame_condition.notify_all()
            return False

        if self._reconnect_count >= self._config.max_reconnect_attempts:
            logger.error("Max reconnection attempts (%d) exceeded.", self._config.max_reconnect_attempts)
            with self._lock:
                self._state = SourceState.ERROR
                self._new_frame_condition.notify_all()
            return False

        self._reconnect_count += 1
        logger.info(
            "Attempting reconnection %d/%d for source '%s' in %.1fs...",
            self._reconnect_count,
            self._config.max_reconnect_attempts,
            self._config.source,
            self._config.reconnect_interval_sec
        )
        
        # Sleep in small slices to respect stop_event
        sleep_elapsed = 0.0
        while sleep_elapsed < self._config.reconnect_interval_sec and not self._stop_event.is_set():
            time.sleep(0.1)
            sleep_elapsed += 0.1

        if self._stop_event.is_set():
            return False

        if self._open_device():
            logger.info("Reconnection succeeded for source '%s'.", self._config.source)
            self._reconnect_count = 0
            self._consecutive_errors = 0
            self._last_frame_time = 0.0
            with self._lock:
                self._state = SourceState.RUNNING
            return True

        return True
