"""SimpleDisplay: a thin local display loop on top of an existing CVService.

SimpleDisplay does NOT run its own CV loop, capture its own camera, or
maintain its own counters. It only:

1. Pulls the latest BGR frame from the existing CameraSource (the same one
   the pipeline is using) so the picture on screen is what the pipeline sees.
2. Pulls the cached LiveState dict + active tracks from the existing
   CVService (the same one the FastAPI backend serves over WebSocket).
3. Hands both to visionqueue.viewer.hud.draw_simple_hud() and shows the
   result in an OpenCV window.

Lifecycle:
- start()  : launch the display worker thread (window NOT created here — must
              be called from the same thread that will call wait()).
- wait()   : create the window, pump the Windows/OpenCV event loop on the
              calling thread, destroy the window on exit.
- stop()   : signal the worker thread to exit.

Thread architecture (Windows-safe):
- Window creation (namedWindow), imshow(), and the Windows message pump
  (waitKey) ALL run on the same thread (the thread that calls wait()).
- The display worker thread only prepares frames (read state + render HUD)
  and stores the result in a thread-safe slot.  The main thread's wait()
  loop picks up the latest prepared frame and calls imshow() + waitKey().
- In headless mode the same threading model applies; no window is created.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from visionqueue.api.service import CVService
from visionqueue.viewer.hud import draw_simple_hud, render_offline_card

logger = logging.getLogger(__name__)


class SimpleDisplay:
    """Local operator display driven by an existing CVService instance."""

    def __init__(
        self,
        service: CVService,
        window_name: str = "VisionQueue Live",
        headless: bool = False,
        width: int = 1024,
        height: int = 768,
        frame_poll_timeout: float = 0.5,
        on_exit_callback: Optional[callable] = None,
    ) -> None:
        """Initialize the display.

        Args:
            service: The production CVService that owns the pipeline and
                camera. The display only reads from this instance.
            window_name: OpenCV window title.
            headless: When True, do not open an OpenCV window. Frames are
                still produced (callable from tests) so the rendering logic
                can be validated without a display server.
            width: Initial window width (GUI mode only).
            height: Initial window height (GUI mode only).
            frame_poll_timeout: Max seconds to wait for a new frame per loop.
            on_exit_callback: Optional callable invoked when the user presses
                Q / ESC. The callback is responsible for initiating the full
                Launcher shutdown (e.g. calling Launcher.stop()). If not
                provided, the display loop exits silently on Q/ESC.
        """
        self._service = service
        self._window_name = window_name
        self._headless = headless
        self._width = width
        self._height = height
        self._frame_poll_timeout = frame_poll_timeout
        self._on_exit_callback = on_exit_callback

        # Actual display dimensions (updated from window size by main thread).
        self._display_w = width
        self._display_h = height

        self._stop_event = threading.Event()
        self._window_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_rendered: Optional[np.ndarray] = None
        self._render_count: int = 0

        # Thread-safe slot: the worker thread writes prepared frames here,
        # and the main thread (wait loop) reads them for imshow().
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_lock = threading.Lock()
        self._pending_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the display worker thread.

        On Windows the window must be created and the event pump must run on
        the SAME thread that calls wait().  Do NOT create the window here.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.info("SimpleDisplay.start() called but already running.")
            return
        self._stop_event.clear()
        self._window_ready.clear()
        self._pending_event.clear()
        self._thread = threading.Thread(target=self._loop, name="VisionQueue-Display", daemon=True)
        self._thread.start()
        if self._headless:
            self._window_ready.set()
        logger.info("SimpleDisplay worker started (headless=%s).", self._headless)

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the loop to exit. Window cleanup happens in wait() on the main thread."""
        self._stop_event.set()
        # Wake the worker if it's blocked on the pending event.
        self._pending_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("SimpleDisplay stopped.")

    def wait(self) -> None:
        """Pump the Windows/GUI event loop on the calling thread.

        MUST be called from the same thread that should own the window.
        On Windows this drives the Windows message pump via waitKey(), which
        is required for the window to redraw and stay responsive.

        This method also performs imshow() for each frame the worker thread
        has prepared, keeping all HighGUI calls on a single thread.

        Safe to call even when headless=True (waitKey returns immediately).
        """
        if self._thread is None:
            return

        if not self._headless:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._window_name, self._width, self._height)
            except cv2.error as exc:
                logger.warning("OpenCV window unavailable (%s); degrading to headless.", exc)
                self._headless = True

        self._window_ready.set()

        if self._headless:
            while not self._stop_event.is_set():
                time.sleep(0.05)
        else:
            while not self._stop_event.is_set():
                # Update display dimensions from actual window size.
                try:
                    rect = cv2.getWindowImageRect(self._window_name)
                    self._display_w = rect[2]
                    self._display_h = rect[3]
                except (cv2.error, AttributeError):
                    pass

                # Pick up the latest prepared frame from the worker thread.
                frame_to_show = None
                with self._pending_lock:
                    if self._pending_frame is not None:
                        frame_to_show = self._pending_frame

                if frame_to_show is not None:
                    cv2.imshow(self._window_name, frame_to_show)

                key = cv2.waitKey(1)
                if key in (ord('q'), ord('Q'), 27):
                    self._stop_event.set()
                    break

        if not self._headless:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Read-only state for tests
    # ------------------------------------------------------------------

    @property
    def last_rendered_frame(self) -> Optional[np.ndarray]:
        """The most recent frame the display produced (HUD-applied)."""
        return self._last_rendered

    @property
    def render_count(self) -> int:
        """Number of frames rendered since start(). Useful for tests."""
        return self._render_count

    # ------------------------------------------------------------------
    # Aspect-ratio-preserving frame transform
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox(
        frame: np.ndarray, target_w: int, target_h: int,
    ) -> Tuple[np.ndarray, float, int, int]:
        """Scale frame to fit *target_w* x *target_h* preserving aspect ratio.

        Uses letterboxing (horizontal bars) when the target is wider than the
        frame aspect, or pillarboxing (vertical bars) when taller.

        Returns:
            (padded_frame, scale, x_offset, y_offset) — the transform
            parameters needed to map original-frame coordinates onto the
            padded canvas.
        """
        fh, fw = frame.shape[:2]
        if fw <= 0 or fh <= 0 or target_w <= 0 or target_h <= 0:
            return frame, 1.0, 0, 0
        scale = min(target_w / fw, target_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        if new_w <= 0 or new_h <= 0:
            return frame, 1.0, 0, 0
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas, scale, x_off, y_off

    @staticmethod
    def _transform_tracks(
        tracks: List[Dict[str, Any]],
        scale: float,
        x_off: int,
        y_off: int,
    ) -> List[Dict[str, Any]]:
        """Map track bounding boxes through the letterbox transform."""
        if not tracks:
            return tracks
        transformed: List[Dict[str, Any]] = []
        for trk in tracks:
            bbox = trk.get("bbox")
            if bbox and len(bbox) == 4:
                new_trk = dict(trk)
                new_trk["bbox"] = [
                    bbox[0] * scale + x_off,
                    bbox[1] * scale + y_off,
                    bbox[2] * scale + x_off,
                    bbox[3] * scale + y_off,
                ]
                transformed.append(new_trk)
            else:
                transformed.append(trk)
        return transformed

    # ------------------------------------------------------------------
    # Main loop (runs on the display worker thread)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Background thread: reads state, renders HUD, and posts frames.

        The worker thread does ALL the slow work (reading camera frames,
        pulling LiveState, running the HUD draw).  The prepared frame is
        stored in _pending_frame for the main thread to pick up via
        imshow().
        """
        self._window_ready.wait(timeout=5.0)

        while not self._stop_event.is_set():
            frame, state, tracks = self._read_state()
            try:
                rendered = self._render(frame, state, tracks)
            except Exception:
                logger.exception("SimpleDisplay render failure")
                time.sleep(0.05)
                continue

            self._last_rendered = rendered
            self._render_count += 1

            # Publish the prepared frame for the main thread's imshow().
            with self._pending_lock:
                self._pending_frame = rendered
            self._pending_event.set()

            time.sleep(0.01)

    def run(self, timeout: Optional[float] = None) -> None:
        """Blocking convenience: start the worker and pump the event loop.

        Returns when stop_event is set, timeout expires, or Q/ESC is pressed.
        """
        self.start()
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.is_running() and not self._stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                self.stop()
                break
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Internal: read state from the CVService
    # ------------------------------------------------------------------

    def _read_state(self) -> tuple[Optional[np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
        """Pull the latest frame + LiveState + tracks from the CVService."""
        state: Dict[str, Any] = self._service.get_latest_state() or {}
        tracks: List[Dict[str, Any]] = self._service.get_latest_tracks() or []

        frame: Optional[np.ndarray] = None
        camera = getattr(self._service.pipeline, "camera", None)
        if camera is not None:
            try:
                fd = camera.get_latest_frame(wait_new=True, timeout=self._frame_poll_timeout)
                if fd is not None and getattr(fd, "frame", None) is not None:
                    frame = fd.frame
            except Exception:
                logger.debug("Display: failed to fetch latest frame", exc_info=True)
        return frame, state, tracks

    def _render(
        self,
        frame: Optional[np.ndarray],
        state: Dict[str, Any],
        tracks: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Render a single annotated frame.

        When a camera frame is provided it is letterboxed to the current
        display-window dimensions *before* the HUD overlay so that the
        aspect ratio is preserved and bounding boxes stay correctly aligned.
        """
        if frame is None:
            rendered = render_offline_card(self._width, self._height, state or {})
            rh, rw = rendered.shape[:2]
            if rw != self._display_w or rh != self._display_h:
                rendered, _, _, _ = self._letterbox(rendered, self._display_w, self._display_h)
            return rendered

        target_w = self._display_w
        target_h = self._display_h
        fh, fw = frame.shape[:2]
        if fw != target_w or fh != target_h:
            frame, scale, x_off, y_off = self._letterbox(frame, target_w, target_h)
            tracks = self._transform_tracks(tracks or [], scale, x_off, y_off)

        return draw_simple_hud(frame, state or {}, tracks or [])
