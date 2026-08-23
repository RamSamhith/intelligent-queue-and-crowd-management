"""Type definitions and data models for the VisionQueue camera subsystem."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np


class SourceState(str, Enum):
    """Lifecycle states of the camera source."""
    INITIALIZING = "INITIALIZING"
    OPENED = "OPENED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    DISCONNECTED = "DISCONNECTED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for video capture source.
    
    Attributes:
        source: Camera device index (e.g. 0), video file path, or stream URL (RTSP/HTTP).
        width: Requested frame width (optional).
        height: Requested frame height (optional).
        fps: Requested frame rate (optional).
        api_preference: OpenCV capture backend (e.g. cv2.CAP_ANY, cv2.CAP_DSHOW).
        buffer_size: OpenCV internal frame buffer capacity (default 1 for minimum latency).
        max_consecutive_errors: Number of consecutive frame read failures before marking disconnected.
        reconnect_interval_sec: Seconds to wait before attempting reconnection when disconnected.
        max_reconnect_attempts: Maximum reconnection retries (0 to disable auto-reconnect).
        loop_video: When True and source is a video file, automatically restart at EOF.
    """
    source: Union[int, str] = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    api_preference: int = 0  # cv2.CAP_ANY
    buffer_size: int = 1
    max_consecutive_errors: int = 10
    reconnect_interval_sec: float = 2.0
    max_reconnect_attempts: int = 5
    loop_video: bool = False


@dataclass(frozen=True)
class FrameData:
    """Immutable container for an acquired frame and its metadata.
    
    Attributes:
        frame: The raw image frame as a BGR numpy array (H, W, C), uint8.
        timestamp: Acquisition timestamp (in seconds, from time.perf_counter() or time.time()).
        frame_id: Monotonically increasing 0-indexed frame sequence number.
        width: Frame width in pixels.
        height: Frame height in pixels.
        source_state: Current state of the video capture source.
        fps: Measured capture frame rate at the moment of acquisition.
    """
    frame: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int
    source_state: SourceState
    fps: float = 0.0

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Return (width, height) tuple."""
        return self.width, self.height

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return numpy shape (H, W, C)."""
        return self.frame.shape
