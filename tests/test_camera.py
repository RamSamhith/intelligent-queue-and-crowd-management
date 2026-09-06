"""Unit and integration tests for VisionQueue Camera subsystem."""

import os
import tempfile
import time
from typing import Generator
import cv2
import numpy as np
import pytest

from visionqueue.camera import CameraConfig, CameraSource, FrameData, SourceState


@pytest.fixture
def synthetic_video_path() -> Generator[str, None, None]:
    """Create a temporary 30-frame synthetic video file fixture."""
    temp_dir = tempfile.mkdtemp()
    video_file = os.path.join(temp_dir, "test_synth.avi")
    
    width, height = 320, 240
    fps = 30.0
    num_frames = 30
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create synthetic frame with frame index drawn
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [(i * 8) % 255, 100, 200]
        cv2.putText(frame, f"F{i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        
    writer.release()
    yield video_file
    
    # Cleanup
    if os.path.exists(video_file):
        try:
            os.remove(video_file)
            os.rmdir(temp_dir)
        except Exception:
            pass


def test_camera_config_defaults():
    """Verify CameraConfig default values."""
    config = CameraConfig()
    assert config.source == 0
    assert config.width is None
    assert config.height is None
    assert config.buffer_size == 1
    assert config.loop_video is False


def test_frame_data_properties():
    """Verify FrameData container properties and immutability."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fd = FrameData(
        frame=dummy_frame,
        timestamp=100.0,
        frame_id=42,
        width=640,
        height=480,
        source_state=SourceState.RUNNING,
        fps=30.0
    )
    assert fd.width == 640
    assert fd.height == 480
    assert fd.dimensions == (640, 480)
    assert fd.shape == (480, 640, 3)
    assert fd.frame_id == 42
    assert fd.source_state == SourceState.RUNNING


def test_camera_source_opens_and_acquires_frames(synthetic_video_path: str):
    """Verify CameraSource successfully opens a video source and acquires valid frames."""
    config = CameraConfig(source=synthetic_video_path, loop_video=True)
    cam = CameraSource(config)
    
    assert cam.state == SourceState.INITIALIZING
    cam.start()
    assert cam.is_running
    assert cam.state == SourceState.RUNNING
    
    # Read first frame
    success, frame_data = cam.read(timeout=2.0)
    assert success is True
    assert frame_data is not None
    assert isinstance(frame_data.frame, np.ndarray)
    assert frame_data.width == 320
    assert frame_data.height == 240
    assert frame_data.dimensions == (320, 240)
    assert frame_data.frame.shape == (240, 320, 3)
    assert frame_data.timestamp > 0.0
    assert frame_data.frame_id >= 0
    
    # Read subsequent frame
    success2, frame_data2 = cam.read(timeout=2.0)
    assert success2 is True
    assert frame_data2 is not None
    assert frame_data2.frame_id > frame_data.frame_id
    
    cam.stop()
    assert not cam.is_running
    assert cam.state == SourceState.STOPPED


def test_camera_source_context_manager(synthetic_video_path: str):
    """Verify context manager automatically starts and stops the capture source."""
    config = CameraConfig(source=synthetic_video_path, loop_video=True)
    with CameraSource(config) as cam:
        assert cam.is_running
        frame_data = cam.get_latest_frame(wait_new=True, timeout=2.0)
        assert frame_data is not None
        assert frame_data.width == 320
        
    assert not cam.is_running
    assert cam.state == SourceState.STOPPED


def test_latest_frame_dropping_behavior(synthetic_video_path: str):
    """Verify that slow consumer drops old frames and always receives the latest frame."""
    config = CameraConfig(source=synthetic_video_path, loop_video=True)
    with CameraSource(config) as cam:
        # Get first frame
        f1 = cam.get_latest_frame(wait_new=True, timeout=2.0)
        assert f1 is not None
        initial_id = f1.frame_id
        
        # Simulate slow consumer (sleep while capture thread runs ahead)
        time.sleep(0.3)  # Capture thread produces ~9 frames in 300ms at 30fps
        
        # Get next latest frame
        f2 = cam.get_latest_frame(wait_new=True, timeout=2.0)
        assert f2 is not None
        # Consumer should have skipped intermediate frames, so frame_id jumps significantly
        assert f2.frame_id > initial_id + 1, f"Expected dropped frames, got f1={initial_id}, f2={f2.frame_id}"


def test_video_eof_handling_without_loop(synthetic_video_path: str):
    """Verify that video file reaches EOF and cleanly transitions to STOPPED state."""
    config = CameraConfig(source=synthetic_video_path, loop_video=False)
    cam = CameraSource(config)
    cam.start()
    
    # Read all 30 frames
    frames_read = 0
    start = time.perf_counter()
    while cam.is_running and (time.perf_counter() - start < 3.0):
        fd = cam.get_latest_frame(wait_new=True, timeout=0.5)
        if fd is not None:
            frames_read += 1
            
    assert frames_read > 0
    # Eventually reaches EOF and stops
    time.sleep(0.2)
    assert cam.state == SourceState.STOPPED or not cam.is_running
    cam.stop()


def test_unavailable_source_handling():
    """Verify that attempting to open an invalid source raises RuntimeError and sets ERROR state."""
    invalid_path = "C:/non_existent_path_to_video_12345.mp4"
    config = CameraConfig(source=invalid_path, max_reconnect_attempts=0)
    cam = CameraSource(config)

    with pytest.raises(RuntimeError, match="Failed to open video source"):
        cam.start()

    assert cam.state == SourceState.ERROR
    assert not cam.is_running
    cam.stop()


def test_reconnect_success_resets_last_frame_time_and_counter():
    """After a successful reconnect, _last_frame_time and _reconnect_count must be reset.

    Frame IDs (_frame_sequence) must remain monotonic across reconnects because
    ByteTrack and downstream counting semantics depend on monotonic frame IDs.
    Resetting them would cause track pileup.
    """
    config = CameraConfig(source="synthetic_path.avi", loop_video=True,
                          max_reconnect_attempts=3, reconnect_interval_sec=0.1)
    cam = CameraSource(config)

    # Simulate pre-reconnect state
    cam._reconnect_count = 2
    cam._last_frame_time = time.time() - 30.0  # 30s ago — would cause FPS spike if not reset
    cam._measured_fps = 25.0
    cam._frame_sequence = 100

    # Patch _open_device to succeed
    cam._open_device = lambda: True

    result = cam._handle_disconnect()
    assert result is True
    assert cam._reconnect_count == 0, "Reconnect counter must reset to 0 on success"
    assert cam._last_frame_time == 0.0, (
        "Last-frame-time must reset to prevent artificial FPS spike on first new frame"
    )
    # _frame_sequence is NOT reset (monotonic across reconnects is required by ByteTrack)
    assert cam._frame_sequence == 100, (
        "Frame IDs must remain monotonic across reconnects — do NOT reset _frame_sequence"
    )
    # _measured_fps is intentionally not reset (EMA recovers naturally)


def test_reconnect_failure_does_not_reset_state():
    """When _handle_disconnect returns False (stop_event set), state must remain untouched."""
    config = CameraConfig(source="synthetic_path.avi", loop_video=True,
                          max_reconnect_attempts=3, reconnect_interval_sec=0.0)
    cam = CameraSource(config)

    cam._reconnect_count = 2
    cam._last_frame_time = time.time() - 30.0
    cam._frame_sequence = 50

    # Set stop event so reconnect is aborted
    cam._stop_event.set()

    result = cam._handle_disconnect()
    assert result is False
    # Counter may have been incremented but state cleanup must not run
    assert cam._last_frame_time != 0.0, (
        "On failed/aborted reconnect, _last_frame_time must not be reset"
    )
