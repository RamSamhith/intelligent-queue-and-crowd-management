import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from visionqueue.camera import CameraConfig, CameraSource, SourceState


def main():
    print("==================================================")
    print("VISIONQUEUE — REAL WEBCAM SMOKE TEST (DEVICE 0)")
    print("==================================================")

    # 1. Initialize camera source on device 0
    config = CameraConfig(
        source=0,
        api_preference=cv2.CAP_DSHOW,  # Windows DirectShow for fast init
        buffer_size=1
    )
    print(f"Opening CameraSource with config: {config}")

    try:
        cam = CameraSource(config)
        cam.start()
    except Exception as exc:
        print(f"Failed to open webcam index 0: {exc}")
        print("Note: Webcam might be in use or unavailable on this system.")
        return

    print("Camera successfully opened and background capture thread started.")
    print(f"Source state: {cam.state}")
    print(f"Hardware reported dimensions: {cam.dimensions} (WxH)")
    print(f"Hardware reported FPS: {cam.reported_fps:.1f}")

    # 2. Acquire real frames and measure arrival behavior
    print("\nAcquiring live frames for 2.0 seconds...")
    frames_acquired = 0
    start_time = time.perf_counter()
    intervals = []
    frame_ages = []
    last_ts = None
    last_frame_id = -1

    while (time.perf_counter() - start_time) < 2.0:
        fetch_start = time.perf_counter()
        frame_data = cam.get_latest_frame(wait_new=True, timeout=1.0)
        fetch_end = time.perf_counter()
        
        if frame_data is None:
            continue
            
        frames_acquired += 1
        age_ms = (fetch_end - frame_data.timestamp) * 1000.0
        frame_ages.append(age_ms)
        
        if last_ts is not None:
            intervals.append((frame_data.timestamp - last_ts) * 1000.0)
        last_ts = frame_data.timestamp
        last_frame_id = frame_data.frame_id

    total_duration = time.perf_counter() - start_time
    measured_fps = frames_acquired / total_duration if total_duration > 0 else 0.0

    print(f"\nLive Capture Telemetry:")
    print(f"- Total Frames Acquired: {frames_acquired}")
    print(f"- Test Duration: {total_duration:.2f} s")
    print(f"- Actual Capture FPS: {measured_fps:.2f}")
    if intervals:
        print(f"- Mean Frame Interval: {np.mean(intervals):.2f} ms (std={np.std(intervals):.2f} ms)")
    if frame_ages:
        print(f"- Mean Frame Age / Fetch Latency: {np.mean(frame_ages):.2f} ms (min={np.min(frame_ages):.2f} ms, max={np.max(frame_ages):.2f} ms)")

    # 3. Test latest-frame dropping behavior with simulated slow consumer
    print("\nTesting Latest-Frame Dropping Behavior:")
    print("Simulating slow consumer (sleeping 200ms between gets)...")
    
    slow_ids = []
    for _ in range(5):
        time.sleep(0.2)  # Simulate slow downstream CV inference
        fd = cam.get_latest_frame(wait_new=False)
        if fd is not None:
            slow_ids.append(fd.frame_id)
            print(f"  Retrieved latest Frame ID: {fd.frame_id} (Width={fd.width}, Height={fd.height})")

    frame_jumps = [slow_ids[i] - slow_ids[i-1] for i in range(1, len(slow_ids))]
    print(f"  Frame ID jumps during slow consumption: {frame_jumps}")
    assert all(j > 1 for j in frame_jumps), "Latest-frame buffer failed to discard stale intermediate frames!"
    print("  Latest-frame single-slot buffer behavior VERIFIED: Zero stale frame accumulation.")

    # 4. Clean shutdown
    print("\nStopping camera and releasing hardware...")
    cam.stop()
    print(f"Source state after stop: {cam.state}")
    assert cam.state == SourceState.STOPPED
    print("Clean shutdown VERIFIED.")

    print("\n==================================================")
    print("SMOKE TEST SUMMARY")
    print("==================================================")
    print(f"Device: Camera Index 0")
    print(f"Resolution: {cam.dimensions[0]}x{cam.dimensions[1]}")
    print(f"Measured Capture FPS: {measured_fps:.2f}")
    print(f"Latest-Frame Dropping: VERIFIED")
    print(f"State Machine Lifecycle: VERIFIED")
    print("Webcam Smoke Test: SUCCESSFUL")


if __name__ == "__main__":
    main()
