"""Live hardware smoke test: CameraSource → YOLO26n PersonDetector.

Runs the integrated pipeline on the real webcam with the real ONNX model on
the RTX 5060 GPU.  Reports per-frame metrics for 30 frames then exits.

Usage:
    python scratch/smoke_camera_detector.py
"""

from __future__ import annotations

import os
import sys
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.camera import CameraSource
from visionqueue.camera.types import CameraConfig
from visionqueue.detection import PersonDetector, DetectorConfig


def main():
    N_FRAMES = 30

    model_path = os.path.join(PROJECT_ROOT, "models", "yolo26n.onnx")
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    # ── Init detector (loads model once) ────────────────────────────
    print("Loading YOLO26n ONNX model...")
    detector = PersonDetector(DetectorConfig(model_path=model_path))
    print(f"  Provider : {detector.active_provider}")
    print(f"  GPU      : {detector.is_gpu}")
    print(f"  Input    : {detector.input_shape}")

    # Warm-up GPU kernels
    detector.warm_up(rounds=5)

    # ── Init camera ─────────────────────────────────────────────────
    cam_cfg = CameraConfig(source=0, width=640, height=480)
    cam = CameraSource(cam_cfg)

    print(f"\nOpening webcam (source={cam_cfg.source})...")
    cam.start()
    print(f"  Resolution : {cam.dimensions}")
    print(f"  Reported FPS: {cam.reported_fps:.1f}")

    # ── Run integrated pipeline ─────────────────────────────────────
    print(f"\n{'Frame':>6}  {'Persons':>7}  {'Infer ms':>9}  {'Total ms':>9}  {'Confs'}")
    print("-" * 60)

    total_infer_ms = 0.0
    total_pipeline_ms = 0.0
    total_persons = 0

    for i in range(N_FRAMES):
        frame_data = cam.get_latest_frame(wait_new=True, timeout=2.0)
        if frame_data is None:
            print(f"  [{i:3d}] No frame (timeout or stopped)")
            continue

        t0 = time.perf_counter()
        detections, infer_ms = detector.detect_timed(frame_data.frame)
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        total_infer_ms += infer_ms
        total_pipeline_ms += pipeline_ms
        n = len(detections)
        total_persons += n

        confs = ", ".join(f"{d.confidence:.2f}" for d in detections[:5])
        print(f"  {i:4d}    {n:5d}    {infer_ms:7.2f}    {pipeline_ms:7.2f}    [{confs}]")

    cam.stop()

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"  Frames processed   : {N_FRAMES}")
    print(f"  Provider           : {detector.active_provider}")
    print(f"  Avg inference      : {total_infer_ms / N_FRAMES:.2f} ms")
    print(f"  Avg pipeline       : {total_pipeline_ms / N_FRAMES:.2f} ms")
    print(f"  Total persons seen : {total_persons}")
    print(f"  Avg persons/frame  : {total_persons / N_FRAMES:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
