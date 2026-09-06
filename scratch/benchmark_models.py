"""Side-by-side benchmark comparing YOLO26n and YOLO26m on NVIDIA RTX 5060."""

from __future__ import annotations

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig


def run_benchmark(model_path: str, model_name: str, test_frames: list[np.ndarray], warmup_rounds: int = 20, iterations: int = 100):
    print(f"\n==================================================")
    print(f" BENCHMARKING: {model_name} ({model_path})")
    print(f"==================================================")
    
    cfg = DetectorConfig(model_path=model_path, confidence_threshold=0.25)
    detector = PersonDetector(cfg)
    print(f"Active Provider : {detector.active_provider}")
    print(f"Is GPU          : {detector.is_gpu}")
    print(f"Input shape     : {detector.input_shape}")
    
    # Warmup
    print(f"Warming up ({warmup_rounds} rounds)...")
    detector.warm_up(warmup_rounds)
    for i in range(warmup_rounds):
        frame = test_frames[i % len(test_frames)]
        detector.detect_timed(frame)
    
    # Timed runs
    latencies = []
    detection_counts = []
    
    # Measure memory before/after if CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        mem_allocated_mb = 0.0
        mem_reserved_mb = 0.0

    print(f"Running {iterations} timed inference iterations...")
    for i in range(iterations):
        frame = test_frames[i % len(test_frames)]
        dets, lat_ms = detector.detect_timed(frame)
        latencies.append(lat_ms)
        detection_counts.append(len(dets))
    
    latencies = np.array(latencies)
    avg_lat = np.mean(latencies)
    p50_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps_est = 1000.0 / avg_lat

    print("\n--- RESULTS ---")
    print(f"Average Latency : {avg_lat:.2f} ms")
    print(f"p50 Latency     : {p50_lat:.2f} ms")
    print(f"p95 Latency     : {p95_lat:.2f} ms")
    print(f"Min Latency     : {min_lat:.2f} ms")
    print(f"Max Latency     : {max_lat:.2f} ms")
    print(f"Inference FPS   : {fps_est:.1f} FPS")
    print(f"Avg Detections  : {np.mean(detection_counts):.1f}")
    
    return {
        "model": model_name,
        "path": model_path,
        "provider": detector.active_provider,
        "is_gpu": detector.is_gpu,
        "avg_ms": avg_lat,
        "p50_ms": p50_lat,
        "p95_ms": p95_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "fps": fps_est,
        "avg_detections": np.mean(detection_counts),
    }


def main():
    # Generate realistic test frames (including synthetic human-like shapes, blank, random noise)
    test_frames = []
    # 1. Blank frame
    test_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    # 2. Noise frame
    np.random.seed(42)
    test_frames.append(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    # 3. Geometric frame with multiple rectangles/ovals
    geo_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    import cv2
    cv2.rectangle(geo_frame, (100, 100), (250, 400), (50, 70, 180), -1)
    cv2.circle(geo_frame, (175, 80), 30, (200, 180, 150), -1)
    cv2.rectangle(geo_frame, (350, 150), (450, 420), (80, 120, 60), -1)
    cv2.circle(geo_frame, (400, 120), 25, (210, 190, 160), -1)
    test_frames.append(geo_frame)

    # 4. If webcam is available, capture a real live frame
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            test_frames.append(frame)
            print("Captured real frame from webcam for benchmark.")
        cap.release()

    res_n = run_benchmark("models/yolo26n.onnx", "YOLO26n", test_frames, warmup_rounds=20, iterations=100)
    res_m = run_benchmark("models/yolo26m.onnx", "YOLO26m", test_frames, warmup_rounds=20, iterations=100)

    print("\n" + "=" * 70)
    print(" SUMMARY BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} | {'YOLO26n (Baseline)':<18} | {'YOLO26m (New)':<18}")
    print("-" * 70)
    print(f"{'Execution Provider':<25} | {res_n['provider']:<18} | {res_m['provider']:<18}")
    print(f"{'Is GPU Accelerated':<25} | {str(res_n['is_gpu']):<18} | {str(res_m['is_gpu']):<18}")
    print(f"{'Average Latency (ms)':<25} | {res_n['avg_ms']:<18.2f} | {res_m['avg_ms']:<18.2f}")
    print(f"{'p50 Latency (ms)':<25} | {res_n['p50_ms']:<18.2f} | {res_m['p50_ms']:<18.2f}")
    print(f"{'p95 Latency (ms)':<25} | {res_n['p95_ms']:<18.2f} | {res_m['p95_ms']:<18.2f}")
    print(f"{'Min Latency (ms)':<25} | {res_n['min_ms']:<18.2f} | {res_m['min_ms']:<18.2f}")
    print(f"{'Max Latency (ms)':<25} | {res_n['max_ms']:<18.2f} | {res_m['max_ms']:<18.2f}")
    print(f"{'Detector FPS (GPU)':<25} | {res_n['fps']:<18.1f} | {res_m['fps']:<18.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
