"""Real Hardware Smoke Test for VisionQueue Unified CV Pipeline.

Runs the complete pipeline:
CameraSource (Real Webcam index 0)
  -> YOLO26n (ONNX Runtime CUDA EP on RTX 5060)
  -> Tracking (ByteTrack)
  -> ROI
  -> Counting
  -> Analytics
  -> Alerts
  -> Reliability
  -> LiveState

Measures and logs actual real-time telemetry over a bounded test duration.
"""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import argparse
import time
import cv2
import numpy as np

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import AnalyticsConfig, CrowdThresholds
from visionqueue.camera.capture import CameraSource
from visionqueue.camera.types import CameraConfig
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import ReliabilityConfig
from visionqueue.roi.types import ROIConfig


def main():
    parser = argparse.ArgumentParser(description="VisionQueue Real Hardware Pipeline Smoke Test")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default="models/yolo26m.onnx", help="ONNX model path (default: models/yolo26m.onnx)")
    parser.add_argument("--frames", type=int, default=100, help="Max frames to process (default: 100)")
    parser.add_argument("--display", action="store_true", help="Show visualization window")
    args = parser.parse_args()

    print("\n========================================================")
    print(" VisionQueue Live Hardware Pipeline Smoke Test")
    print("========================================================")

    # 1. Pipeline Configuration
    pipeline_config = CVPipelineConfig(
        camera=CameraConfig(source=args.source, buffer_size=1),
        detector=DetectorConfig(
            model_path=args.model,
            confidence_threshold=0.25,
        ),
        roi=ROIConfig(x=50, y=50, width=540, height=380),
        virtual_line=VirtualLine(pt1=(320.0, 50.0), pt2=(320.0, 430.0)),
        analytics=AnalyticsConfig(
            capacity=5,
            thresholds=CrowdThresholds(moderate_threshold=40.0, high_threshold=70.0, critical_threshold=90.0),
            debounce_frames=3,
        ),
        alerts=AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        ),
        reliability=ReliabilityConfig(min_starting_frames=3),
        enable_face_detection=False,
    )

    print(f"Loading Person detector ({args.model}) on ONNX Runtime...")
    detector = PersonDetector(pipeline_config.detector)
    print(f"Active Provider: {detector.active_provider} (GPU Accelerated: {detector.is_gpu})")
    print(f"Warming up CUDA kernels (3 rounds)...")
    detector.warm_up(3)

    camera = CameraSource(pipeline_config.camera)
    pipeline = CVPipeline(config=pipeline_config, camera=camera, detector=detector)

    print(f"Opening camera source {args.source}...")
    pipeline.start()

    print("\nExecuting live pipeline loop (target: %d frames)..." % args.frames)
    print("-" * 100)
    print(f"{'Frame':>5} | {'State':<9} | {'Count':>5} | {'Occ %':>5} | {'Crowd':<8} | {'YOLO ms':>7} | {'Loop ms':>7} | {'Age ms':>6} | {'FPS':>5}")
    print("-" * 100)

    latencies_yolo = []
    latencies_loop = []
    frame_ages = []
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while frame_count < args.frames:
            loop_t0 = time.perf_counter()
            state: LiveState = pipeline.step()
            loop_elapsed_ms = (time.perf_counter() - loop_t0) * 1000.0

            frame_count += 1
            infer_ms = state.performance.inference_latency_ms
            age_ms = state.performance.frame_age_ms
            fps = state.performance.processing_fps

            latencies_yolo.append(infer_ms)
            latencies_loop.append(loop_elapsed_ms)
            frame_ages.append(age_ms)

            occ_str = f"{state.occupancy['percent']}%" if state.occupancy['percent'] is not None else "N/A"

            print(
                f"{frame_count:5d} | "
                f"{state.system_state.value:<9} | "
                f"{state.counts['current']:5d} | "
                f"{occ_str:>5} | "
                f"{state.crowd['level']:<8} | "
                f"{infer_ms:7.2f} | "
                f"{loop_elapsed_ms:7.2f} | "
                f"{age_ms:6.1f} | "
                f"{fps:5.1f}"
            )

            # Brief sleep to match typical camera cadence if reading fast
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSmoke test interrupted by user.")
    finally:
        pipeline.stop()

    total_duration_sec = time.perf_counter() - t_start
    avg_pipeline_fps = frame_count / total_duration_sec if total_duration_sec > 0 else 0.0

    print("-" * 100)
    print("\n========================================================")
    print(" REAL HARDWARE PERFORMANCE SUMMARY")
    print("========================================================")
    print(f"Total Frames Processed   : {frame_count}")
    print(f"Total Test Wall Time     : {total_duration_sec:.2f} s")
    print(f"Effective Pipeline FPS   : {avg_pipeline_fps:.1f} FPS")
    if latencies_yolo:
        print(f"YOLO26n Inference Latency: Avg={np.mean(latencies_yolo):.2f} ms | p50={np.median(latencies_yolo):.2f} ms | p95={np.percentile(latencies_yolo, 95):.2f} ms")
    if latencies_loop:
        print(f"Total Step Loop Latency  : Avg={np.mean(latencies_loop):.2f} ms | p50={np.median(latencies_loop):.2f} ms | p95={np.percentile(latencies_loop, 95):.2f} ms")
    if frame_ages:
        print(f"Frame Age (Staleness)    : Avg={np.mean(frame_ages):.2f} ms | Max={np.max(frame_ages):.2f} ms")
    print(f"Final System State       : {state.system_state.value}")
    print(f"Final Count / Track Instances: Current={state.counts['current']} | Track Instances={state.counts.get('track_instances', 0)}")
    print("========================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
