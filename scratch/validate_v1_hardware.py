"""VisionQueue V1 Whole-Frame CV Hardware Validation Script.

Validates the full unified CV Pipeline on live hardware with default V1 mode:
- Whole camera frame is counting area (enable_roi=False)
- No ROI restriction required
- Line crossing independent
- Real-time performance profiling
- Structured telemetry and verification checkpoints
"""

import os
import sys
import time
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import AnalyticsConfig, CrowdThresholds
from visionqueue.camera.capture import CameraSource
from visionqueue.camera.types import CameraConfig
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import ReliabilityConfig, SystemState


def main():
    parser = argparse.ArgumentParser(description="VisionQueue V1 Whole-Frame CV Hardware Validation")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default="models/yolo26m.onnx", help="ONNX model path (default: models/yolo26m.onnx)")
    parser.add_argument("--duration", type=float, default=20.0, help="Test duration in seconds (default: 20.0)")
    parser.add_argument("--capacity", type=int, default=2, help="Venue capacity for occupancy calculation (default: 2)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" VISIONQUEUE V1 WHOLE-FRAME COMPUTER VISION HARDWARE VALIDATION")
    print("=" * 70)
    print(f"Camera Device Source    : {args.source}")
    print(f"Model Path              : {args.model}")
    print(f"Target Validation Time  : {args.duration:.1f} seconds")
    print(f"Configured Capacity     : {args.capacity}")
    print(f"Operating Mode          : WHOLE-FRAME (enable_roi=False)")
    print("-" * 70)

    # 1. Pipeline Configuration with V1 Default (enable_roi=False, no ROI required)
    config = CVPipelineConfig(
        camera=CameraConfig(source=args.source, buffer_size=1),
        detector=DetectorConfig(
            model_path=args.model,
            confidence_threshold=0.25,
        ),
        enable_roi=False,  # V1 DEFAULT: Whole-Frame Counting
        virtual_line=VirtualLine(pt1=(320.0, 50.0), pt2=(320.0, 430.0)),
        analytics=AnalyticsConfig(
            capacity=args.capacity,
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

    print(f"Initializing Person Detector ({args.model}) on ONNX Runtime...")
    detector = PersonDetector(config.detector)
    print(f"Active Execution Provider: {detector.active_provider} (GPU: {detector.is_gpu})")
    print("Warming up CUDA EP execution kernels...")
    detector.warm_up(3)

    camera = CameraSource(config.camera)
    pipeline = CVPipeline(config=config, camera=camera, detector=detector)

    print(f"Starting unified CVPipeline on camera source {args.source}...")
    pipeline.start()

    print("\nExecuting live hardware validation stream...")
    print("-" * 105)
    print(f"{'Time':>6} | {'State':<8} | {'Dets':>4} | {'Tracks':>6} | {'Whole-Count':>11} | {'Occ %':>5} | {'Crowd':<8} | {'YOLO ms':>7} | {'Loop ms':>7} | {'FPS':>5}")
    print("-" * 105)

    t_start = time.perf_counter()
    frame_count = 0
    yolo_latencies = []
    loop_latencies = []
    frame_ages = []
    states_observed = set()
    max_count_observed = 0
    active_alerts_fired = 0

    try:
        while True:
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            if elapsed >= args.duration:
                break

            loop_t0 = time.perf_counter()
            state: LiveState = pipeline.step()
            loop_ms = (time.perf_counter() - loop_t0) * 1000.0

            frame_count += 1
            infer_ms = state.performance.inference_latency_ms
            age_ms = state.performance.frame_age_ms
            fps = state.performance.processing_fps

            yolo_latencies.append(infer_ms)
            loop_latencies.append(loop_ms)
            frame_ages.append(age_ms)
            states_observed.add(state.system_state.value)

            cur_cnt = state.counts["current"]
            if cur_cnt > max_count_observed:
                max_count_observed = cur_cnt

            if len(state.alerts) > 0:
                active_alerts_fired += len(state.alerts)

            occ_str = f"{state.occupancy['percent']}%" if state.occupancy['percent'] is not None else "N/A"

            if frame_count % 3 == 0 or frame_count == 1:
                print(
                    f"{elapsed:5.1f}s | "
                    f"{state.system_state.value:<8} | "
                    f"{state.detections_count:4d} | "
                    f"{state.tracks_count:6d} | "
                    f"{cur_cnt:11d} | "
                    f"{occ_str:>5} | "
                    f"{state.crowd['level']:<8} | "
                    f"{infer_ms:7.2f} | "
                    f"{loop_ms:7.2f} | "
                    f"{fps:5.1f}"
                )

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
    finally:
        pipeline.stop()

    total_wall_time = time.perf_counter() - t_start
    effective_fps = frame_count / total_wall_time if total_wall_time > 0 else 0.0

    print("-" * 105)
    print("\n" + "=" * 70)
    print(" V1 WHOLE-FRAME HARDWARE VALIDATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Frames Evaluated     : {frame_count}")
    print(f"Total Wall Duration        : {total_wall_time:.2f} s")
    print(f"Effective End-to-End FPS   : {effective_fps:.1f} FPS")
    if yolo_latencies:
        print(f"Detector GPU Latency (p50) : {np.median(yolo_latencies):.2f} ms (Avg: {np.mean(yolo_latencies):.2f} ms, p95: {np.percentile(yolo_latencies, 95):.2f} ms)")
    if loop_latencies:
        print(f"Pipeline Loop Latency (p50): {np.median(loop_latencies):.2f} ms (Avg: {np.mean(loop_latencies):.2f} ms, p95: {np.percentile(loop_latencies, 95):.2f} ms)")
    if frame_ages:
        print(f"Frame Age (Staleness Avg)  : {np.mean(frame_ages):.2f} ms (Max: {np.max(frame_ages):.2f} ms)")
    print("-" * 70)
    print("TELEMETRY & COUNTING RESULTS:")
    print(f"  * System States Observed : {states_observed}")
    print(f"  * Final System State     : {state.system_state.value}")
    print(f"  * Max Frame Count Seen   : {max_count_observed}")
    print(f"  * Final Current Count    : {state.counts['current']}")
    print(f"  * Total Entries Observed : {state.counts['entries']}")
    print(f"  * Total Exits Observed   : {state.counts['exits']}")
    print(f"  * Net Count (Entries-Exit: {state.counts['net_count']}")
    print(f"  * Approx Unique Visitors : {state.counts['unique_session_approx']}")
    print(f"  * Final Crowd Level      : {state.crowd['level']}")
    print(f"  * Final Occupancy Pct    : {state.occupancy['percent']}%")
    print(f"  * Alert Count Fired      : {active_alerts_fired}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
