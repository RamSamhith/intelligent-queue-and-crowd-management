"""V1.1 latency benchmark."""
import time
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import AnalyticsConfig
from visionqueue.camera.types import FrameData, SourceState
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig
from visionqueue.reliability.types import ReliabilityConfig
from visionqueue.roi.types import ROIConfig


def main():
    print("=== VisionQueue V1.1 Latency Benchmark ===")

    detector = PersonDetector(
        DetectorConfig(
            model_path=os.path.join(PROJECT_ROOT, "models", "yolo26m.onnx"),
            confidence_threshold=0.25,
        )
    )
    print(f"Active Provider: {detector.active_provider} (GPU Accelerated: {detector.is_gpu})")
    detector.warm_up(3)

    config = CVPipelineConfig(
        virtual_line=VirtualLine(pt1=(320.0, 40.0), pt2=(320.0, 440.0)),
        queue_roi=ROIConfig(x=200, y=200, width=200, height=200),
        analytics=AnalyticsConfig(capacity=10),
        reliability=ReliabilityConfig(min_starting_frames=2),
        enable_face_detection=False,
    )
    pipeline = CVPipeline(config=config, detector=detector)

    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Warm up
    for i in range(3):
        fd = FrameData(
            frame=raw_frame, timestamp=time.time(), frame_id=i + 1, width=640, height=480,
            source_state=SourceState.RUNNING, fps=30.0,
        )
        pipeline.process_frame(fd, timestamp=time.time())

    # Benchmark
    N = 50
    infer_latencies = []
    loop_latencies = []
    fps_samples = []
    for i in range(N):
        fd = FrameData(
            frame=raw_frame, timestamp=time.time(), frame_id=i + 10, width=640, height=480,
            source_state=SourceState.RUNNING, fps=30.0,
        )
        t0 = time.perf_counter()
        state = pipeline.process_frame(fd, timestamp=time.time())
        t1 = time.perf_counter()
        infer_latencies.append(state.performance.inference_latency_ms)
        loop_latencies.append((t1 - t0) * 1000.0)
        fps_samples.append(state.performance.processing_fps)

    def stats(arr, name):
        a = np.array(arr)
        print(f"  {name:<30}: mean={a.mean():.2f} p50={np.percentile(a, 50):.2f} p95={np.percentile(a, 95):.2f} max={a.max():.2f}")

    print(f"\nBenchmark: {N} frames on synthetic 640x480 BGR")
    stats(infer_latencies, "YOLO inference (ms)")
    stats(loop_latencies, "Total loop (ms)")
    stats(fps_samples, "Processing FPS")

    # Verify V1.1 fields are present in final state
    print(f"\nV1.1 schema verification on final state:")
    d = state.to_dict()
    assert "track_instances" in d["counts"]
    assert "unique_session_approx" in d["counts"]  # deprecated alias
    assert "queue_people" in d
    print("  [OK] track_instances present")
    print("  [OK] unique_session_approx alias present (deprecated)")
    print("  [OK] queue_people present")

    # No alerts expected (no real people, no failure)
    assert len(d["alerts"]) == 0
    print("  [OK] No spurious alerts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
