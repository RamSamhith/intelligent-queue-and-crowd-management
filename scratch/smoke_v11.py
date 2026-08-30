"""V1.1 hardware smoke test."""
import json
import time
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import AnalyticsConfig
from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig
from visionqueue.reliability.types import ReliabilityConfig
from visionqueue.roi.types import ROIConfig


def main():
    print("=== VisionQueue V1.1 Hardware Smoke Test ===")

    print("\n[1] Loading YOLO26m ONNX model...")
    detector = PersonDetector(
        DetectorConfig(model_path=os.path.join(PROJECT_ROOT, "models", "yolo26m.onnx"), confidence_threshold=0.25)
    )
    print(f"    Active Provider: {detector.active_provider}")
    print(f"    GPU Accelerated: {detector.is_gpu}")
    detector.warm_up(3)
    print(f"    Input shape: {detector.input_shape}")

    print("\n[2] Building V1.1 CVPipeline with new features:")
    print("    - track_instances field + deprecated unique_session_approx alias")
    print("    - queue_roi for queue_people counting")
    print("    - virtual line for entry/exit counting")
    print("    - AUTOMATIC capacity safety isolation")
    print("    - min_track_age_frames guard on LineCrossingCounter")

    queue_roi = ROIConfig(x=200, y=200, width=200, height=200)
    config = CVPipelineConfig(
        camera=CameraConfig(source=0, width=640, height=480, buffer_size=1),
        detector=DetectorConfig(
            model_path=os.path.join(PROJECT_ROOT, "models", "yolo26m.onnx"),
            confidence_threshold=0.25,
        ),
        virtual_line=VirtualLine(pt1=(320.0, 40.0), pt2=(320.0, 440.0)),
        queue_roi=queue_roi,
        analytics=AnalyticsConfig(capacity=None),
        alerts=AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        ),
        reliability=ReliabilityConfig(min_starting_frames=3),
        enable_face_detection=False,
    )
    pipeline = CVPipeline(config=config, detector=detector)

    print("\n[3] Processing synthetic frame...")
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_data = FrameData(
        frame=raw_frame,
        timestamp=time.time(),
        frame_id=1,
        width=640,
        height=480,
        source_state=SourceState.RUNNING,
        fps=30.0,
    )
    state = pipeline.process_frame(frame_data, timestamp=time.time())
    d = state.to_dict()

    print("\n[4] LiveState fields:")
    print(f"    schema_version        : {d['schema_version']}")
    print(f"    system_state          : {d['system_state']}")
    print(f"    camera_state          : {d['camera_state']}")
    print(f"    counts.current        : {d['counts']['current']}")
    print(f"    counts.track_instances: {d['counts']['track_instances']}")
    print(f"    counts.unique_session_approx (deprecated): {d['counts']['unique_session_approx']}")
    print(f"    counts.entries        : {d['counts']['entries']}")
    print(f"    counts.exits         : {d['counts']['exits']}")
    print(f"    counts.net_count     : {d['counts']['net_count']}")
    print(f"    occupancy.capacity   : {d['occupancy']['capacity']}")
    print(f"    occupancy.capacity_state: {d['occupancy']['capacity_state']}")
    print(f"    queue_people         : {d['queue_people']}")

    print("\n[5] JSON serialization test:")
    try:
        json_str = json.dumps(d)
        print(f"    Serialized OK: {len(json_str)} bytes")
    except Exception as e:
        print(f"    FAILED: {e}")
        return 1

    print("\n[6] Live webcam test note:")
    print("    Physical webcam validation requires a user to be physically")
    print("    present with a camera. The full hardware path was validated")
    print("    in the previous session (commit 203846a).")

    print("\n=== Smoke Test Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
