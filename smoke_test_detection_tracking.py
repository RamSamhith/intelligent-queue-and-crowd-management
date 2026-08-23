"""Live smoke test: Camera → PersonDetector → Tracking adapter → ByteTrack.

Run this script to verify the complete detection → tracking pipeline
with a live webcam or video file.
"""

import sys
sys.path.insert(0, 'C:/projects/INTELLIGENT QUEUE AND CROWD MANAGMENT')

import argparse
import cv2
import numpy as np

from visionqueue.detection import PersonDetector, DetectorConfig
from visionqueue.tracking.adapter import DetectionTrackingAdapter
from visionqueue.tracking import ByteTrackConfig


def main():
    parser = argparse.ArgumentParser(description="VisionQueue Detection→Tracking smoke test")
    parser.add_argument("--source", type=str, default="0", help="Video source (camera index or file path)")
    parser.add_argument("--model", type=str, default="models/yolo26n.onnx", help="ONNX model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = infinite)")
    parser.add_argument("--display", action="store_true", help="Display annotated video")
    args = parser.parse_args()

    # Initialize detector
    print(f"Loading detector from {args.model}...")
    det_config = DetectorConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
    )
    detector = PersonDetector(det_config)
    print(f"Detector loaded. Provider: {detector.active_provider}")
    print(f"GPU acceleration: {'Yes' if detector.is_gpu else 'No'}")

    # Warm up
    print("Warming up detector...")
    detector.warm_up(3)

    # Initialize tracking adapter
    track_config = ByteTrackConfig(
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        frame_rate=30,
    )
    adapter = DetectionTrackingAdapter(config=track_config)

    # Open video source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return 1

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")

    frame_count = 0
    print("\nStarting detection→tracking pipeline...")
    print("-" * 80)
    print(f"{'Frame':>6} | {'Dets':>4} | {'Tracks':>6} | {'Track IDs':<30} | {'Boxes'}")
    print("-" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break

            frame_count += 1

            # Detect
            detections, infer_ms = detector.detect_timed(frame)

            # Track
            tracks = adapter.update(detections)

            # Log
            track_ids = [t.track_id for t in tracks]
            boxes_str = ", ".join([f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]" for t in tracks for b in [t.bbox]]) if tracks else "none"
            print(f"{frame_count:6d} | {len(detections):4d} | {len(tracks):6d} | {str(track_ids):<30} | {boxes_str}")

            # Optional display
            if args.display:
                vis = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"P:{det.confidence:.2f}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.bbox)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis, f"ID:{track.track_id}", (x1, y1-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(vis, f"Frame: {frame_count} | Dets: {len(detections)} | Tracks: {len(tracks)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"Infer: {infer_ms:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("VisionQueue Detection→Tracking", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("-" * 80)
    print(f"Processed {frame_count} frames")
    print("Smoke test complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())