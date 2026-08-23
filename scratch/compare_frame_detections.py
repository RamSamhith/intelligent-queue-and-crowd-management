"""Same-frame detection comparison between YOLO26n and YOLO26m."""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig


def create_synthetic_test_scenes():
    scenes = {}
    
    # Scene 1: Multi-scale people (Near, Medium, Far/Small)
    s1 = np.ones((720, 1280, 3), dtype=np.uint8) * 220
    # Far/Small figure
    cv2.rectangle(s1, (200, 300), (230, 380), (30, 30, 100), -1) # torso/legs
    cv2.circle(s1, (215, 290), 10, (180, 160, 140), -1) # head
    # Near figure
    cv2.rectangle(s1, (500, 200), (660, 680), (40, 50, 120), -1)
    cv2.circle(s1, (580, 150), 45, (190, 170, 150), -1)
    # Edge-of-frame figure (partially cut off on left edge)
    cv2.rectangle(s1, (0, 250), (60, 600), (60, 40, 80), -1)
    cv2.circle(s1, (30, 210), 30, (190, 170, 150), -1)
    scenes["Multi-Scale & Edge Scene"] = s1

    # Scene 2: Overlapping / Occluded figures
    s2 = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    # Person A
    cv2.rectangle(s2, (400, 250), (520, 650), (80, 40, 40), -1)
    cv2.circle(s2, (460, 200), 40, (190, 170, 150), -1)
    # Person B (partially overlapping Person A)
    cv2.rectangle(s2, (480, 270), (600, 650), (40, 80, 40), -1)
    cv2.circle(s2, (540, 220), 40, (190, 170, 150), -1)
    scenes["Overlapping Figures"] = s2

    # Scene 3: Live webcam capture if available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            scenes["Live Webcam Capture"] = frame
        cap.release()

    return scenes


def main():
    print("==================================================")
    print(" SAME-FRAME DETECTION COMPARISON (YOLO26n vs YOLO26m)")
    print("==================================================")

    det_n = PersonDetector(DetectorConfig(model_path="models/yolo26n.onnx", confidence_threshold=0.25))
    det_m = PersonDetector(DetectorConfig(model_path="models/yolo26m.onnx", confidence_threshold=0.25))
    
    det_n.warm_up(5)
    det_m.warm_up(5)

    scenes = create_synthetic_test_scenes()

    for name, frame in scenes.items():
        print(f"\n--- Scenario: {name} (Shape: {frame.shape}) ---")
        dets_n, lat_n = det_n.detect_timed(frame)
        dets_m, lat_m = det_m.detect_timed(frame)

        print(f"YOLO26n ({lat_n:.2f} ms): {len(dets_n)} detections")
        for idx, d in enumerate(dets_n):
            print(f"  [n-{idx+1}] bbox={d.bbox}, conf={d.confidence:.4f}, class={d.class_id}")

        print(f"YOLO26m ({lat_m:.2f} ms): {len(dets_m)} detections")
        for idx, d in enumerate(dets_m):
            print(f"  [m-{idx+1}] bbox={d.bbox}, conf={d.confidence:.4f}, class={d.class_id}")


if __name__ == "__main__":
    main()
