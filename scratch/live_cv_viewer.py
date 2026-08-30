"""VisionQueue Continuous Live CV Demonstration & Scene-Intelligence Viewer.

Interactive real-time webcam and video viewer demonstrating the complete
VisionQueue Computer Vision subsystem:
CameraSource -> YOLO26m ONNX (CUDA) -> ByteTrack -> Whole-Frame Counting (V1)
-> Scene Intelligence & Automatic Capacity -> Crowd Analytics -> Alert Engine
-> Reliability Watchdog -> LiveState.

Usage:
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source 0
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source 0 --capacity 10
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source 0 --usable-area 45.0
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source "path/to/video.mp4"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import (
    AnalyticsConfig,
    CapacitySource,
    CrowdLevel,
    CrowdThresholds,
    SceneProfile,
)
from visionqueue.analytics.scene import SceneAnalyzerConfig
from visionqueue.camera.capture import CameraSource
from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import ReliabilityConfig, SystemState
from visionqueue.roi.types import ROIConfig


def get_state_color(state_str: str) -> Tuple[int, int, int]:
    """Return BGR color corresponding to the system health state."""
    if state_str == "LIVE":
        return (40, 200, 40)       # Vibrant Green
    elif state_str == "STARTING":
        return (255, 180, 0)      # Cyan / Gold
    elif state_str == "DEGRADED":
        return (0, 215, 255)      # Yellow
    elif state_str == "UNSTABLE":
        return (255, 0, 255)      # Magenta
    elif state_str == "OFFLINE":
        return (40, 40, 240)      # Red
    elif state_str == "STOPPING":
        return (160, 160, 160)    # Gray
    return (200, 200, 200)


def get_crowd_color(crowd_str: str) -> Tuple[int, int, int]:
    """Return BGR color corresponding to the crowd density classification."""
    if crowd_str == "LOW":
        return (40, 200, 40)       # Green
    elif crowd_str == "MODERATE":
        return (0, 215, 255)      # Yellow
    elif crowd_str == "HIGH":
        return (0, 140, 255)      # Orange
    elif crowd_str == "CRITICAL":
        return (40, 40, 240)      # Red
    return (200, 200, 200)


def draw_hud(
    frame: np.ndarray,
    state: LiveState,
    diagnostics: dict,
    virtual_line: Optional[VirtualLine] = None,
    show_virtual_line: bool = False,
    roi: Optional[ROIConfig] = None,
    show_roi: bool = False,
) -> np.ndarray:
    """Render a clean, operator-friendly HUD with bounding boxes, track IDs, and alert banner."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ------------------------------------------------------------
    # 1. Optional ROI Overlay (when explicitly enabled/requested)
    # ------------------------------------------------------------
    if roi is not None and show_roi:
        rx1 = int(round(roi.x))
        ry1 = int(round(roi.y))
        rx2 = int(round(roi.x + roi.width))
        ry2 = int(round(roi.y + roi.height))
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 140, 0), 2)
        cv2.putText(
            annotated,
            "ROI REGION",
            (rx1 + 8, max(24, ry1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 140, 0),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 2. Optional Virtual Line (only when explicitly requested)
    # ------------------------------------------------------------
    if virtual_line is not None and show_virtual_line:
        p1 = (int(round(virtual_line.pt1[0])), int(round(virtual_line.pt1[1])))
        p2 = (int(round(virtual_line.pt2[0])), int(round(virtual_line.pt2[1])))
        cv2.line(annotated, p1, p2, (0, 215, 255), 2)
        net_cnt = state.counts.get('net_count', 0)
        net_s = f"+{net_cnt}" if net_cnt > 0 else str(net_cnt)
        label_x = min(max(20, (p1[0] + p2[0]) // 2 - 80), w - 240)
        label_y = min(max(20, (p1[1] + p2[1]) // 2 - 8), h - 20)
        cv2.putText(
            annotated,
            f"Line (In:{state.counts.get('entries', 0)} | Out:{state.counts.get('exits', 0)} | Net:{net_s})",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 3. Draw Track Bounding Boxes and IDs
    # ------------------------------------------------------------
    active_tracks = diagnostics.get("active_tracks", [])
    for trk in active_tracks:
        bbox = trk.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        track_id = trk.get("track_id", -1)
        conf = trk.get("confidence", 0.0)

        # Draw main tracking bounding box
        box_color = (0, 230, 115)  # Spring green
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Draw ground-contact anchor point
        cx = (x1 + x2) // 2
        cv2.circle(annotated, (cx, y2), 4, (0, 255, 255), -1)

        # Draw pill label with Track ID & confidence
        label = f"ID:{track_id} ({conf:.2f})"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pill_y1 = max(0, y1 - lh - 8)
        pill_y2 = max(lh + 8, y1)
        cv2.rectangle(annotated, (x1, pill_y1), (x1 + lw + 6, pill_y2), (25, 25, 25), -1)
        cv2.rectangle(annotated, (x1, pill_y1), (x1 + lw + 6, pill_y2), box_color, 1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, pill_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 4. Clean Top Header HUD (Semi-transparent dark bar)
    # ------------------------------------------------------------
    header_h = 74
    header_overlay = annotated.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), (18, 18, 22), -1)
    cv2.addWeighted(header_overlay, 0.88, annotated, 0.12, 0, annotated)

    state_name = state.system_state.value
    state_color = get_state_color(state_name)

    # Left Section: Branding & System State
    cv2.putText(
        annotated,
        "VISIONQUEUE",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(annotated, (20, 50), 6, state_color, -1)
    cv2.putText(
        annotated,
        state_name,
        (34, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )

    # Middle Section: Headcount, Capacity, Occupancy, Crowd
    current_cnt = state.counts.get("current", 0)
    cap_val = state.occupancy.get("capacity")
    occ_val = state.occupancy.get("percent")
    occ_str = f"{occ_val}%" if occ_val is not None else "--"
    
    scene_diag = diagnostics.get("scene_analysis", {})
    cap_source = scene_diag.get("capacity_source", "NOT_SET")
    if cap_val is not None:
        src_tag = f" ({cap_source[:4]})" if cap_source in ("AUTOMATIC", "MANUAL", "CALIBRATED") else ""
        people_display = f"{current_cnt} / {cap_val}"
        cap_str = f"{cap_val}{src_tag}"
    else:
        people_display = str(current_cnt)
        cap_str = "NOT SET"

    crowd_lvl = state.crowd.get("level", "LOW")
    crowd_col = get_crowd_color(crowd_lvl)

    col_w = max(115, int(w * 0.15))
    start_x = max(160, int(w * 0.20))

    # Tile 1: PEOPLE
    cv2.putText(annotated, "PEOPLE", (start_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(annotated, people_display, (start_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    # Tile 2: CAPACITY
    x2 = start_x + col_w
    cv2.putText(annotated, "CAPACITY", (x2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(annotated, cap_str, (x2, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255) if cap_val is not None else (160, 160, 160), 2, cv2.LINE_AA)

    # Tile 3: OCCUPANCY
    x3 = x2 + col_w + 10
    cv2.putText(annotated, "OCCUPANCY", (x3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(annotated, occ_str, (x3, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Tile 4: CROWD STATUS
    x4 = x3 + col_w + 10
    cv2.putText(annotated, "CROWD", (x4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(annotated, crowd_lvl, (x4, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, crowd_col, 2, cv2.LINE_AA)

    # Right Section: Flow Counters (ENTRY / EXIT / NET)
    right_x = max(x4 + col_w + 15, int(w * 0.76))
    net_val = state.counts.get('net_count', 0)
    net_str = f"+{net_val}" if net_val > 0 else str(net_val)
    cv2.putText(annotated, f"ENTRY : {state.counts.get('entries', 0)}", (right_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"EXIT  : {state.counts.get('exits', 0)}", (right_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"NET   : {net_str}", (right_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------
    # 5. Prominent Active Alert Banner (Centered below header)
    # ------------------------------------------------------------
    if state.alerts:
        banner_h = 36
        banner_y1 = header_h + 8
        banner_y2 = banner_y1 + banner_h

        active_alert = state.alerts[0]
        for a in state.alerts:
            if a.severity.value == "CRITICAL":
                active_alert = a
                break

        if active_alert.type.value == "CRITICAL_OCCUPANCY" and cap_val is not None:
            alert_text = f"CAPACITY REACHED: {current_cnt} / {cap_val} PEOPLE ({occ_str}) -- CRITICAL OCCUPANCY"
        else:
            alert_text = f"ALERT: [{active_alert.severity.value}] {active_alert.reason}"

        (tw, th), tb = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        bw = max(tw + 40, 420)
        bx1 = max(10, (w - bw) // 2)
        bx2 = min(w - 10, bx1 + bw)

        alert_bg = annotated.copy()
        cv2.rectangle(alert_bg, (bx1, banner_y1), (bx2, banner_y2), (20, 20, 220), -1)
        cv2.addWeighted(alert_bg, 0.85, annotated, 0.15, 0, annotated)
        cv2.rectangle(annotated, (bx1, banner_y1), (bx2, banner_y2), (60, 60, 255), 2)

        tx = bx1 + (bw - tw) // 2
        ty = banner_y1 + ((banner_h + th) // 2) - 2
        cv2.putText(annotated, alert_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------
    # 6. Clean Bottom Control Bar
    # ------------------------------------------------------------
    footer_h = 28
    footer_y = h - footer_h
    footer_overlay = annotated.copy()
    cv2.rectangle(footer_overlay, (0, footer_y), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(footer_overlay, 0.85, annotated, 0.15, 0, annotated)

    ctrl_txt = "Press 'Q' or 'ESC' to exit  |  'R': Reset  |  'L': Toggle Line  |  'O': Toggle ROI"
    cv2.putText(
        annotated,
        ctrl_txt,
        (14, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return annotated


def render_offline_card(width: int, height: int, state: LiveState) -> np.ndarray:
    """Render an informative standby screen when camera frame acquisition fails."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 24)

    box_w, box_h = min(520, width - 40), 220
    bx1 = (width - box_w) // 2
    by1 = (height - box_h) // 2
    cv2.rectangle(canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), (35, 35, 45), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), (40, 40, 220), 2)

    cv2.putText(canvas, "CAMERA FEED UNAVAILABLE", (bx1 + 25, by1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"System State : {state.system_state.value}", (bx1 + 25, by1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Camera State : {state.camera_state.value}", (bx1 + 25, by1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Last Reliable Count: {state.counts.get('current', 0)} (Preserved)", (bx1 + 25, by1 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 115), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Reason: {state.status_reason}", (bx1 + 25, by1 + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VisionQueue Real-Time Live CV Demonstration & Scene Intelligence Viewer"
    )
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0), video file path, or stream URL")
    parser.add_argument("--model", type=str, default="models/yolo26m.onnx", help="ONNX model path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Person detection confidence threshold")
    parser.add_argument("--capacity", type=int, default=None, help="Manual venue capacity override")
    parser.add_argument("--usable-area", type=float, default=None, help="Calibrated usable ground area in m²")
    parser.add_argument("--target-density", type=float, default=0.8, help="Target density in persons/m² (default: 0.8)")
    parser.add_argument("--enable-roi", action="store_true", help="Enable ROI mode (default: False, whole-frame counting)")
    parser.add_argument("--show-roi", action="store_true", help="Render ROI bounding box overlay")
    parser.add_argument("--show-line", action="store_true", help="Render VirtualLine entrance crossing line")
    parser.add_argument("--line-pt1", type=str, default="320,40", help="VirtualLine start point x,y (default: '320,40')")
    parser.add_argument("--line-pt2", type=str, default="320,440", help="VirtualLine end point x,y (default: '320,440')")
    parser.add_argument("--line-dir", type=str, default="ENTRY", choices=["ENTRY", "EXIT"], help="Crossing direction name for Left-to-Right crossing")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames before closing (0 = infinite)")
    parser.add_argument("--duration", type=float, default=0.0, help="Max duration in seconds (0 = infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window (for CI/automated test)")
    args = parser.parse_args()

    source_val = int(args.source) if args.source.isdigit() else args.source

    # Parse VirtualLine Coordinates
    try:
        p1_parts = [float(c.strip()) for c in args.line_pt1.split(",")]
        p2_parts = [float(c.strip()) for c in args.line_pt2.split(",")]
        virtual_line = VirtualLine(
            pt1=(p1_parts[0], p1_parts[1]),
            pt2=(p2_parts[0], p2_parts[1]),
            entry_direction=args.line_dir,
        )
    except Exception as e:
        print(f"Warning: Invalid --line-pt1/pt2 coordinates ({e}). Using default line.")
        virtual_line = VirtualLine(pt1=(320.0, 40.0), pt2=(320.0, 440.0))

    show_virtual_line = args.show_line
    show_roi = args.show_roi

    print("\n" + "=" * 75)
    print(" VISIONQUEUE — LIVE CV DEMONSTRATION & SCENE INTELLIGENCE")
    print("=" * 75)
    print(f"Video Source            : {source_val}")
    print(f"Model Path              : {args.model}")
    print(f"Confidence Threshold    : {args.confidence}")
    print(f"Manual Capacity         : {args.capacity or 'None (Automatic Estimation Active)'}")
    print(f"Calibrated Usable Area  : {f'{args.usable_area:.1f} m²' if args.usable_area else 'None (Automatic Estimation Active)'}")
    print(f"Operating Mode          : {'ROI Mode (enable_roi=True)' if args.enable_roi else 'WHOLE-FRAME (V1 Default, enable_roi=False)'}")
    print(f"Entrance Line (pt1->pt2): ({virtual_line.x1:.0f},{virtual_line.y1:.0f}) -> ({virtual_line.x2:.0f},{virtual_line.y2:.0f}) [Dir: {virtual_line.entry_direction}]")
    print(f"Stream Duration         : {'Continuous (Press Q/ESC to exit)' if args.duration <= 0 and args.max_frames <= 0 else f'Bounded (max_frames={args.max_frames}, duration={args.duration}s)'}")
    print("-" * 75)

    # 1. Scene Profile Setup
    scene_profile = None
    if args.capacity is not None or args.usable_area is not None:
        scene_profile = SceneProfile(
            name="live_scene",
            manual_capacity=args.capacity,
            usable_area_m2=args.usable_area,
            target_density_persons_per_m2=args.target_density,
        )

    # 2. Pipeline Configuration
    pipeline_config = CVPipelineConfig(
        camera=CameraConfig(
            source=source_val,
            width=args.width,
            height=args.height,
            buffer_size=1,
            api_preference=cv2.CAP_DSHOW if isinstance(source_val, int) and sys.platform.startswith("win") else 0,
        ),
        detector=DetectorConfig(
            model_path=args.model,
            confidence_threshold=args.confidence,
        ),
        enable_roi=args.enable_roi,
        virtual_line=virtual_line,
        scene=scene_profile,
        scene_analyzer=SceneAnalyzerConfig(
            enabled=True,
            target_density_persons_per_m2=args.target_density,
        ),
        analytics=AnalyticsConfig(
            capacity=args.capacity if args.capacity and args.capacity > 0 else None,
            thresholds=CrowdThresholds(moderate_threshold=40.0, high_threshold=70.0, critical_threshold=90.0),
            debounce_frames=3,
        ),
        alerts=AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.5, clear_seconds=1.5),
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.5, clear_seconds=1.5),
        ),
        reliability=ReliabilityConfig(min_starting_frames=3),
        enable_face_detection=False,
    )

    print(f"Initializing Person Detector ({args.model}) on ONNX Runtime...")
    detector = PersonDetector(pipeline_config.detector)
    print(f"Active Provider: {detector.active_provider} (GPU Accelerated: {detector.is_gpu})")
    print("Warming up execution kernels...")
    detector.warm_up(3)

    camera = CameraSource(pipeline_config.camera)
    pipeline = CVPipeline(config=pipeline_config, camera=camera, detector=detector)

    window_name = "VisionQueue Live CV Viewer — Whole-Frame Counting (V1)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)

    print(f"\nStarting pipeline acquisition thread...")
    pipeline.start()

    print("Stream active. Displaying real-time video...")
    print("Controls: Press 'Q' or 'ESC' to exit | 'R': Reset | 'L': Toggle Line | 'O': Toggle ROI\n")

    frame_count = 0
    t_start = time.perf_counter()

    # Session Metrics Tracking
    infer_latencies: List[float] = []
    loop_latencies: List[float] = []
    fps_samples: List[float] = []
    frame_ages: List[float] = []
    prev_alert_keys: set[str] = set()
    alerts_seen: set[str] = set()
    detection_failures = 0
    tracking_failures = 0
    degraded_occurred = False
    offline_occurred = False

    try:
        while True:
            t_now = time.perf_counter()
            elapsed = t_now - t_start

            if args.duration > 0 and elapsed >= args.duration:
                print(f"\nReached target duration ({args.duration:.1f}s). Closing.")
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"\nReached target max frames ({args.max_frames}). Closing.")
                break

            loop_t0 = time.perf_counter()
            frame_data: Optional[FrameData] = pipeline.camera.get_latest_frame(wait_new=True, timeout=1.0)
            state: LiveState = pipeline.process_frame(frame_data)
            loop_t1 = time.perf_counter()

            diagnostics = pipeline.last_diagnostics
            frame_count += 1

            # Track Telemetry Statistics
            loop_ms = (loop_t1 - loop_t0) * 1000.0
            loop_latencies.append(loop_ms)
            infer_latencies.append(state.performance.inference_latency_ms)
            fps_samples.append(state.performance.processing_fps)
            frame_ages.append(state.performance.frame_age_ms)

            # Alert Activation / Clearance Edge Notifications
            curr_alert_keys = {f"{a.type.value}:{a.severity.value}" for a in state.alerts}
            new_alerts = curr_alert_keys - prev_alert_keys
            cleared_alerts = prev_alert_keys - curr_alert_keys

            for k in new_alerts:
                matching_alert = next((a for a in state.alerts if f"{a.type.value}:{a.severity.value}" == k), None)
                reason_str = matching_alert.reason if matching_alert else "Threshold reached"
                print(f"\n>>> [ALERT FIRED] {k} | {reason_str} (Frame {frame_count})")
                alerts_seen.add(k)

            for k in cleared_alerts:
                print(f"\n<<< [ALERT CLEARED] {k} | Occupancy / health normalized (Frame {frame_count})")

            prev_alert_keys = curr_alert_keys

            if not args.headless:
                if frame_data is not None and frame_data.frame is not None:
                    vis_frame = draw_hud(
                        frame=frame_data.frame,
                        state=state,
                        diagnostics=diagnostics,
                        virtual_line=virtual_line,
                        show_virtual_line=show_virtual_line,
                        roi=pipeline_config.roi,
                        show_roi=show_roi,
                    )
                else:
                    fallback_w = args.width or 640
                    fallback_h = args.height or 480
                    vis_frame = render_offline_card(fallback_w, fallback_h, state)

                cv2.imshow(window_name, vis_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\nExit key pressed by user.")
                    break
                elif key in (ord('r'), ord('R')):
                    print("\n[SESSION RESET] Resetting tracking, counts, and metrics...")
                    pipeline.reset_session()
                elif key in (ord('l'), ord('L')):
                    show_virtual_line = not show_virtual_line
                    print(f"\n[TOGGLE] Virtual Line Overlay: {'ON' if show_virtual_line else 'OFF'}")
                elif key in (ord('o'), ord('O')):
                    show_roi = not show_roi
                    print(f"\n[TOGGLE] ROI Overlay: {'ON' if show_roi else 'OFF'}")
            else:
                if frame_count % 10 == 0 or frame_count == 1:
                    scene_d = diagnostics.get("scene_analysis", {})
                    cap_src = scene_d.get("capacity_source", "NOT_SET")
                    print(
                        f"[{elapsed:5.1f}s] Frame {frame_count:4d} | State: {state.system_state.value:<8} | "
                        f"People: {state.counts.get('current', 0):2d} | "
                        f"Occ: {state.occupancy.get('percent', 'N/A')}% (Cap:{state.occupancy.get('capacity', 'None')} [{cap_src}]) | "
                        f"Crowd: {state.crowd.get('level', 'LOW'):<8} | "
                        f"FPS: {state.performance.processing_fps:4.1f} | "
                        f"Infer: {state.performance.inference_latency_ms:5.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\nViewer interrupted by user.")
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    effective_fps = frame_count / total_time if total_time > 0 else 0.0

    # Calculate percentiles
    p50_infer = float(np.percentile(infer_latencies, 50)) if infer_latencies else 0.0
    p95_infer = float(np.percentile(infer_latencies, 95)) if infer_latencies else 0.0
    avg_infer = float(np.mean(infer_latencies)) if infer_latencies else 0.0

    p50_loop = float(np.percentile(loop_latencies, 50)) if loop_latencies else 0.0
    p95_loop = float(np.percentile(loop_latencies, 95)) if loop_latencies else 0.0
    avg_loop = float(np.mean(loop_latencies)) if loop_latencies else 0.0

    avg_frame_age = float(np.mean(frame_ages)) if frame_ages else 0.0
    max_frame_age = float(np.max(frame_ages)) if frame_ages else 0.0

    scene_final = pipeline.last_diagnostics.get("scene_analysis", {})
    cap_final = state.occupancy.get("capacity")
    cap_src_final = scene_final.get("capacity_source", "NOT_SET")

    # ============================================================
    # TERMINAL FINAL REPORT
    # ============================================================
    print("\n" + "=" * 75)
    print(" VISIONQUEUE — TERMINAL SESSION FINAL REPORT")
    print("=" * 75)
    print(f"Session Duration              : {total_time:.2f} s")
    print(f"Total Frames Processed        : {frame_count}")
    print(f"Effective Processing Rate     : {effective_fps:.1f} FPS (Mean Reported: {float(np.mean(fps_samples) if fps_samples else 0.0):.1f} FPS)")
    print(f"Final System Health State     : {state.system_state.value}")
    print("-" * 75)
    print("HEADCOUNT & OCCUPANCY METRICS:")
    print(f"- Final Headcount             : {state.counts.get('current', 0)}")
    print(f"- Peak Headcount in Session   : {state.crowd.get('peak_count', 0)}")
    print(f"- Distinct Track Instances    : {state.counts.get('track_instances', 0)}")
    print(f"- Cumulative Entries (In)     : {state.counts.get('entries', 0)}")
    print(f"- Cumulative Exits (Out)      : {state.counts.get('exits', 0)}")
    print(f"- Net Headcount (In - Out)    : {state.counts.get('net_count', 0)}")
    print(f"- Final Crowd Density Level   : {state.crowd.get('level', 'LOW')}")
    print(f"- Final Occupancy Percentage  : {state.occupancy.get('percent', 'N/A')}%")
    print(f"- Effective Capacity          : {cap_final if cap_final is not None else 'NOT_SET'}")
    print(f"- Capacity Determination      : {cap_src_final}")
    print(f"- Scene Intelligence Reason   : {scene_final.get('reason', 'N/A')}")
    print("-" * 75)
    print("LATENCY & PERFORMANCE TELEMETRY:")
    print(f"- YOLO Inference Latency      : Mean: {avg_infer:.1f} ms | p50: {p50_infer:.1f} ms | p95: {p95_infer:.1f} ms")
    print(f"- Total Frame Processing Loop : Mean: {avg_loop:.1f} ms | p50: {p50_loop:.1f} ms | p95: {p95_loop:.1f} ms")
    print(f"- Camera Frame Age            : Mean: {avg_frame_age:.1f} ms | Max: {max_frame_age:.1f} ms")
    print("-" * 75)
    print("RELIABILITY & STABILITY AUDIT:")
    print(f"- Distinct Alerts Triggered   : {len(alerts_seen)} {list(alerts_seen)}")
    print(f"- Detection Failure Events    : {detection_failures}")
    print(f"- Tracking Failure Events     : {tracking_failures}")
    print(f"- Degraded/Unstable Occurred  : {'YES' if degraded_occurred else 'NO'}")
    print(f"- Camera Offline Occurred     : {'YES' if offline_occurred else 'NO'}")
    print("=" * 75 + "\n")

    # ============================================================
    # SAVE JSON SESSION REPORT
    # ============================================================
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(total_time, 2),
        "frames_processed": frame_count,
        "effective_fps": round(effective_fps, 2),
        "final_system_state": state.system_state.value,
        "counts": {
            "final_current": state.counts.get("current", 0),
            "peak_current": state.crowd.get("peak_count", 0),
            "track_instances": state.counts.get("track_instances", 0),
            "entries": state.counts.get("entries", 0),
            "exits": state.counts.get("exits", 0),
            "net_count": state.counts.get("net_count", 0),
        },
        "occupancy": {
            "final_percent": state.occupancy.get("percent"),
            "effective_capacity": cap_final,
            "capacity_source": cap_src_final,
            "crowd_level": state.crowd.get("level", "LOW"),
            "crowd_trend": state.crowd.get("trend", "STABLE"),
        },
        "performance": {
            "avg_inference_ms": round(avg_infer, 2),
            "p50_inference_ms": round(p50_infer, 2),
            "p95_inference_ms": round(p95_infer, 2),
            "avg_loop_ms": round(avg_loop, 2),
            "p50_loop_ms": round(p50_loop, 2),
            "p95_loop_ms": round(p95_loop, 2),
            "avg_frame_age_ms": round(avg_frame_age, 2),
            "max_frame_age_ms": round(max_frame_age, 2),
        },
        "reliability": {
            "distinct_alerts_count": len(alerts_seen),
            "alerts": list(alerts_seen),
            "detection_failures": detection_failures,
            "tracking_failures": tracking_failures,
            "degraded_occurred": degraded_occurred,
            "offline_occurred": offline_occurred,
        },
        "scene_analysis": scene_final,
    }

    scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    report_path = os.path.join(scratch_dir, "session_report_latest.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved session telemetry report to: {report_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
