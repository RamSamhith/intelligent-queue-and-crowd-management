"""VisionQueue Continuous Live CV Viewer & Real-Time Demonstration Tool.

Interactive real-time webcam and video viewer demonstrating the complete,
frozen VisionQueue Computer Vision subsystem:
CameraSource -> YOLO26m ONNX Runtime (CUDA) -> ByteTrack -> Whole-Frame Counting (V1)
-> Crowd Analytics -> Alert Engine -> Reliability Watchdog -> LiveState.

Usage:
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source 0
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source 0 --capacity 5 --confidence 0.25
    .venv\\Scripts\\python scratch/live_cv_viewer.py --source "path/to/video.mp4"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.alerts.types import AlertEngineConfig, AlertRuleConfig
from visionqueue.analytics.types import AnalyticsConfig, CrowdLevel, CrowdThresholds
from visionqueue.camera.capture import CameraSource
from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.reliability.types import ReliabilityConfig, SystemState


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
    show_virtual_line: bool = True,
) -> np.ndarray:
    """Render bounding boxes, track IDs, telemetry overlays, and HUD on the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ------------------------------------------------------------
    # 1. Draw Virtual Line (if configured)
    # ------------------------------------------------------------
    if virtual_line is not None and show_virtual_line:
        p1 = (int(virtual_line.pt1[0]), int(virtual_line.pt1[1]))
        p2 = (int(virtual_line.pt2[0]), int(virtual_line.pt2[1]))
        cv2.line(annotated, p1, p2, (0, 215, 255), 2)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            annotated,
            f"Line (In:{state.counts.get('entries', 0)} | Out:{state.counts.get('exits', 0)})",
            (p1[0] - 60, max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 2. Draw Track Bounding Boxes and IDs
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

        # Draw ground-center anchor circle
        cx = (x1 + x2) // 2
        cv2.circle(annotated, (cx, y2), 4, (0, 255, 255), -1)

        # Draw pill label with Track ID & confidence
        label = f"ID:{track_id} ({conf:.2f})"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pill_y1 = max(0, y1 - lh - 8)
        pill_y2 = max(lh + 8, y1)
        cv2.rectangle(
            annotated,
            (x1, pill_y1),
            (x1 + lw + 6, pill_y2),
            (25, 25, 25),
            -1,
        )
        cv2.rectangle(
            annotated,
            (x1, pill_y1),
            (x1 + lw + 6, pill_y2),
            box_color,
            1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, pill_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 3. Top Status Header HUD (Semi-transparent background bar)
    # ------------------------------------------------------------
    header_h = 75
    header_overlay = annotated.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), (15, 15, 18), -1)
    cv2.addWeighted(header_overlay, 0.85, annotated, 0.15, 0, annotated)

    state_name = state.system_state.value
    state_color = get_state_color(state_name)

    # System State Badge
    cv2.putText(
        annotated,
        "VISIONQUEUE V1",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"STATE: {state_name}",
        (12, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )
    if state.is_frozen:
        cv2.putText(
            annotated,
            "[COUNT FROZEN - SAFE RECOVERY]",
            (12, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            annotated,
            f"Camera: {state.camera_state.value}",
            (12, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # Center: Counting & Occupancy metrics
    col2_x = max(240, int(w * 0.32))
    current_cnt = state.counts.get("current", 0)
    unique_approx = state.counts.get("unique_session_approx", 0)
    occ_val = state.occupancy.get("percent")
    occ_str = f"{occ_val}%" if occ_val is not None else "N/A"
    cap_val = state.occupancy.get("capacity")
    cap_str = f"(Cap: {cap_val})" if cap_val is not None else ""

    cv2.putText(
        annotated,
        f"People (Whole-Frame): {current_cnt}",
        (col2_x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Occupancy: {occ_str} {cap_str}",
        (col2_x, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Session Unique Approx: {unique_approx} | In: {state.counts.get('entries', 0)} | Out: {state.counts.get('exits', 0)}",
        (col2_x, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Right: Crowd Level & Performance Telemetry
    col3_x = max(col2_x + 280, int(w * 0.72))
    crowd_lvl = state.crowd.get("level", "LOW")
    crowd_col = get_crowd_color(crowd_lvl)
    trend_str = state.crowd.get("trend", "STABLE")

    cv2.putText(
        annotated,
        f"CROWD: {crowd_lvl}",
        (col3_x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        crowd_col,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Trend: {trend_str}",
        (col3_x, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"FPS: {state.performance.processing_fps:.1f} | Infer: {state.performance.inference_latency_ms:.1f}ms",
        (col3_x, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------
    # 4. Bottom Active Alerts & Keybinding Footer
    # ------------------------------------------------------------
    footer_h = 32
    footer_y = h - footer_h
    footer_overlay = annotated.copy()
    cv2.rectangle(footer_overlay, (0, footer_y), (w, h), (15, 15, 18), -1)
    cv2.addWeighted(footer_overlay, 0.85, annotated, 0.15, 0, annotated)

    # Active alerts (if any)
    if state.alerts:
        alert_msg = " | ".join([f"[{a.severity.value}] {a.type.value}: {a.reason}" for a in state.alerts])
        cv2.putText(
            annotated,
            f"ALERT: {alert_msg}",
            (12, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            annotated,
            "Mode: Whole-Frame Counting (V1 Default, No ROI Required) | Press 'Q' or 'ESC' to exit | 'R' to reset session",
            (12, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    return annotated


def render_offline_card(
    width: int,
    height: int,
    state: LiveState,
) -> np.ndarray:
    """Render an informative standby screen when camera frame acquisition fails."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 24)

    # Center Alert Box
    box_w, box_h = min(500, width - 40), 220
    bx1 = (width - box_w) // 2
    by1 = (height - box_h) // 2
    cv2.rectangle(canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), (35, 35, 45), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), (40, 40, 220), 2)

    cv2.putText(
        canvas,
        "CAMERA FEED UNAVAILABLE",
        (bx1 + 25, by1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 40, 240),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"System State : {state.system_state.value}",
        (bx1 + 25, by1 + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Camera State : {state.camera_state.value}",
        (bx1 + 25, by1 + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Last Reliable Count: {state.counts.get('current', 0)} (Preserved)",
        (bx1 + 25, by1 + 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 115),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Reason: {state.status_reason}",
        (bx1 + 25, by1 + 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VisionQueue Real-Time Live CV Viewer & Demonstration Tool"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera device index (e.g. 0), video file path, or stream URL (default: '0')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo26m.onnx",
        help="ONNX model path (default: 'models/yolo26m.onnx')",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Person detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=5,
        help="Venue capacity for occupancy percentage calculation (default: 5)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Requested capture frame width (optional)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Requested capture frame height (optional)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before closing (default: 0 for continuous live stream)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Maximum duration in seconds before closing (default: 0.0 for continuous live stream)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV window (for CI/automated testing)",
    )
    args = parser.parse_args()

    # Parse camera source argument
    source_val = int(args.source) if args.source.isdigit() else args.source

    print("\n" + "=" * 75)
    print(" VISIONQUEUE — CONTINUOUS LIVE COMPUTER VISION VIEWER")
    print("=" * 75)
    print(f"Video Source            : {source_val}")
    print(f"Model Path              : {args.model}")
    print(f"Confidence Threshold    : {args.confidence}")
    print(f"Venue Capacity          : {args.capacity}")
    print(f"Operating Mode          : WHOLE-FRAME (V1 Default, enable_roi=False)")
    print(f"Stream Duration         : {'Continuous (Press Q/ESC to exit)' if args.duration <= 0 and args.max_frames <= 0 else f'Bounded (max_frames={args.max_frames}, duration={args.duration}s)'}")
    print("-" * 75)

    # 1. Pipeline Configuration (Using standard public contracts)
    virtual_line = VirtualLine(pt1=(320.0, 40.0), pt2=(320.0, 440.0))
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
        enable_roi=False,  # V1 Default: Whole-Frame Counting
        virtual_line=virtual_line,
        analytics=AnalyticsConfig(
            capacity=args.capacity if args.capacity > 0 else None,
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
    print("Controls: Press 'Q' or 'ESC' to exit | 'R' to reset session counts\n")

    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            t_now = time.perf_counter()
            elapsed = t_now - t_start

            # Check termination criteria
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\nReached target duration ({args.duration:.1f}s). Closing.")
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"\nReached target max frames ({args.max_frames}). Closing.")
                break

            # Fetch newest frame data from CameraSource
            frame_data: Optional[FrameData] = pipeline.camera.get_latest_frame(wait_new=True, timeout=1.0)

            # Process through unified CVPipeline
            state: LiveState = pipeline.process_frame(frame_data)
            diagnostics = pipeline.last_diagnostics
            frame_count += 1

            if not args.headless:
                if frame_data is not None and frame_data.frame is not None:
                    vis_frame = draw_hud(
                        frame=frame_data.frame,
                        state=state,
                        diagnostics=diagnostics,
                        virtual_line=virtual_line,
                    )
                else:
                    # Camera offline / disconnected standby card
                    fallback_w = args.width or 640
                    fallback_h = args.height or 480
                    vis_frame = render_offline_card(fallback_w, fallback_h, state)

                cv2.imshow(window_name, vis_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                    print("\nExit key pressed by user.")
                    break
                elif key in (ord('r'), ord('R')):
                    print("\n[SESSION RESET] Resetting tracking, counts, and metrics...")
                    pipeline.reset_session()
            else:
                # Headless console log
                if frame_count % 10 == 0 or frame_count == 1:
                    print(
                        f"[{elapsed:5.1f}s] Frame {frame_count:4d} | State: {state.system_state.value:<8} | "
                        f"People: {state.counts.get('current', 0):2d} | "
                        f"Occupancy: {state.occupancy.get('percent', 'N/A')}% | "
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

    print("-" * 75)
    print("SESSION EXECUTION SUMMARY:")
    print(f"- Total Frames Evaluated  : {frame_count}")
    print(f"- Total Wall Clock Time   : {total_time:.2f} s")
    print(f"- Effective Processing FPS: {effective_fps:.1f} FPS")
    print(f"- Final System State      : {state.system_state.value}")
    print(f"- Final Headcount         : {state.counts.get('current', 0)}")
    print(f"- Total Session Visitors  : {state.counts.get('unique_session_approx', 0)}")
    print("=" * 75 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
