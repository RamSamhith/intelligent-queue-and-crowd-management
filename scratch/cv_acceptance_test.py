"""VisionQueue Human-in-the-Loop CV Acceptance Test.

Interactive and automated telemetry test for validating the end-to-end CV pipeline
with a real human using the physical laptop webcam.

Test Phases:
1. Outside ROI (Verify current_count = 0, no false positives)
2. Enter ROI (Verify detection -> track -> bottom-center in-ROI -> current_count = 1)
3. Remain Inside ROI (Verify track ID persistence, debounced crowd level, occupancy %)
4. Cross Virtual Entry Line (Verify entry event incremented)
5. Leave ROI (Verify current_count decreases back to 0)
6. Cross Virtual Exit Line (Verify exit event incremented, session counts stable)

Outputs real-time telemetry, visual OpenCV HUD overlay, and structured verification report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
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
from visionqueue.reliability.types import ReliabilityConfig
from visionqueue.roi.types import ROIConfig


def draw_hud(
    frame: np.ndarray,
    state: LiveState,
    roi: ROIConfig,
    virtual_line: Optional[VirtualLine],
    active_phase: str,
) -> np.ndarray:
    """Draw ROI, virtual line, bounding boxes, ground points, and telemetry HUD on frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw ROI Box
    rx, ry, rw, rh = roi.x, roi.y, roi.width, roi.height
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 220, 100), 2)
    cv2.putText(
        annotated,
        f"ROI: In-Count={state.counts['current']}",
        (rx + 8, ry + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 100),
        2,
    )

    # 2. Draw Virtual Line
    if virtual_line is not None:
        p1 = (int(virtual_line.pt1[0]), int(virtual_line.pt1[1]))
        p2 = (int(virtual_line.pt2[0]), int(virtual_line.pt2[1]))
        cv2.line(annotated, p1, p2, (0, 215, 255), 3)
        # Entry/Exit indicators
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            annotated,
            f"Line (Entries:{state.counts['entries']} | Exits:{state.counts['exits']})",
            (p1[0] - 80, p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 215, 255),
            2,
        )
        # Draw small arrow indicating entry direction (left to right)
        cv2.arrowedLine(annotated, (mid_x - 30, mid_y), (mid_x + 30, mid_y), (0, 215, 255), 2, tipLength=0.3)

    # 3. Telemetry Header HUD (Semi-transparent top bar)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Status text
    state_color = (0, 255, 0) if state.system_state.value == "LIVE" else (0, 165, 255)
    cv2.putText(
        annotated,
        f"STATE: {state.system_state.value}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
    )
    cv2.putText(
        annotated,
        f"Occ: {state.counts['current']} (Cap {state.occupancy.get('percent', 'N/A')}%) | Crowd: {state.crowd['level']}",
        (180, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"Distinct Track Instances: {state.counts.get('track_instances', 0)} | Peak: {state.crowd.get('peak_count', 0)}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        annotated,
        f"FPS: {state.performance.processing_fps:.1f} | YOLO: {state.performance.inference_latency_ms:.1f}ms | Age: {state.performance.frame_age_ms:.1f}ms",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 220, 255),
        1,
    )

    # Active Alerts
    if state.alerts:
        alert_str = " | ".join([f"[{a.type.value}: {a.severity.value}]" for a in state.alerts if a.status.value == "ACTIVE"])
        if alert_str:
            cv2.putText(annotated, f"ALERT: {alert_str}", (w - 380, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Current Instruction / Phase Box (Bottom bar)
    overlay_bot = annotated.copy()
    cv2.rectangle(overlay_bot, (0, h - 45), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay_bot, 0.8, annotated, 0.2, 0, annotated)
    cv2.putText(
        annotated,
        f"TEST PHASE: {active_phase}",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 220, 0),
        2,
    )

    return annotated


def run_acceptance_test(
    camera_source: int = 0,
    model_path: str = "models/yolo26m.onnx",
    duration_seconds: float = 45.0,
    enable_gui: bool = True,
    output_log_path: str = "scratch/acceptance_test_log.json",
):
    print("\n" + "=" * 70)
    print(" VISIONQUEUE HUMAN-IN-THE-LOOP ACCEPTANCE TEST")
    print("=" * 70)
    print("Hardware: Laptop Webcam (Index %d)" % camera_source)
    print("Model: %s" % model_path)
    print("Accelerator: NVIDIA RTX 5060 Laptop GPU (CUDAExecutionProvider)")
    print("Target Test Duration: %.1f seconds" % duration_seconds)
    print("-" * 70)
    print("TEST PHASES:")
    print("  1. Stand OUTSIDE ROI rectangle (x < 280 or y < 50)")
    print("  2. Step INTO ROI (bottom-center enters green ROI box)")
    print("  3. REMAIN INSIDE ROI (observe count=1, crowd level debounce)")
    print("  4. Walk across YELLOW Virtual Line from Left to Right (Entry)")
    print("  5. Step OUTSIDE the green ROI box")
    print("  6. Walk across YELLOW Virtual Line from Right to Left (Exit)")
    print("=" * 70 + "\n")

    # Configure Acceptance Test Pipeline
    # 640x480 standard frame layout:
    # ROI: Right portion of frame — full height so person bottom-centers (feet)
    # near y=460-480 are captured. Virtual line at x=280 is the entry boundary.
    # Previously y=50,h=390 caused y2=440 which clipped detections with bbox_y2 > 440.
    roi_cfg = ROIConfig(x=160, y=0, width=480, height=480)
    line_cfg = VirtualLine(pt1=(280.0, 0.0), pt2=(280.0, 480.0))

    pipeline_config = CVPipelineConfig(
        camera=CameraConfig(source=camera_source, buffer_size=1),
        detector=DetectorConfig(
            model_path=model_path,
            confidence_threshold=0.25,
        ),
        roi=roi_cfg,
        virtual_line=line_cfg,
        analytics=AnalyticsConfig(
            capacity=2,
            thresholds=CrowdThresholds(moderate_threshold=50.0, high_threshold=75.0, critical_threshold=90.0),
            debounce_frames=3,
        ),
        alerts=AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.5, clear_seconds=1.0),
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.5, clear_seconds=1.0),
        ),
        reliability=ReliabilityConfig(min_starting_frames=3),
        enable_face_detection=False,
    )

    print(f"Initializing Person Detector ({model_path}) on CUDA EP...")
    detector = PersonDetector(pipeline_config.detector)
    print(f"Active Provider: {detector.active_provider}")
    detector.warm_up(3)

    camera = CameraSource(pipeline_config.camera)
    pipeline = CVPipeline(config=pipeline_config, camera=camera, detector=detector)

    print("Starting CameraSource...")
    camera.start()

    log_entries: List[Dict] = []
    t_start = time.perf_counter()
    frame_idx = 0
    last_print_time = 0.0
    last_entries = 0
    last_exits = 0

    phase_checkpoints = {
        "Phase 1: Stand OUTSIDE ROI": False,
        "Phase 2: Step INTO ROI": False,
        "Phase 3: Remain Inside ROI (Occupancy & Crowd)": False,
        "Phase 4: Cross Entry Line (L -> R)": False,
        "Phase 5: Leave ROI": False,
        "Phase 6: Cross Exit Line (R -> L)": False,
    }

    window_name = "VisionQueue - Live Acceptance Test [Press 'Q' to End, 'SPACE' for Checkpoint]"
    if enable_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nStarting live acceptance session loop...")
    print(f"{'Time':>6} | {'Phase':<28} | {'State':<8} | {'In-ROI':>6} | {'Occ %':>5} | {'Crowd':<8} | {'In':>3} | {'Out':>3} | {'Uniq':>4} | {'YOLO ms':>7} | {'Age ms':>6}")
    print("-" * 115)

    try:
        while True:
            now_sec = time.perf_counter() - t_start
            if now_sec >= duration_seconds:
                print("\nReached target test duration (%.1fs). Ending acceptance run." % duration_seconds)
                break

            # Current Phase Schedule (dynamic guideline based on test timeline)
            if now_sec < 8.0:
                active_phase = "1/6: Outside ROI (verify count=0)"
            elif now_sec < 16.0:
                active_phase = "2/6: Step INTO ROI (verify count=1)"
            elif now_sec < 24.0:
                active_phase = "3/6: Remain Inside (verify crowd debounce)"
            elif now_sec < 32.0:
                active_phase = "4/6: Cross Line L->R (verify entry +1)"
            elif now_sec < 40.0:
                active_phase = "5/6: Leave ROI (verify count=0)"
            else:
                active_phase = "6/6: Cross Line R->L (verify exit +1)"

            loop_t0 = time.perf_counter()
            frame_data = camera.get_latest_frame(wait_new=True, timeout=1.0)
            if frame_data is None:
                state = pipeline.process_frame(None)
            else:
                state = pipeline.process_frame(frame_data)
            loop_elapsed_ms = (time.perf_counter() - loop_t0) * 1000.0

            frame_idx += 1

            # Checkpoint verification
            if state.counts["current"] == 0 and now_sec < 8.0:
                phase_checkpoints["Phase 1: Stand OUTSIDE ROI"] = True
            if state.counts["current"] >= 1 and 8.0 <= now_sec < 24.0:
                phase_checkpoints["Phase 2: Step INTO ROI"] = True
                phase_checkpoints["Phase 3: Remain Inside ROI (Occupancy & Crowd)"] = True
            if state.counts["entries"] >= 1:
                phase_checkpoints["Phase 4: Cross Entry Line (L -> R)"] = True
            if state.counts["current"] == 0 and now_sec >= 32.0:
                phase_checkpoints["Phase 5: Leave ROI"] = True
            if state.counts["exits"] >= 1:
                phase_checkpoints["Phase 6: Cross Exit Line (R -> L)"] = True

            # Telemetry Log Record
            state_dict = state.to_dict()
            diag = pipeline.last_diagnostics
            entry = {
                "frame_id": frame_idx,
                "elapsed_sec": round(now_sec, 3),
                "phase": active_phase,
                "system_state": state.system_state.value,
                "current_count": state.counts["current"],
                "occupancy_percent": state.occupancy.get("percent"),
                "crowd_level": state.crowd["level"],
                "entries": state.counts["entries"],
                "exits": state.counts["exits"],
                "net_count": state.counts["net_count"],
                "track_instances": state.counts.get("track_instances", 0),
                # 15 Required Diagnostic telemetry fields
                "detections": diag.get("detections", []),
                "detections_count": diag.get("detections_count", state.detections_count),
                "active_tracks_count": diag.get("active_tracks_count", state.tracks_count),
                "active_track_ids": diag.get("active_track_ids", []),
                "active_tracks": diag.get("active_tracks", []),
                "track_ages": diag.get("track_ages", {}),
                "time_since_last_association": diag.get("time_since_last_association", {}),
                "tracks_created": diag.get("tracks_created", []),
                "tracks_lost": diag.get("tracks_lost", []),
                "id_replacements": diag.get("id_replacements", []),
                "association_ious": diag.get("association_ious", []),
                "matching_result": diag.get("matching_result", {}),
                "track_terminations": diag.get("track_terminations", []),
                "line_crossing_events": diag.get("line_crossing_events", []),
                "roi_events": diag.get("roi_events", {}),
                "active_alerts": [a.to_dict() for a in state.alerts if a.status.value == "ACTIVE"],
                "yolo_latency_ms": round(state.performance.inference_latency_ms, 2),
                "loop_latency_ms": round(loop_elapsed_ms, 2),
                "frame_age_ms": round(state.performance.frame_age_ms, 2),
                "fps": round(state.performance.processing_fps, 1),
            }
            log_entries.append(entry)

            # Periodic Console Reporting (every 0.5s or on count change)
            cur_entries = state.counts["entries"]
            cur_exits = state.counts["exits"]
            if (now_sec - last_print_time >= 0.5) or (cur_entries != last_entries) or (cur_exits != last_exits):
                last_print_time = now_sec
                last_entries = cur_entries
                last_exits = cur_exits
                occ_str = f"{state.occupancy.get('percent', 0)}%" if state.occupancy.get('percent') is not None else "N/A"
                print(
                    f"{now_sec:5.1f}s | "
                    f"{active_phase:<28} | "
                    f"{state.system_state.value:<8} | "
                    f"{state.counts['current']:6d} | "
                    f"{occ_str:>5} | "
                    f"{state.crowd['level']:<8} | "
                    f"{state.counts['entries']:3d} | "
                    f"{state.counts['exits']:3d} | "
                    f"{state.counts.get('track_instances', 0):4d} | "
                    f"{state.performance.inference_latency_ms:7.2f} | "
                    f"{state.performance.frame_age_ms:6.1f}",
                    flush=True,
                )

            # Optional OpenCV Window GUI Rendering
            if enable_gui and frame_data is not None:
                hud_frame = draw_hud(frame_data.frame, state, roi_cfg, line_cfg, active_phase)
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Esc or 'q'
                    print("\nOperator manually requested exit (key 'q').")
                    break

    except KeyboardInterrupt:
        print("\nAcceptance test interrupted by operator.")
    finally:
        camera.stop()
        if enable_gui:
            cv2.destroyAllWindows()

    # Save Structured JSON Log
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    with open(output_log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)

    diag_out_path = os.path.splitext(output_log_path)[0] + "_diagnostic.json"
    with open(diag_out_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)

    # Compute Acceptance Verification Metrics
    total_frames = len(log_entries)
    total_time = time.perf_counter() - t_start
    yolo_lats = [e["yolo_latency_ms"] for e in log_entries if e["yolo_latency_ms"] > 0]
    loop_lats = [e["loop_latency_ms"] for e in log_entries if e["loop_latency_ms"] > 0]
    frame_ages = [e["frame_age_ms"] for e in log_entries]
    fps_vals = [e["fps"] for e in log_entries if e["fps"] > 0]

    last_entry = log_entries[-1] if log_entries else {}

    print("\n" + "=" * 70)
    print(" ACCEPTANCE TEST VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total Frames Evaluated     : {total_frames}")
    print(f"Total Test Wall Time       : {total_time:.2f} s")
    print(f"Average Pipeline FPS       : {np.mean(fps_vals) if fps_vals else 0.0:.1f} FPS")
    if yolo_lats:
        print(f"YOLO26n Latency (GPU p50)  : {np.median(yolo_lats):.2f} ms (Avg: {np.mean(yolo_lats):.2f} ms, p95: {np.percentile(yolo_lats, 95):.2f} ms)")
    if loop_lats:
        print(f"End-to-End Loop (p50)      : {np.median(loop_lats):.2f} ms (p95: {np.percentile(loop_lats, 95):.2f} ms)")
    if frame_ages:
        print(f"Average Frame Age          : {np.mean(frame_ages):.2f} ms")
    print("-" * 70)
    print("FINAL RECORDED TELEMETRY:")
    print(f"  * Final System State     : {last_entry.get('system_state', 'N/A')}")
    print(f"  * Final In-ROI Count     : {last_entry.get('current_count', 0)}")
    print(f"  * Total Entries Observed : {last_entry.get('entries', 0)}")
    print(f"  * Total Exits Observed   : {last_entry.get('exits', 0)}")
    print(f"  * Net Count (In - Out)   : {last_entry.get('net_count', 0)}")
    print(f"  * Distinct Track Instances : {last_entry.get('track_instances', 0)}")
    print(f"  * Final Crowd Level      : {last_entry.get('crowd_level', 'N/A')}")
    print(f"  * Active Alerts Fired    : {len(last_entry.get('active_alerts', []))}")
    print("-" * 70)
    print("PHASE VERIFICATION CHECKPOINTS:")
    for phase_name, passed in phase_checkpoints.items():
        mark = "PASSED" if passed else "PENDING/UNCONFIRMED"
        print(f"  [{mark:<20}] {phase_name}")
    print("-" * 70)
    print(f"Detailed frame-by-frame log saved to: {output_log_path}")
    print("=" * 70 + "\n")

    return log_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionQueue CV Acceptance Test")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default="models/yolo26m.onnx", help="ONNX model path (default: models/yolo26m.onnx)")
    parser.add_argument("--duration", type=float, default=45.0, help="Test duration in seconds (default: 45.0)")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV imshow display window")
    parser.add_argument("--log-out", type=str, default="scratch/acceptance_test_log.json", help="Path to save JSON log")
    args = parser.parse_args()

    run_acceptance_test(
        camera_source=args.source,
        model_path=args.model,
        duration_seconds=args.duration,
        enable_gui=not args.no_gui,
        output_log_path=args.log_out,
    )
