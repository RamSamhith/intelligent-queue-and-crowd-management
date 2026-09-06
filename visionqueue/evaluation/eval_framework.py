"""VisionQueue Offline Evaluation Framework.

Processes pre-recorded video through the production CVPipeline and compares
outputs with human ground-truth annotations, producing accuracy and performance
metrics.

IMPORTANT DESIGN PRINCIPLE:
- Offline accuracy mode processes EVERY frame sequentially without frame-dropping.
- Real-time performance benchmarking is a separate measurement.
- These two modes must NOT be conflated.

Usage:
    python scratch/eval_framework.py --video clip.mp4 --gt gt.json --out ./eval_output
    python scratch/eval_framework.py --video clip.mp4 --gt gt.json --out ./eval_output --model models/yolo26n.onnx
    python scratch/eval_framework.py --video clip.mp4 --gt gt.json --out ./eval_output --compare-model models/yolo26m.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import CrossingDirection, VirtualLine
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.types import DetectorConfig
from visionqueue.pipeline.coordinator import CVPipeline
from visionqueue.pipeline.types import CVPipelineConfig
from visionqueue.analytics.types import AnalyticsConfig
from visionqueue.reliability.types import ReliabilityConfig
from visionqueue.roi.types import ROIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Ground Truth Loading & Validation
# =============================================================================

def load_ground_truth(gt_path: str) -> dict:
    """Load and validate a ground-truth JSON file."""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    schema_version = gt.get("schema_version", 1)
    if schema_version != 1:
        logger.warning("Unexpected GT schema version %d (expected 1). Proceeding anyway.", schema_version)

    meta = gt.get("metadata", {})
    fps = meta.get("fps", 30.0)
    frame_width = meta.get("frame_width", 0)
    frame_height = meta.get("frame_height", 0)
    total_frames = meta.get("total_frames", 0)

    if fps <= 0:
        logger.warning("Invalid GT fps %s, defaulting to 30.0", fps)
        fps = 30.0

    protocol = gt.get("annotation_protocol", {})
    crossing_tolerance_sec = protocol.get("crossing_tolerance_seconds", 1.0)
    if crossing_tolerance_sec <= 0:
        logger.warning("Invalid crossing_tolerance_seconds %s, defaulting to 1.0", crossing_tolerance_sec)
        crossing_tolerance_sec = 1.0

    occupancy_by_frame: Dict[int, int] = {}
    for entry in gt.get("occupancy", []):
        frame = entry["frame"]
        count = entry["person_count"]
        occupancy_by_frame[frame] = count

    gt_crossings: List[dict] = gt.get("crossings", [])
    gt_metrics = gt.get("metrics", {})

    return {
        "schema_version": schema_version,
        "metadata": meta,
        "protocol": protocol,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "total_frames": total_frames,
        "crossing_tolerance_sec": crossing_tolerance_sec,
        "occupancy_by_frame": occupancy_by_frame,
        "gt_crossings": gt_crossings,
        "gt_metrics": gt_metrics,
        "critical_events": gt.get("critical_events", []),
        "trajectories": gt.get("trajectories", []),
    }


# =============================================================================
# Metrics Computation
# =============================================================================

def frame_to_timestamp(frame: int, fps: float) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame / fps if fps > 0 else 0.0


def timestamp_to_frame(ts: float, fps: float) -> int:
    """Convert timestamp in seconds to nearest frame number."""
    return int(round(ts * fps)) if fps > 0 else 0


def compute_occupancy_metrics(
    predicted_by_frame: Dict[int, int],
    gt_by_frame: Dict[int, int],
    fps: float,
) -> dict:
    """Compute occupancy accuracy metrics against ground truth.

    Compares predicted counts to GT counts for all frames where GT is available.
    Uses per-frame absolute error, no percentage-based errors (MAPE not meaningful
    when GT count can be 0).

    Returns:
        Dictionary of occupancy metrics with MAE, RMSE, max error, bias, error distribution.
    """
    frames = sorted(set(predicted_by_frame.keys()) & set(gt_by_frame.keys()))
    if not frames:
        return {
            "mae": None,
            "rmse": None,
            "max_absolute_error": None,
            "bias": None,
            "bias_direction": None,
            "frames_evaluated": 0,
            "frames_with_gt": len(gt_by_frame),
            "note": "No overlapping frames between predictions and ground truth",
        }

    errors: List[int] = []
    absolute_errors: List[int] = []
    for f in frames:
        pred = predicted_by_frame[f]
        gt = gt_by_frame[f]
        err = pred - gt
        errors.append(err)
        absolute_errors.append(abs(err))

    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
    max_abs_err = max(absolute_errors) if absolute_errors else None
    bias = sum(errors) / len(errors) if errors else None

    if bias is not None:
        bias_dir = "OVERCOUNT" if bias > 0.5 else ("UNDERCOUNT" if bias < -0.5 else "BALANCED")
    else:
        bias_dir = None

    below_1 = sum(1 for e in absolute_errors if e <= 1)
    below_2 = sum(1 for e in absolute_errors if e <= 2)
    below_3 = sum(1 for e in absolute_errors if e <= 3)
    n = len(absolute_errors)

    return {
        "mae": round(mae, 4) if mae is not None else None,
        "rmse": round(rmse, 4) if rmse is not None else None,
        "max_absolute_error": max_abs_err,
        "bias": round(bias, 4) if bias is not None else None,
        "bias_direction": bias_dir,
        "frames_evaluated": n,
        "frames_with_gt": len(gt_by_frame),
        "error_within_1": f"{below_1}/{n} ({round(100*below_1/n, 1)}%)",
        "error_within_2": f"{below_2}/{n} ({round(100*below_2/n, 1)}%)",
        "error_within_3": f"{below_3}/{n} ({round(100*below_3/n, 1)}%)",
    }


def compute_peak_occupancy_metrics(
    predicted_peak: int,
    gt_peak: int,
) -> dict:
    """Compute peak occupancy accuracy metrics."""
    error = predicted_peak - gt_peak
    abs_error = abs(error)
    return {
        "predicted_peak": predicted_peak,
        "gt_peak": gt_peak,
        "peak_error": error,
        "peak_absolute_error": abs_error,
    }


def compute_crossing_metrics(
    predicted_crossings: List[dict],
    gt_crossings: List[dict],
    fps: float,
    tolerance_sec: float,
) -> dict:
    """Compute entry/exit event accuracy metrics.

    Uses temporal matching: a predicted crossing matches a GT crossing if:
    1. Same direction (ENTRY or EXIT)
    2. |pred_timestamp - gt_timestamp| <= tolerance_sec

    One-to-one matching: each GT event can match at most one predicted event.
    Matching is greedy (earliest available GT event first).

    Returns precision, recall, F1 for ENTRY, EXIT, and combined.
    """
    tolerance_frames = int(round(tolerance_sec * fps)) if fps > 0 else int(tolerance_sec * 30)

    def get_events_by_dir(crossings: List[dict], direction: str) -> List[dict]:
        return [c for c in crossings if c.get("direction", "").upper() == direction.upper()]

    def greedy_match(
        preds: List[dict],
        gts: List[dict],
        tolerance_f: int,
    ) -> Tuple[int, int, int]:
        """Return (true_positives, false_positives, false_negatives) via greedy temporal matching."""
        gt_used = [False] * len(gts)
        tp = 0
        for pred in preds:
            p_frame = pred.get("frame", 0)
            best_d = tolerance_f + 1
            best_idx = -1
            for i, gt in enumerate(gts):
                if gt_used[i]:
                    continue
                g_frame = gt.get("frame", 0)
                d = abs(p_frame - g_frame)
                if d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx >= 0 and best_d <= tolerance_f:
                gt_used[best_idx] = True
                tp += 1
        fp = len(preds) - tp
        fn = len(gts) - tp
        return tp, fp, fn

    results: Dict[str, Any] = {}

    for direction in ["ENTRY", "EXIT"]:
        preds = get_events_by_dir(predicted_crossings, direction)
        gts = get_events_by_dir(gt_crossings, direction)
        tp, fp, fn = greedy_match(preds, gts, tolerance_frames)

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        results[direction] = {
            "predicted": len(preds),
            "gt": len(gts),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    all_preds = predicted_crossings
    all_gts = gt_crossings
    tp_all, fp_all, fn_all = greedy_match(all_preds, all_gts, tolerance_frames)
    p_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else None
    r_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else None
    f1_all = (2 * p_all * r_all / (p_all + r_all)) if (p_all is not None and r_all is not None and (p_all + r_all) > 0) else None

    results["COMBINED"] = {
        "predicted": len(all_preds),
        "gt": len(all_gts),
        "true_positives": tp_all,
        "false_positives": fp_all,
        "false_negatives": fn_all,
        "precision": round(p_all, 4) if p_all is not None else None,
        "recall": round(r_all, 4) if r_all is not None else None,
        "f1": round(f1_all, 4) if f1_all is not None else None,
    }

    results["parameters"] = {
        "tolerance_seconds": tolerance_sec,
        "tolerance_frames": tolerance_frames,
        "fps": fps,
    }

    return results


def compute_track_diagnostics(
    track_events: List[dict],
    total_frames: int,
    fps: float,
) -> dict:
    """Compute track instance diagnostic metrics.

    IMPORTANT: These track diagnostics describe TRACK INSTANCES (ByteTrack IDs),
    NOT unique human beings. Do not report these as human identity metrics.
    """
    if not track_events:
        return {
            "total_track_instances": 0,
            "note": "No track events recorded",
        }

    created = [e for e in track_events if e.get("event") == "created"]
    lost = [e for e in track_events if e.get("event") == "lost"]
    terminated = [e for e in track_events if e.get("event") == "terminated"]

    unique_ids = set()
    for e in created:
        if "track_id" in e:
            unique_ids.add(e["track_id"])

    return {
        "total_track_instances": len(unique_ids),
        "total_created": len(created),
        "total_lost": len(lost),
        "total_terminated": len(terminated),
        "net_change": len(created) - len(terminated),
        "total_frames": total_frames,
        "note": "Track instance diagnostics reflect ByteTrack IDs, NOT human identities.",
    }


# =============================================================================
# Pipeline Runner (Offline Accuracy Mode)
# =============================================================================

def run_offline_evaluation(
    video_path: str,
    gt_path: Optional[str],
    model_path: Optional[str],
    virtual_line_config: Optional[dict],
    capacity: Optional[int],
    roi_config: Optional[dict],
    queue_roi_config: Optional[dict],
    output_dir: str,
    sample_every_n: int = 1,
    verbose: bool = False,
    max_frames: Optional[int] = None,
) -> dict:
    """Run the CV pipeline on a video file in offline accuracy mode.

    Processes every frame sequentially (no frame dropping).
    Collects per-frame predictions and aggregates them.
    Ensures finite sequential pass without background loop at EOF.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or math.isnan(actual_fps):
        actual_fps = 30.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Video: %s (%dx%d @ %.2f FPS, %d frames)",
        Path(video_path).name,
        vid_width,
        vid_height,
        actual_fps,
        total_vid_frames,
    )

    frame_count_mismatch: Optional[Dict[str, int]] = None
    gt: Optional[dict] = None
    if gt_path:
        gt = load_ground_truth(gt_path)
        gt_total_frames = gt.get("total_frames", 0)
        if total_vid_frames > 0 and gt_total_frames > 0 and total_vid_frames != gt_total_frames:
            frame_count_mismatch = {
                "video_frames": total_vid_frames,
                "gt_frames": gt_total_frames,
            }
            logger.warning(
                "Frame count mismatch: video has %d frames, GT metadata specifies %d frames.",
                total_vid_frames,
                gt_total_frames,
            )

    # Initialize PersonDetector directly using the production detector
    detector_cfg = DetectorConfig(
        model_path=model_path or "models/yolo26m.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    detector = PersonDetector(detector_cfg)
    logger.info(
        "Detector initialized. Model: %s, Active provider: %s, is_gpu: %s",
        detector_cfg.model_path,
        detector.active_provider,
        detector.is_gpu,
    )

    # Configure camera as non-looping offline video source
    camera_cfg = CameraConfig(
        source=video_path,
        loop_video=False,
        width=vid_width or None,
        height=vid_height or None,
    )

    analytics_cfg = AnalyticsConfig(capacity=capacity)

    roi = None
    if roi_config:
        roi = ROIConfig(
            x=roi_config.get("x", 0),
            y=roi_config.get("y", 0),
            width=roi_config.get("width", vid_width),
            height=roi_config.get("height", vid_height),
        )

    queue_roi = None
    if queue_roi_config:
        queue_roi = ROIConfig(
            x=queue_roi_config.get("x", 0),
            y=queue_roi_config.get("y", 0),
            width=queue_roi_config.get("width", vid_width),
            height=queue_roi_config.get("height", vid_height),
        )

    virtual_line = None
    if virtual_line_config:
        virtual_line = VirtualLine(
            pt1=tuple(virtual_line_config.get("pt1", [0.0, 0.0])),
            pt2=tuple(virtual_line_config.get("pt2", [0.0, 0.0])),
        )

    pipeline_cfg = CVPipelineConfig(
        camera=camera_cfg,
        detector=detector_cfg,
        analytics=analytics_cfg,
        enable_roi=(roi is not None),
        roi=roi,
        queue_roi=queue_roi,
        virtual_line=virtual_line,
        reliability=ReliabilityConfig(min_starting_frames=1),
    )

    # Inject the initialized detector.
    # Note: We do NOT call pipeline.start() to prevent background threaded capture from looping.
    pipeline = CVPipeline(config=pipeline_cfg, detector=detector)

    per_frame_predictions: Dict[int, dict] = {}
    predicted_crossings: List[dict] = []
    track_events: List[dict] = []
    inference_latencies: List[float] = []
    loop_latencies: List[float] = []
    start_wall = time.perf_counter()

    sample_step = max(1, sample_every_n)
    effective_max = total_vid_frames if total_vid_frames > 0 else None
    if max_frames is not None:
        effective_max = min(effective_max, max_frames) if effective_max is not None else max_frames

    try:
        frame_idx = 0
        while True:
            if effective_max is not None and frame_idx >= effective_max:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = frame_idx / actual_fps
            frame_data = FrameData(
                frame=frame,
                timestamp=timestamp,
                frame_id=frame_idx,
                width=vid_width,
                height=vid_height,
                source_state=SourceState.RUNNING,
                fps=actual_fps,
            )

            state = pipeline.process_frame(frame_data, timestamp=timestamp)
            state_dict = state.to_dict()

            if frame_idx % sample_step == 0:
                per_frame_predictions[frame_idx] = {
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "current": state_dict["counts"]["current"],
                    "system_state": state_dict["system_state"],
                    "queue_people": state_dict.get("queue_people", 0),
                    "inference_latency_ms": state_dict["performance"]["inference_latency_ms"],
                    "processing_fps": state_dict["performance"]["processing_fps"],
                    "is_healthy": state_dict["is_healthy"],
                    "is_frozen": state_dict["is_frozen"],
                    "peak_count": state_dict["crowd"]["peak_count"],
                    "entries": state_dict["counts"]["entries"],
                    "exits": state_dict["counts"]["exits"],
                    "track_instances": state_dict["counts"]["track_instances"],
                }

            diag = pipeline.last_diagnostics
            for crossing in diag.get("line_crossing_events", []):
                predicted_crossings.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "track_id": crossing.get("track_id"),
                    "direction": crossing.get("direction"),
                    "crossing_point": crossing.get("crossing_point"),
                })

            for tc in diag.get("tracks_created", []):
                track_events.append({"event": "created", "frame": frame_idx, "track_id": tc.get("track_id")})
            for tl in diag.get("tracks_lost", []):
                track_events.append({"event": "lost", "frame": frame_idx, "track_id": tl.get("track_id")})
            for tt in diag.get("track_terminations", []):
                track_events.append({"event": "terminated", "frame": frame_idx, "track_id": tt.get("track_id")})

            inference_latencies.append(state_dict["performance"]["inference_latency_ms"])
            loop_latencies.append(
                state_dict["performance"]["inference_latency_ms"]
                + state_dict["performance"].get("frame_age_ms", 0)
            )

            frame_idx += 1

    finally:
        pipeline.stop()
        cap.release()

    end_wall = time.perf_counter()
    wall_elapsed = end_wall - start_wall

    final_state = pipeline.last_state
    final_dict = final_state.to_dict() if final_state else {}

    final_counts = {
        "predicted_peak": max((p["current"] for p in per_frame_predictions.values()), default=0),
        "predicted_total_entries": final_dict.get("counts", {}).get("entries", 0),
        "predicted_total_exits": final_dict.get("counts", {}).get("exits", 0),
        "predicted_net": final_dict.get("counts", {}).get("net_count", 0),
        "total_track_instances": final_dict.get("counts", {}).get("track_instances", 0),
    }

    results: Dict[str, Any] = {
        "video": {
            "path": video_path,
            "filename": Path(video_path).name,
            "fps": actual_fps,
            "frame_width": vid_width,
            "frame_height": vid_height,
            "total_frames_processed": frame_idx,
            "elapsed_wall_seconds": round(wall_elapsed, 3),
            "effective_fps": round(frame_idx / wall_elapsed, 2) if wall_elapsed > 0 else 0,
        },
        "pipeline": {
            "model_path": detector_cfg.model_path,
            "active_provider": detector.active_provider,
            "is_gpu": detector.is_gpu,
            "capacity": capacity,
            "has_roi": roi is not None,
            "has_queue_roi": queue_roi is not None,
            "has_virtual_line": virtual_line is not None,
        },
        "performance": _compute_performance_summary(inference_latencies, loop_latencies, frame_idx),
        "final_counts": final_counts,
        "predicted_crossings_count": len(predicted_crossings),
        "per_frame_predictions": per_frame_predictions,
    }

    if frame_count_mismatch is not None:
        results["frame_count_mismatch"] = frame_count_mismatch

    if gt_path and gt is not None:
        results["ground_truth"] = {
            "path": gt_path,
            "schema_version": gt["schema_version"],
            "metadata": gt["metadata"],
            "total_entries": gt["gt_metrics"].get("total_entries", 0),
            "total_exits": gt["gt_metrics"].get("total_exits", 0),
            "peak_occupancy": gt["gt_metrics"].get("peak_occupancy", 0),
            "annotation_ids": gt["gt_metrics"].get("annotation_id_count", 0),
            "frames_with_gt": len(gt["occupancy_by_frame"]),
            "crossings_tolerance_seconds": gt["crossing_tolerance_sec"],
        }

        predicted_by_frame = {f: p["current"] for f, p in per_frame_predictions.items()}
        results["occupancy_accuracy"] = compute_occupancy_metrics(
            predicted_by_frame,
            gt["occupancy_by_frame"],
            actual_fps,
        )

        results["peak_occupancy_accuracy"] = compute_peak_occupancy_metrics(
            final_counts["predicted_peak"],
            gt["gt_metrics"].get("peak_occupancy", 0),
        )

        results["crossing_accuracy"] = compute_crossing_metrics(
            predicted_crossings,
            gt["gt_crossings"],
            actual_fps,
            gt["crossing_tolerance_sec"],
        )

        results["track_diagnostics"] = compute_track_diagnostics(
            track_events,
            frame_idx,
            actual_fps,
        )

        results["evaluation_note"] = (
            "Occupancy and crossing accuracy metrics are computed only against annotated frames. "
            "Track diagnostics describe tracker-created track instances (ByteTrack IDs), "
            "NOT unique human identities. Do not interpret these as human visitor counts."
        )

    return results


def _compute_performance_summary(
    inference_latencies: List[float],
    loop_latencies: List[float],
    total_frames: int,
) -> dict:
    """Compute latency/FPS statistics from per-frame measurements."""
    def percentile(data: List[float], p: float) -> Optional[float]:
        if not data:
            return None
        s = sorted(data)
        if len(s) == 1:
            return round(s[0], 2)
        rank = (p * (len(s) - 1))
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return round(s[lo], 2)
        frac = rank - lo
        val = s[lo] * (1 - frac) + s[hi] * frac
        return round(val, 2)

    if not inference_latencies:
        return {"note": "No latency data collected"}

    return {
        "total_frames": total_frames,
        "inference_latency_ms": {
            "mean": round(sum(inference_latencies) / len(inference_latencies), 2),
            "min": round(min(inference_latencies), 2),
            "max": round(max(inference_latencies), 2),
            "p50": percentile(inference_latencies, 0.50),
            "p95": percentile(inference_latencies, 0.95),
            "p99": percentile(inference_latencies, 0.99),
        },
        "loop_latency_ms": {
            "mean": round(sum(loop_latencies) / len(loop_latencies), 2) if loop_latencies else None,
            "p95": percentile(loop_latencies, 0.95) if loop_latencies else None,
        },
    }


def print_summary(results: dict) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 70)
    print("VISIONQUEUE OFFLINE EVALUATION SUMMARY")
    print("=" * 70)

    vid = results.get("video", {})
    print(f"\nVideo: {vid.get('filename', '?')}  |  {vid.get('frame_width', 0)}x{vid.get('frame_height', 0)}  |  {vid.get('fps', 0):.1f} FPS  |  {vid.get('total_frames_processed', 0)} frames  |  {vid.get('elapsed_wall_seconds', 0):.2f}s wall")

    pipe = results.get("pipeline", {})
    print(f"  Detector: {pipe.get('model_path', '?')} | Provider: {pipe.get('active_provider', 'Unknown')} (GPU: {pipe.get('is_gpu', False)})")

    if "frame_count_mismatch" in results:
        fcm = results["frame_count_mismatch"]
        print(f"  WARNING: Frame count mismatch detected (Video frames: {fcm.get('video_frames')}, GT frames: {fcm.get('gt_frames')})")

    perf = results.get("performance", {})
    inf = perf.get("inference_latency_ms", {})
    print(f"\n  Performance (inference latency):")
    print(f"    Mean: {inf.get('mean', '?'):.2f} ms  |  p50: {inf.get('p50', '?'):.2f} ms  |  p95: {inf.get('p95', '?'):.2f} ms  |  p99: {inf.get('p99', '?'):.2f} ms")

    fc = results.get("final_counts", {})
    print(f"\n  Final Counts:")
    print(f"    Peak occupancy: {fc.get('predicted_peak', 0)}")
    print(f"    Total entries: {fc.get('predicted_total_entries', 0)}")
    print(f"    Total exits:   {fc.get('predicted_total_exits', 0)}")
    print(f"    Net:           {fc.get('predicted_net', 0)}")
    print(f"    Track instances: {fc.get('total_track_instances', 0)}  [NOTE: tracker IDs, NOT human identities]")

    if "occupancy_accuracy" in results:
        occ = results["occupancy_accuracy"]
        print(f"\n  Occupancy Accuracy (PROVISIONAL ENGINEERING TARGET — not validated):")
        if occ.get("mae") is not None:
            print(f"    Frames evaluated: {occ.get('frames_evaluated', 0)}")
            print(f"    MAE:              {occ['mae']:.2f}  |  RMSE: {occ.get('rmse', '?'):.2f}  |  Max: {occ.get('max_absolute_error', '?')}")
            print(f"    Bias:             {occ.get('bias', '?'):.2f} ({occ.get('bias_direction', '?')})")
            print(f"    Error <= 1:      {occ.get('error_within_1', '?')}")
            print(f"    Error <= 2:      {occ.get('error_within_2', '?')}")
            print(f"    Error <= 3:      {occ.get('error_within_3', '?')}")
        else:
            print(f"    {occ.get('note', 'No overlapping frames')}")

    if "peak_occupancy_accuracy" in results:
        pk = results["peak_occupancy_accuracy"]
        gt_sec = results.get("ground_truth", {})
        print(f"\n  Peak Occupancy:")
        print(f"    Predicted: {pk.get('predicted_peak', 0)}  |  GT: {gt_sec.get('peak_occupancy', '?')}  |  Error: {pk.get('peak_error', '?')}")

    if "crossing_accuracy" in results:
        print(f"\n  Crossing Accuracy:")
        for direction in ["ENTRY", "EXIT", "COMBINED"]:
            d = results["crossing_accuracy"].get(direction, {})
            if "precision" not in d:
                continue
            p_str = f"{d['precision']:.4f}" if d.get("precision") is not None else "N/A"
            r_str = f"{d['recall']:.4f}" if d.get("recall") is not None else "N/A"
            f1_str = f"{d['f1']:.4f}" if d.get("f1") is not None else "N/A"
            print(
                f"    {direction:10s}  |  Pred: {d.get('predicted', 0):3d}  |  GT: {d.get('gt', 0):3d}"
                f"  |  TP: {d.get('true_positives', 0):3d}  |  FP: {d.get('false_positives', 0):3d}"
                f"  |  FN: {d.get('false_negatives', 0):3d}"
                f"  |  P: {p_str}  |  R: {r_str}  |  F1: {f1_str}"
            )

    if "track_diagnostics" in results:
        td = results["track_diagnostics"]
        print(f"\n  Track Diagnostics (tracker IDs, NOT human identities):")
        print(f"    Total track instances: {td.get('total_track_instances', 0)}")
        print(f"    Created: {td.get('total_created', 0)}  |  Lost: {td.get('total_lost', 0)}  |  Terminated: {td.get('total_terminated', 0)}")
        print(f"    {td.get('note', '')}")

    if results.get("evaluation_note"):
        print(f"\n  NOTE: {results['evaluation_note']}")

    print("\n" + "=" * 70)


# =============================================================================
# Model Comparison
# =============================================================================

def run_model_comparison(
    video_path: str,
    gt_path: Optional[str],
    baseline_model: str,
    compare_model: str,
    output_dir: str,
    capacity: Optional[int] = None,
) -> dict:
    """Run evaluation on two models and compare results.

    Both models are evaluated on the SAME video and SAME ground truth to enable
    direct comparison.
    """
    logger.info("Running baseline model: %s", baseline_model)
    results_baseline = run_offline_evaluation(
        video_path=video_path,
        gt_path=gt_path,
        model_path=baseline_model,
        virtual_line_config=None,
        capacity=capacity,
        roi_config=None,
        queue_roi_config=None,
        output_dir=str(Path(output_dir) / "baseline"),
        verbose=False,
    )

    logger.info("Running comparison model: %s", compare_model)
    results_compare = run_offline_evaluation(
        video_path=video_path,
        gt_path=gt_path,
        model_path=compare_model,
        virtual_line_config=None,
        capacity=capacity,
        roi_config=None,
        queue_roi_config=None,
        output_dir=str(Path(output_dir) / "compare"),
        verbose=False,
    )

    comparison: Dict[str, Any] = {
        "baseline_model": baseline_model,
        "compare_model": compare_model,
        "video": results_baseline.get("video", {}),
    }

    b_perf = results_baseline.get("performance", {}).get("inference_latency_ms", {})
    c_perf = results_compare.get("performance", {}).get("inference_latency_ms", {})

    comparison["inference_latency"] = {
        "baseline_mean_ms": b_perf.get("mean"),
        "compare_mean_ms": c_perf.get("mean"),
        "speedup": round(b_perf.get("mean", 0) / c_perf.get("mean", 1), 3) if c_perf.get("mean") else None,
        "baseline_p95_ms": b_perf.get("p95"),
        "compare_p95_ms": c_perf.get("p95"),
    }

    comparison["final_counts"] = {
        "baseline": results_baseline.get("final_counts", {}),
        "compare": results_compare.get("final_counts", {}),
    }

    if gt_path:
        b_occ = results_baseline.get("occupancy_accuracy", {})
        c_occ = results_compare.get("occupancy_accuracy", {})
        comparison["occupancy_comparison"] = {
            "baseline_mae": b_occ.get("mae"),
            "compare_mae": c_occ.get("mae"),
            "mae_improvement": round(b_occ.get("mae", 0) - c_occ.get("mae", 0), 4) if b_occ.get("mae") is not None and c_occ.get("mae") is not None else None,
            "baseline_rmse": b_occ.get("rmse"),
            "compare_rmse": c_occ.get("rmse"),
        }

        b_x = results_baseline.get("crossing_accuracy", {}).get("COMBINED", {})
        c_x = results_compare.get("crossing_accuracy", {}).get("COMBINED", {})
        comparison["crossing_comparison"] = {
            "baseline_f1": b_x.get("f1"),
            "compare_f1": c_x.get("f1"),
            "baseline_precision": b_x.get("precision"),
            "compare_precision": c_x.get("precision"),
            "baseline_recall": b_x.get("recall"),
            "compare_recall": c_x.get("recall"),
        }

        b_pk = results_baseline.get("peak_occupancy_accuracy", {})
        c_pk = results_compare.get("peak_occupancy_accuracy", {})
        comparison["peak_occupancy_comparison"] = {
            "baseline_error": b_pk.get("peak_error"),
            "compare_error": c_pk.get("peak_error"),
            "baseline_absolute_error": b_pk.get("peak_absolute_error"),
            "compare_absolute_error": c_pk.get("peak_absolute_error"),
        }

    return comparison


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionQueue Offline Evaluation Framework. "
        "Processes pre-recorded video through the CV pipeline and compares against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to evaluation video file (.mp4, .avi, etc.)")
    parser.add_argument("--gt", dest="gt_path", help="Path to ground-truth JSON annotation file")
    parser.add_argument("--out", default="scratch/eval_output", help="Output directory for results")
    parser.add_argument("--model", dest="model_path", default=None, help="Path to ONNX model (default: models/yolo26m.onnx)")
    parser.add_argument("--compare-model", dest="compare_model", default=None, help="Second model path for comparison")
    parser.add_argument("--capacity", type=int, default=None, help="Venue capacity for analytics")
    parser.add_argument("--roi-x", type=int, default=None, help="ROI x coordinate")
    parser.add_argument("--roi-y", type=int, default=None, help="ROI y coordinate")
    parser.add_argument("--roi-w", type=int, default=None, help="ROI width")
    parser.add_argument("--roi-h", type=int, default=None, help="ROI height")
    parser.add_argument("--virtual-line-pt1", dest="vl_pt1", nargs=2, type=float, default=None, metavar=("X", "Y"), help="Virtual line pt1")
    parser.add_argument("--virtual-line-pt2", dest="vl_pt2", nargs=2, type=float, default=None, metavar=("X", "Y"), help="Virtual line pt2")
    parser.add_argument("--sample-every", type=int, default=1, help="Sample predictions every N frames (default: 1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.compare_model:
        comparison = run_model_comparison(
            video_path=args.video,
            gt_path=args.gt_path,
            baseline_model=args.model_path or "models/yolo26m.onnx",
            compare_model=args.compare_model,
            output_dir=args.out,
            capacity=args.capacity,
        )
        comp_path = Path(args.out) / "model_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nModel comparison saved to: {comp_path}")
        print(f"  Baseline ({comparison['baseline_model']}): MAE={comparison.get('occupancy_comparison', {}).get('baseline_mae', 'N/A')}")
        print(f"  Compare  ({comparison['compare_model']}): MAE={comparison.get('occupancy_comparison', {}).get('compare_mae', 'N/A')}")
        return

    roi_cfg = None
    if all(v is not None for v in [args.roi_x, args.roi_y, args.roi_w, args.roi_h]):
        roi_cfg = {"x": args.roi_x, "y": args.roi_y, "width": args.roi_w, "height": args.roi_h}

    vl_cfg = None
    if args.vl_pt1 and args.vl_pt2:
        vl_cfg = {"pt1": args.vl_pt1, "pt2": args.vl_pt2}

    results = run_offline_evaluation(
        video_path=args.video,
        gt_path=args.gt_path,
        model_path=args.model_path,
        virtual_line_config=vl_cfg,
        capacity=args.capacity,
        roi_config=roi_cfg,
        queue_roi_config=None,
        output_dir=args.out,
        sample_every_n=args.sample_every,
        verbose=args.verbose,
        max_frames=args.max_frames,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{Path(args.video).stem}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results saved to: %s", json_path)

    csv_path = out_dir / f"{Path(args.video).stem}_per_frame.csv"
    per_frame = results.get("per_frame_predictions", {})
    if per_frame:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if per_frame:
                fieldnames = list(next(iter(per_frame.values())).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for frame_data in per_frame.values():
                    writer.writerow(frame_data)
        logger.info("Per-frame CSV saved to: %s", csv_path)

    print_summary(results)


if __name__ == "__main__":
    main()
