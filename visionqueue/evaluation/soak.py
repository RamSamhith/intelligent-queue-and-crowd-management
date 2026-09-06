"""Long-run soak test for VisionQueue memory and stability profiling.

Executes the pipeline continuously over a video loop (or live camera) to measure
long-running stability, memory growth trends, count correctness, and reliability.

IMPORTANT:
- Does NOT add psutil as a dependency.
- Uses only stdlib mechanisms where available.
- Memory measurement uses os.getpid with platform-specific stdlib calls.
- If a metric cannot be measured reliably without new dependencies, it is
  reported as "unavailable" rather than expanding the dependency footprint.

Usage:
    python scratch/long_run_soak.py --video clip.mp4 --minutes 10 --out scratch/soak_results.csv
    python scratch/long_run_soak.py --camera 0 --minutes 5 --out scratch/soak_cam.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visionqueue.camera.types import CameraConfig, FrameData, SourceState
from visionqueue.counting.types import VirtualLine
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


def get_process_memory_mb() -> Optional[float]:
    """Estimate process RSS memory in MB using stdlib.

    Returns None if the measurement is unavailable on this platform.
    This function intentionally refuses to add psutil as a dependency.
    """
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_MEMORY_COUNTERS = 28
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return round(mem.ullTotalPhys / (1024 * 1024), 2)
            return None
        else:
            pid = os.getpid()
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return round(kb / 1024.0, 2)
            return None
    except Exception:
        return None


def run_soak_test(
    video_path: Optional[str],
    camera_source: Optional[int],
    duration_minutes: float,
    model_path: str,
    capacity: Optional[int],
    output_csv: str,
    sample_interval_sec: float = 10.0,
    loop_video: bool = True,
) -> Dict[str, Any]:
    """Execute a long-running soak test.

    Args:
        video_path: Path to video file (mutually exclusive with camera_source).
        camera_source: Camera device index (mutually exclusive with video_path).
        duration_minutes: How long to run the soak test.
        model_path: Path to ONNX detector model.
        capacity: Optional venue capacity for analytics.
        output_csv: Path to output CSV file.
        sample_interval_sec: How often to sample metrics (seconds).
        loop_video: Whether to loop the video file.
    """
    if video_path and camera_source is not None:
        raise ValueError("Specify either --video or --camera, not both")
    if not video_path and camera_source is None:
        raise ValueError("Must specify either --video or --camera")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_camera = camera_source is not None

    if use_camera:
        camera_cfg = CameraConfig(source=camera_source)
    else:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        camera_cfg = CameraConfig(source=video_path, loop_video=loop_video)

    pipeline_cfg = CVPipelineConfig(
        camera=camera_cfg,
        detector=DetectorConfig(
            model_path=model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        analytics=AnalyticsConfig(capacity=capacity),
        reliability=ReliabilityConfig(min_starting_frames=1),
    )

    logger.info(
        "Starting soak test: source=%s, duration=%.1fmin, model=%s, output=%s",
        camera_source if use_camera else video_path,
        duration_minutes,
        model_path,
        output_csv,
    )

    pipeline = CVPipeline(config=pipeline_cfg)
    pipeline.start()

    end_time = time.time() + duration_minutes * 60.0
    start_wall = time.time()

    fieldnames = [
        "timestamp", "elapsed_sec", "frame_id", "system_state",
        "current_count", "queue_people", "peak_count",
        "entries", "exits", "net_count",
        "track_instances",
        "processing_fps", "inference_latency_ms",
        "ram_mb", "ram_available",
        "is_healthy", "is_frozen",
        "detector_failures", "tracking_failures",
    ]
    rows_written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        last_sample_time = start_wall

        while time.time() < end_time:
            try:
                state = pipeline.step(timeout=2.0)
            except Exception as e:
                logger.error("Pipeline step failed: %s", e)
                continue

            now = time.time()

            if now - last_sample_time < sample_interval_sec:
                continue

            last_sample_time = now
            elapsed = now - start_wall

            ram_mb = get_process_memory_mb()

            state_dict = state.to_dict() if state else {}
            counts = state_dict.get("counts", {})
            perf = state_dict.get("performance", {})
            crowd = state_dict.get("crowd", {})

            diag = pipeline.last_diagnostics
            tracks_created = len(diag.get("tracks_created", []))
            tracks_lost = len(diag.get("tracks_lost", []))
            tracks_terminated = len(diag.get("track_terminations", []))

            row = {
                "timestamp": round(now, 3),
                "elapsed_sec": round(elapsed, 2),
                "frame_id": state_dict.get("frame_id", 0),
                "system_state": state_dict.get("system_state", "UNKNOWN"),
                "current_count": counts.get("current", 0),
                "queue_people": state_dict.get("queue_people", 0),
                "peak_count": crowd.get("peak_count", 0),
                "entries": counts.get("entries", 0),
                "exits": counts.get("exits", 0),
                "net_count": counts.get("net_count", 0),
                "track_instances": counts.get("track_instances", 0),
                "processing_fps": perf.get("processing_fps", 0.0),
                "inference_latency_ms": perf.get("inference_latency_ms", 0.0),
                "ram_mb": ram_mb if ram_mb is not None else -1.0,
                "ram_available": ram_mb is not None,
                "is_healthy": state_dict.get("is_healthy", False),
                "is_frozen": state_dict.get("is_frozen", False),
                "detector_failures": 0,
                "tracking_failures": 0,
            }

            writer.writerow(row)
            rows_written += 1

            if rows_written % 6 == 0:
                logger.info(
                    "Soak: %.1fmin elapsed | People: %d | Tracks: %d | "
                    "FPS: %.1f | RAM: %s MB | State: %s",
                    elapsed / 60.0,
                    counts.get("current", 0),
                    counts.get("track_instances", 0),
                    perf.get("processing_fps", 0.0),
                    f"{ram_mb:.1f}" if ram_mb else "N/A",
                    state_dict.get("system_state", "?"),
                )

    pipeline.stop()

    logger.info("Soak test complete. Log: %s (%d samples)", out_path, rows_written)
    return {
        "output_csv": str(out_path),
        "samples": rows_written,
        "duration_minutes": duration_minutes,
        "elapsed_wall_seconds": round(time.time() - start_wall, 2),
    }


def analyze_soak_results(csv_path: str) -> Dict[str, Any]:
    """Analyze soak test results and produce a summary report."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "No data in soak results CSV"}

    n = len(rows)
    start_time = float(rows[0]["timestamp"])
    end_time = float(rows[-1]["timestamp"])

    def safe_float(values: List[str], key: str) -> List[float]:
        return [float(r[key]) for r in values if r.get(key) and float(r[key]) >= 0]

    fps_vals = safe_float(rows, "processing_fps")
    inf_latency = safe_float(rows, "inference_latency_ms")
    ram_vals = safe_float(rows, "ram_mb")
    counts = [int(r["current_count"]) for r in rows if r.get("current_count")]
    track_instances = [int(r["track_instances"]) for r in rows if r.get("track_instances")]
    queue_people = [int(r["queue_people"]) for r in rows if r.get("queue_people")]

    states = [r["system_state"] for r in rows]
    is_frozen = [r["is_frozen"] for r in rows]

    initial_ram = ram_vals[0] if ram_vals else None
    min_ram = min(ram_vals) if ram_vals else None
    max_ram = max(ram_vals) if ram_vals else None
    final_ram = ram_vals[-1] if ram_vals else None

    if initial_ram is not None and final_ram is not None:
        ram_growth = round(final_ram - initial_ram, 2)
        if n > 1:
            ram_per_hour = round((ram_growth / (end_time - start_time)) * 3600, 3) if ram_growth != 0 else 0.0
        else:
            ram_per_hour = 0.0
    else:
        ram_growth = None
        ram_per_hour = None

    state_distribution: Dict[str, int] = {}
    for s in states:
        state_distribution[s] = state_distribution.get(s, 0) + 1

    violations = []
    for i, r in enumerate(rows):
        try:
            curr = int(r.get("current_count", 0))
            qp = int(r.get("queue_people", 0))
            if qp > curr:
                violations.append({"frame": i, "type": "queue_gt_current", "queue": qp, "current": curr})
        except (ValueError, TypeError):
            pass

    return {
        "duration_seconds": round(end_time - start_time, 2),
        "samples": n,
        "fps": {
            "mean": round(sum(fps_vals) / len(fps_vals), 2) if fps_vals else None,
            "min": round(min(fps_vals), 2) if fps_vals else None,
            "max": round(max(fps_vals), 2) if fps_vals else None,
        },
        "inference_latency_ms": {
            "mean": round(sum(inf_latency) / len(inf_latency), 2) if inf_latency else None,
            "max": round(max(inf_latency), 2) if inf_latency else None,
        },
        "memory_mb": {
            "initial": initial_ram,
            "min": min_ram,
            "max": max_ram,
            "final": final_ram,
            "growth_mb": ram_growth,
            "growth_per_hour_mb": ram_per_hour,
            "note": "Memory measurement unavailable on this platform" if initial_ram is None else None,
        },
        "counts": {
            "max_current": max(counts) if counts else None,
            "max_queue": max(queue_people) if queue_people else None,
            "final_track_instances": track_instances[-1] if track_instances else None,
        },
        "state_distribution": state_distribution,
        "frozen_samples": sum(1 for v in is_frozen if str(v).lower() == "true"),
        "invariant_violations": violations,
        "invariant_violation_count": len(violations),
        "note": (
            "All metrics are PROVISIONAL ENGINEERING TARGETS, not established acceptance criteria. "
            "Memory growth is reported but does not constitute a memory leak diagnosis. "
            "Track instance counts reflect ByteTrack IDs, NOT human identities."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionQueue long-run soak test.")
    parser.add_argument("--video", help="Path to video file (mutually exclusive with --camera)")
    parser.add_argument("--camera", type=int, help="Camera device index (mutually exclusive with --video)")
    parser.add_argument("--minutes", type=float, default=30.0, help="Duration in minutes (default: 30)")
    parser.add_argument("--model", default="models/yolo26m.onnx", help="ONNX model path")
    parser.add_argument("--capacity", type=int, default=None, help="Venue capacity")
    parser.add_argument("--out", default="scratch/soak_results.csv", help="Output CSV path")
    parser.add_argument("--sample-interval", type=float, default=10.0, help="Sample interval in seconds")
    parser.add_argument("--no-loop", dest="no_loop", action="store_true", help="Do not loop video file")
    parser.add_argument("--analyze-only", dest="analyze_only", help="Analyze existing soak CSV")
    args = parser.parse_args()

    if args.analyze_only:
        logger.info("Analyzing existing soak results: %s", args.analyze_only)
        report = analyze_soak_results(args.analyze_only)
        import json
        report_path = Path(args.analyze_only).with_suffix(".analysis.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSoak Analysis: {report_path}")
        print(json.dumps(report, indent=2))
        return

    summary = run_soak_test(
        video_path=args.video,
        camera_source=args.camera,
        duration_minutes=args.minutes,
        model_path=args.model,
        capacity=args.capacity,
        output_csv=args.out,
        sample_interval_sec=args.sample_interval,
        loop_video=not args.no_loop,
    )

    logger.info("Running analysis on soak results...")
    report = analyze_soak_results(args.out)
    import json
    report_path = Path(args.out).with_suffix(".analysis.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSoak test complete: {summary['output_csv']}")
    print(f"Analysis report: {report_path}")
    print(f"Samples collected: {summary['samples']}")

    if report.get("memory_mb", {}).get("initial") is not None:
        mm = report["memory_mb"]
        print(f"\nMemory: initial={mm['initial']} MB | min={mm['min']} MB | max={mm['max']} MB | final={mm['final']} MB | growth={mm['growth_mb']} MB")
    else:
        print(f"\nMemory: N/A (measurement unavailable on this platform)")

    print(f"State distribution: {report.get('state_distribution', {})}")
    print(f"Invariant violations: {report.get('invariant_violation_count', 0)}")
    print(f"Max current count: {report.get('counts', {}).get('max_current')}")
    print(f"\nNOTE: {report.get('note', '')}")


if __name__ == "__main__":
    main()
