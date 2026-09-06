"""Unit tests for the VisionQueue long-run soak analysis.

Tests the analysis function with synthetic CSV data, the memory measurement
utility, and invariant checks. Does not require GPU or actual video.
"""

import csv
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scratch"))
import long_run_soak


# =============================================================================
# Memory Measurement Tests
# =============================================================================

class TestGetProcessMemory:
    """Tests for the stdlib-only memory measurement."""

    def test_returns_value_or_none(self):
        result = long_run_soak.get_process_memory_mb()
        if platform.system() == "Windows":
            assert result is None or isinstance(result, (int, float))
            if result is not None:
                assert result >= 0
        elif platform.system() == "Linux":
            assert result is None or isinstance(result, (int, float))

    def test_no_psutil_imported(self):
        """Verify the soak tool does not require psutil."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'scratch'); "
             "import long_run_soak; "
             "assert 'psutil' not in sys.modules, 'psutil was imported'; "
             "print('no_psutil_ok')"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            assert "no_psutil_ok" in result.stdout


# =============================================================================
# Soak Analysis Tests
# =============================================================================

class TestAnalyzeSoakResults:
    """Tests for analyze_soak_results with synthetic CSV data."""

    def _write_synthetic_csv(self, tmp_path, rows):
        path = tmp_path / "soak.csv"
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
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return str(path)

    def test_empty_csv_returns_error(self, tmp_path):
        path = self._write_synthetic_csv(tmp_path, [])
        result = long_run_soak.analyze_soak_results(path)
        assert "error" in result

    def test_basic_summary(self, tmp_path):
        rows = [
            {
                "timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 1,
                "system_state": "LIVE", "current_count": 3, "queue_people": 2,
                "peak_count": 3, "entries": 0, "exits": 0, "net_count": 0,
                "track_instances": 1, "processing_fps": 30.0,
                "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
                "is_healthy": True, "is_frozen": False,
            },
            {
                "timestamp": 10.0, "elapsed_sec": 10.0, "frame_id": 300,
                "system_state": "LIVE", "current_count": 3, "queue_people": 2,
                "peak_count": 3, "entries": 0, "exits": 0, "net_count": 0,
                "track_instances": 1, "processing_fps": 29.5,
                "inference_latency_ms": 7.2, "ram_mb": 202.0, "ram_available": True,
                "is_healthy": True, "is_frozen": False,
            },
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["samples"] == 2
        assert result["duration_seconds"] == 10.0
        assert result["fps"]["mean"] == 29.75
        assert result["counts"]["max_current"] == 3
        assert result["counts"]["max_queue"] == 2
        assert result["state_distribution"]["LIVE"] == 2

    def test_memory_growth_tracking(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "LIVE",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
            {"timestamp": 3600.0, "elapsed_sec": 3600.0, "frame_id": 100000, "system_state": "LIVE",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 250.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["memory_mb"]["initial"] == 200.0
        assert result["memory_mb"]["final"] == 250.0
        assert result["memory_mb"]["growth_mb"] == 50.0
        assert result["memory_mb"]["growth_per_hour_mb"] == 50.0

    def test_memory_unavailable_handling(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "LIVE",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": -1.0, "ram_available": False,
             "is_healthy": True, "is_frozen": False},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["memory_mb"]["initial"] is None
        assert "unavailable" in (result["memory_mb"].get("note") or "")

    def test_invariant_violations_detected(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "LIVE",
             "current_count": 3, "queue_people": 5, "peak_count": 5, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["invariant_violation_count"] == 1
        assert result["invariant_violations"][0]["type"] == "queue_gt_current"

    def test_no_invariant_violations(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "LIVE",
             "current_count": 5, "queue_people": 3, "peak_count": 5, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["invariant_violation_count"] == 0

    def test_state_distribution(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "STARTING",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": False, "is_frozen": False},
            {"timestamp": 1.0, "elapsed_sec": 1.0, "frame_id": 30, "system_state": "LIVE",
             "current_count": 3, "queue_people": 2, "peak_count": 3, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 2, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
            {"timestamp": 2.0, "elapsed_sec": 2.0, "frame_id": 60, "system_state": "DEGRADED",
             "current_count": 2, "queue_people": 1, "peak_count": 3, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 2, "processing_fps": 15.0,
             "inference_latency_ms": 50.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": False, "is_frozen": True},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["state_distribution"].get("STARTING") == 1
        assert result["state_distribution"].get("LIVE") == 1
        assert result["state_distribution"].get("DEGRADED") == 1
        assert result["frozen_samples"] == 1

    def test_fps_summary(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "LIVE",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 25.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
            {"timestamp": 10.0, "elapsed_sec": 10.0, "frame_id": 250, "system_state": "LIVE",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 35.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert result["fps"]["min"] == 25.0
        assert result["fps"]["max"] == 35.0
        assert result["fps"]["mean"] == 30.0

    def test_note_included_in_analysis(self, tmp_path):
        rows = [
            {"timestamp": 0.0, "elapsed_sec": 0.0, "frame_id": 0, "system_state": "LIVE",
             "current_count": 0, "queue_people": 0, "peak_count": 0, "entries": 0, "exits": 0,
             "net_count": 0, "track_instances": 0, "processing_fps": 30.0,
             "inference_latency_ms": 7.0, "ram_mb": 200.0, "ram_available": True,
             "is_healthy": True, "is_frozen": False},
        ]
        path = self._write_synthetic_csv(tmp_path, rows)
        result = long_run_soak.analyze_soak_results(path)
        assert "note" in result
        assert "PROVISIONAL" in result["note"]


# =============================================================================
# Argument Validation Tests
# =============================================================================

class TestArgumentValidation:
    """Tests for argument validation in run_soak_test."""

    def test_video_and_camera_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            long_run_soak.run_soak_test(
                video_path="dummy.mp4",
                camera_source=0,
                duration_minutes=0.1,
                model_path="models/yolo26m.onnx",
                capacity=None,
                output_csv="dummy.csv",
            )

    def test_neither_video_nor_camera_raises(self):
        with pytest.raises(ValueError, match="Must specify"):
            long_run_soak.run_soak_test(
                video_path=None,
                camera_source=None,
                duration_minutes=0.1,
                model_path="models/yolo26m.onnx",
                capacity=None,
                output_csv="dummy.csv",
            )
