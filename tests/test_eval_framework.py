"""Unit tests for the VisionQueue offline evaluation framework.

Tests the pure metric computation functions in scratch/eval_framework.py
without requiring GPU, video files, or the full CV pipeline.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scratch"))
import eval_framework


# =============================================================================
# Ground Truth Schema Validation Tests
# =============================================================================

class TestGroundTruthSchema:
    """Tests for the ground truth JSON schema."""

    def test_valid_gt_structure(self, tmp_path):
        gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test.mp4",
                "fps": 30.0,
                "frame_width": 1920,
                "frame_height": 1080,
                "total_frames": 300,
            },
            "annotation_protocol": {
                "track_identity_protocol": "New ID per entry; no re-identification assumed.",
                "crossing_tolerance_seconds": 1.0,
            },
            "metrics": {
                "total_entries": 10,
                "total_exits": 8,
                "peak_occupancy": 5,
            },
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        assert loaded["schema_version"] == 1
        assert loaded["fps"] == 30.0
        assert loaded["crossing_tolerance_sec"] == 1.0
        assert loaded["gt_metrics"]["total_entries"] == 10

    def test_missing_schema_version_defaults_to_1(self, tmp_path):
        gt = {
            "metadata": {
                "video_filename": "test.mp4",
                "fps": 25.0,
                "frame_width": 640,
                "frame_height": 480,
                "total_frames": 100,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test.",
                "crossing_tolerance_seconds": 2.0,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 0},
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        assert loaded["schema_version"] == 1
        assert loaded["fps"] == 25.0

    def test_invalid_fps_defaults_to_30(self, tmp_path):
        gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test.mp4",
                "fps": -1,
                "frame_width": 640,
                "frame_height": 480,
                "total_frames": 100,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test.",
                "crossing_tolerance_seconds": 1.0,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 0},
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        assert loaded["fps"] == 30.0

    def test_invalid_tolerance_defaults_to_1(self, tmp_path):
        gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test.mp4",
                "fps": 30.0,
                "frame_width": 640,
                "frame_height": 480,
                "total_frames": 100,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test.",
                "crossing_tolerance_seconds": -5,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 0},
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        assert loaded["crossing_tolerance_sec"] == 1.0

    def test_occupancy_parsed_by_frame(self, tmp_path):
        gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test.mp4",
                "fps": 30.0,
                "frame_width": 640,
                "frame_height": 480,
                "total_frames": 300,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test.",
                "crossing_tolerance_seconds": 1.0,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 3},
            "occupancy": [
                {"frame": 0, "person_count": 0, "confidence": "HIGH"},
                {"frame": 30, "person_count": 1, "confidence": "HIGH"},
                {"frame": 60, "person_count": 3, "confidence": "MEDIUM"},
            ],
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        occ = loaded["occupancy_by_frame"]
        assert occ[0] == 0
        assert occ[30] == 1
        assert occ[60] == 3

    def test_missing_occupancy_returns_empty_dict(self, tmp_path):
        gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test.mp4",
                "fps": 30.0,
                "frame_width": 640,
                "frame_height": 480,
                "total_frames": 100,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test.",
                "crossing_tolerance_seconds": 1.0,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 0},
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        assert loaded["occupancy_by_frame"] == {}
        assert loaded["gt_crossings"] == []

    def test_missing_optional_fields_handled(self, tmp_path):
        gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test.mp4",
                "fps": 30.0,
                "frame_width": 640,
                "frame_height": 480,
                "total_frames": 100,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test.",
                "crossing_tolerance_seconds": 1.0,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 0},
            "critical_events": [],
            "trajectories": [],
        }
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))
        loaded = eval_framework.load_ground_truth(str(path))
        assert loaded["critical_events"] == []
        assert loaded["trajectories"] == []


# =============================================================================
# Occupancy Accuracy Metrics Tests
# =============================================================================

class TestOccupancyMetrics:
    """Tests for compute_occupancy_metrics."""

    def test_perfect_match(self):
        pred = {0: 0, 30: 1, 60: 2, 90: 2, 120: 1}
        gt = {0: 0, 30: 1, 60: 2, 90: 2, 120: 1}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["max_absolute_error"] == 0
        assert result["bias"] == 0.0
        assert result["bias_direction"] == "BALANCED"

    def test_undercount(self):
        pred = {0: 0, 30: 1, 60: 1}
        gt = {0: 0, 30: 2, 60: 3}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["mae"] == pytest.approx(1.0)
        assert result["bias"] == pytest.approx(-1.0)
        assert result["bias_direction"] == "UNDERCOUNT"

    def test_overcount(self):
        pred = {0: 2, 30: 3, 60: 3}
        gt = {0: 0, 30: 1, 60: 2}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["mae"] == pytest.approx(1.667, rel=0.01)
        assert result["bias"] == pytest.approx(1.667, rel=0.01)
        assert result["bias_direction"] == "OVERCOUNT"

    def test_no_overlapping_frames(self):
        pred = {0: 0, 30: 1}
        gt = {60: 2, 90: 3}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["mae"] is None
        assert result["rmse"] is None
        assert result["frames_evaluated"] == 0

    def test_partial_overlap(self):
        pred = {0: 1, 30: 2, 60: 3, 90: 4}
        gt = {30: 2, 60: 3, 120: 5, 150: 5}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["frames_evaluated"] == 2
        assert result["frames_with_gt"] == 4
        assert result["mae"] == 0.0

    def test_zero_gt_count(self):
        pred = {0: 1, 30: 0, 60: 0}
        gt = {0: 0, 30: 0, 60: 0}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["mae"] == pytest.approx(1.0 / 3.0, rel=0.01)
        assert result["bias"] == pytest.approx(1.0 / 3.0, rel=0.01)

    def test_error_distribution(self):
        pred = {0: 4, 30: 5, 60: 6, 90: 7, 120: 8}
        gt = {0: 0, 30: 0, 60: 0, 90: 0, 120: 0}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["error_within_1"] == "0/5 (0.0%)"
        assert result["error_within_2"] == "0/5 (0.0%)"
        assert result["error_within_3"] == "0/5 (0.0%)"

    def test_all_within_1(self):
        pred = {0: 1, 30: 1, 60: 2, 90: 3, 120: 2}
        gt = {0: 0, 30: 2, 60: 3, 90: 2, 120: 3}
        result = eval_framework.compute_occupancy_metrics(pred, gt, fps=30.0)
        assert result["mae"] == pytest.approx(1.0)
        assert result["error_within_1"] == "5/5 (100.0%)"
        assert result["error_within_2"] == "5/5 (100.0%)"


# =============================================================================
# Peak Occupancy Metrics Tests
# =============================================================================

class TestPeakOccupancyMetrics:
    """Tests for compute_peak_occupancy_metrics."""

    def test_exact_match(self):
        result = eval_framework.compute_peak_occupancy_metrics(5, 5)
        assert result["predicted_peak"] == 5
        assert result["gt_peak"] == 5
        assert result["peak_error"] == 0
        assert result["peak_absolute_error"] == 0

    def test_under_peak(self):
        result = eval_framework.compute_peak_occupancy_metrics(4, 7)
        assert result["peak_error"] == -3
        assert result["peak_absolute_error"] == 3

    def test_over_peak(self):
        result = eval_framework.compute_peak_occupancy_metrics(9, 5)
        assert result["peak_error"] == 4
        assert result["peak_absolute_error"] == 4


# =============================================================================
# Crossing Metrics Tests
# =============================================================================

class TestCrossingMetrics:
    """Tests for compute_crossing_metrics."""

    def test_perfect_match_single_entry(self):
        preds = [{"frame": 30, "direction": "ENTRY"}]
        gts = [{"frame": 30, "direction": "ENTRY", "annotation_id": "P1"}]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        entry = result["ENTRY"]
        assert entry["true_positives"] == 1
        assert entry["false_positives"] == 0
        assert entry["false_negatives"] == 0
        assert entry["precision"] == 1.0
        assert entry["recall"] == 1.0
        assert entry["f1"] == 1.0

    def test_false_positive(self):
        preds = [{"frame": 30, "direction": "ENTRY"}, {"frame": 60, "direction": "ENTRY"}]
        gts = [{"frame": 30, "direction": "ENTRY", "annotation_id": "P1"}]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        entry = result["ENTRY"]
        assert entry["true_positives"] == 1
        assert entry["false_positives"] == 1
        assert entry["false_negatives"] == 0
        assert entry["precision"] == pytest.approx(0.5)

    def test_false_negative(self):
        preds = [{"frame": 30, "direction": "ENTRY"}]
        gts = [
            {"frame": 30, "direction": "ENTRY", "annotation_id": "P1"},
            {"frame": 60, "direction": "ENTRY", "annotation_id": "P2"},
        ]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        entry = result["ENTRY"]
        assert entry["true_positives"] == 1
        assert entry["false_positives"] == 0
        assert entry["false_negatives"] == 1
        assert entry["recall"] == pytest.approx(0.5)

    def test_temporal_tolerance_within_window(self):
        preds = [{"frame": 32, "direction": "ENTRY"}]
        gts = [{"frame": 30, "direction": "ENTRY", "annotation_id": "P1"}]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["true_positives"] == 1

    def test_temporal_tolerance_outside_window(self):
        preds = [{"frame": 100, "direction": "ENTRY"}]
        gts = [{"frame": 30, "direction": "ENTRY", "annotation_id": "P1"}]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["true_positives"] == 0
        assert result["ENTRY"]["false_positives"] == 1
        assert result["ENTRY"]["false_negatives"] == 1

    def test_one_to_one_matching(self):
        preds = [{"frame": 30, "direction": "ENTRY"}, {"frame": 90, "direction": "ENTRY"}]
        gts = [
            {"frame": 31, "direction": "ENTRY", "annotation_id": "P1"},
            {"frame": 95, "direction": "ENTRY", "annotation_id": "P2"},
        ]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        entry = result["ENTRY"]
        assert entry["true_positives"] == 2
        assert entry["false_positives"] == 0
        assert entry["false_negatives"] == 0
        assert entry["recall"] == 1.0
        assert entry["precision"] == 1.0

    def test_closest_gt_chosen_when_multiple(self):
        preds = [{"frame": 30, "direction": "ENTRY"}]
        gts = [
            {"frame": 30, "direction": "ENTRY", "annotation_id": "P1"},
            {"frame": 60, "direction": "ENTRY", "annotation_id": "P2"},
        ]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["true_positives"] == 1
        assert result["ENTRY"]["false_negatives"] == 1

    def test_entry_exit_separation(self):
        preds = [{"frame": 30, "direction": "ENTRY"}, {"frame": 60, "direction": "EXIT"}]
        gts = [
            {"frame": 30, "direction": "ENTRY", "annotation_id": "P1"},
            {"frame": 60, "direction": "EXIT", "annotation_id": "P1"},
        ]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["true_positives"] == 1
        assert result["EXIT"]["true_positives"] == 1
        assert result["COMBINED"]["true_positives"] == 2

    def test_empty_predictions(self):
        preds = []
        gts = [{"frame": 30, "direction": "ENTRY", "annotation_id": "P1"}]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["precision"] is None
        assert result["ENTRY"]["recall"] == 0.0
        assert result["ENTRY"]["f1"] is None

    def test_empty_ground_truth(self):
        preds = [{"frame": 30, "direction": "ENTRY"}]
        gts = []
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["precision"] == 0.0
        assert result["ENTRY"]["recall"] is None
        assert result["ENTRY"]["f1"] is None

    def test_case_insensitive_direction(self):
        preds = [{"frame": 30, "direction": "entry"}]
        gts = [{"frame": 30, "direction": "ENTRY", "annotation_id": "P1"}]
        result = eval_framework.compute_crossing_metrics(preds, gts, fps=30.0, tolerance_sec=1.0)
        assert result["ENTRY"]["true_positives"] == 1


# =============================================================================
# Track Diagnostics Tests
# =============================================================================

class TestTrackDiagnostics:
    """Tests for compute_track_diagnostics."""

    def test_no_track_events(self):
        result = eval_framework.compute_track_diagnostics([], 300, fps=30.0)
        assert result["total_track_instances"] == 0

    def test_single_track_created_and_terminated(self):
        events = [
            {"event": "created", "frame": 0, "track_id": 0},
            {"event": "terminated", "frame": 299, "track_id": 0},
        ]
        result = eval_framework.compute_track_diagnostics(events, 300, fps=30.0)
        assert result["total_track_instances"] == 1
        assert result["total_created"] == 1
        assert result["total_terminated"] == 1
        assert result["net_change"] == 0

    def test_multiple_tracks(self):
        events = [
            {"event": "created", "frame": 0, "track_id": 0},
            {"event": "created", "frame": 30, "track_id": 1},
            {"event": "created", "frame": 60, "track_id": 2},
            {"event": "lost", "frame": 150, "track_id": 0},
            {"event": "terminated", "frame": 180, "track_id": 0},
        ]
        result = eval_framework.compute_track_diagnostics(events, 300, fps=30.0)
        assert result["total_track_instances"] == 3
        assert result["total_created"] == 3
        assert result["total_lost"] == 1
        assert result["total_terminated"] == 1

    def test_note_is_included(self):
        result = eval_framework.compute_track_diagnostics([], 300, fps=30.0)
        assert "note" in result


# =============================================================================
# Performance Summary Tests
# =============================================================================

class TestPerformanceSummary:
    """Tests for _compute_performance_summary."""

    def test_empty_latencies(self):
        result = eval_framework._compute_performance_summary([], [], 0)
        assert result["note"] == "No latency data collected"

    def test_single_latency(self):
        result = eval_framework._compute_performance_summary([8.0], [10.0], 1)
        assert result["inference_latency_ms"]["mean"] == 8.0
        assert result["inference_latency_ms"]["min"] == 8.0
        assert result["inference_latency_ms"]["max"] == 8.0
        assert result["inference_latency_ms"]["p50"] == 8.0
        assert result["inference_latency_ms"]["p95"] == 8.0

    def test_percentiles(self):
        latencies = list(range(1, 101))
        result = eval_framework._compute_performance_summary(latencies, latencies, 100)
        assert result["inference_latency_ms"]["p50"] == pytest.approx(50.5, rel=0.01)
        assert result["inference_latency_ms"]["p95"] == pytest.approx(95.05, rel=0.01)
        assert result["inference_latency_ms"]["p99"] == pytest.approx(99.01, rel=0.01)

    def test_total_frames_preserved(self):
        result = eval_framework._compute_performance_summary([5.0, 6.0, 7.0], [8.0, 9.0, 10.0], 3)
        assert result["total_frames"] == 3


# =============================================================================
# Frame / Timestamp Conversion Tests
# =============================================================================

class TestFrameTimestampConversions:
    """Tests for frame/timestamp conversion utilities."""

    def test_frame_to_timestamp_30fps(self):
        assert eval_framework.frame_to_timestamp(30, 30.0) == pytest.approx(1.0)
        assert eval_framework.frame_to_timestamp(60, 30.0) == pytest.approx(2.0)
        assert eval_framework.frame_to_timestamp(0, 30.0) == 0.0

    def test_timestamp_to_frame_30fps(self):
        assert eval_framework.timestamp_to_frame(1.0, 30.0) == 30
        assert eval_framework.timestamp_to_frame(2.0, 30.0) == 60
        assert eval_framework.timestamp_to_frame(0.0, 30.0) == 0

    def test_zero_fps_handling(self):
        assert eval_framework.frame_to_timestamp(30, 0.0) == 0.0
        assert eval_framework.timestamp_to_frame(1.0, 0.0) == 0

    def test_fractional_round_trip(self):
        for fps in [15.0, 25.0, 29.97, 30.0, 60.0]:
            for frame in [0, 1, 30, 100, 1000]:
                ts = eval_framework.frame_to_timestamp(frame, fps)
                recovered = eval_framework.timestamp_to_frame(ts, fps)
                assert abs(recovered - frame) <= 1


# =============================================================================
# Queue Occupancy Invariant Tests
# =============================================================================

class TestQueueOccupancyInvariant:
    """Tests that queue_people <= current_people is maintained in evaluation output."""

    def test_queue_lte_current_in_sample(self):
        pred = {0: {"current": 3, "queue_people": 2}, 30: {"current": 5, "queue_people": 5}}
        for frame, data in pred.items():
            assert data["queue_people"] <= data["current"]


# =============================================================================
# Offline Evaluation Execution & Regression Tests
# =============================================================================

class TestOfflineEvaluationExecution:
    """Regression tests for offline evaluation finite execution, EOF handling, and detector usage."""

    @pytest.fixture
    def synthetic_eval_data(self, tmp_path):
        import cv2
        import numpy as np

        vid_path = tmp_path / "test_synth.mp4"
        width, height = 320, 240
        fps = 25.0
        num_frames = 10

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (width, height))
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f"F{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()

        gt_data = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test_synth.mp4",
                "video_id": "test_synth",
                "fps": fps,
                "frame_width": width,
                "frame_height": height,
                "total_frames": num_frames,
            },
            "annotation_protocol": {
                "track_identity_protocol": "Standard identity test",
                "crossing_tolerance_seconds": 1.0,
            },
            "occupancy": [{"frame": i, "person_count": 0, "confidence": "HIGH"} for i in range(num_frames)],
            "metrics": {
                "total_entries": 0,
                "total_exits": 0,
                "peak_occupancy": 0,
            },
        }
        gt_path = tmp_path / "test_synth_gt.json"
        gt_path.write_text(json.dumps(gt_data))

        return str(vid_path), str(gt_path), num_frames, fps, width, height

    def test_offline_eval_finite_video_terminates(self, synthetic_eval_data, tmp_path):
        vid_path, gt_path, num_frames, fps, width, height = synthetic_eval_data
        out_dir = tmp_path / "eval_out"

        results = eval_framework.run_offline_evaluation(
            video_path=vid_path,
            gt_path=gt_path,
            model_path="models/yolo26m.onnx",
            virtual_line_config=None,
            capacity=None,
            roi_config=None,
            queue_roi_config=None,
            output_dir=str(out_dir),
            sample_every_n=1,
            verbose=False,
        )

        assert results["video"]["total_frames_processed"] == num_frames
        assert len(results["per_frame_predictions"]) == num_frames
        assert sorted(results["per_frame_predictions"].keys()) == list(range(num_frames))
        assert "active_provider" in results["pipeline"]
        assert "is_gpu" in results["pipeline"]

    def test_offline_eval_no_eof_looping_or_duplicates(self, synthetic_eval_data, tmp_path):
        vid_path, gt_path, num_frames, _, _, _ = synthetic_eval_data
        out_dir = tmp_path / "eval_out_no_loop"

        results = eval_framework.run_offline_evaluation(
            video_path=vid_path,
            gt_path=None,
            model_path="models/yolo26m.onnx",
            virtual_line_config=None,
            capacity=None,
            roi_config=None,
            queue_roi_config=None,
            output_dir=str(out_dir),
            sample_every_n=1,
            verbose=False,
        )

        # Processed exactly the finite length of the video
        assert results["video"]["total_frames_processed"] == num_frames
        pred_frames = list(results["per_frame_predictions"].keys())
        assert len(pred_frames) == num_frames
        assert len(set(pred_frames)) == num_frames

    def test_offline_eval_sample_every_n(self, synthetic_eval_data, tmp_path):
        vid_path, gt_path, num_frames, _, _, _ = synthetic_eval_data
        out_dir = tmp_path / "eval_out_sampled"

        results = eval_framework.run_offline_evaluation(
            video_path=vid_path,
            gt_path=gt_path,
            model_path="models/yolo26m.onnx",
            virtual_line_config=None,
            capacity=None,
            roi_config=None,
            queue_roi_config=None,
            output_dir=str(out_dir),
            sample_every_n=3,
            verbose=False,
        )

        # Total frames sequentially processed is all 10 frames
        assert results["video"]["total_frames_processed"] == num_frames
        # But sampled predictions are only frames 0, 3, 6, 9
        sampled_frames = sorted(results["per_frame_predictions"].keys())
        assert sampled_frames == [0, 3, 6, 9]

    def test_offline_eval_gt_frame_count_mismatch_detected(self, synthetic_eval_data, tmp_path):
        vid_path, _, num_frames, fps, width, height = synthetic_eval_data
        out_dir = tmp_path / "eval_out_mismatch"

        mismatched_gt = {
            "schema_version": 1,
            "metadata": {
                "video_filename": "test_synth.mp4",
                "video_id": "test_synth",
                "fps": fps,
                "frame_width": width,
                "frame_height": height,
                "total_frames": num_frames + 15,  # Mismatch: 25 vs 10
            },
            "annotation_protocol": {
                "track_identity_protocol": "Test",
                "crossing_tolerance_seconds": 1.0,
            },
            "metrics": {"total_entries": 0, "total_exits": 0, "peak_occupancy": 0},
        }
        gt_path = tmp_path / "mismatched_gt.json"
        gt_path.write_text(json.dumps(mismatched_gt))

        results = eval_framework.run_offline_evaluation(
            video_path=vid_path,
            gt_path=str(gt_path),
            model_path="models/yolo26m.onnx",
            virtual_line_config=None,
            capacity=None,
            roi_config=None,
            queue_roi_config=None,
            output_dir=str(out_dir),
            sample_every_n=1,
            verbose=False,
        )

        assert "frame_count_mismatch" in results
        assert results["frame_count_mismatch"]["video_frames"] == num_frames
        assert results["frame_count_mismatch"]["gt_frames"] == num_frames + 15

    def test_offline_eval_cuda_provider_selection_mocked(self, synthetic_eval_data, tmp_path, monkeypatch):
        from visionqueue.detection.detector import PersonDetector

        orig_init = PersonDetector._load_model

        def mock_load_model(self):
            orig_init(self)
            self._active_provider = "CUDAExecutionProvider"

        monkeypatch.setattr(PersonDetector, "_load_model", mock_load_model)

        vid_path, _, _, _, _, _ = synthetic_eval_data
        out_dir = tmp_path / "eval_out_cuda"

        results = eval_framework.run_offline_evaluation(
            video_path=vid_path,
            gt_path=None,
            model_path="models/yolo26m.onnx",
            virtual_line_config=None,
            capacity=None,
            roi_config=None,
            queue_roi_config=None,
            output_dir=str(out_dir),
            sample_every_n=1,
            verbose=False,
        )

        assert results["pipeline"]["active_provider"] == "CUDAExecutionProvider"
        assert results["pipeline"]["is_gpu"] is True

    def test_offline_eval_cpu_fallback_valid(self, synthetic_eval_data, tmp_path):
        vid_path, _, _, _, _, _ = synthetic_eval_data
        out_dir = tmp_path / "eval_out_cpu"

        results = eval_framework.run_offline_evaluation(
            video_path=vid_path,
            gt_path=None,
            model_path="models/yolo26m.onnx",
            virtual_line_config=None,
            capacity=None,
            roi_config=None,
            queue_roi_config=None,
            output_dir=str(out_dir),
            sample_every_n=1,
            verbose=False,
        )

        # When CUDA DLLs are unavailable on the host, CPU fallback is cleanly reported
        assert results["pipeline"]["active_provider"] in ("CPUExecutionProvider", "CUDAExecutionProvider")
        if results["pipeline"]["active_provider"] == "CPUExecutionProvider":
            assert results["pipeline"]["is_gpu"] is False

    def test_live_camera_looping_behavior_preserved(self, synthetic_eval_data):
        from visionqueue.camera import CameraConfig, CameraSource, SourceState
        import time

        vid_path, _, num_frames, _, _, _ = synthetic_eval_data
        # Explicit live-camera replay with loop_video=True
        cfg = CameraConfig(source=vid_path, loop_video=True)
        source = CameraSource(cfg)
        source.start()

        try:
            time.sleep(0.5)
            assert source.state == SourceState.RUNNING
            frame = source.get_latest_frame()
            assert frame is not None
        finally:
            source.stop()

    def test_offline_eval_missing_video_fails(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            eval_framework.run_offline_evaluation(
                video_path=str(tmp_path / "non_existent.mp4"),
                gt_path=None,
                model_path="models/yolo26m.onnx",
                virtual_line_config=None,
                capacity=None,
                roi_config=None,
                queue_roi_config=None,
                output_dir=str(tmp_path / "out"),
            )

