"""Unit tests for the VisionQueue PersonDetector module.

Tests cover:
- Detection types (Detection, DetectorConfig)
- PersonDetector construction and model loading
- Preprocessing (letterbox correctness)
- Postprocessing (coordinate mapping, filtering, clipping)
- Empty/invalid input handling
- detect_timed() latency measurement
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure visionqueue package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visionqueue.detection.types import Detection, DetectorConfig
from visionqueue.detection.detector import PersonDetector


# ===================================================================
# Detection data-class tests
# ===================================================================

class TestDetection:
    """Tests for the Detection frozen dataclass."""

    def test_to_dict_basic(self):
        det = Detection(bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.92, class_id=0)
        d = det.to_dict()
        assert d == {"bbox": [10.0, 20.0, 100.0, 200.0], "confidence": 0.92, "class_id": 0}

    def test_immutable(self):
        det = Detection(bbox=(0, 0, 1, 1), confidence=0.5, class_id=0)
        with pytest.raises(AttributeError):
            det.confidence = 0.9  # type: ignore

    def test_to_dict_roundtrip(self):
        det = Detection(bbox=(1.23, 4.56, 7.89, 10.11), confidence=0.1234, class_id=0)
        d = det.to_dict()
        det2 = Detection(bbox=tuple(d["bbox"]), confidence=d["confidence"], class_id=d["class_id"])
        assert det == det2


# ===================================================================
# DetectorConfig tests
# ===================================================================

class TestDetectorConfig:
    """Tests for DetectorConfig defaults and customization."""

    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.model_path == "models/yolo26m_crowd.onnx"
        assert cfg.input_size == (640, 640)
        assert cfg.confidence_threshold == 0.25
        assert cfg.person_class_id == 0
        assert "CUDAExecutionProvider" in cfg.providers
        assert "CPUExecutionProvider" in cfg.providers

    def test_custom_threshold(self):
        cfg = DetectorConfig(confidence_threshold=0.5)
        assert cfg.confidence_threshold == 0.5

    def test_custom_providers_cpu_only(self):
        cfg = DetectorConfig(providers=["CPUExecutionProvider"])
        assert cfg.providers == ["CPUExecutionProvider"]

    def test_staging_env_var_override(self, monkeypatch):
        monkeypatch.setenv("VISIONQUEUE_DETECTOR_MODEL_PATH", "models/custom_staging.onnx")
        cfg = DetectorConfig()
        assert cfg.model_path == "models/custom_staging.onnx"

    def test_rollback_env_var_override(self, monkeypatch):
        monkeypatch.setenv("VISIONQUEUE_DETECTOR_MODEL_PATH", "models/yolo26m.onnx")
        cfg = DetectorConfig()
        assert cfg.model_path == "models/yolo26m.onnx"

    def test_explicit_model_path_precedence(self, monkeypatch):
        monkeypatch.setenv("VISIONQUEUE_DETECTOR_MODEL_PATH", "models/yolo26m_crowd.onnx")
        cfg = DetectorConfig(model_path="models/custom.onnx")
        assert cfg.model_path == "models/custom.onnx"

    def test_env_var_unset_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("VISIONQUEUE_DETECTOR_MODEL_PATH", raising=False)
        cfg = DetectorConfig()
        assert cfg.model_path == "models/yolo26m_crowd.onnx"


# ===================================================================
# PersonDetector construction / model loading
# ===================================================================

def _make_mock_session():
    """Create a mock ort.InferenceSession."""
    sess = MagicMock()
    sess.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    inp = MagicMock()
    inp.name = "images"
    inp.shape = [1, 3, 640, 640]
    sess.get_inputs.return_value = [inp]

    # Default empty detections
    sess.run.return_value = [np.zeros((1, 300, 6), dtype=np.float32)]
    return sess


class TestPersonDetectorInit:
    """Tests for PersonDetector initialization."""

    def test_model_not_found_raises(self):
        cfg = DetectorConfig(model_path="nonexistent.onnx")
        with pytest.raises(FileNotFoundError, match="nonexistent.onnx"):
            PersonDetector(cfg)

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_loads_model_and_reports_provider(self, mock_isfile, mock_ort_cls):
        mock_ort_cls.return_value = _make_mock_session()
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        assert det.active_provider == "CUDAExecutionProvider"
        assert det.input_shape == [1, 3, 640, 640]
        assert det.is_gpu is True

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_cpu_fallback_detected(self, mock_isfile, mock_ort_cls):
        sess = _make_mock_session()
        sess.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort_cls.return_value = sess
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        assert det.active_provider == "CPUExecutionProvider"
        assert det.is_gpu is False


# ===================================================================
# Preprocessing
# ===================================================================

class TestPreprocessing:
    """Tests for the static _preprocess method."""

    def test_square_input_no_padding(self):
        """A 640x640 frame should produce no padding."""
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        blob, scale, pad_l, pad_t = PersonDetector._preprocess(frame, 640, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert scale == pytest.approx(1.0)
        assert pad_l == 0
        assert pad_t == 0

    def test_landscape_frame_padding(self):
        """A 640x480 (landscape) frame should get vertical padding (letterbox bars on top/bottom)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, scale, pad_l, pad_t = PersonDetector._preprocess(frame, 640, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert scale == pytest.approx(1.0)
        assert pad_l == 0
        assert pad_t > 0  # top padding for centering

    def test_portrait_frame_padding(self):
        """A 480x640 (portrait) frame should get horizontal padding."""
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        blob, scale, pad_l, pad_t = PersonDetector._preprocess(frame, 640, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert pad_l > 0  # left padding for centering

    def test_normalization_range(self):
        """Pixel values should be normalized to [0, 1]."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        blob, _, _, _ = PersonDetector._preprocess(frame, 640, 640)
        assert blob.max() <= 1.0
        assert blob.min() >= 0.0

    def test_output_is_float32(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        blob, _, _, _ = PersonDetector._preprocess(frame, 640, 640)
        assert blob.dtype == np.float32

    def test_output_is_contiguous(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        blob, _, _, _ = PersonDetector._preprocess(frame, 640, 640)
        assert blob.flags["C_CONTIGUOUS"]


# ===================================================================
# Postprocessing
# ===================================================================

class TestPostprocessing:
    """Tests for postprocessing output to Detection list."""

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def _make_detector(self, mock_isfile, mock_ort_cls, **kwargs):
        mock_ort_cls.return_value = _make_mock_session()
        return PersonDetector(DetectorConfig(model_path="fake.onnx", **kwargs))

    def test_filters_below_confidence(self):
        det = self._make_detector(confidence_threshold=0.5)
        # row: [x1, y1, x2, y2, conf, class_id]
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [50, 50, 200, 200, 0.3, 0]  # below threshold
        raw[0, 1] = [50, 50, 200, 200, 0.7, 0]  # above threshold
        results = det._postprocess(raw, 640, 480, 1.0, 0, 80)
        assert len(results) == 1
        assert results[0].confidence == pytest.approx(0.7, abs=0.01)

    def test_filters_non_person(self):
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [10, 10, 100, 100, 0.9, 0]  # person
        raw[0, 1] = [10, 10, 100, 100, 0.9, 2]  # car → filtered
        raw[0, 2] = [10, 10, 100, 100, 0.9, 7]  # truck → filtered
        results = det._postprocess(raw, 640, 480, 1.0, 0, 0)
        assert len(results) == 1
        assert results[0].class_id == 0

    def test_clips_to_frame_boundaries(self):
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        # bbox extends outside frame after de-padding
        raw[0, 0] = [-50, -50, 700, 700, 0.9, 0]
        results = det._postprocess(raw, 640, 480, 1.0, 0, 0)
        assert len(results) == 1
        x1, y1, x2, y2 = results[0].bbox
        assert x1 >= 0.0
        assert y1 >= 0.0
        assert x2 <= 640.0
        assert y2 <= 480.0

    def test_empty_when_no_valid_detections(self):
        det = self._make_detector(confidence_threshold=0.5)
        raw = np.zeros((1, 300, 6), dtype=np.float32)  # all zeros → conf=0
        results = det._postprocess(raw, 640, 480, 1.0, 0, 0)
        assert results == []

    def test_coordinate_remapping_with_scale(self):
        """Verify coordinates are correctly unscaled from letterboxed space."""
        det = self._make_detector(confidence_threshold=0.1)
        # Simulate scale=0.5, pad_left=10, pad_top=20
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [110, 120, 210, 220, 0.95, 0]  # in letterboxed space
        results = det._postprocess(raw, 640, 480, 0.5, 10, 20)
        assert len(results) == 1
        x1, y1, x2, y2 = results[0].bbox
        # (110 - 10) / 0.5 = 200, (120 - 20) / 0.5 = 200
        # (210 - 10) / 0.5 = 400, (220 - 20) / 0.5 = 400
        assert x1 == pytest.approx(200.0, abs=0.5)
        assert y1 == pytest.approx(200.0, abs=0.5)
        assert x2 == pytest.approx(400.0, abs=0.5)
        assert y2 == pytest.approx(400.0, abs=0.5)

    def test_rejects_nan_output(self):
        """NaN bbox or confidence must not propagate to Detection."""
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [50, 50, 200, 200, float("nan"), 0]
        assert det._postprocess(raw, 640, 480, 1.0, 0, 0) == []

    def test_rejects_pos_inf_output(self):
        """+Inf must not propagate (would break IoU/Kalman)."""
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [float("inf"), float("inf"), float("inf"), float("inf"), 0.9, 0]
        assert det._postprocess(raw, 640, 480, 1.0, 0, 0) == []

    def test_rejects_neg_inf_output(self):
        """-Inf must not propagate."""
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [float("-inf"), float("-inf"), float("-inf"), float("-inf"), 0.9, 0]
        assert det._postprocess(raw, 640, 480, 1.0, 0, 0) == []

    def test_mixed_valid_and_invalid_rejects_batch(self):
        """Whole-array guard rejects entire batch if any row is non-finite."""
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [50, 50, 200, 200, 0.95, 0]  # valid row
        raw[0, 1] = [60, 60, 210, 210, float("nan"), 0]  # invalid row
        # Policy: any non-finite → entire batch rejected to protect downstream.
        assert det._postprocess(raw, 640, 480, 1.0, 0, 0) == []

    def test_valid_output_unchanged_by_guard(self):
        """Fully finite output must still produce detections."""
        det = self._make_detector(confidence_threshold=0.1)
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [100, 100, 300, 400, 0.85, 0]
        raw[0, 1] = [200, 150, 350, 450, 0.60, 0]
        results = det._postprocess(raw, 640, 480, 1.0, 0, 80)
        assert len(results) == 2


# ===================================================================
# detect() and detect_timed() end-to-end (mocked inference)
# ===================================================================

class TestDetectAPI:
    """Tests for detect() and detect_timed() with mocked ONNX session."""

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_detect_returns_list(self, mock_isfile, mock_ort_cls):
        mock_ort_cls.return_value = _make_mock_session()
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_detect_with_persons(self, mock_isfile, mock_ort_cls):
        sess = _make_mock_session()
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [100, 100, 300, 400, 0.85, 0]  # person
        output[0, 1] = [200, 150, 350, 450, 0.60, 0]  # person
        sess.run.return_value = [output]
        mock_ort_cls.return_value = sess

        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 2
        assert all(r.class_id == 0 for r in results)

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_detect_none_frame(self, mock_isfile, mock_ort_cls):
        mock_ort_cls.return_value = _make_mock_session()
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        assert det.detect(None) == []

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_detect_empty_frame(self, mock_isfile, mock_ort_cls):
        mock_ort_cls.return_value = _make_mock_session()
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        assert det.detect(np.array([])) == []

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_detect_timed_returns_latency(self, mock_isfile, mock_ort_cls):
        mock_ort_cls.return_value = _make_mock_session()
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results, ms = det.detect_timed(frame)
        assert isinstance(results, list)
        assert ms >= 0.0

    @patch("visionqueue.detection.detector.ort.InferenceSession")
    @patch("os.path.isfile", return_value=True)
    def test_warm_up_calls_session(self, mock_isfile, mock_ort_cls):
        sess = _make_mock_session()
        mock_ort_cls.return_value = sess
        det = PersonDetector(DetectorConfig(model_path="fake.onnx"))
        det.warm_up(rounds=5)
        assert sess.run.call_count >= 5
