"""Regression tests for staging evaluator telemetry and model configuration invariants."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scratch"))

from visionqueue.detection.types import (
    DetectorConfig,
    DEFAULT_MODEL_PATH,
    ROLLBACK_MODEL_PATH,
    STAGING_MODEL_ENV_VAR,
)
import evaluate_staging_comparison as esc


class TestEvaluatorTelemetry:
    """Test memory and hardware telemetry functions in evaluate_staging_comparison."""

    def test_get_cpu_ram_mb_positive(self):
        """get_cpu_ram_mb must return a realistic positive process RSS in MB."""
        rss = esc.get_cpu_ram_mb()
        assert isinstance(rss, float)
        assert rss > 0.0, f"Expected positive process RSS, got {rss}"
        # A running Python process with loaded modules should be at least 5 MB and under 64 GB
        assert 5.0 <= rss <= 65536.0, f"RSS {rss} MB is outside reasonable bounds"

    def test_get_gpu_vram_mb(self):
        """get_gpu_vram_mb must return a non-negative float."""
        vram = esc.get_gpu_vram_mb()
        assert isinstance(vram, float)
        assert vram >= 0.0


class TestModelSelectionInvariants:
    """Verify production safety invariants for detector model selection."""

    def test_default_is_production_model(self, monkeypatch):
        monkeypatch.delenv(STAGING_MODEL_ENV_VAR, raising=False)
        cfg = DetectorConfig()
        assert cfg.model_path == DEFAULT_MODEL_PATH
        assert cfg.model_path == "models/yolo26m_crowd.onnx"
        assert DEFAULT_MODEL_PATH == "models/yolo26m_crowd.onnx"

    def test_rollback_env_override(self, monkeypatch):
        monkeypatch.setenv(STAGING_MODEL_ENV_VAR, ROLLBACK_MODEL_PATH)
        cfg = DetectorConfig()
        assert cfg.model_path == "models/yolo26m.onnx"
        assert ROLLBACK_MODEL_PATH == "models/yolo26m.onnx"

    def test_staging_env_override(self, monkeypatch):
        monkeypatch.setenv(STAGING_MODEL_ENV_VAR, "models/custom_staging.onnx")
        cfg = DetectorConfig()
        assert cfg.model_path == "models/custom_staging.onnx"

    def test_explicit_model_path_precedence(self, monkeypatch):
        monkeypatch.setenv(STAGING_MODEL_ENV_VAR, "models/yolo26m_crowd.onnx")
        cfg = DetectorConfig(model_path="models/custom.onnx")
        assert cfg.model_path == "models/custom.onnx"
