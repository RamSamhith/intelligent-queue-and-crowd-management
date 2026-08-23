"""YOLO26n ONNX Runtime person detector for VisionQueue.

Loads the exported YOLO26n end-to-end ONNX model once, runs GPU-accelerated
inference on individual BGR frames, and returns person detections in the
VisionQueue Detection Contract format.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure torch CUDA DLLs are discoverable by ONNX Runtime on Windows
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
except Exception:
    pass

import onnxruntime as ort

from visionqueue.detection.types import Detection, DetectorConfig

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLO26n ONNX Runtime person detector with single-load, reusable session.

    Usage::

        detector = PersonDetector(DetectorConfig(model_path="models/yolo26n.onnx"))
        detections = detector.detect(bgr_frame)
        for det in detections:
            print(det.to_dict())
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self._config = config or DetectorConfig()
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = ""
        self._input_shape: List[int] = []
        self._active_provider: str = ""

        self._load_model()

    @property
    def config(self) -> DetectorConfig:
        return self._config

    @property
    def active_provider(self) -> str:
        """Return the actual ONNX Runtime execution provider in use."""
        return self._active_provider

    @property
    def input_shape(self) -> List[int]:
        """Return the ONNX model input shape [B, C, H, W]."""
        return list(self._input_shape)

    @property
    def is_gpu(self) -> bool:
        """Return True if CUDAExecutionProvider is the primary active provider."""
        return "CUDA" in self._active_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run person detection on a single BGR frame.

        Args:
            frame: OpenCV BGR image as uint8 numpy array (H, W, 3).

        Returns:
            List of Detection instances for persons above the confidence threshold.
        """
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        input_h, input_w = self._config.input_size

        # Preprocess: letterbox + normalize
        input_tensor, scale, pad_left, pad_top = self._preprocess(
            frame, input_w, input_h
        )

        # Inference (model loaded once in __init__)
        raw_output = self._session.run(None, {self._input_name: input_tensor})[0]

        # Postprocess: extract person detections in original coords
        detections = self._postprocess(
            raw_output, orig_w, orig_h, scale, pad_left, pad_top
        )
        return detections

    def detect_timed(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """Run person detection and return (detections, inference_ms).

        The returned latency measures only ONNX Runtime session.run(),
        excluding pre/post-processing.
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        orig_h, orig_w = frame.shape[:2]
        input_h, input_w = self._config.input_size

        input_tensor, scale, pad_left, pad_top = self._preprocess(
            frame, input_w, input_h
        )

        t0 = time.perf_counter()
        raw_output = self._session.run(None, {self._input_name: input_tensor})[0]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._postprocess(
            raw_output, orig_w, orig_h, scale, pad_left, pad_top
        )
        return detections, inference_ms

    def warm_up(self, rounds: int = 3) -> None:
        """Run warm-up inferences to initialize CUDA kernels."""
        dummy = np.zeros(
            [1, 3, self._config.input_size[0], self._config.input_size[1]],
            dtype=np.float32,
        )
        for _ in range(rounds):
            self._session.run(None, {self._input_name: dummy})
        logger.info("PersonDetector warm-up complete (%d rounds).", rounds)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load ONNX model and create InferenceSession (called once)."""
        model_path = self._config.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        logger.info("Loading ONNX model from '%s'...", model_path)
        self._session = ort.InferenceSession(
            model_path, providers=self._config.providers
        )

        active_providers = self._session.get_providers()
        self._active_provider = active_providers[0] if active_providers else "Unknown"

        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_shape = inp.shape

        logger.info(
            "ONNX model loaded. Input: '%s' %s. Active provider: %s",
            self._input_name,
            self._input_shape,
            self._active_provider,
        )

    @staticmethod
    def _preprocess(
        frame: np.ndarray, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, float, int, int]:
        """Letterbox resize, BGR→RGB, normalize, CHW, batch.

        Returns:
            (input_tensor, scale, pad_left, pad_top)
        """
        orig_h, orig_w = frame.shape[:2]
        scale = min(target_h / orig_h, target_w / orig_w)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        pad_w = (target_w - new_w) / 2.0
        pad_h = (target_h - new_h) / 2.0
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # BGR → RGB, uint8 → float32 [0,1], HWC → CHW, add batch
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        blob = np.ascontiguousarray(blob)

        return blob, scale, left, top

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_left: int,
        pad_top: int,
    ) -> List[Detection]:
        """Decode YOLO26n end-to-end output [1, 300, 6] into Detection list.

        Each row: [x1, y1, x2, y2, confidence, class_id] in letterboxed space.
        """
        conf_thresh = self._config.confidence_threshold
        person_id = self._config.person_class_id
        detections: List[Detection] = []

        proposals = raw[0]  # (300, 6)
        for row in proposals:
            conf = float(row[4])
            if conf < conf_thresh:
                continue
            cls = int(row[5])
            if cls != person_id:
                continue

            # Map from letterboxed 640×640 back to original frame coords
            x1 = (float(row[0]) - pad_left) / scale
            y1 = (float(row[1]) - pad_top) / scale
            x2 = (float(row[2]) - pad_left) / scale
            y2 = (float(row[3]) - pad_top) / scale

            # Clip to frame boundaries
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            detections.append(Detection(
                bbox=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                confidence=round(conf, 4),
                class_id=cls,
            ))

        return detections
