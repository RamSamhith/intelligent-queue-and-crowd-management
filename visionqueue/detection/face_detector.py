"""YuNet face detector wrapper for VisionQueue.

Runs OpenCV FaceDetectorYN for face presence detection without biometric identity,
facial recognition, or embedding extraction.
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetection:
    """A single face detection record (bounding box and confidence).

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in original frame coordinates.
        confidence: Detection confidence score in [0.0, 1.0].
    """
    bbox: Tuple[float, float, float, float]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class FaceDetectorConfig:
    """Configuration for YuNet face detector.

    Attributes:
        model_path: Path to YuNet ONNX model file.
        confidence_threshold: Minimum face detection confidence.
        nms_threshold: Non-maximum suppression threshold.
        top_k: Keep top-k bounding boxes before NMS.
    """
    model_path: str = "models/face_detection_yunet_2023mar.onnx"
    confidence_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000


class FaceDetector:
    """OpenCV YuNet face detector with single-load reusable session.

    CRITICAL PRIVACY NOTICE:
    This detector detects face presence and bounding boxes only. It does NOT
    extract facial embeddings, create face identity profiles, or match identities.
    """

    def __init__(self, config: Optional[FaceDetectorConfig] = None) -> None:
        self._config = config or FaceDetectorConfig()
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._current_size: Tuple[int, int] = (0, 0)

        self._load_model()

    @property
    def config(self) -> FaceDetectorConfig:
        return self._config

    def _load_model(self) -> None:
        model_path = self._config.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YuNet ONNX model not found: {model_path}")

        logger.info("Loading YuNet face detector from '%s'...", model_path)
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self._config.confidence_threshold,
            nms_threshold=self._config.nms_threshold,
            top_k=self._config.top_k,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a BGR frame.

        Args:
            frame: OpenCV BGR image as uint8 numpy array (H, W, 3).

        Returns:
            List of FaceDetection records.
        """
        if frame is None or frame.size == 0 or self._detector is None:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self._current_size:
            self._detector.setInputSize((w, h))
            self._current_size = (w, h)

        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None:
            return []

        detections: List[FaceDetection] = []
        for face in raw_faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, bw, bh = face[0:4]
            conf = float(face[-1])
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(w), float(x + bw))
            y2 = min(float(h), float(y + bh))

            detections.append(FaceDetection(
                bbox=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                confidence=round(conf, 4),
            ))

        return detections
