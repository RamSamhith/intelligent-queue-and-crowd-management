"""Type definitions for the VisionQueue detection subsystem."""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

DEFAULT_MODEL_PATH: str = "models/yolo26m_crowd.onnx"
ROLLBACK_MODEL_PATH: str = "models/yolo26m.onnx"
STAGING_MODEL_ENV_VAR: str = "VISIONQUEUE_DETECTOR_MODEL_PATH"


@dataclass(frozen=True)
class Detection:
    """A single object detection in the VisionQueue Detection Contract.

    Coordinates are in original (un-letterboxed) frame pixel space, clipped
    to the frame boundaries.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in original frame coordinates.
        confidence: Detection confidence score in [0.0, 1.0].
        class_id: COCO class id (0 = person).
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int

    def to_dict(self) -> dict:
        """Serialize to the VisionQueue detection contract dict."""
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
        }


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the YOLO26n ONNX person detector.

    Attributes:
        model_path: Path to the ONNX model file. Defaults to models/yolo26m_crowd.onnx
            (production crowd-specialized model), or the path specified by the
            VISIONQUEUE_DETECTOR_MODEL_PATH environment variable (which can be used
            to select models/yolo26m.onnx for rollback or staging evaluation).
        input_size: Model input spatial dimensions (height, width).
        confidence_threshold: Minimum detection confidence to accept.
        person_class_id: COCO class id for person filtering.
        providers: Ordered list of ONNX Runtime execution providers to request.
    """
    model_path: str = field(
        default_factory=lambda: os.environ.get(STAGING_MODEL_ENV_VAR, DEFAULT_MODEL_PATH)
    )
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    person_class_id: int = 0
    providers: List[str] = field(default_factory=lambda: [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])
