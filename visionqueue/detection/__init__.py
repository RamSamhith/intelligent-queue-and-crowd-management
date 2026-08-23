"""VisionQueue detection subsystem.

Provides:
- PersonDetector (YOLO26n ONNX Runtime person detector)
- FaceDetector (YuNet ONNX face presence detector)
- Detection & FaceDetection contract models
"""

from visionqueue.detection.types import Detection, DetectorConfig
from visionqueue.detection.detector import PersonDetector
from visionqueue.detection.face_detector import FaceDetection, FaceDetectorConfig, FaceDetector

__all__ = [
    "Detection",
    "DetectorConfig",
    "PersonDetector",
    "FaceDetection",
    "FaceDetectorConfig",
    "FaceDetector",
]
