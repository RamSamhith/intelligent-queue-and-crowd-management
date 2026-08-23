"""Type definitions for the VisionQueue ROI subsystem."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class ROIConfig:
    """Configuration for a rectangular Region of Interest.

    Coordinates are in pixel space, top-left origin (OpenCV convention).

    Attributes:
        x: Left edge (inclusive).
        y: Top edge (inclusive).
        width: Width in pixels (> 0).
        height: Height in pixels (> 0).
    """
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"ROI width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"ROI height must be > 0, got {self.height}")

    @property
    def x2(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge (exclusive)."""
        return self.y + self.height

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) with x2, y2 exclusive."""
        return (self.x, self.y, self.x2, self.y2)

    def contains_point(self, px: float, py: float) -> bool:
        """Check if point (px, py) is inside ROI.

        Uses half-open interval [x, x2) × [y, y2) to match image pixel semantics.
        """
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def clamp_to_frame(self, frame_width: int, frame_height: int) -> "ROIConfig":
        """Return a new ROI clamped to frame boundaries.

        If ROI is completely outside frame, returns a minimal valid ROI at (0,0,1,1).
        """
        x1 = max(0, min(self.x, frame_width - 1))
        y1 = max(0, min(self.y, frame_height - 1))
        x2 = max(x1 + 1, min(self.x2, frame_width))
        y2 = max(y1 + 1, min(self.y2, frame_height))

        return ROIConfig(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def is_valid_for_frame(self, frame_width: int, frame_height: int) -> bool:
        """Check if ROI has any overlap with frame."""
        return (self.x < frame_width and self.y < frame_height and
                self.x2 > 0 and self.y2 > 0)

    def area(self) -> int:
        """Return ROI area in pixels."""
        return self.width * self.height

    def intersection_area(self, other: "ROIConfig") -> int:
        """Compute intersection area with another ROI."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def iou(self, other: "ROIConfig") -> float:
        """Compute IoU with another ROI."""
        inter = self.intersection_area(other)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackROIStatus:
    """Status of a track relative to an ROI.

    Attributes:
        track_id: The track identifier.
        in_roi: True if track's bottom-center point is inside ROI.
        bottom_center: (cx, bottom_y) coordinates used for the check.
        bbox: Original track bounding box [x1, y1, x2, y2].
    """
    track_id: int
    in_roi: bool
    bottom_center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]