"""VisionQueue Counting Subsystem.

Provides modular, lightweight components for:
1. Instantaneous In-ROI Occupancy Counting (OccupancyCounter)
2. Virtual Line Crossing Entry/Exit Counting (LineCrossingCounter)
3. Approximate Session-Unique Visitor Counting (SessionCounter)
"""

from visionqueue.counting.types import (
    CrossingDirection,
    CrossingEvent,
    LineCrossingCounts,
    OccupancyState,
    Point2D,
    SessionCounts,
    VirtualLine,
)
from visionqueue.counting.occupancy import OccupancyCounter
from visionqueue.counting.line_crossing import LineCrossingCounter
from visionqueue.counting.session import SessionCounter

__all__ = [
    "CrossingDirection",
    "CrossingEvent",
    "LineCrossingCounts",
    "OccupancyState",
    "Point2D",
    "SessionCounts",
    "VirtualLine",
    "OccupancyCounter",
    "LineCrossingCounter",
    "SessionCounter",
]
