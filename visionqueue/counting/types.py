"""Type definitions and contracts for the VisionQueue counting subsystem."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple


class CrossingDirection(str, Enum):
    """Direction of a virtual line crossing event."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Point2D:
    """2D coordinate in pixel space."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class VirtualLine:
    """Virtual line segment used for entry/exit crossing detection.

    The line is defined from pt1 (start) to pt2 (end).
    Direction convention:
    - Looking from pt1 to pt2:
      - Trajectory crossing from LEFT to RIGHT is defined by entry_direction.
      - Default: LEFT-to-RIGHT crossing is ENTRY, RIGHT-to-LEFT is EXIT.

    Attributes:
        pt1: Start point (x, y) of the line segment.
        pt2: End point (x, y) of the line segment.
        entry_direction: Direction name for LEFT-to-RIGHT crossing (default: CrossingDirection.ENTRY).
    """
    pt1: Tuple[float, float]
    pt2: Tuple[float, float]
    entry_direction: CrossingDirection = CrossingDirection.ENTRY

    def __post_init__(self):
        if self.pt1 == self.pt2:
            raise ValueError(f"VirtualLine start and end points cannot be identical: {self.pt1}")

    @property
    def x1(self) -> float:
        return self.pt1[0]

    @property
    def y1(self) -> float:
        return self.pt1[1]

    @property
    def x2(self) -> float:
        return self.pt2[0]

    @property
    def y2(self) -> float:
        return self.pt2[1]

    @property
    def length(self) -> float:
        """Euclidean length of the line segment."""
        import math
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def to_dict(self) -> dict:
        """Serialize VirtualLine configuration to standard dictionary."""
        return {
            "pt1": [round(float(c), 2) for c in self.pt1],
            "pt2": [round(float(c), 2) for c in self.pt2],
            "entry_direction": (
                self.entry_direction.value
                if isinstance(self.entry_direction, CrossingDirection)
                else str(self.entry_direction)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VirtualLine":
        """Deserialize VirtualLine from configuration dictionary."""
        pt1 = (float(data["pt1"][0]), float(data["pt1"][1]))
        pt2 = (float(data["pt2"][0]), float(data["pt2"][1]))
        entry_dir = data.get("entry_direction", "ENTRY")
        entry_direction = (
            CrossingDirection(entry_dir)
            if isinstance(entry_dir, str) and entry_dir in CrossingDirection._value2member_map_
            else CrossingDirection.ENTRY
        )
        return cls(pt1=pt1, pt2=pt2, entry_direction=entry_direction)


@dataclass(frozen=True)
class CrossingEvent:
    """Event emitted when a track crosses a virtual line.

    Attributes:
        track_id: ID of the track that crossed.
        direction: Direction of crossing (ENTRY or EXIT).
        frame_id: Sequence frame number at moment of crossing.
        timestamp: Acquisition / detection timestamp.
        crossing_point: Estimated (x, y) point of crossing.
        bbox: Bounding box [x1, y1, x2, y2] at moment of crossing.
    """
    track_id: int
    direction: CrossingDirection
    frame_id: int
    timestamp: float
    crossing_point: Tuple[float, float]
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> dict:
        """Serialize crossing event to dictionary."""
        return {
            "track_id": self.track_id,
            "direction": self.direction.value,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "crossing_point": list(self.crossing_point),
            "bbox": list(self.bbox),
        }


@dataclass(frozen=True)
class OccupancyState:
    """Current in-ROI occupancy state at a given frame.

    Attributes:
        current_count: Number of unique active tracks whose bottom-center is inside ROI.
        active_track_ids: Sorted list of track IDs currently inside the ROI.
        frame_id: Monotonically increasing frame sequence number.
        timestamp: Timestamp of the frame.
    """
    current_count: int
    active_track_ids: List[int]
    frame_id: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "current_count": self.current_count,
            "active_track_ids": list(self.active_track_ids),
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class LineCrossingCounts:
    """Cumulative line crossing statistics.

    Attributes:
        entries: Total count of verified ENTRY events.
        exits: Total count of verified EXIT events.
        net_count: Net count (entries - exits).
    """
    entries: int = 0
    exits: int = 0

    @property
    def net_count(self) -> int:
        return self.entries - self.exits

    def to_dict(self) -> dict:
        return {
            "entries": self.entries,
            "exits": self.exits,
            "net_count": self.net_count,
        }


@dataclass(frozen=True)
class SessionCounts:
    """Session-level tracking metrics.

    IMPORTANT:
    This metric provides an APPROXIMATE count of unique visitors based strictly
    on temporary tracking session IDs assigned by ByteTrack. It does NOT perform
    biometric recognition, facial identity matching, or persistent re-identification.
    If a person leaves the camera FOV and re-enters, a new tracking ID will be generated
    and counted.

    Attributes:
    approximate_unique_count: Total unique track IDs observed in this session.
    currently_active_count: Number of active tracks in the latest frame.
    active_track_ids: Track IDs active in the latest frame.
    """
    cumulative_track_instances: int = 0
    approximate_unique_count: int = 0
    currently_active_count: int = 0
    active_track_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cumulative_track_instances": self.cumulative_track_instances,
            "approximate_unique_count": self.approximate_unique_count,
            "currently_active_count": self.currently_active_count,
            "active_track_ids": list(self.active_track_ids),
        }
