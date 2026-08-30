"""Virtual line crossing entry/exit counter for VisionQueue.

Detects when track trajectories (bottom-center points) cross a configured
virtual line segment, determines direction (ENTRY vs EXIT), and suppresses
duplicate triggers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from visionqueue.counting.types import (
    CrossingDirection,
    CrossingEvent,
    LineCrossingCounts,
    VirtualLine,
)
from visionqueue.roi import track_bottom_center
from visionqueue.tracking.adapter import Track


def _cross_product(
    ax: float, ay: float,
    bx: float, by: float,
    px: float, py: float,
) -> float:
    """Signed cross product of vector AB with vector AP.

    Returns:
        > 0 if P is on the left side of directed line AB.
        < 0 if P is on the right side of directed line AB.
        = 0 if P is collinear with line AB.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    q1: Tuple[float, float], q2: Tuple[float, float],
    eps: float = 1e-6,
) -> Tuple[bool, Optional[Tuple[float, float]], int]:
    """Check if line segment p1-p2 intersects segment q1-q2.

    Args:
        p1: Trajectory start point (x, y).
        p2: Trajectory end point (x, y).
        q1: Virtual line start point (x, y).
        q2: Virtual line end point (x, y).
        eps: Small epsilon for numerical stability.

    Returns:
        (intersects, intersection_point, side_transition)
        where side_transition is:
            +1: crossed from Left (+ side) to Right (- side)
            -1: crossed from Right (- side) to Left (+ side)
             0: no valid crossing
    """
    ax, ay = q1
    bx, by = q2
    p1x, p1y = p1
    p2x, p2y = p2

    # Side of line q1->q2 for trajectory endpoints
    cp1 = _cross_product(ax, ay, bx, by, p1x, p1y)
    cp2 = _cross_product(ax, ay, bx, by, p2x, p2y)

    # Must cross the line (opposite signs or strictly moving from one side across)
    if abs(cp1) < eps and abs(cp2) < eps:
        # Collinear movement along the line - not a crossing
        return False, None, 0

    if cp1 > eps and cp2 < -eps:
        transition = 1  # Left -> Right
    elif cp1 < -eps and cp2 > eps:
        transition = -1  # Right -> Left
    elif abs(cp1) <= eps and cp2 < -eps:
        transition = 1  # Started on line, moved right
    elif abs(cp1) <= eps and cp2 > eps:
        transition = -1  # Started on line, moved left
    elif cp1 > eps and abs(cp2) <= eps:
        # Landed on line from left - we don't trigger yet until it moves across
        return False, None, 0
    elif cp1 < -eps and abs(cp2) <= eps:
        # Landed on line from right - we don't trigger yet until it moves across
        return False, None, 0
    else:
        # Both on same side
        return False, None, 0

    # Check if the intersection point lies within the segment q1->q2
    # Parameter t along trajectory p1->p2
    denom = cp1 - cp2
    if abs(denom) < eps:
        return False, None, 0

    t = cp1 / denom
    if not (-eps <= t <= 1.0 + eps):
        return False, None, 0

    ix = p1x + t * (p2x - p1x)
    iy = p1y + t * (p2y - p1y)

    # Check if (ix, iy) falls within the bounds of virtual line segment q1->q2
    # Check projection along segment AB
    ab_len_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if ab_len_sq < eps:
        return False, None, 0

    u = ((ix - ax) * (bx - ax) + (iy - ay) * (by - ay)) / ab_len_sq
    if -eps <= u <= 1.0 + eps:
        return True, (ix, iy), transition

    return False, None, 0


@dataclass
class _TrackLineState:
    """Internal state for a track relative to the virtual line."""
    last_point: Tuple[float, float]
    last_side: int  # +1 (left), -1 (right), 0 (unknown/on line)
    last_frame_id: int
    first_frame_id: int
    last_crossed_direction: Optional[CrossingDirection] = None


class LineCrossingCounter:
    """Virtual line crossing entry/exit counter.

    Features:
    - Trajectory tracking using bottom-center points of tracks.
    - Robust 2D segment intersection.
    - Direction determination (ENTRY vs EXIT) based on VirtualLine configuration.
    - Anti-chatter suppression (prevents duplicate triggers when hovering on/near the line).
    - Configurable minimum track age to suppress crossing events for brand-new tracks
      whose very first observation movement would otherwise trigger a false crossing.
    - Automatic history cleanup for expired tracks.
    """

    def __init__(
        self,
        virtual_line: VirtualLine,
        track_max_age: int = 30,
        min_track_age_frames: int = 0,
    ) -> None:
        """Initialize the line crossing counter.

        Args:
            virtual_line: Configured VirtualLine segment.
            track_max_age: Number of consecutive inactive frames before purging track history.
            min_track_age_frames: Minimum number of frames a track must be observed before
                it is eligible to emit a crossing event. Default 0 preserves legacy behavior.
                Set to a small value (e.g. 1 or 2) to suppress first-frame false crossings
                when a track is born exactly on the line and its first move happens to
                cross the line in a single step.
        """
        if min_track_age_frames < 0:
            raise ValueError(
                f"min_track_age_frames must be >= 0, got {min_track_age_frames}"
            )
        self._line = virtual_line
        self._track_max_age = track_max_age
        self._min_track_age_frames = min_track_age_frames
        self._track_states: Dict[int, _TrackLineState] = {}
        self._entries: int = 0
        self._exits: int = 0
        self._total_events: List[CrossingEvent] = []

    @property
    def line(self) -> VirtualLine:
        """Configured VirtualLine segment."""
        return self._line

    @property
    def counts(self) -> LineCrossingCounts:
        """Current cumulative entry/exit counts."""
        return LineCrossingCounts(entries=self._entries, exits=self._exits)

    @property
    def entries(self) -> int:
        return self._entries

    @property
    def exits(self) -> int:
        return self._exits

    @property
    def net_count(self) -> int:
        return self._entries - self._exits

    @property
    def events(self) -> List[CrossingEvent]:
        """All emitted crossing events."""
        return list(self._total_events)

    def update_line(self, new_line: VirtualLine) -> None:
        """Update the virtual line configuration and reset active trajectory side states."""
        self._line = new_line
        self._track_states.clear()

    def update(
        self,
        tracks: List[Track],
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> List[CrossingEvent]:
        """Process tracks for a single frame, detecting and returning new crossing events.

        Args:
            tracks: List of active Track objects.
            frame_id: Current sequence frame number.
            timestamp: Current frame timestamp.

        Returns:
            List of CrossingEvent objects detected in this frame.
        """
        new_events: List[CrossingEvent] = []
        active_ids = set()

        for track in tracks:
            active_ids.add(track.track_id)
            curr_pt = track_bottom_center(track.bbox)
            curr_cp = _cross_product(
                self._line.x1, self._line.y1,
                self._line.x2, self._line.y2,
                curr_pt[0], curr_pt[1],
            )
            curr_side = 1 if curr_cp > 1e-6 else (-1 if curr_cp < -1e-6 else 0)

            if track.track_id not in self._track_states:
                # First time seeing this track - record position and side without crossing
                self._track_states[track.track_id] = _TrackLineState(
                    last_point=curr_pt,
                    last_side=curr_side,
                    last_frame_id=frame_id,
                    first_frame_id=frame_id,
                )
                continue

            state = self._track_states[track.track_id]
            prev_pt = state.last_point

            # Check for segment crossing
            intersects, inter_pt, transition = _segments_intersect(
                prev_pt, curr_pt,
                self._line.pt1, self._line.pt2,
            )

            if intersects and inter_pt is not None and transition != 0:
                # Minimum-track-age guard:
                # Suppress crossing events for brand-new tracks whose very first
                # trajectory segment happens to cross the line. This prevents
                # single-frame false positives when a track is born directly on
                # the line and its initial movement is interpreted as a crossing.
                track_age_frames = frame_id - state.first_frame_id
                if track_age_frames < self._min_track_age_frames:
                    state.last_point = curr_pt
                    if curr_side != 0:
                        state.last_side = curr_side
                    state.last_frame_id = frame_id
                    continue

                # Determine direction
                if transition == 1:
                    direction = self._line.entry_direction
                else:
                    direction = (
                        CrossingDirection.EXIT
                        if self._line.entry_direction == CrossingDirection.ENTRY
                        else CrossingDirection.ENTRY
                    )

                # Anti-chatter: suppress if this track already triggered this exact direction without leaving
                if state.last_crossed_direction != direction:
                    event = CrossingEvent(
                        track_id=track.track_id,
                        direction=direction,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        crossing_point=inter_pt,
                        bbox=track.bbox,
                    )
                    new_events.append(event)
                    self._total_events.append(event)

                    if direction == CrossingDirection.ENTRY:
                        self._entries += 1
                    else:
                        self._exits += 1

                    state.last_crossed_direction = direction

            # Update track state
            state.last_point = curr_pt
            if curr_side != 0:
                state.last_side = curr_side
            state.last_frame_id = frame_id

        # Clean up stale tracks to prevent unbounded memory growth
        stale_ids = [
            tid for tid, s in self._track_states.items()
            if frame_id - s.last_frame_id > self._track_max_age
        ]
        for tid in stale_ids:
            del self._track_states[tid]

        return new_events

    def reset(self) -> None:
        """Reset all counters and internal trajectory state."""
        self._entries = 0
        self._exits = 0
        self._total_events.clear()
        self._track_states.clear()
