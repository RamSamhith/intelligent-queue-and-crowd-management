"""Approximate session-unique visitor counter for VisionQueue.

Tracks cumulative unique tracking IDs observed across an active camera session.

IMPORTANT PRIVACY & ACCURACY NOTICE:
This module computes an APPROXIMATE count based strictly on the temporary
track IDs assigned by ByteTrack during the active tracking session.
It does NOT use biometric recognition, face recognition, feature embeddings,
or cross-camera re-identification. If a person exits the camera view and re-enters
after their track buffer expires, ByteTrack will assign a new track ID and this
counter will increment.
"""

from __future__ import annotations
from typing import List, Set

from visionqueue.counting.types import SessionCounts
from visionqueue.tracking.adapter import Track


class SessionCounter:
    """Approximate session-unique visitor counter.

    Maintains a set of observed track IDs during an active session to estimate
    total distinct visitors seen.

    Guarantees:
    - Lightweight O(1) set lookups per track per frame.
    - Zero external network or model overhead.
    - Deterministic and privacy-preserving.
    """

    def __init__(self) -> None:
        """Initialize an empty session counter."""
        self._seen_track_ids: Set[int] = set()
        self._last_active_ids: List[int] = []

    @property
    def cumulative_track_instances(self) -> int:
        """Total distinct track IDs observed in current session.

        Canonical new name. Replaces the misleading "approximate_unique_count"
        terminology. Track IDs are not human identities; this counts tracker
        instances only.
        """
        return len(self._seen_track_ids)

    @property
    def approximate_unique_count(self) -> int:
        """Backward-compatible alias for cumulative_track_instances.

        Preserved so existing consumers (older tests, downstream code) that
        reference the legacy name continue to work. New code should use
        cumulative_track_instances for clarity.
        """
        return len(self._seen_track_ids)

    @property
    def seen_track_ids(self) -> Set[int]:
        """Set of all track IDs recorded in current session."""
        return set(self._seen_track_ids)

    def update(self, tracks: List[Track]) -> SessionCounts:
        """Process tracks for a frame, updating the session accumulator.

        Args:
            tracks: List of active Track objects in the current frame.

        Returns:
            SessionCounts containing cumulative unique count and active metrics.
        """
        current_ids = [t.track_id for t in tracks]
        for tid in current_ids:
            self._seen_track_ids.add(tid)

        self._last_active_ids = sorted(list(set(current_ids)))

        n = len(self._seen_track_ids)
        return SessionCounts(
            cumulative_track_instances=n,
            approximate_unique_count=n,
            currently_active_count=len(self._last_active_ids),
            active_track_ids=self._last_active_ids,
        )

    def reset(self) -> None:
        """Clear all session tracking state and start a fresh session."""
        self._seen_track_ids.clear()
        self._last_active_ids.clear()

    def new_session(self) -> None:
        """Alias for reset() to explicitly demarcate session boundaries."""
        self.reset()
