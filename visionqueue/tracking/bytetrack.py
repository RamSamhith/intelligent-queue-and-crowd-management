"""
ByteTrack Multi-Object Tracker

Implements the ByteTrack algorithm:
1. High-confidence detections associated with existing tracks (first stage)
2. Low-confidence detections used to recover occluded/lost tracks (second stage)
3. New tracks created from unmatched high-confidence detections
4. Track lifecycle management with configurable buffer
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray

from .kalman import KalmanFilter
from .track import Track, TrackState
from .matching import matching, iou_distance


@dataclass
class ByteTrackConfig:
    """Configuration for ByteTrack."""
    track_buffer: int = 30           # Frames to keep lost tracks
    match_thresh: float = 0.8        # IoU threshold for matching
    track_high_thresh: float = 0.5   # High confidence threshold
    track_low_thresh: float = 0.1    # Low confidence threshold
    new_track_thresh: float = 0.6    # Threshold for creating new tracks
    frame_rate: int = 30             # Video frame rate


class ByteTrack:
    """
    ByteTrack tracker for person detection.

    Input: detections as (N, 6) array [x1, y1, x2, y2, confidence, class_id]
    Output: list of track dicts {track_id, bbox: [x1,y1,x2,y2], confidence}
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None):
        self.config = config or ByteTrackConfig()

        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []

        self.frame_id = 0
        self.track_id_counter = 0
        self.kf = KalmanFilter()

        # Derived threshold
        self.det_thresh = self.config.track_high_thresh + 0.1
        self.max_time_lost = int(self.config.frame_rate / 30.0 * self.config.track_buffer)

        # Non-invasive diagnostic telemetry state
        self._last_diagnostics: Dict[str, Any] = {}
        self._recent_inactive_tracks: deque[dict] = deque(maxlen=150)

    @property
    def last_diagnostics(self) -> Dict[str, Any]:
        """Return the most recent frame tracking diagnostic record."""
        return self._last_diagnostics

    def update(self, detections: NDArray[np.float32]) -> list[dict]:
        """
        Update tracker with new frame detections.

        Args:
            detections: (N, 6) array [x1, y1, x2, y2, confidence, class_id]
                       Empty array (0, 6) if no detections

        Returns:
            List of track dicts: {track_id, bbox: [x1,y1,x2,y2], confidence}
        """
        self.frame_id += 1

        # Diagnostic collectors
        tracks_created_diag: List[Dict[str, Any]] = []
        tracks_lost_diag: List[Dict[str, Any]] = []
        tracks_terminated_diag: List[Dict[str, Any]] = []
        id_replacements_diag: List[Dict[str, Any]] = []

        # Filter by class_id if needed (default: person = 0)
        if len(detections) > 0:
            # Keep all classes for now - filtering can be done externally
            pass

        # Separate high and low confidence detections
        if len(detections) > 0:
            scores = detections[:, 4]
            high_mask = scores >= self.config.track_high_thresh
            low_mask = (scores > self.config.track_low_thresh) & (scores < self.config.track_high_thresh)

            dets_high = detections[high_mask]
            dets_low = detections[low_mask]
        else:
            dets_high = np.empty((0, 6), dtype=np.float32)
            dets_low = np.empty((0, 6), dtype=np.float32)

        # ============================================================
        # Step 1: Predict all tracked and lost tracks
        # ============================================================
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        # ============================================================
        # Step 2: First association - high confidence detections
        # ============================================================
        # Pool: tracked + lost tracks
        strack_pool = self.tracked_tracks + self.lost_tracks
        strack_tlbr = np.array([t.tlbr for t in strack_pool], dtype=np.float32) if strack_pool else np.empty((0, 4), dtype=np.float32)

        stage1_matches_diag: List[Dict[str, Any]] = []
        stage1_ious_diag: List[Dict[str, Any]] = []

        if len(dets_high) > 0:
            det_tlbr_high = dets_high[:, :4]

            # Record raw IoU matrix for diagnostics
            if len(strack_tlbr) > 0:
                raw_dist = iou_distance(strack_tlbr, det_tlbr_high)
                raw_ious = 1.0 - raw_dist
                for ti, trk in enumerate(strack_pool):
                    for di in range(len(dets_high)):
                        iou_val = float(raw_ious[ti, di])
                        if iou_val > 0.01:
                            stage1_ious_diag.append({
                                "track_id": trk.track_id,
                                "track_state": trk.state.name,
                                "det_idx": di,
                                "iou": round(iou_val, 4),
                                "det_conf": round(float(dets_high[di, 4]), 4),
                            })

            matches, u_track, u_det = matching(
                strack_tlbr, det_tlbr_high,
                thresh=self.config.match_thresh,
                detections_full=None  # Pure IoU for first association
            )

            # Update matched tracks
            for track_idx, det_idx in matches:
                track = strack_pool[track_idx]
                was_lost = track.state == TrackState.LOST
                det = dets_high[det_idx]
                det_tlwh = self.kf.tlbr_to_tlwh(det[:4])
                
                iou_m = 0.0
                if len(strack_tlbr) > track_idx:
                    inter_dist = iou_distance(strack_tlbr[track_idx:track_idx+1], det[:4].reshape(1, 4))
                    iou_m = float(1.0 - inter_dist[0, 0])

                stage1_matches_diag.append({
                    "track_id": track.track_id,
                    "det_idx": int(det_idx),
                    "iou": round(iou_m, 4),
                    "confidence": round(float(det[4]), 4),
                    "was_lost": was_lost,
                })

                track.update(det_tlwh, float(det[4]), self.frame_id)

                # If matched track was LOST, re-activate it (move to tracked_tracks)
                if was_lost:
                    track.state = TrackState.TRACKED
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)
                    if track not in self.tracked_tracks:
                        self.tracked_tracks.append(track)

            # Unmatched high-confidence detections -> potential new tracks
            unmatched_high_dets = dets_high[u_det] if len(u_det) > 0 else np.empty((0, 6), dtype=np.float32)
        else:
            matches = np.empty((0, 2), dtype=np.int32)
            u_track = np.arange(len(strack_pool), dtype=np.int32)
            u_det = np.array([], dtype=np.int32)
            unmatched_high_dets = np.empty((0, 6), dtype=np.float32)

        # ============================================================
        # Step 3: Second association - low confidence detections
        # Recover occluded/lost tracks (TRACKED and eligible LOST tracks)
        # ============================================================
        # Consider unmatched tracks from Stage 1 that are in TRACKED or LOST state
        r_pool_indices = [i for i in u_track if strack_pool[i].state in (TrackState.TRACKED, TrackState.LOST)]
        r_tracked = [strack_pool[i] for i in r_pool_indices]
        r_tracked_tlbr = np.array([t.tlbr for t in r_tracked], dtype=np.float32) if r_tracked else np.empty((0, 4), dtype=np.float32)

        stage2_matches_diag: List[Dict[str, Any]] = []

        if len(dets_low) > 0 and len(r_tracked) > 0:
            det_tlbr_low = dets_low[:, :4]
            matches_low, u_track_low, u_det_low = matching(
                r_tracked_tlbr, det_tlbr_low,
                thresh=0.5,  # Pure IoU distance threshold (dist <= 0.5, i.e. IoU >= 0.5)
                detections_full=None  # Pure IoU distance (no score fusion)
            )

            # Update recovered tracks
            for track_idx, det_idx in matches_low:
                track = r_tracked[track_idx]
                was_lost = track.state == TrackState.LOST
                det = dets_low[det_idx]
                det_tlwh = self.kf.tlbr_to_tlwh(det[:4])
                
                iou_m = 0.0
                if len(r_tracked_tlbr) > track_idx:
                    inter_dist = iou_distance(r_tracked_tlbr[track_idx:track_idx+1], det[:4].reshape(1, 4))
                    iou_m = float(1.0 - inter_dist[0, 0])

                stage2_matches_diag.append({
                    "track_id": track.track_id,
                    "det_idx": int(det_idx),
                    "iou": round(iou_m, 4),
                    "confidence": round(float(det[4]), 4),
                    "was_lost": was_lost,
                })

                track.update(det_tlwh, float(det[4]), self.frame_id)

                # If matched track was LOST, re-activate it (move to tracked_tracks)
                if was_lost:
                    track.state = TrackState.TRACKED
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)
                    if track not in self.tracked_tracks:
                        self.tracked_tracks.append(track)

            # Update unmatched lists - map back to strack_pool indices
            newly_matched_pool = np.array([r_pool_indices[i] for i in matches_low[:, 0]], dtype=np.int32) if len(matches_low) > 0 else np.array([], dtype=np.int32)
            u_track = np.setdiff1d(u_track, newly_matched_pool)
            u_det_low = u_det_low  # unused low detections
        else:
            u_det_low = np.arange(len(dets_low), dtype=np.int32)

        # ============================================================
        # Step 4: Handle unconfirmed tracks (NEW state) that are still unmatched
        # ============================================================
        unconfirmed_pool_indices = [i for i in u_track if strack_pool[i].state == TrackState.NEW]
        unconfirmed = [strack_pool[i] for i in unconfirmed_pool_indices]
        unconfirmed_tlbr = np.array([t.tlbr for t in unconfirmed], dtype=np.float32) if unconfirmed else np.empty((0, 4), dtype=np.float32)

        if len(unmatched_high_dets) > 0 and len(unconfirmed) > 0:
            det_tlbr = unmatched_high_dets[:, :4]
            matches, u_unconfirmed, u_det = matching(
                unconfirmed_tlbr, det_tlbr,
                thresh=0.7,  # Max distance 0.7 (IoU >= 0.3)
                detections_full=None
            )

            for track_idx, det_idx in matches:
                track = unconfirmed[track_idx]
                det = unmatched_high_dets[det_idx]
                det_tlwh = self.kf.tlbr_to_tlwh(det[:4])
                track.update(det_tlwh, float(det[4]), self.frame_id)

            # Remove unmatched unconfirmed tracks
            for idx in u_unconfirmed:
                unconfirmed[idx].mark_removed()
                self.removed_tracks.append(unconfirmed[idx])
                tracks_terminated_diag.append({
                    "track_id": unconfirmed[idx].track_id,
                    "reason": "unconfirmed_new_track_unmatched",
                    "state": "REMOVED",
                })
                self._recent_inactive_tracks.append({
                    "track_id": unconfirmed[idx].track_id,
                    "tlbr": list(map(float, unconfirmed[idx].tlbr)),
                    "end_frame": self.frame_id,
                })

            unmatched_high_dets = unmatched_high_dets[u_det] if len(u_det) > 0 else np.empty((0, 6), dtype=np.float32)

        # ============================================================
        # Step 5: Initialize new tracks from remaining high-confidence detections
        # ============================================================
        if len(unmatched_high_dets) > 0:
            # Only create new tracks above new_track_thresh
            new_mask = unmatched_high_dets[:, 4] >= self.config.new_track_thresh
            new_dets = unmatched_high_dets[new_mask]

            for det in new_dets:
                track = Track(
                    track_id=self.track_id_counter,
                    tlwh=self.kf.tlbr_to_tlwh(det[:4]),
                    score=float(det[4]),
                    class_id=int(det[5])
                )
                track.start_frame = self.frame_id
                track.frame_id = self.frame_id
                track.end_frame = self.frame_id
                self.tracked_tracks.append(track)

                # Diagnostic: Check if this new track replaces a recently lost/removed track
                new_box = det[:4].reshape(1, 4)
                for prev in self._recent_inactive_tracks:
                    prev_box = np.array(prev["tlbr"], dtype=np.float32).reshape(1, 4)
                    dist_prev = iou_distance(prev_box, new_box)
                    iou_with_prev = float(1.0 - dist_prev[0, 0])
                    if iou_with_prev > 0.05:
                        id_replacements_diag.append({
                            "new_track_id": self.track_id_counter,
                            "replaced_lost_id": prev["track_id"],
                            "iou": round(iou_with_prev, 4),
                            "frames_since_lost": self.frame_id - prev["end_frame"],
                            "confidence": round(float(det[4]), 4),
                            "new_bbox": [round(float(c), 2) for c in det[:4]],
                            "lost_bbox": [round(float(c), 2) for c in prev["tlbr"]],
                        })

                tracks_created_diag.append({
                    "track_id": self.track_id_counter,
                    "bbox": [round(float(c), 2) for c in det[:4]],
                    "confidence": round(float(det[4]), 4),
                    "reason": f"unmatched_high_detection (conf={det[4]:.3f} >= new_track_thresh={self.config.new_track_thresh})",
                })

                self.track_id_counter += 1

        # ============================================================
        # Step 6: Update track states
        # ============================================================
        # Move lost tracks that weren't matched
        for idx in u_track:
            track = strack_pool[idx]
            if track.state == TrackState.TRACKED:
                track.mark_lost()
                self.lost_tracks.append(track)
                tracks_lost_diag.append({
                    "track_id": track.track_id,
                    "reason": "unmatched_in_stage1_and_stage2",
                    "age": track.age,
                    "bbox": [round(float(c), 2) for c in track.tlbr],
                })
                self._recent_inactive_tracks.append({
                    "track_id": track.track_id,
                    "tlbr": list(map(float, track.tlbr)),
                    "end_frame": self.frame_id,
                })

        # Remove expired lost tracks
        expired_tracks = [t for t in self.lost_tracks if self._is_expired(t)]
        self.lost_tracks = [t for t in self.lost_tracks if not self._is_expired(t)]
        for t in expired_tracks:
            t.mark_removed()
            self.removed_tracks.append(t)
            tracks_terminated_diag.append({
                "track_id": t.track_id,
                "reason": f"expired_buffer (lost_frames={self.frame_id - t.end_frame} > max_time_lost={self.max_time_lost})",
                "state": "REMOVED",
            })
            self._recent_inactive_tracks.append({
                "track_id": t.track_id,
                "tlbr": list(map(float, t.tlbr)),
                "end_frame": t.end_frame,
            })

        # Clean up tracked tracks: remove LOST and REMOVED tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED or t.state == TrackState.NEW]

        # ============================================================
        # Step 7: Prepare output & Record Diagnostics
        # ============================================================
        output_tracks = []
        for track in self.tracked_tracks:
            if track.state != TrackState.REMOVED and track.state != TrackState.LOST:
                output_tracks.append(track.to_dict())

        # Construct frame diagnostic record
        self._last_diagnostics = {
            "frame_id": self.frame_id,
            "detections": [
                {
                    "bbox": [round(float(c), 2) for c in d[:4]],
                    "confidence": round(float(d[4]), 4),
                    "class_id": int(d[5]),
                }
                for d in detections
            ] if len(detections) > 0 else [],
            "detections_count": len(detections),
            "active_tracks_count": len(output_tracks),
            "active_track_ids": [t["track_id"] for t in output_tracks],
            "active_tracks": [
                {
                    "track_id": t.track_id,
                    "bbox": [round(float(c), 2) for c in t.tlbr],
                    "confidence": round(float(t.score), 4),
                    "state": t.state.name,
                    "age": t.age,
                    "hit_streak": t.hit_streak,
                    "time_since_update_frames": t.age,
                }
                for t in self.tracked_tracks
                if t.state != TrackState.REMOVED and t.state != TrackState.LOST
            ],
            "track_ages": {str(t.track_id): t.age for t in self.tracked_tracks},
            "time_since_last_association": {str(t.track_id): t.age for t in self.tracked_tracks},
            "tracks_created": tracks_created_diag,
            "tracks_lost": tracks_lost_diag,
            "id_replacements": id_replacements_diag,
            "association_ious": stage1_ious_diag,
            "matching_result": {
                "stage1_high_conf": {
                    "matches": stage1_matches_diag,
                    "unmatched_track_ids": [strack_pool[i].track_id for i in u_track if i < len(strack_pool)],
                    "unmatched_det_indices": [int(i) for i in u_det],
                },
                "stage2_low_conf": {
                    "matches": stage2_matches_diag,
                },
            },
            "track_terminations": tracks_terminated_diag,
        }

        return output_tracks

    def _is_expired(self, track: Track) -> bool:
        """Check if lost track has exceeded buffer."""
        return self.frame_id - track.end_frame > self.max_time_lost

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self._recent_inactive_tracks.clear()
        self._last_diagnostics.clear()
        self.frame_id = 0
        self.track_id_counter = 0