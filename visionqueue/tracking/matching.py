"""
Matching utilities for ByteTrack: IoU calculation and Hungarian assignment.
Pure NumPy implementation.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment as scipy_lsa


def iou_distance(tracks_tlbr: NDArray[np.float32], detections_tlbr: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Compute IoU distance matrix between tracks and detections.

    Args:
        tracks_tlbr: (N, 4) array of track boxes in [x1, y1, x2, y2] format
        detections_tlbr: (M, 4) array of detection boxes in [x1, y1, x2, y2] format

    Returns:
        (N, M) distance matrix where distance = 1 - IoU
    """
    if len(tracks_tlbr) == 0 or len(detections_tlbr) == 0:
        return np.empty((len(tracks_tlbr), len(detections_tlbr)), dtype=np.float32)

    # Expand for broadcasting
    # tracks: (N, 1, 4), detections: (1, M, 4)
    tracks = tracks_tlbr[:, np.newaxis, :]
    detections = detections_tlbr[np.newaxis, :, :]

    # Intersection coordinates
    x1 = np.maximum(tracks[:, :, 0], detections[:, :, 0])
    y1 = np.maximum(tracks[:, :, 1], detections[:, :, 1])
    x2 = np.minimum(tracks[:, :, 2], detections[:, :, 2])
    y2 = np.minimum(tracks[:, :, 3], detections[:, :, 3])

    # Intersection area
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    # Union area
    track_area = (tracks[:, :, 2] - tracks[:, :, 0]) * (tracks[:, :, 3] - tracks[:, :, 1])
    det_area = (detections[:, :, 2] - detections[:, :, 0]) * (detections[:, :, 3] - detections[:, :, 1])
    union = track_area + det_area - intersection

    # IoU
    iou = np.where(union > 0, intersection / union, 0.0)

    # Distance = 1 - IoU
    return 1.0 - iou


def fuse_score(distance: NDArray[np.float32], detections: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Fuse detection scores into distance matrix (lower score = higher distance).

    Args:
        distance: (N, M) IoU distance matrix
        detections: (M, 6) detection array [x1, y1, x2, y2, conf, class_id]

    Returns:
        Fused distance matrix
    """
    if len(detections) == 0:
        return distance

    scores = detections[:, 4]  # confidence
    # Fuse: distance = 1 - (1 - distance) * score = distance + (1 - distance) * (1 - score)
    # Simpler: new_dist = 1 - IoU * score = distance + (1 - distance) * (1 - score)
    # Actually ByteTrack uses: fused_dist = 1 - IoU * score
    iou = 1.0 - distance
    fused_iou = iou * scores[np.newaxis, :]
    return 1.0 - fused_iou


def linear_sum_assignment(cost_matrix: NDArray[np.float32]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Hungarian algorithm for linear sum assignment using scipy.optimize.
    """
    if cost_matrix.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
    # We replace any inf/1e6 costs with a high value but not inf
    # to allow scipy to solve the assignment gracefully
    cleaned_cost = np.where(cost_matrix >= 1e6, 1e6, cost_matrix)
    
    row_ind, col_ind = scipy_lsa(cleaned_cost)
    return row_ind.astype(np.int32), col_ind.astype(np.int32)


def matching(
    tracks_tlbr: NDArray[np.float32],
    detections_tlbr: NDArray[np.float32],
    thresh: float = 0.8,
    detections_full: NDArray[np.float32] | None = None
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """
    Perform matching between tracks and detections using IoU + Hungarian.

    Args:
        tracks_tlbr: (N, 4) track boxes
        detections_tlbr: (M, 4) detection boxes
        thresh: IoU threshold for valid matches (distance <= 1 - thresh)
        detections_full: (M, 6) full detections [x1,y1,x2,y2,conf,cls] for score fusion

    Returns:
        matches: (K, 2) array of [track_idx, det_idx]
        unmatched_tracks: array of track indices
        unmatched_detections: array of detection indices
    """
    if len(tracks_tlbr) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(len(tracks_tlbr), dtype=np.int32),
            np.arange(len(detections_tlbr), dtype=np.int32)
        )

    if len(detections_tlbr) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(len(tracks_tlbr), dtype=np.int32),
            np.empty(0, dtype=np.int32)
        )

    # Compute distance matrix
    dist = iou_distance(tracks_tlbr, detections_tlbr)

    # Fuse scores if provided
    if detections_full is not None and len(detections_full) == len(detections_tlbr):
        dist = fuse_score(dist, detections_full)

    # Threshold: max allowed distance (distance = 1 - IoU <= thresh)
    max_dist = thresh
    dist[dist > max_dist] = 1e6  # Mark invalid

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(dist)

    # Filter by threshold
    valid = dist[row_ind, col_ind] <= max_dist
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    matches = np.column_stack([row_ind, col_ind]) if len(row_ind) > 0 else np.empty((0, 2), dtype=np.int32)

    all_tracks = np.arange(len(tracks_tlbr), dtype=np.int32)
    all_dets = np.arange(len(detections_tlbr), dtype=np.int32)

    matched_tracks = matches[:, 0] if len(matches) > 0 else np.array([], dtype=np.int32)
    matched_dets = matches[:, 1] if len(matches) > 0 else np.array([], dtype=np.int32)

    unmatched_tracks = np.setdiff1d(all_tracks, matched_tracks)
    unmatched_detections = np.setdiff1d(all_dets, matched_dets)

    return matches, unmatched_tracks, unmatched_detections