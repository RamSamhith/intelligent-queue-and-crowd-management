"""Clean, operator-friendly HUD rendering for VisionQueue.

This module is intentionally separate from CameraSource and CVPipeline so
that visualization concerns stay out of the CV runtime. It draws bounding
boxes for the active tracks and a minimal five-tile operator HUD. It does
not draw developer internals (GPU, ONNX path, FPS, latency, session UUID,
frame ID, etc.) in the normal view.

Inputs are pure data (LiveState-like dict and a tracks list). The same dict
shape is produced by CVPipeline._build_live_state() and serialized by
LiveState.to_dict().
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color helpers (BGR tuples, matching OpenCV convention)
# ---------------------------------------------------------------------------

def get_crowd_color(crowd_str: str) -> Tuple[int, int, int]:
    """Return BGR color for a crowd level string."""
    if crowd_str == "LOW":
        return (40, 200, 40)        # Green
    if crowd_str == "MODERATE":
        return (0, 215, 255)       # Yellow
    if crowd_str == "HIGH":
        return (0, 140, 255)       # Orange
    if crowd_str == "CRITICAL":
        return (40, 40, 240)       # Red
    return (200, 200, 200)         # Neutral


def get_state_color(state_str: str) -> Tuple[int, int, int]:
    """Return BGR color for a system state string."""
    if state_str == "LIVE":
        return (40, 200, 40)
    if state_str == "STARTING":
        return (255, 180, 0)
    if state_str == "DEGRADED":
        return (0, 215, 255)
    if state_str == "UNSTABLE":
        return (255, 0, 255)
    if state_str == "OFFLINE":
        return (40, 40, 240)
    if state_str == "STOPPING":
        return (160, 160, 160)
    return (200, 200, 200)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_simple_hud(
    frame: np.ndarray,
    state: Dict[str, Any],
    tracks: Optional[Iterable[Dict[str, Any]]] = None,
) -> np.ndarray:
    """Draw the operator HUD on top of a BGR frame.

    Args:
        frame: BGR uint8 image (as produced by CameraSource).
        state: LiveState-shaped dict (as produced by LiveState.to_dict()).
            Only the operator-relevant fields are read:
              counts.current, counts.entries, counts.exits,
              crowd.level, system_state.
        tracks: Optional iterable of track dicts. Each track dict must have
            keys: bbox (length-4 [x1,y1,x2,y2]) and optionally track_id and
            confidence. Tracks are drawn exactly as reported by the pipeline;
            no second inference or counting happens here.

    Returns:
        New uint8 BGR frame with HUD overlay. The input frame is not mutated.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    counts = state.get("counts", {}) if isinstance(state, dict) else {}
    crowd = state.get("crowd", {}) if isinstance(state, dict) else {}
    system_state = state.get("system_state", "UNKNOWN") if isinstance(state, dict) else "UNKNOWN"

    people_now = int(counts.get("current", 0) or 0)
    entered = int(counts.get("entries", 0) or 0)
    left = int(counts.get("exits", 0) or 0)
    crowd_level = str(crowd.get("level", "LOW"))

    # --- 1. Person bounding boxes (from existing pipeline tracks) ---
    if tracks:
        for trk in tracks:
            bbox = trk.get("bbox") if isinstance(trk, dict) else None
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
            # Guard against out-of-frame boxes from any future change.
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 115), 2)

    # --- 2. Top header bar (operator-only, semi-transparent) ---
    header_h = 86
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)

    cv2.putText(
        annotated,
        "VISIONQUEUE",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    state_col = get_state_color(str(system_state))
    cv2.circle(annotated, (22, 58), 7, state_col, -1)
    cv2.putText(
        annotated,
        f"STATUS: {str(system_state)}",
        (38, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        state_col,
        2,
        cv2.LINE_AA,
    )

    # --- 3. The five required operator tiles ---
    crowd_col = get_crowd_color(crowd_level)

    tile_y_label = 18
    tile_y_value = 62
    tile_x0 = min(int(w * 0.28), 200)
    tile_w = max(1, (w - tile_x0 - 14) // 5)
    label_scale = min(0.45, tile_w / 190.0)
    value_scale = min(0.85, tile_w / 120.0)

    def tile(x: int, label: str, value: str, color: Tuple[int, int, int]) -> None:
        cv2.putText(
            annotated, label, (x, tile_y_label),
            cv2.FONT_HERSHEY_SIMPLEX, label_scale, (170, 170, 170), 1, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, value, (x, tile_y_value),
            cv2.FONT_HERSHEY_SIMPLEX, value_scale, color, 2, cv2.LINE_AA,
        )

    tile(tile_x0 + 0 * tile_w, "PEOPLE NOW", str(people_now), (255, 255, 255))
    tile(tile_x0 + 1 * tile_w, "ENTERED", str(entered), (0, 220, 255))
    tile(tile_x0 + 2 * tile_w, "LEFT", str(left), (0, 220, 255))
    tile(tile_x0 + 3 * tile_w, "CROWD LEVEL", crowd_level, crowd_col)
    # Tile 5 reuses the system state so the rightmost column always shows
    # current health, even if a non-technical viewer missed the header dot.
    tile(tile_x0 + 4 * tile_w, "STATUS", str(system_state), state_col)

    # --- 4. Bottom control bar (single line, friendly) ---
    footer_h = 26
    footer_y = h - footer_h
    footer_overlay = annotated.copy()
    cv2.rectangle(footer_overlay, (0, footer_y), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(footer_overlay, 0.88, annotated, 0.12, 0, annotated)
    cv2.putText(
        annotated,
        "Press 'Q' or ESC to close",
        (14, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    return annotated


def render_offline_card(width: int, height: int, state: Dict[str, Any]) -> np.ndarray:
    """Render a friendly standby card when no camera frame is available.

    Uses only the same operator fields as the normal HUD. No internal
    diagnostics are shown.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 24)

    box_w, box_h = min(520, width - 40), 200
    bx1 = (width - box_w) // 2
    by1 = (height - box_h) // 2
    cv2.rectangle(canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), (35, 35, 45), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), (60, 60, 220), 2)

    counts = state.get("counts", {}) if isinstance(state, dict) else {}
    system_state = state.get("system_state", "UNKNOWN") if isinstance(state, dict) else "UNKNOWN"
    people_now = int(counts.get("current", 0) or 0)

    cv2.putText(
        canvas, "CAMERA NOT AVAILABLE",
        (bx1 + 25, by1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 240), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"Status: {system_state}",
        (bx1 + 25, by1 + 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"People now: {people_now} (last reliable)",
        (bx1 + 25, by1 + 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 115), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Waiting for camera feed...",
        (bx1 + 25, by1 + 175),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA,
    )
    return canvas
