"""VisionQueue local live display.

A lightweight, modular OpenCV display that renders ONLY the human-readable
information a non-technical operator needs, on top of the *same* LiveState
and active tracks already produced by the production CV pipeline.

Design rules:
- Reads from the CVService's cached state (CVService.get_latest_state() and
  CVService.get_latest_tracks()). No independent counting, no independent
  analytics, no duplicate CV logic.
- Renders exactly the required five operator metrics:
    1. People now   (current headcount)
    2. Entered      (cumulative entries from line-crossing / session unique)
    3. Left         (cumulative exits)
    4. Crowd level  (LOW / MODERATE / HIGH / CRITICAL)
    5. Status       (system_state)
- Plus a current person bounding box overlay (drawn from the existing tracks).
- Does NOT show GPU, CUDA provider, ONNX model path, inference latency, frame
  ID, session UUID, memory, or other developer internals in the normal display.
  Those remain available through the API and the existing diagnostics.
"""

from visionqueue.viewer.hud import (
    draw_simple_hud,
    render_offline_card,
    get_crowd_color,
    get_state_color,
)
from visionqueue.viewer.display import SimpleDisplay

__all__ = [
    "draw_simple_hud",
    "render_offline_card",
    "get_crowd_color",
    "get_state_color",
    "SimpleDisplay",
]
