"""Unit tests for the clean live CV viewer UX, HUD rendering, and capacity alert notifications."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from visionqueue.alerts.types import Alert, AlertSeverity, AlertStatus, AlertType
from visionqueue.analytics.types import CrowdLevel, CrowdTrend
from visionqueue.pipeline.types import LiveState
from visionqueue.reliability.types import CameraHealthState, PerformanceMetrics, SystemState, VisionHealthSummary
from scratch.live_cv_viewer import draw_hud, get_crowd_color, get_state_color


def make_dummy_live_state(
    current_count: int = 5,
    capacity: int = 10,
    occupancy_percent: int = 50,
    crowd_level: str = "MODERATE",
    alerts: list = None,
    system_state: SystemState = SystemState.LIVE,
) -> LiveState:
    """Helper to construct dummy LiveState objects for HUD rendering tests."""
    return LiveState(
        schema_version=1,
        timestamp=100.0,
        session_id="test-session",
        frame_id=120,
        system_state=system_state,
        camera_state=CameraHealthState.CONNECTED,
        performance=PerformanceMetrics(processing_fps=30.0, inference_latency_ms=12.5, frame_age_ms=18.0),
        vision=VisionHealthSummary(person_detection="OK", face_detection="OK", tracking="OK"),
        counts={"current": current_count, "unique_session_approx": 15, "entries": 10, "exits": 5, "net_count": 5},
        occupancy={"capacity": capacity, "capacity_state": "SET" if capacity else "NOT_SET", "percent": occupancy_percent},
        crowd={"level": crowd_level, "raw_level": crowd_level, "trend": CrowdTrend.STABLE.value, "peak_count": 8, "peak_occupancy_percent": 80, "peak_timestamp": 90.0},
        alerts=alerts or [],
        status_reason="Nominal",
        is_healthy=True,
        is_frozen=False,
    )


class TestViewerHUD:
    """Tests for clean HUD rendering without technical clutter."""

    def test_draw_hud_renders_without_error(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = make_dummy_live_state(current_count=5, capacity=10, occupancy_percent=50)
        diagnostics = {
            "active_tracks": [{"track_id": 1, "bbox": [100.0, 100.0, 200.0, 300.0], "confidence": 0.88}],
            "scene_analysis": {"capacity_source": "AUTOMATIC", "effective_capacity": 10},
        }

        output_frame = draw_hud(frame, state, diagnostics)
        assert output_frame is not None
        assert output_frame.shape == (480, 640, 3)
        # Verify output is modified (has drawings)
        assert np.any(output_frame != 0)

    def test_draw_hud_active_alert_banner_rendered(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        alert = Alert(
            id="alert-occ-1",
            type=AlertType.CRITICAL_OCCUPANCY,
            severity=AlertSeverity.CRITICAL,
            fired_at=100.0,
            cleared_at=None,
            status=AlertStatus.ACTIVE,
            reason="Capacity reached 10/10 (100%)",
        )
        state = make_dummy_live_state(
            current_count=10,
            capacity=10,
            occupancy_percent=100,
            crowd_level="CRITICAL",
            alerts=[alert],
        )
        diagnostics = {
            "active_tracks": [],
            "scene_analysis": {"capacity_source": "MANUAL", "effective_capacity": 10},
        }

        output_frame = draw_hud(frame, state, diagnostics)
        assert output_frame is not None
        assert np.any(output_frame != 0)

    def test_get_crowd_color_mappings(self):
        assert get_crowd_color("LOW") == (40, 200, 40)
        assert get_crowd_color("MODERATE") == (0, 215, 255)
        assert get_crowd_color("HIGH") == (0, 140, 255)
        assert get_crowd_color("CRITICAL") == (40, 40, 240)

    def test_get_state_color_mappings(self):
        assert get_state_color("LIVE") == (40, 200, 40)
        assert get_state_color("DEGRADED") == (0, 215, 255)
        assert get_state_color("OFFLINE") == (40, 40, 240)


class TestVirtualLineContracts:
    """Tests for VirtualLine configuration, serialization, and arbitrary geometry."""

    def test_virtual_line_to_dict_and_from_dict(self):
        from visionqueue.counting.types import VirtualLine, CrossingDirection
        line = VirtualLine(pt1=(100.0, 200.0), pt2=(300.0, 400.0), entry_direction=CrossingDirection.ENTRY)
        d = line.to_dict()
        assert d["pt1"] == [100.0, 200.0]
        assert d["pt2"] == [300.0, 400.0]
        assert d["entry_direction"] == "ENTRY"

        restored = VirtualLine.from_dict(d)
        assert restored.pt1 == (100.0, 200.0)
        assert restored.pt2 == (300.0, 400.0)
        assert restored.entry_direction == CrossingDirection.ENTRY
        assert round(restored.length, 2) == round(line.length, 2)

    def test_arbitrary_line_orientations(self):
        from visionqueue.counting.types import VirtualLine, CrossingDirection
        # Vertical line
        v_line = VirtualLine(pt1=(320.0, 50.0), pt2=(320.0, 450.0))
        assert v_line.length == 400.0

        # Horizontal line
        h_line = VirtualLine(pt1=(50.0, 240.0), pt2=(590.0, 240.0))
        assert h_line.length == 540.0

        # Diagonal line
        d_line = VirtualLine(pt1=(100.0, 100.0), pt2=(400.0, 500.0))
        assert round(d_line.length, 2) == 500.0

    def test_draw_hud_virtual_line_and_roi_toggles(self):
        from visionqueue.counting.types import VirtualLine
        from visionqueue.roi.types import ROIConfig
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = make_dummy_live_state(current_count=3, capacity=None, occupancy_percent=None)
        diagnostics = {"active_tracks": [], "scene_analysis": {}}
        line = VirtualLine(pt1=(100.0, 100.0), pt2=(500.0, 100.0))
        roi = ROIConfig(x=50.0, y=50.0, width=540.0, height=380.0)

        # 1. Clean V1 Whole-Frame Default (no line, no ROI on frame)
        f_clean = draw_hud(frame, state, diagnostics, virtual_line=line, show_virtual_line=False, roi=roi, show_roi=False)
        assert f_clean is not None

        # 2. Line overlay toggled ON
        f_line = draw_hud(frame, state, diagnostics, virtual_line=line, show_virtual_line=True, roi=roi, show_roi=False)
        assert f_line is not None

        # 3. ROI overlay toggled ON
        f_roi = draw_hud(frame, state, diagnostics, virtual_line=line, show_virtual_line=False, roi=roi, show_roi=True)
        assert f_roi is not None
