"""VisionQueue Analytics Subsystem.

Provides modular, lightweight components for:
1. Safe Capacity-Based Occupancy Percentage Evaluation
2. Debounced Crowd-Level Classification (LOW, MODERATE, HIGH, CRITICAL)
3. Core+ Rolling Crowd Trend Analysis (INCREASING, STABLE, DECREASING)
4. Core+ Session Peak Metrics Tracking
"""

from visionqueue.analytics.types import (
    AnalyticsConfig,
    CapacitySource,
    CapacityState,
    CrowdAnalyticsState,
    CrowdLevel,
    CrowdThresholds,
    CrowdTrend,
    SceneProfile,
)
from visionqueue.analytics.crowd import CrowdAnalyticsEngine

__all__ = [
    "AnalyticsConfig",
    "CapacitySource",
    "CapacityState",
    "CrowdAnalyticsState",
    "CrowdLevel",
    "CrowdThresholds",
    "CrowdTrend",
    "CrowdAnalyticsEngine",
    "SceneProfile",
]
