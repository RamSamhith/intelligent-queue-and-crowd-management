"""VisionQueue Reliability and System Health Subsystem.

Provides:
1. Normalized System States (STARTING, LIVE, DEGRADED, UNSTABLE, OFFLINE, STOPPING)
2. Normalized Camera & Vision Health Summaries
3. Telemetry and Latency Metrics (FPS, Inference Latency, Frame Age)
4. State Transitions with Last-Reliable-Count Preservation
"""

from visionqueue.reliability.types import (
    CameraHealthState,
    PerformanceMetrics,
    ReliabilityConfig,
    SystemReliabilityState,
    SystemState,
    VisionHealthState,
    VisionHealthSummary,
)
from visionqueue.reliability.manager import ReliabilityManager

__all__ = [
    "CameraHealthState",
    "PerformanceMetrics",
    "ReliabilityConfig",
    "SystemReliabilityState",
    "SystemState",
    "VisionHealthState",
    "VisionHealthSummary",
    "ReliabilityManager",
]
