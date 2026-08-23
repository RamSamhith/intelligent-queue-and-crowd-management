"""VisionQueue Alert Subsystem.

Provides symmetric debouncing and state management for P0 alerts:
1. CRITICAL_OCCUPANCY
2. CAMERA_OR_DETECTION_FAILURE
"""

from visionqueue.alerts.types import (
    Alert,
    AlertEngineConfig,
    AlertRuleConfig,
    AlertSeverity,
    AlertStatus,
    AlertType,
)
from visionqueue.alerts.debouncer import SymmetricAlertDebouncer
from visionqueue.alerts.engine import AlertEngine

__all__ = [
    "Alert",
    "AlertEngineConfig",
    "AlertRuleConfig",
    "AlertSeverity",
    "AlertStatus",
    "AlertType",
    "SymmetricAlertDebouncer",
    "AlertEngine",
]
