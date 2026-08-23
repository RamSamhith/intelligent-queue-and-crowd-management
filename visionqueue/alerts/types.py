"""Type definitions and contracts for the VisionQueue alert engine."""

from __future__ import annotations
from dataclasses import dataclass, field
import uuid
from enum import Enum
from typing import Optional


class AlertType(str, Enum):
    """P0 Alert category identifiers."""
    CRITICAL_OCCUPANCY = "CRITICAL_OCCUPANCY"
    CAMERA_OR_DETECTION_FAILURE = "CAMERA_OR_DETECTION_FAILURE"


class AlertSeverity(str, Enum):
    """Severity classification of alerts."""
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """Lifecycle status of an alert."""
    ACTIVE = "ACTIVE"
    CLEARED = "CLEARED"


@dataclass(frozen=True)
class Alert:
    """Standardized immutable alert record matching the VisionQueue contract.

    Attributes:
        id: Unique identifier for this alert instance.
        type: Category of alert (CRITICAL_OCCUPANCY or CAMERA_OR_DETECTION_FAILURE).
        severity: Severity level (WARNING or CRITICAL).
        fired_at: Epoch timestamp when the alert transitioned to ACTIVE.
        cleared_at: Epoch timestamp when the alert cleared, or None if currently active.
        status: ACTIVE or CLEARED.
        reason: Human-readable operational explanation of the alert.
    """
    id: str
    type: AlertType
    severity: AlertSeverity
    fired_at: float
    cleared_at: Optional[float]
    status: AlertStatus
    reason: str

    def to_dict(self) -> dict:
        """Serialize alert to standard dictionary format."""
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "fired_at": self.fired_at,
            "cleared_at": self.cleared_at,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AlertRuleConfig:
    """Timing and debounce configuration for an individual alert rule.

    Attributes:
        debounce_seconds: Seconds condition must remain continuously True before firing.
        clear_seconds: Seconds condition must remain continuously False before clearing.
        severity: Severity assigned to fired alerts.
    """
    debounce_seconds: float = 2.0
    clear_seconds: float = 2.0
    severity: AlertSeverity = AlertSeverity.CRITICAL

    def __post_init__(self):
        if self.debounce_seconds < 0:
            raise ValueError(f"debounce_seconds must be >= 0, got {self.debounce_seconds}")
        if self.clear_seconds < 0:
            raise ValueError(f"clear_seconds must be >= 0, got {self.clear_seconds}")


@dataclass(frozen=True)
class AlertEngineConfig:
    """Global configuration for the Alert Engine.

    Attributes:
        critical_occupancy: Configuration for CRITICAL_OCCUPANCY alerts.
        camera_detection_failure: Configuration for CAMERA_OR_DETECTION_FAILURE alerts.
    """
    critical_occupancy: AlertRuleConfig = field(default_factory=lambda: AlertRuleConfig(
        debounce_seconds=2.0,
        clear_seconds=2.0,
        severity=AlertSeverity.CRITICAL,
    ))
    camera_detection_failure: AlertRuleConfig = field(default_factory=lambda: AlertRuleConfig(
        debounce_seconds=2.0,
        clear_seconds=2.0,
        severity=AlertSeverity.CRITICAL,
    ))
