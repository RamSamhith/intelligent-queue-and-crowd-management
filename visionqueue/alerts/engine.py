"""P0 Alert Engine for VisionQueue.

Orchestrates symmetric debouncing and state management for:
1. CRITICAL_OCCUPANCY
2. CAMERA_OR_DETECTION_FAILURE
"""

from __future__ import annotations
from typing import Dict, List, Optional

from visionqueue.alerts.debouncer import SymmetricAlertDebouncer
from visionqueue.alerts.types import (
    Alert,
    AlertEngineConfig,
    AlertStatus,
    AlertType,
)
from visionqueue.analytics.types import CrowdLevel


class AlertEngine:
    """Manages P0 alerts with symmetric hysteresis debounce.

    Guarantees:
    - Exactly two P0 alert types managed with independent debouncers.
    - Symmetric hysteresis: alert must clear past clear_seconds before re-firing.
    - No duplicate firing while an alert is active.
    - Thread-safe / lightweight per-frame execution.
    """

    def __init__(self, config: Optional[AlertEngineConfig] = None) -> None:
        """Initialize AlertEngine with configuration.

        Args:
            config: Optional AlertEngineConfig with per-rule timing parameters.
        """
        self._config: AlertEngineConfig = config or AlertEngineConfig()
        self._debouncers: Dict[AlertType, SymmetricAlertDebouncer] = {
            AlertType.CRITICAL_OCCUPANCY: SymmetricAlertDebouncer(
                alert_type=AlertType.CRITICAL_OCCUPANCY,
                config=self._config.critical_occupancy,
            ),
            AlertType.CAMERA_OR_DETECTION_FAILURE: SymmetricAlertDebouncer(
                alert_type=AlertType.CAMERA_OR_DETECTION_FAILURE,
                config=self._config.camera_detection_failure,
            ),
        }
        self._all_events: List[Alert] = []

    @property
    def config(self) -> AlertEngineConfig:
        return self._config

    @property
    def active_alerts(self) -> List[Alert]:
        """List of all currently active Alert records."""
        return [
            debouncer.active_alert
            for debouncer in self._debouncers.values()
            if debouncer.active_alert is not None
        ]

    @property
    def alert_history(self) -> List[Alert]:
        """Chronological list of all emitted alert events (ACTIVE and CLEARED)."""
        return list(self._all_events)

    def is_active(self, alert_type: AlertType) -> bool:
        """Check if a specific alert type is currently active."""
        debouncer = self._debouncers.get(alert_type)
        return debouncer.is_active if debouncer is not None else False

    def update(
        self,
        crowd_level: CrowdLevel | str,
        camera_offline: bool = False,
        detection_failed: bool = False,
        timestamp: float = 0.0,
    ) -> List[Alert]:
        """Process a frame's status and return any state-transition Alert events.

        Args:
            crowd_level: Active (debounced) crowd tier.
            camera_offline: True if camera source is offline or disconnected.
            detection_failed: True if person detector is failing / unhealthy.
            timestamp: Frame acquisition / processing timestamp.

        Returns:
            List of Alert objects that transitioned (newly fired or newly cleared) in this frame.
        """
        emitted_events: List[Alert] = []

        # 1. Evaluate CRITICAL_OCCUPANCY condition
        is_critical_crowd = (
            crowd_level == CrowdLevel.CRITICAL
            if isinstance(crowd_level, CrowdLevel)
            else str(crowd_level).upper() == "CRITICAL"
        )
        crit_event = self._debouncers[AlertType.CRITICAL_OCCUPANCY].update(
            condition=is_critical_crowd,
            timestamp=timestamp,
            reason="Crowd occupancy reached CRITICAL tier",
        )
        if crit_event is not None:
            emitted_events.append(crit_event)
            self._all_events.append(crit_event)

        # 2. Evaluate CAMERA_OR_DETECTION_FAILURE condition
        is_failure = camera_offline or detection_failed
        failure_reasons = []
        if camera_offline:
            failure_reasons.append("Camera offline / disconnected")
        if detection_failed:
            failure_reasons.append("Person detector failure")
        failure_reason_str = " & ".join(failure_reasons) if failure_reasons else "Hardware or detection failure"

        fail_event = self._debouncers[AlertType.CAMERA_OR_DETECTION_FAILURE].update(
            condition=is_failure,
            timestamp=timestamp,
            reason=failure_reason_str,
        )
        if fail_event is not None:
            emitted_events.append(fail_event)
            self._all_events.append(fail_event)

        return emitted_events

    def reset(self) -> None:
        """Reset all debouncers and clear alert history."""
        for debouncer in self._debouncers.values():
            debouncer.reset()
        self._all_events.clear()

    def new_session(self) -> None:
        """Alias for reset() to demarcate session boundaries."""
        self.reset()
