"""Symmetric hysteresis alert debouncer for VisionQueue."""

from __future__ import annotations
import uuid
from typing import Optional, Tuple

from visionqueue.alerts.types import (
    Alert,
    AlertRuleConfig,
    AlertSeverity,
    AlertStatus,
    AlertType,
)


class SymmetricAlertDebouncer:
    """Stateful debouncer enforcing symmetric hysteresis for an alert condition.

    Lifecycle:
    1. INACTIVE State:
       - Condition becomes True -> starts debounce timer.
       - If condition remains continuously True for >= debounce_seconds:
         transitions to ACTIVE and emits an ACTIVE Alert.
       - If condition becomes False before debounce window expires: timer resets.
    2. ACTIVE State:
       - Condition becomes False -> starts clear timer.
       - If condition remains continuously False for >= clear_seconds:
         transitions to INACTIVE, sets cleared_at, and emits a CLEARED Alert.
       - If condition becomes True before clear window expires: clear timer resets.
       - Alert CANNOT re-fire while in ACTIVE state.
    """

    def __init__(
        self,
        alert_type: AlertType,
        config: Optional[AlertRuleConfig] = None,
    ) -> None:
        """Initialize the debouncer.

        Args:
            alert_type: AlertType category.
            config: AlertRuleConfig defining debounce and clear time windows.
        """
        self._alert_type: AlertType = alert_type
        self._config: AlertRuleConfig = config or AlertRuleConfig()

        self._is_active: bool = False
        self._active_alert: Optional[Alert] = None
        self._true_start_time: Optional[float] = None
        self._false_start_time: Optional[float] = None

    @property
    def alert_type(self) -> AlertType:
        return self._alert_type

    @property
    def config(self) -> AlertRuleConfig:
        return self._config

    @property
    def is_active(self) -> bool:
        """True if the alert is currently in ACTIVE status."""
        return self._is_active

    @property
    def active_alert(self) -> Optional[Alert]:
        """The currently active Alert instance, or None if inactive."""
        return self._active_alert

    def update(
        self,
        condition: bool,
        timestamp: float,
        reason: str = "",
    ) -> Optional[Alert]:
        """Process condition for current timestamp and return an Alert if state changed.

        Args:
            condition: Instantaneous state of the alert condition (True = triggering, False = healthy).
            timestamp: Monotonic or epoch timestamp in seconds.
            reason: Contextual explanation for the alert.

        Returns:
            Alert with status=ACTIVE if newly fired,
            Alert with status=CLEARED if newly cleared,
            None if no state transition occurred.
        """
        if not self._is_active:
            # Currently INACTIVE
            self._false_start_time = None

            if condition:
                if self._true_start_time is None:
                    self._true_start_time = timestamp

                elapsed = timestamp - self._true_start_time
                if elapsed >= self._config.debounce_seconds:
                    # Fire alert!
                    self._is_active = True
                    self._active_alert = Alert(
                        id=str(uuid.uuid4()),
                        type=self._alert_type,
                        severity=self._config.severity,
                        fired_at=timestamp,
                        cleared_at=None,
                        status=AlertStatus.ACTIVE,
                        reason=reason or f"{self._alert_type.value} threshold exceeded",
                    )
                    self._true_start_time = None
                    return self._active_alert
            else:
                self._true_start_time = None

            return None

        else:
            # Currently ACTIVE
            self._true_start_time = None

            if not condition:
                if self._false_start_time is None:
                    self._false_start_time = timestamp

                elapsed = timestamp - self._false_start_time
                if elapsed >= self._config.clear_seconds:
                    # Clear alert!
                    self._is_active = False
                    assert self._active_alert is not None
                    cleared_alert = Alert(
                        id=self._active_alert.id,
                        type=self._active_alert.type,
                        severity=self._active_alert.severity,
                        fired_at=self._active_alert.fired_at,
                        cleared_at=timestamp,
                        status=AlertStatus.CLEARED,
                        reason=self._active_alert.reason,
                    )
                    self._active_alert = None
                    self._false_start_time = None
                    return cleared_alert
            else:
                # Condition returned True before clear window expired - reset clear timer
                self._false_start_time = None

            return None

    def reset(self) -> None:
        """Reset internal debouncer timers and active state."""
        self._is_active = False
        self._active_alert = None
        self._true_start_time = None
        self._false_start_time = None
