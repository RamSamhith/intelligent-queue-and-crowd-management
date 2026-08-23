"""Comprehensive unit tests for the VisionQueue alert subsystem."""

import pytest
from visionqueue.alerts import (
    Alert,
    AlertEngine,
    AlertEngineConfig,
    AlertRuleConfig,
    AlertSeverity,
    AlertStatus,
    AlertType,
    SymmetricAlertDebouncer,
)
from visionqueue.analytics import CrowdLevel


class TestSymmetricAlertDebouncer:
    """Unit tests for the underlying SymmetricAlertDebouncer."""

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            AlertRuleConfig(debounce_seconds=-1.0)
        with pytest.raises(ValueError):
            AlertRuleConfig(clear_seconds=-0.5)

    def test_condition_becomes_true_debounce_not_reached(self):
        debouncer = SymmetricAlertDebouncer(
            alert_type=AlertType.CRITICAL_OCCUPANCY,
            config=AlertRuleConfig(debounce_seconds=2.0, clear_seconds=2.0),
        )

        # Condition becomes True at t=1.0
        ev1 = debouncer.update(condition=True, timestamp=1.0)
        assert ev1 is None
        assert not debouncer.is_active
        assert debouncer.active_alert is None

        # At t=2.0 (elapsed 1.0s < 2.0s debounce)
        ev2 = debouncer.update(condition=True, timestamp=2.0)
        assert ev2 is None
        assert not debouncer.is_active

    def test_debounce_reached_transitions_to_active(self):
        debouncer = SymmetricAlertDebouncer(
            alert_type=AlertType.CRITICAL_OCCUPANCY,
            config=AlertRuleConfig(debounce_seconds=2.0, clear_seconds=2.0),
        )

        # t=1.0: Start condition
        debouncer.update(condition=True, timestamp=1.0)

        # t=3.0: Elapsed 2.0s -> Fires!
        alert = debouncer.update(condition=True, timestamp=3.0, reason="Critical crowd")
        assert alert is not None
        assert alert.type == AlertType.CRITICAL_OCCUPANCY
        assert alert.status == AlertStatus.ACTIVE
        assert alert.fired_at == 3.0
        assert alert.cleared_at is None
        assert alert.reason == "Critical crowd"
        assert debouncer.is_active
        assert debouncer.active_alert == alert

    def test_repeated_true_does_not_refire_while_active(self):
        debouncer = SymmetricAlertDebouncer(
            alert_type=AlertType.CRITICAL_OCCUPANCY,
            config=AlertRuleConfig(debounce_seconds=2.0, clear_seconds=2.0),
        )

        debouncer.update(condition=True, timestamp=1.0)
        first_alert = debouncer.update(condition=True, timestamp=3.0)
        assert first_alert is not None

        # Multiple subsequent frames with condition=True
        assert debouncer.update(condition=True, timestamp=4.0) is None
        assert debouncer.update(condition=True, timestamp=5.0) is None
        assert debouncer.update(condition=True, timestamp=10.0) is None
        assert debouncer.is_active

    def test_condition_clears_with_clear_debounce(self):
        debouncer = SymmetricAlertDebouncer(
            alert_type=AlertType.CRITICAL_OCCUPANCY,
            config=AlertRuleConfig(debounce_seconds=2.0, clear_seconds=3.0),
        )

        # Fire alert at t=3.0
        debouncer.update(condition=True, timestamp=1.0)
        active_alert = debouncer.update(condition=True, timestamp=3.0)
        assert active_alert is not None

        # t=4.0: Condition becomes False
        assert debouncer.update(condition=False, timestamp=4.0) is None
        assert debouncer.is_active  # Still active during clear window

        # t=6.0: Elapsed 2.0s (< 3.0s clear window)
        assert debouncer.update(condition=False, timestamp=6.0) is None
        assert debouncer.is_active

        # t=7.0: Elapsed 3.0s -> Cleared!
        cleared_alert = debouncer.update(condition=False, timestamp=7.0)
        assert cleared_alert is not None
        assert cleared_alert.id == active_alert.id
        assert cleared_alert.status == AlertStatus.CLEARED
        assert cleared_alert.fired_at == 3.0
        assert cleared_alert.cleared_at == 7.0
        assert not debouncer.is_active
        assert debouncer.active_alert is None

    def test_clear_timer_resets_if_condition_returns_true(self):
        debouncer = SymmetricAlertDebouncer(
            alert_type=AlertType.CRITICAL_OCCUPANCY,
            config=AlertRuleConfig(debounce_seconds=2.0, clear_seconds=3.0),
        )

        debouncer.update(condition=True, timestamp=1.0)
        debouncer.update(condition=True, timestamp=3.0)
        assert debouncer.is_active

        # Condition False at t=4.0
        debouncer.update(condition=False, timestamp=4.0)
        # Condition returns True at t=5.0 (before 3.0s clear window)
        debouncer.update(condition=True, timestamp=5.0)

        # Condition goes False again at t=6.0
        debouncer.update(condition=False, timestamp=6.0)

        # At t=8.0: elapsed from t=6.0 is 2.0s < 3.0s -> not cleared
        assert debouncer.update(condition=False, timestamp=8.0) is None
        assert debouncer.is_active

        # At t=9.0: elapsed from t=6.0 is 3.0s -> Cleared!
        cleared = debouncer.update(condition=False, timestamp=9.0)
        assert cleared is not None
        assert cleared.status == AlertStatus.CLEARED

    def test_retrigger_after_clearing(self):
        debouncer = SymmetricAlertDebouncer(
            alert_type=AlertType.CRITICAL_OCCUPANCY,
            config=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        )

        # 1. Fire
        debouncer.update(condition=True, timestamp=1.0)
        a1 = debouncer.update(condition=True, timestamp=2.0)
        assert a1 is not None

        # 2. Clear
        debouncer.update(condition=False, timestamp=3.0)
        c1 = debouncer.update(condition=False, timestamp=4.0)
        assert c1 is not None

        # 3. Fire again
        debouncer.update(condition=True, timestamp=5.0)
        a2 = debouncer.update(condition=True, timestamp=6.0)
        assert a2 is not None
        assert a2.id != a1.id  # New distinct alert instance
        assert a2.fired_at == 6.0


class TestAlertEngine:
    """Unit tests for the integrated P0 AlertEngine."""

    def test_critical_occupancy_alert_lifecycle(self):
        engine = AlertEngine(AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=2.0, clear_seconds=2.0),
        ))

        # t=1.0: Crowd level reaches CRITICAL
        events1 = engine.update(crowd_level=CrowdLevel.CRITICAL, timestamp=1.0)
        assert len(events1) == 0
        assert not engine.is_active(AlertType.CRITICAL_OCCUPANCY)

        # t=3.0: Stays CRITICAL for 2.0s -> Fires!
        events2 = engine.update(crowd_level=CrowdLevel.CRITICAL, timestamp=3.0)
        assert len(events2) == 1
        ev = events2[0]
        assert ev.type == AlertType.CRITICAL_OCCUPANCY
        assert ev.status == AlertStatus.ACTIVE
        assert engine.is_active(AlertType.CRITICAL_OCCUPANCY)
        assert len(engine.active_alerts) == 1

        # t=4.0: Dips to HIGH
        events3 = engine.update(crowd_level=CrowdLevel.HIGH, timestamp=4.0)
        assert len(events3) == 0

        # t=6.0: Stays HIGH for 2.0s -> Cleared!
        events4 = engine.update(crowd_level=CrowdLevel.HIGH, timestamp=6.0)
        assert len(events4) == 1
        assert events4[0].status == AlertStatus.CLEARED
        assert not engine.is_active(AlertType.CRITICAL_OCCUPANCY)
        assert len(engine.active_alerts) == 0

    def test_camera_failure_alert(self):
        engine = AlertEngine(AlertEngineConfig(
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        ))

        # Camera goes offline
        engine.update(crowd_level=CrowdLevel.LOW, camera_offline=True, timestamp=1.0)
        events = engine.update(crowd_level=CrowdLevel.LOW, camera_offline=True, timestamp=2.0)
        assert len(events) == 1
        assert events[0].type == AlertType.CAMERA_OR_DETECTION_FAILURE
        assert events[0].status == AlertStatus.ACTIVE
        assert "Camera offline" in events[0].reason

    def test_detection_failure_alert(self):
        engine = AlertEngine(AlertEngineConfig(
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        ))

        # Person detector fails
        engine.update(crowd_level=CrowdLevel.LOW, detection_failed=True, timestamp=1.0)
        events = engine.update(crowd_level=CrowdLevel.LOW, detection_failed=True, timestamp=2.0)
        assert len(events) == 1
        assert events[0].type == AlertType.CAMERA_OR_DETECTION_FAILURE
        assert events[0].status == AlertStatus.ACTIVE
        assert "Person detector failure" in events[0].reason

    def test_multiple_independent_alerts_simultaneously(self):
        engine = AlertEngine(AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
            camera_detection_failure=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        ))

        # Both conditions start at t=1.0
        engine.update(crowd_level=CrowdLevel.CRITICAL, camera_offline=True, timestamp=1.0)

        # Both fire at t=2.0
        events = engine.update(crowd_level=CrowdLevel.CRITICAL, camera_offline=True, timestamp=2.0)
        assert len(events) == 2
        active_types = {a.type for a in engine.active_alerts}
        assert active_types == {
            AlertType.CRITICAL_OCCUPANCY,
            AlertType.CAMERA_OR_DETECTION_FAILURE,
        }

    def test_reset_and_new_session(self):
        engine = AlertEngine(AlertEngineConfig(
            critical_occupancy=AlertRuleConfig(debounce_seconds=1.0, clear_seconds=1.0),
        ))

        engine.update(crowd_level=CrowdLevel.CRITICAL, timestamp=1.0)
        engine.update(crowd_level=CrowdLevel.CRITICAL, timestamp=2.0)
        assert engine.is_active(AlertType.CRITICAL_OCCUPANCY)

        engine.new_session()
        assert not engine.is_active(AlertType.CRITICAL_OCCUPANCY)
        assert len(engine.active_alerts) == 0
        assert len(engine.alert_history) == 0

    def test_alert_serialization_to_dict(self):
        alert = Alert(
            id="test-uuid",
            type=AlertType.CRITICAL_OCCUPANCY,
            severity=AlertSeverity.CRITICAL,
            fired_at=10.5,
            cleared_at=15.0,
            status=AlertStatus.CLEARED,
            reason="Occupancy exceeded 90%",
        )
        d = alert.to_dict()
        assert d == {
            "id": "test-uuid",
            "type": "CRITICAL_OCCUPANCY",
            "severity": "CRITICAL",
            "fired_at": 10.5,
            "cleared_at": 15.0,
            "status": "CLEARED",
            "reason": "Occupancy exceeded 90%",
        }
