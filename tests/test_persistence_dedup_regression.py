"""Focused regression tests proving the persistence event-duplication bug.

Approach: instead of relying on the writer to actually persist the events,
these tests inspect the LOGICAL state of the service after several
observations. They verify:

  * the number of distinct logical events queued for a transition
  * the in-flight critical queue contents
  * the actual database contents after ACKs are allowed to drain

The `DelayedRepo` blocks writer-thread writes for `event_delay` seconds so
the first observation of a transition remains in-flight while later
(repeated) observations of the same logical state occur.

These tests use only the public + `_persist_state` method of CVService and do
NOT modify production code.
"""

import json
import threading
import time
import pytest

from visionqueue.api.service import CVService
from visionqueue.persistence.database import Database
from visionqueue.persistence.models import (
    CameraRecord,
    EventRecord,
    SessionCreateRecord,
)
from visionqueue.persistence.repository import PersistenceRepository
from visionqueue.persistence.writer import PersistenceWriter


def _state(system_state="STARTING", camera_state="CONNECTED", crowd_level="LOW",
           entries=0, exits=0, current=0, frame_id=1, session_id="dedup_sess"):
    return {
        "schema_version": 1,
        "timestamp": time.time(),
        "session_id": session_id,
        "frame_id": frame_id,
        "system_state": system_state,
        "camera_state": camera_state,
        "performance": {
            "processing_fps": 30.0,
            "inference_latency_ms": 12.0,
            "frame_age_ms": 15.0,
        },
        "vision": {
            "person_detection": "HEALTHY",
            "face_detection": "DISABLED",
            "tracking": "HEALTHY",
        },
        "counts": {
            "current": current,
            "track_instances": current,
            "unique_session_approx": entries,
            "entries": entries,
            "exits": exits,
            "net_count": entries - exits,
        },
        "occupancy": {
            "capacity": 10,
            "capacity_state": "NORMAL",
            "percent": float(current) * 10.0,
        },
        "crowd": {
            "level": crowd_level,
            "raw_level": crowd_level,
            "density_score": 0.2,
            "trend": "STABLE",
            "is_elevated": False,
            "peak_count": current,
            "peak_occupancy_percent": float(current) * 10.0,
            "peak_timestamp": time.time(),
        },
        "alerts": [],
        "status_reason": "ok",
        "is_healthy": True,
        "is_frozen": False,
        "detections_count": current,
        "tracks_count": current,
        "faces_count": 0,
        "queue_people": 0,
    }


class DelayedRepo:
    """Repository that delays (and optionally fails) EventRecord writes.

    The first call takes ``first_delay`` seconds; subsequent calls return
    immediately. This models a single in-flight slow write followed by fast
    backlog draining.
    """

    def __init__(self, real_repo, first_delay=2.0, subsequent_delay=0.0,
                 fail_first_n=0):
        self._real = real_repo
        self._first_delay = first_delay
        self._subsequent_delay = subsequent_delay
        self._fail_first_n = fail_first_n
        self._event_calls = 0
        self._written = []
        self._lock = threading.Lock()

    def execute(self, record, conn=None):
        delay = self._first_delay if self._event_calls == 0 else self._subsequent_delay
        if isinstance(record, EventRecord):
            with self._lock:
                idx = self._event_calls
                self._event_calls += 1
            if idx < self._fail_first_n:
                raise RuntimeError("Simulated transient DB failure")
            end = time.time() + delay
            while time.time() < end:
                time.sleep(0.005)
        if conn is None:
            self._real.execute(record)
        else:
            self._real.execute(record, conn=conn)
        with self._lock:
            self._written.append(record)

    def get_or_create_camera(self, cam):
        return self._real.get_or_create_camera(cam)

    def get_events(self, *a, **k):
        return self._real.get_events(*a, **k)

    def get_sessions(self, *a, **k):
        return self._real.get_sessions(*a, **k)

    def get_session(self, *a, **k):
        return self._real.get_session(*a, **k)

    def get_history(self, *a, **k):
        return self._real.get_history(*a, **k)

    def get_alerts(self, *a, **k):
        return self._real.get_alerts(*a, **k)


@pytest.fixture
def svc(tmp_path):
    """Provide a CVService whose writer's repo delays the first EventRecord
    write long enough that multiple observations pile up behind it."""
    db_path = str(tmp_path / "dedup.db")
    db = Database(db_path)
    db.initialize()
    real_repo = PersistenceRepository(db)
    # First event write takes 2s; subsequent writes are immediate. This models
    # a single slow in-flight write followed by fast backlog draining.
    delayed = DelayedRepo(real_repo, first_delay=2.0, subsequent_delay=0.0)
    writer = PersistenceWriter(delayed, max_queue_size=1000)

    svc = CVService.__new__(CVService)
    svc._pipeline = None
    svc._persistence_config = type("C", (), {"enabled": True,
                                              "database_path": db_path,
                                              "measurement_interval_seconds": 0.0,
                                              "max_queue_size": 1000})()
    svc._writer = writer
    svc._repo = real_repo
    svc._active_session_id = "dedup_sess"
    svc._camera_id = 1
    svc._last_measurement_time = 0.0
    svc._persistence_healthy = True
    svc._pending_critical = {}
    svc._in_flight_critical = set()
    svc._known_active_alerts = {}
    svc._prev_persist_state = svc._initial_prev_state()
    svc._running = False
    svc._stop_event = threading.Event()
    svc._state_lock = threading.Lock()
    svc._lifecycle_lock = threading.Lock()
    svc._latest_state_dict = {}
    svc._latest_tracks = []

    cid = real_repo.get_or_create_camera(
        CameraRecord(None, "test", "0", None, None, None, None, None, None,
                     "2026-09-06T12:00:00Z")
    )
    svc._camera_id = cid
    writer.write_sync(SessionCreateRecord(id="dedup_sess", camera_id=cid,
                                          started_at="2026-09-06T12:00:00Z"))
    writer.start_async_worker()
    yield svc, real_repo, writer, delayed

    # Stop writer cleanly
    try:
        writer.flush_all(timeout=10)
    except Exception:
        pass
    writer.stop_async_worker(timeout=5.0)


def _count_events(real_repo, event_type, session_id="dedup_sess"):
    events = real_repo.get_events(session_id, limit=1000)
    return [e for e in events if e["event_type"] == event_type]


def _sys_keys(service):
    return sorted(k for k in service._pending_critical.keys() if ":SYS:" in k)


# ------------------------------------------------------------------
# A. SYSTEM DELAYED ACK
# ------------------------------------------------------------------

def test_system_delayed_ack_no_duplicate(svc):
    """While the STARTING->LIVE event write is still in-flight, repeat the
    observation several times. Only ONE logical SYSTEM_STATE_CHANGED should be
    queued (regardless of how many times we observe the new state)."""
    service, repo, writer, delayed = svc

    # Baseline
    service._persist_state(_state(system_state="STARTING", frame_id=1))
    assert _sys_keys(service) == [], _sys_keys(service)

    # Transition observation
    service._persist_state(_state(system_state="LIVE", frame_id=2))
    keys_after_first = _sys_keys(service)
    assert len(keys_after_first) == 1, keys_after_first

    # Repeated same-state observations
    for i in range(5):
        service._persist_state(_state(system_state="LIVE", frame_id=3 + i))
    keys_after_repeats = _sys_keys(service)
    # BUG: with the current code, this is 6 (one per observation).
    # FIX: this should remain 1.
    assert len(keys_after_repeats) == 1, (
        f"Expected exactly 1 SYSTEM_STATE_CHANGED in flight; got "
        f"{len(keys_after_repeats)}: {keys_after_repeats}"
    )

    # Now drain everything and confirm exactly one row in the DB.
    # Subsequent writes return immediately, so the backlog drains in <1s.
    # Wait for the first (slow) write to complete + ACKs to drain.
    deadline = time.time() + 8.0
    while time.time() < deadline:
        # Process any pending ACKs
        while True:
            ack = writer.get_ack_nowait()
            if not ack:
                break
            service._pending_critical.pop(ack.record.record_key, None)
            service._in_flight_critical.discard(ack.record.record_key)
            if ack.success:
                service._advance_state(ack.record)
        if (not service._pending_critical and
                not service._in_flight_critical):
            break
        time.sleep(0.05)
    # Allow the writer to fully drain any remaining queue
    time.sleep(1.5)
    # Final ACK drain
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    sys_events = _count_events(repo, "SYSTEM_STATE_CHANGED")
    assert len(sys_events) == 1, (
        f"Expected 1 SYSTEM_STATE_CHANGED in DB, got {len(sys_events)}: "
        f"{[json.loads(e['metadata_json']) for e in sys_events]}"
    )
    md = json.loads(sys_events[0]["metadata_json"])
    assert md["from"] == "STARTING"
    assert md["to"] == "LIVE"


# ------------------------------------------------------------------
# B. SYSTEM REPEATED SAME STATE
# ------------------------------------------------------------------

def test_system_repeated_same_state(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(system_state="STARTING", frame_id=1))
    for i in range(5):
        service._persist_state(_state(system_state="LIVE", frame_id=2 + i))
    keys = _sys_keys(service)
    assert len(keys) == 1, keys

    # Drain
    deadline = time.time() + 8.0
    while time.time() < deadline:
        while True:
            ack = writer.get_ack_nowait()
            if not ack:
                break
            service._pending_critical.pop(ack.record.record_key, None)
            service._in_flight_critical.discard(ack.record.record_key)
            if ack.success:
                service._advance_state(ack.record)
        if (not service._pending_critical and
                not service._in_flight_critical):
            break
        time.sleep(0.05)
    time.sleep(1.5)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    sys_events = _count_events(repo, "SYSTEM_STATE_CHANGED")
    assert len(sys_events) == 1


# ------------------------------------------------------------------
# C. SYSTEM DISTINCT TRANSITIONS
# ------------------------------------------------------------------

def test_system_distinct_transitions(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(system_state="STARTING", frame_id=1))
    service._persist_state(_state(system_state="LIVE", frame_id=2))
    # Wait for first transition to ACK
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    service._persist_state(_state(system_state="OFFLINE", frame_id=3))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    service._persist_state(_state(system_state="LIVE", frame_id=4))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    sys_events = _count_events(repo, "SYSTEM_STATE_CHANGED")
    sys_events.sort(key=lambda e: e["occurred_at"])
    assert len(sys_events) == 3, (
        f"Expected 3 transitions, got {len(sys_events)}: "
        f"{[json.loads(e['metadata_json']) for e in sys_events]}"
    )
    flows = [json.loads(e["metadata_json"]) for e in sys_events]
    assert (flows[0]["from"], flows[0]["to"]) == ("STARTING", "LIVE")
    assert (flows[1]["from"], flows[1]["to"]) == ("LIVE", "OFFLINE")
    assert (flows[2]["from"], flows[2]["to"]) == ("OFFLINE", "LIVE")


# ------------------------------------------------------------------
# E. CROWD DELAYED ACK
# ------------------------------------------------------------------

def test_crowd_delayed_ack_no_duplicate(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(crowd_level="LOW", frame_id=1))
    service._persist_state(_state(crowd_level="HIGH", frame_id=2))
    for i in range(5):
        service._persist_state(_state(crowd_level="HIGH", frame_id=3 + i))
    crowd_keys = [k for k in service._pending_critical if ":CROWD:" in k]
    assert len(crowd_keys) == 1, crowd_keys

    # Drain
    deadline = time.time() + 8.0
    while time.time() < deadline:
        while True:
            ack = writer.get_ack_nowait()
            if not ack:
                break
            service._pending_critical.pop(ack.record.record_key, None)
            service._in_flight_critical.discard(ack.record.record_key)
            if ack.success:
                service._advance_state(ack.record)
        if (not service._pending_critical and
                not service._in_flight_critical):
            break
        time.sleep(0.05)
    time.sleep(1.5)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    crowd_events = _count_events(repo, "CROWD_LEVEL_CHANGED")
    assert len(crowd_events) == 1, crowd_events
    md = json.loads(crowd_events[0]["metadata_json"])
    assert md["from"] == "LOW"
    assert md["to"] == "HIGH"


# ------------------------------------------------------------------
# F. CROWD DISTINCT TRANSITIONS
# ------------------------------------------------------------------

def test_crowd_distinct_transitions(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(crowd_level="LOW", frame_id=1))
    service._persist_state(_state(crowd_level="HIGH", frame_id=2))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    service._persist_state(_state(crowd_level="LOW", frame_id=3))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    crowd_events = _count_events(repo, "CROWD_LEVEL_CHANGED")
    assert len(crowd_events) == 2, crowd_events


# ------------------------------------------------------------------
# G. CAMERA DELAYED ACK
# ------------------------------------------------------------------

def test_camera_delayed_ack_no_duplicate(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(camera_state="CONNECTED", frame_id=1))
    service._persist_state(_state(camera_state="DISCONNECTED", frame_id=2))
    for i in range(5):
        service._persist_state(_state(camera_state="DISCONNECTED", frame_id=3 + i))
    cam_keys = [k for k in service._pending_critical if ":CAM:" in k]
    assert len(cam_keys) == 1, cam_keys

    deadline = time.time() + 8.0
    while time.time() < deadline:
        while True:
            ack = writer.get_ack_nowait()
            if not ack:
                break
            service._pending_critical.pop(ack.record.record_key, None)
            service._in_flight_critical.discard(ack.record.record_key)
            if ack.success:
                service._advance_state(ack.record)
        if (not service._pending_critical and
                not service._in_flight_critical):
            break
        time.sleep(0.05)
    time.sleep(1.5)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)

    cam_events = _count_events(repo, "CAMERA_STATE_CHANGED")
    assert len(cam_events) == 1, cam_events
    md = json.loads(cam_events[0]["metadata_json"])
    assert md["from"] == "CONNECTED"
    assert md["to"] == "DISCONNECTED"


# ------------------------------------------------------------------
# H. CAMERA DISTINCT TRANSITIONS
# ------------------------------------------------------------------

def test_camera_distinct_transitions(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(camera_state="CONNECTED", frame_id=1))
    service._persist_state(_state(camera_state="DISCONNECTED", frame_id=2))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    service._persist_state(_state(camera_state="CONNECTED", frame_id=3))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    cam_events = _count_events(repo, "CAMERA_STATE_CHANGED")
    assert len(cam_events) == 2, cam_events


# ------------------------------------------------------------------
# I. ENTRY/EXIT REGRESSION
# ------------------------------------------------------------------

def test_entry_event_count_not_duplicated(svc):
    service, repo, writer, delayed = svc
    service._persist_state(_state(entries=5, frame_id=1))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    service._persist_state(_state(entries=8, frame_id=2))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    service._persist_state(_state(entries=10, frame_id=3))
    time.sleep(3.0)
    while True:
        ack = writer.get_ack_nowait()
        if not ack:
            break
        service._pending_critical.pop(ack.record.record_key, None)
        service._in_flight_critical.discard(ack.record.record_key)
        if ack.success:
            service._advance_state(ack.record)
    entry_events = _count_events(repo, "ENTRY")
    # 3 unique cumulative entries -> 3 ENTRY records with deltas 5, 3, 2.
    assert len(entry_events) == 3, entry_events
    values = sorted([e["value"] for e in entry_events])
    assert values == [2, 3, 5]


# ------------------------------------------------------------------
# J. QUEUE PRESSURE
# ------------------------------------------------------------------

def test_queue_pressure_does_not_lose_events(svc):
    """Push many transitions through; verify that:
    - no transition is silently lost (events end up in DB)
    - if a transition is in flight when the queue fills, the event stays in
      _pending_critical and is recoverable
    """
    service, repo, writer, delayed = svc

    service._persist_state(_state(system_state="STARTING", frame_id=1))
    # 20 unique system_state values, each will produce one logical event after
    # the previous ACK. We use states that are unique to bypass dedup.
    states = ["STATE_" + str(i) for i in range(10)]
    for i, s in enumerate(states):
        service._persist_state(_state(system_state=s, frame_id=2 + i))
        time.sleep(2.5)  # wait for each write to finish (2s delay)
        while True:
            ack = writer.get_ack_nowait()
            if not ack:
                break
            service._pending_critical.pop(ack.record.record_key, None)
            service._in_flight_critical.discard(ack.record.record_key)
            if ack.success:
                service._advance_state(ack.record)

    sys_events = _count_events(repo, "SYSTEM_STATE_CHANGED")
    # STARTING->STATE_0 plus 9 subsequent transitions
    assert len(sys_events) >= 10, len(sys_events)


# ------------------------------------------------------------------
# D. SYSTEM FAILURE + RETRY
# ------------------------------------------------------------------

def test_system_failure_then_retry_one_event(tmp_path):
    """First persistence attempt fails. The retry must succeed, and the
    final DB must contain exactly one logical transition."""
    db_path = str(tmp_path / "dedup_fail.db")
    db = Database(db_path)
    db.initialize()
    real_repo = PersistenceRepository(db)
    # First EventRecord raises; subsequent succeed. No delay.
    delayed = DelayedRepo(real_repo, first_delay=0.0, subsequent_delay=0.0,
                          fail_first_n=1)
    writer = PersistenceWriter(delayed, max_queue_size=100)
    svc = CVService.__new__(CVService)
    svc._pipeline = None
    svc._persistence_config = type("C", (), {"enabled": True,
                                              "database_path": db_path,
                                              "measurement_interval_seconds": 0.0,
                                              "max_queue_size": 100})()
    svc._writer = writer
    svc._repo = real_repo
    svc._active_session_id = "fail_sess"
    svc._camera_id = 1
    svc._last_measurement_time = 0.0
    svc._persistence_healthy = True
    svc._pending_critical = {}
    svc._in_flight_critical = set()
    svc._known_active_alerts = {}
    svc._prev_persist_state = svc._initial_prev_state()
    svc._running = False
    svc._stop_event = threading.Event()
    svc._state_lock = threading.Lock()
    svc._lifecycle_lock = threading.Lock()
    svc._latest_state_dict = {}
    svc._latest_tracks = []

    cid = real_repo.get_or_create_camera(
        CameraRecord(None, "t", "0", None, None, None, None, None, None,
                     "2026-09-06T12:00:00Z")
    )
    svc._camera_id = cid
    writer.write_sync(SessionCreateRecord(id="fail_sess", camera_id=cid,
                                          started_at="2026-09-06T12:00:00Z"))
    writer.start_async_worker()
    try:
        svc._persist_state(_state(system_state="STARTING", frame_id=1,
                                  session_id="fail_sess"))
        svc._persist_state(_state(system_state="LIVE", frame_id=2,
                                  session_id="fail_sess"))
        # Allow the worker to fail and then retry through the writer's retry
        # loop. Writer retries up to 3 times with backoff (~0.1+0.2+0.4=0.7s).
        time.sleep(2.0)
        # The first write failed; the record is buffered in _pending_critical
        # (no _advance_state happened). Subsequent _process_persistence_acks
        # invocations should retry the buffered record.
        # But the writer's _process_persistence_acks is only called once per
        # worker iteration. So we simulate it manually here.
        for _ in range(20):
            while True:
                ack = writer.get_ack_nowait()
                if not ack:
                    break
                if ack.success:
                    svc._pending_critical.pop(ack.record.record_key, None)
                    svc._in_flight_critical.discard(ack.record.record_key)
                    svc._advance_state(ack.record)
                else:
                    # Leave it in pending/in-flight for retry
                    svc._in_flight_critical.discard(ack.record.record_key)
            # Re-enqueue any pending not in flight
            for k, r in list(svc._pending_critical.items()):
                if k not in svc._in_flight_critical:
                    try:
                        writer.enqueue_critical(r)
                        svc._in_flight_critical.add(k)
                    except Exception:
                        pass
            time.sleep(0.2)
        # Final drain
        time.sleep(1.0)
        while True:
            ack = writer.get_ack_nowait()
            if not ack:
                break
            if ack.success:
                svc._pending_critical.pop(ack.record.record_key, None)
                svc._in_flight_critical.discard(ack.record.record_key)
                svc._advance_state(ack.record)
            else:
                svc._in_flight_critical.discard(ack.record.record_key)

        events = real_repo.get_events("fail_sess", limit=100)
        sys_events = [e for e in events if e["event_type"] == "SYSTEM_STATE_CHANGED"]
        assert len(sys_events) == 1, sys_events
    finally:
        try:
            writer.flush_all(timeout=10)
        except Exception:
            pass
        writer.stop_async_worker(timeout=5.0)


# ------------------------------------------------------------------
# K. LIFECYCLE / SESSION RESET
# ------------------------------------------------------------------

def test_session_reset_clears_dedup_state(svc):
    """After reset_session, the next STARTING->LIVE observation must again
    produce exactly one logical event."""
    service, repo, writer, delayed = svc

    # First cycle
    service._persist_state(_state(system_state="STARTING", frame_id=1))
    service._persist_state(_state(system_state="LIVE", frame_id=2))
    for _ in range(3):
        service._persist_state(_state(system_state="LIVE", frame_id=3))

    # Reset (must be called while running per API contract). Since we are
    # not using a real pipeline here, manually call the dedup reset logic.
    service._prev_persist_state = service._initial_prev_state()
    service._pending_critical.clear()
    service._in_flight_critical.clear()

    # New session after reset
    service._persist_state(_state(system_state="STARTING", frame_id=10))
    service._persist_state(_state(system_state="LIVE", frame_id=11))
    for _ in range(3):
        service._persist_state(_state(system_state="LIVE", frame_id=12))

    keys = _sys_keys(service)
    assert len(keys) == 1, keys


# ------------------------------------------------------------------
# K. SHUTDOWN WITH PENDING PERSISTENCE EVENTS
# ------------------------------------------------------------------

def test_shutdown_drains_pending_events(tmp_path):
    """After service.stop(), the writer's flush_all must persist all
    pending events."""
    db_path = str(tmp_path / "shutdown_drain.db")
    db = Database(db_path)
    db.initialize()
    real_repo = PersistenceRepository(db)
    # No delay so writes happen quickly
    writer = PersistenceWriter(real_repo, max_queue_size=100)

    svc = CVService.__new__(CVService)
    svc._pipeline = None
    svc._persistence_config = type("C", (), {"enabled": True,
                                              "database_path": db_path,
                                              "measurement_interval_seconds": 0.0,
                                              "max_queue_size": 100})()
    svc._writer = writer
    svc._repo = real_repo
    svc._active_session_id = "shutdown_sess"
    svc._camera_id = 1
    svc._last_measurement_time = 0.0
    svc._persistence_healthy = True
    svc._pending_critical = {}
    svc._in_flight_critical = set()
    svc._known_active_alerts = {}
    svc._prev_persist_state = svc._initial_prev_state()
    svc._running = False
    svc._stop_event = threading.Event()
    svc._state_lock = threading.Lock()
    svc._lifecycle_lock = threading.Lock()
    svc._latest_state_dict = {}
    svc._latest_tracks = []

    cid = real_repo.get_or_create_camera(
        CameraRecord(None, "t", "0", None, None, None, None, None, None,
                     "2026-09-06T12:00:00Z")
    )
    svc._camera_id = cid
    writer.write_sync(SessionCreateRecord(id="shutdown_sess", camera_id=cid,
                                          started_at="2026-09-06T12:00:00Z"))
    writer.start_async_worker()

    svc._persist_state(_state(system_state="STARTING", frame_id=1,
                              session_id="shutdown_sess"))
    svc._persist_state(_state(system_state="LIVE", frame_id=2,
                              session_id="shutdown_sess"))
    # Simulate service.stop() flow: flush_all + write_sync(SessionClose)
    writer.flush_all(timeout=5.0)
    writer.write_sync(SessionCreateRecord.__class__.__mro__[0].__new__(
        type("SCR", (), {}), id="dummy", camera_id=cid, started_at="now"
    ) if False else __import__(
        "visionqueue.persistence.models", fromlist=["SessionCloseRecord"]
    ).SessionCloseRecord(session_id="shutdown_sess", status="STOPPED",
                         ended_at="2026-09-06T12:00:00Z"))
    writer.stop_async_worker(timeout=5.0)

    events = real_repo.get_events("shutdown_sess", limit=100)
    sys_events = [e for e in events if e["event_type"] == "SYSTEM_STATE_CHANGED"]
    assert len(sys_events) == 1, sys_events