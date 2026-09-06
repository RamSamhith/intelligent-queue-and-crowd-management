-- cameras
CREATE TABLE IF NOT EXISTS cameras (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    source          TEXT NOT NULL,
    location        TEXT,
    capacity        INTEGER CHECK(capacity IS NULL OR capacity > 0),
    roi_x           INTEGER,
    roi_y           INTEGER,
    roi_width       INTEGER,
    roi_height      INTEGER,
    created_at      TEXT NOT NULL
);

-- sessions
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    camera_id       INTEGER NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    status          TEXT NOT NULL CHECK(status IN ('RUNNING','COMPLETED','FAILED','STOPPED')),
    FOREIGN KEY(camera_id) REFERENCES cameras(id)
);

-- measurements
CREATE TABLE IF NOT EXISTS measurements (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT NOT NULL,
    recorded_at             TEXT NOT NULL,
    current_count           INTEGER NOT NULL,
    unique_count            INTEGER NOT NULL,
    entries                 INTEGER NOT NULL,
    exits                   INTEGER NOT NULL,
    net_count               INTEGER NOT NULL,
    occupancy_percent       REAL,
    crowd_level             TEXT,
    crowd_trend             TEXT,
    peak_count              INTEGER,
    peak_occupancy_percent  REAL,
    peak_timestamp          TEXT,
    processing_latency_ms   REAL,
    system_state            TEXT,
    camera_state            TEXT,
    queue_people            INTEGER,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

-- events
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    value           INTEGER,
    zone            TEXT,
    metadata_json   TEXT,
    dedup_key       TEXT UNIQUE,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

-- alerts
CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    type            TEXT NOT NULL,
    severity        TEXT NOT NULL,
    fired_at        TEXT NOT NULL,
    cleared_at      TEXT,
    status          TEXT NOT NULL,
    reason          TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_camera ON sessions(camera_id);
CREATE INDEX IF NOT EXISTS idx_measurements_session_time ON measurements(session_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_alerts_session_time ON alerts(session_id, fired_at);
