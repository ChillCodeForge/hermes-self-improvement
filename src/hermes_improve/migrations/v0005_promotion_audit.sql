-- Signed operator approvals and the release audit log, both append-only.
CREATE TABLE IF NOT EXISTS promotion_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    candidate_hash TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    operator TEXT NOT NULL,
    key_fingerprint TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    signature TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS promotion_approvals_append_only_update
BEFORE UPDATE ON promotion_approvals
BEGIN SELECT RAISE(ABORT, 'promotion approvals are append-only'); END;
CREATE TRIGGER IF NOT EXISTS promotion_approvals_append_only_delete
BEFORE DELETE ON promotion_approvals
BEGIN SELECT RAISE(ABORT, 'promotion approvals are append-only'); END;
CREATE TABLE IF NOT EXISTS release_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id TEXT NOT NULL,
    event TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS release_events_append_only_update
BEFORE UPDATE ON release_events
BEGIN SELECT RAISE(ABORT, 'release events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS release_events_append_only_delete
BEFORE DELETE ON release_events
BEGIN SELECT RAISE(ABORT, 'release events are append-only'); END;
