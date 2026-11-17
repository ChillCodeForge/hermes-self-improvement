-- Immutable benchmark replay evidence.
CREATE TABLE IF NOT EXISTS benchmark_replays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id TEXT NOT NULL,
    benchmark_version INTEGER NOT NULL,
    benchmark_digest TEXT NOT NULL,
    commit_id TEXT NOT NULL,
    runs INTEGER NOT NULL,
    reproducible INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    outcome_digest TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
