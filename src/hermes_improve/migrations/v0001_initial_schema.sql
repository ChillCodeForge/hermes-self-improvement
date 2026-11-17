-- Initial experiment ledger layout (first release of the store), plus the
-- append-only migration audit table. Every statement is IF NOT EXISTS so that
-- databases created before schema versioning (user_version 0) adopt it.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS schema_migrations_append_only_update
BEFORE UPDATE ON schema_migrations
BEGIN SELECT RAISE(ABORT, 'schema migrations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS schema_migrations_append_only_delete
BEFORE DELETE ON schema_migrations
BEGIN SELECT RAISE(ABORT, 'schema migrations are append-only'); END;
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    lineage_id TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    source_sessions TEXT NOT NULL DEFAULT '[]',
    candidate_type TEXT NOT NULL,
    target TEXT NOT NULL,
    risk TEXT NOT NULL,
    baseline_commit TEXT NOT NULL,
    benchmark_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    decision TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_type TEXT NOT NULL,
    target TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    commit_id TEXT NOT NULL,
    diff_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    candidate_hash TEXT NOT NULL,
    benchmark_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    passed INTEGER NOT NULL,
    reasons TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_results (
    evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(id),
    task_id TEXT NOT NULL,
    split TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (evaluation_id, task_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    digest TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    decision_type TEXT NOT NULL,
    input_json TEXT NOT NULL,
    options_json TEXT NOT NULL,
    selected_option TEXT NOT NULL,
    confidence REAL,
    source TEXT NOT NULL,
    ground_truth TEXT,
    ground_truth_source TEXT,
    correct INTEGER,
    model_version TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS releases (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
