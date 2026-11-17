-- Evolution runs and the generated-variant registry.
CREATE TABLE IF NOT EXISTS evolution_runs (
    id TEXT PRIMARY KEY,
    root_candidate_id TEXT NOT NULL REFERENCES candidates(id),
    lineage_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    stop_reason TEXT,
    report_json TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS generated_variants (
    target TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    origin TEXT NOT NULL,
    generation INTEGER NOT NULL,
    parent_candidate_id TEXT,
    candidate_id TEXT REFERENCES candidates(id),
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    scores_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (target, content_sha256)
);
