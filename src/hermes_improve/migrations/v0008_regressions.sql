-- Spec section 26 Regression entity: one row per regression an evaluation
-- detected on one benchmark task. Rows are append-only.
CREATE TABLE IF NOT EXISTS regressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_run_id TEXT NOT NULL REFERENCES evaluation_runs(id),
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    benchmark_id TEXT NOT NULL,
    suite TEXT NOT NULL,
    split TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('reported', 'task_failure')),
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium')),
    count INTEGER NOT NULL CHECK (count > 0),
    baseline_outcome_json TEXT NOT NULL,
    candidate_outcome_json TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    UNIQUE (evaluation_run_id, benchmark_id, kind, severity)
);
CREATE INDEX IF NOT EXISTS regressions_experiment ON regressions(experiment_id);
CREATE INDEX IF NOT EXISTS regressions_candidate ON regressions(candidate_id);
CREATE TRIGGER IF NOT EXISTS regressions_append_only_update
BEFORE UPDATE ON regressions
BEGIN SELECT RAISE(ABORT, 'regressions are append-only'); END;
