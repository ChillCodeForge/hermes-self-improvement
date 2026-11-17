-- Spec section 26: failed experiments are never deleted. The experiment
-- ledger and its evaluation evidence refuse every DELETE.
CREATE TRIGGER IF NOT EXISTS experiments_never_delete
BEFORE DELETE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS candidates_never_delete
BEFORE DELETE ON candidates
BEGIN SELECT RAISE(ABORT, 'candidates are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS evaluation_runs_never_delete
BEFORE DELETE ON evaluation_runs
BEGIN SELECT RAISE(ABORT, 'evaluation runs are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS task_results_never_delete
BEFORE DELETE ON task_results
BEGIN SELECT RAISE(ABORT, 'task results are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS run_observations_never_delete
BEFORE DELETE ON run_observations
BEGIN SELECT RAISE(ABORT, 'run observations are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS regressions_never_delete
BEFORE DELETE ON regressions
BEGIN SELECT RAISE(ABORT, 'regressions are never deleted'); END;
