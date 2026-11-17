-- Spec section 26: the experiment row carries the evaluated candidate, the
-- execution identity of its deciding evaluation and the gate outcome.
--   candidate_commit, candidate_hash: set when the candidate is created and
--     re-stated by the evaluation that decides the experiment.
--   model_config_digest, environment_digest, seeds, aggregate_metrics: set by
--     the evaluation that decides the experiment.
--   gate_result: JSON object; "policy" is written by the deciding evaluation,
--     "promotion" by the recorded promotion.
-- Rows created before this migration keep NULL: the data was never recorded.
ALTER TABLE experiments ADD COLUMN candidate_commit TEXT;
ALTER TABLE experiments ADD COLUMN candidate_hash TEXT;
ALTER TABLE experiments ADD COLUMN model_config_digest TEXT;
ALTER TABLE experiments ADD COLUMN environment_digest TEXT;
ALTER TABLE experiments ADD COLUMN seeds TEXT;
ALTER TABLE experiments ADD COLUMN aggregate_metrics TEXT;
ALTER TABLE experiments ADD COLUMN gate_result TEXT;
