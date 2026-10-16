from __future__ import annotations

import pytest

from hermes_improve.artifacts import ArtifactStore
from hermes_improve.errors import IntegrityError
from hermes_improve.models import Candidate, CandidateType, EvaluationReport, Split, TaskOutcome
from hermes_improve.store import ExperimentStore


def test_artifact_store_is_content_addressed_and_verified(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    digest, path = store.put(b"trajectory")
    assert store.get(digest) == b"trajectory"
    path.write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        store.get(digest)


def test_experiment_candidate_evaluation_history_is_persisted(tmp_path):
    store = ExperimentStore(tmp_path / "state.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="lineage-1", hypothesis="improve skill",
        candidate_type="SKILL_PATCH", target="brain/skills/demo", risk="LOW",
        baseline_commit="abc", benchmark_version="bench-v1", policy_version="1",
    )
    candidate = Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "brain/skills/demo", "abc", "def", "diff")
    store.add_candidate(candidate)
    outcome = TaskOutcome("t1", Split.REGRESSION, "FULL", True, True, 1, 5)
    report = EvaluationReport("cand-1", candidate.content_hash, "bench-v1", "1", (outcome,), True, ())
    store.record_evaluation("eval-1", "cand-1", report)
    assert store.show_experiment("exp-1")["candidates"][0]["content_hash"] == candidate.content_hash


def test_evaluation_with_candidate_hash_mismatch_is_rejected(tmp_path):
    store = ExperimentStore(tmp_path / "state.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="lineage-1", hypothesis="improve skill",
        candidate_type="SKILL_PATCH", target="skill", risk="LOW",
        baseline_commit="abc", benchmark_version="bench-v1", policy_version="1",
    )
    candidate = Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "skill", "abc", "def", "diff")
    store.add_candidate(candidate)
    report = EvaluationReport("cand-1", "wrong-hash", "bench-v1", "1", (), False, ("bad",))
    with pytest.raises(ValueError, match="hash"):
        store.record_evaluation("eval-1", "cand-1", report)
