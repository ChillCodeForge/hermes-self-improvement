from __future__ import annotations

import sqlite3

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


def test_benchmark_candidates_deduplicate_but_retain_all_sources(tmp_path):
    from hermes_improve.miner import mine_session

    store = ExperimentStore(tmp_path / "state.sqlite3")
    one = mine_session({"session_id": "s1", "task": {"prompt": "Fix the bug"}})
    two = mine_session({"session_id": "s2", "task": {"prompt": "fix   the BUG"}})
    assert store.save_benchmark_candidate(one, "s1") is True
    assert store.save_benchmark_candidate(two, "s2") is False
    rows = store.list_benchmark_candidates()
    assert len(rows) == 1
    assert rows[0]["source_sessions"] == ["s1", "s2"]


def test_benchmark_candidate_review_is_persistent_immutable_and_secret_checked(tmp_path):
    from hermes_improve.miner import mine_session

    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = mine_session({"session_id": "s1", "task": {"prompt": "Review this task"}})
    store.save_benchmark_candidate(candidate, "s1")

    pending = store.get_benchmark_candidate(candidate.task_fingerprint)
    assert pending["review_status"] == "PENDING"
    assert pending["reviewer"] is None

    reviewed = store.review_benchmark_candidate(
        candidate.task_fingerprint,
        decision="ACCEPTED",
        reviewer="operator@example.invalid",
        rationale="Task is reproducible and has a reviewable source.",
    )
    assert reviewed["review_status"] == "ACCEPTED"
    assert reviewed["reviewed_at"]

    repeated = store.review_benchmark_candidate(
        candidate.task_fingerprint,
        decision="ACCEPTED",
        reviewer="operator@example.invalid",
        rationale="Task is reproducible and has a reviewable source.",
    )
    assert repeated["reviewed_at"] == reviewed["reviewed_at"]
    with pytest.raises(ValueError, match="immutable"):
        store.review_benchmark_candidate(
            candidate.task_fingerprint,
            decision="REJECTED",
            reviewer="operator@example.invalid",
            rationale="Changed opinion",
        )
    with pytest.raises(ValueError, match="credential-like"):
        store.review_benchmark_candidate(
            "0" * 64,
            decision="REJECTED",
            reviewer="operator",
            rationale="api_key=not-a-real-secret",
        )


def test_existing_benchmark_candidates_migrate_to_pending_review(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    fingerprint = "a" * 64
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TABLE benchmark_candidates (
                task_fingerprint TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                eligibility TEXT NOT NULL,
                reproducibility_score REAL NOT NULL,
                ground_truth_type TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                source_sessions_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        db.execute(
            """INSERT INTO benchmark_candidates
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fingerprint, "candidate-1", fingerprint, "SOFT_BENCHMARK", 0.5,
             "structured", "[]", '["session-1"]', "2025-01-01T00:00:00Z"),
        )

    candidate = ExperimentStore(path).get_benchmark_candidate(fingerprint)
    assert candidate["review_status"] == "PENDING"
    assert candidate["reviewer"] is None
    assert candidate["reviewed_at"] is None
