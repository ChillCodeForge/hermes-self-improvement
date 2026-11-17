"""Versioned ledger migrations, Regression entity and delete guards (spec §26, §37)."""

from __future__ import annotations

import sqlite3
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve import migrations
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.migrations import (
    MIGRATIONS, SCHEMA_VERSION, Migration, SchemaVersionError, apply_migrations,
)
from hermes_improve.models import Role
from hermes_improve.policy import PromotionPolicy
from hermes_improve.store import ExperimentStore
from test_evaluator import FixtureRunner, candidate, setup_store

REPO = Path(__file__).resolve().parents[2]

# The layout written by the first release of the store (no metrics_json or
# signature, no run observations, no benchmark proposals, no audit tables).
_FIRST_RELEASE_LAYOUT = """
CREATE TABLE experiments (
    id TEXT PRIMARY KEY, parent_id TEXT, lineage_id TEXT NOT NULL, hypothesis TEXT NOT NULL,
    source_sessions TEXT NOT NULL DEFAULT '[]', candidate_type TEXT NOT NULL, target TEXT NOT NULL,
    risk TEXT NOT NULL, baseline_commit TEXT NOT NULL, benchmark_version TEXT NOT NULL,
    policy_version TEXT NOT NULL, decision TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE candidates (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_type TEXT NOT NULL, target TEXT NOT NULL, base_commit TEXT NOT NULL,
    commit_id TEXT NOT NULL, diff_hash TEXT NOT NULL, content_hash TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE evaluation_runs (
    id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES candidates(id),
    candidate_hash TEXT NOT NULL, benchmark_version TEXT NOT NULL, policy_version TEXT NOT NULL,
    passed INTEGER NOT NULL, reasons TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE task_results (
    evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(id), task_id TEXT NOT NULL,
    split TEXT NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY (evaluation_id, task_id)
);
CREATE TABLE artifacts (digest TEXT PRIMARY KEY, path TEXT NOT NULL, media_type TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL REFERENCES experiments(id),
    decision_type TEXT NOT NULL, input_json TEXT NOT NULL, options_json TEXT NOT NULL,
    selected_option TEXT NOT NULL, confidence REAL, source TEXT NOT NULL, ground_truth TEXT,
    ground_truth_source TEXT, correct INTEGER, model_version TEXT, created_at TEXT NOT NULL
);
CREATE TABLE releases (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_hash TEXT NOT NULL, status TEXT NOT NULL, manifest_json TEXT NOT NULL, created_at TEXT NOT NULL
);
"""

# What the last unversioned store had added on top through CREATE IF NOT
# EXISTS and inline ALTER TABLE statements.
_LAST_UNVERSIONED_ADDITIONS = """
ALTER TABLE evaluation_runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE evaluation_runs ADD COLUMN signature TEXT;
CREATE TABLE run_observations (
    evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(id), task_id TEXT NOT NULL,
    commit_id TEXT NOT NULL, seed INTEGER NOT NULL, observation_json TEXT NOT NULL,
    PRIMARY KEY (evaluation_id, task_id, commit_id, seed)
);
CREATE TABLE benchmark_candidates (
    task_fingerprint TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, cluster_id TEXT NOT NULL,
    eligibility TEXT NOT NULL, reproducibility_score REAL NOT NULL, ground_truth_type TEXT NOT NULL,
    reasons_json TEXT NOT NULL, source_sessions_json TEXT NOT NULL, created_at TEXT NOT NULL
);
ALTER TABLE benchmark_candidates ADD COLUMN review_status TEXT NOT NULL DEFAULT 'PENDING';
ALTER TABLE benchmark_candidates ADD COLUMN reviewer TEXT;
ALTER TABLE benchmark_candidates ADD COLUMN review_rationale TEXT;
ALTER TABLE benchmark_candidates ADD COLUMN reviewed_at TEXT;
CREATE TABLE promotion_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL REFERENCES experiments(id),
    candidate_id TEXT NOT NULL REFERENCES candidates(id), candidate_hash TEXT NOT NULL,
    policy_version TEXT NOT NULL, operator TEXT NOT NULL, key_fingerprint TEXT NOT NULL,
    payload_json TEXT NOT NULL, signature TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TRIGGER promotion_approvals_append_only_delete BEFORE DELETE ON promotion_approvals
BEGIN SELECT RAISE(ABORT, 'promotion approvals are append-only'); END;
"""

_LEDGER_TABLES = ("experiments", "candidates", "evaluation_runs", "task_results", "run_observations", "regressions")
_EXPERIMENT_FIELDS = (
    "candidate_commit", "candidate_hash", "model_config_digest", "environment_digest",
    "seeds", "aggregate_metrics", "gate_result",
)


def _columns(path, table):
    with sqlite3.connect(path) as db:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _objects(path, kind):
    with sqlite3.connect(path) as db:
        return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type=?", (kind,))}


def _version(path):
    with sqlite3.connect(path) as db:
        return db.execute("PRAGMA user_version").fetchone()[0]


def _applied(path):
    with sqlite3.connect(path) as db:
        return [tuple(row) for row in db.execute("SELECT version, name, sha256 FROM schema_migrations ORDER BY version")]


def _assert_current_schema(path):
    assert _version(path) == SCHEMA_VERSION
    assert _applied(path) == [(item.version, item.name, item.sha256) for item in MIGRATIONS]
    assert set(_EXPERIMENT_FIELDS) <= _columns(path, "experiments")
    assert {"metrics_json", "signature"} <= _columns(path, "evaluation_runs")
    assert {"review_status", "reviewer", "review_rationale", "reviewed_at"} <= _columns(path, "benchmark_candidates")
    assert {
        "id", "evaluation_run_id", "experiment_id", "candidate_id", "benchmark_id", "severity",
        "baseline_outcome_json", "candidate_outcome_json", "detected_at",
    } <= _columns(path, "regressions")
    assert {
        "evolution_runs", "generated_variants", "benchmark_replays", "promotion_approvals", "release_events",
    } <= _objects(path, "table")
    assert {f"{table}_never_delete" for table in _LEDGER_TABLES} <= _objects(path, "trigger")


def test_migrations_are_numbered_contiguously_and_shipped_as_package_data():
    assert [item.version for item in MIGRATIONS] == list(range(1, SCHEMA_VERSION + 1))
    package = Path(migrations.__file__).parent
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    patterns = pyproject["tool"]["setuptools"]["package-data"]["hermes_improve.migrations"]
    sql_files = sorted(path.name for path in package.glob("v*.sql"))
    assert sql_files and all(any(Path(name).match(pattern) for pattern in patterns) for name in sql_files)
    assert {item.filename for item in MIGRATIONS} == {
        path.name for path in package.iterdir() if path.name.startswith("v") and path.suffix in {".sql", ".py"}
    }


def test_fresh_database_gets_every_migration_once(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ExperimentStore(path)
    _assert_current_schema(path)
    # Re-opening is idempotent: no migration runs or is recorded twice.
    ExperimentStore(path)
    assert apply_migrations(path) == SCHEMA_VERSION
    _assert_current_schema(path)


@pytest.mark.parametrize("layout", ["first_release", "last_unversioned"])
def test_unversioned_database_upgrades_and_keeps_its_rows(tmp_path, layout):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(_FIRST_RELEASE_LAYOUT)
        if layout == "last_unversioned":
            db.executescript(_LAST_UNVERSIONED_ADDITIONS)
        db.execute(
            """INSERT INTO experiments (id,lineage_id,hypothesis,candidate_type,target,risk,baseline_commit,
               benchmark_version,policy_version,decision,created_at)
               VALUES ('exp-old','exp-old','legacy','SKILL_PATCH','brain/skills/a','LOW','base','b1','p1',
                       'rejected','2025-01-01T00:00:00Z')"""
        )
        db.execute(
            """INSERT INTO candidates VALUES ('cand-old','exp-old','SKILL_PATCH','brain/skills/a','base',
               'cand-commit','diff-hash','content-hash','2025-01-01T00:00:00Z')"""
        )
        db.execute(
            """INSERT INTO evaluation_runs (id,candidate_id,candidate_hash,benchmark_version,policy_version,
               passed,reasons,created_at)
               VALUES ('eval-old','cand-old','content-hash','b1','p1',0,'[]','2025-01-01T00:00:00Z')"""
        )
    assert _version(path) == 0

    store = ExperimentStore(path)

    _assert_current_schema(path)
    old = store.show_experiment("exp-old")
    assert old["decision"] == "rejected" and old["candidates"][0]["id"] == "cand-old"
    # Never recorded for legacy rows, so never invented.
    assert all(old[field] is None for field in _EXPERIMENT_FIELDS)
    evaluation = store.candidate_evaluations("cand-old")[0]["evaluation"]
    assert evaluation["metrics_json"] == "{}" and evaluation["signature"] is None
    with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match="never deleted"):
        db.execute("DELETE FROM experiments WHERE id='exp-old'")


def test_failing_migration_rolls_back_its_whole_step(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    assert apply_migrations(path, MIGRATIONS[:2]) == 2

    def broken(db):
        db.execute("CREATE TABLE half_applied (id INTEGER)")
        raise RuntimeError("migration failed midway")

    with pytest.raises(RuntimeError, match="midway"):
        apply_migrations(path, (*MIGRATIONS[:2], Migration(3, "broken", "v0003_broken.py", "0" * 64, broken)))
    assert _version(path) == 2
    assert "half_applied" not in _objects(path, "table")
    assert [row[0] for row in _applied(path)] == [1, 2]
    # The real migrations still continue from version 2.
    assert apply_migrations(path) == SCHEMA_VERSION
    _assert_current_schema(path)


def test_database_newer_than_the_code_is_refused(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ExperimentStore(path)
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(SchemaVersionError, match="newer than this code supports"):
        ExperimentStore(path)


def _signing_key():
    return Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )


class RegressingRunner(FixtureRunner):
    """Candidate runs fail the task the baseline solves and report verifier regressions."""

    def run(self, commit, task, seed):
        observation = replace(super().run(commit, task, seed), model_config_digest="model-digest")
        if commit == "base":
            return observation
        return replace(observation, success=False, quality=1, critical_regressions=1, high_regressions=2)


def _evaluated_store(tmp_path, runner):
    benchmarks, access = setup_store(tmp_path)
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="exp-1", hypothesis="test regressions", candidate_type="SKILL_PATCH",
        target="skill/a", risk="LOW", baseline_commit="base", benchmark_version=benchmarks.version,
        policy_version="1",
    )
    subject = candidate()
    store.add_candidate(subject)
    evaluator = PairedEvaluator(
        benchmarks, runner, PromotionPolicy(), store, evaluator_signing_key=_signing_key(),
    )
    report = evaluator.evaluate(
        subject, baseline_commit="base", benchmark_ids=["full-task", "holdout-task"], seeds=[1, 2],
        suite="FULL", benchmark_version=benchmarks.version,
        holdout_grant=access.evaluator_holdout_grant(Role.EVALUATOR), evaluation_id="eval-1",
    )
    return store, subject, report


def test_evaluate_records_one_regression_row_per_detected_regression(tmp_path):
    store, subject, report = _evaluated_store(tmp_path, RegressingRunner())
    assert not report.passed
    rows = store.list_regressions(experiment_id="exp-1")
    found = {(row["benchmark_id"], row["kind"], row["severity"], row["count"]) for row in rows}
    assert found == {
        (task, kind, severity, count)
        for task in ("full-task", "holdout-task")
        for kind, severity, count in (("reported", "critical", 1), ("reported", "high", 2), ("task_failure", "high", 1))
    }
    for row in rows:
        assert row["evaluation_run_id"] == "eval-1" and row["candidate_id"] == subject.candidate_id
        assert row["detected_at"] == report.created_at
        assert row["baseline_outcome"]["success"] is True and row["baseline_outcome"]["quality"] == 5
        assert row["candidate_outcome"]["success"] is False and row["candidate_outcome"]["run_count"] == 2
    holdout = [row for row in rows if row["benchmark_id"] == "holdout-task"]
    assert {row["split"] for row in holdout} == {"SEALED_HOLDOUT"}
    assert {row["suite"] for row in holdout} == {"SEALED_HOLDOUT"}
    assert len(store.list_regressions(severity="critical")) == 2
    assert store.list_regressions(candidate_id="other") == []
    with pytest.raises(ValueError, match="unknown regression severity"):
        store.list_regressions(severity="low")

    experiment = store.show_experiment("exp-1")
    assert experiment["decision"] == "rejected"
    assert experiment["gate_result"]["policy"]["regressions"] == 6
    assert experiment["model_config_digest"] == "model-digest"


def test_clean_evaluation_records_no_regressions_and_fills_experiment_evidence(tmp_path):
    store, subject, report = _evaluated_store(tmp_path, FixtureRunner())
    assert report.passed
    assert store.list_regressions(evaluation_run_id="eval-1") == []
    experiment = store.show_experiment("exp-1")
    assert experiment["decision"] == "passed"
    assert (experiment["candidate_commit"], experiment["candidate_hash"]) == (subject.commit, subject.content_hash)
    assert experiment["environment_digest"] == "fixture-digest"
    assert experiment["model_config_digest"] is None  # the fixture runner reports no model configuration
    assert experiment["seeds"] == {"full-task": [1, 2], "holdout-task": [1, 2]}
    assert experiment["aggregate_metrics"]["suite"] == "FULL"
    assert experiment["aggregate_metrics"]["final_tasks"] == report.metrics["final_tasks"]
    assert experiment["gate_result"] == {"policy": {
        "evaluation_id": "eval-1", "candidate_id": subject.candidate_id, "suite": "FULL",
        "policy_version": report.policy_version, "benchmark_version": report.benchmark_version,
        "passed": True, "reasons": [], "regressions": 0, "decided_at": report.created_at,
    }}


def test_ledger_rows_refuse_deletion(tmp_path):
    store, _, _ = _evaluated_store(tmp_path, RegressingRunner())
    with sqlite3.connect(store.path) as db:
        before = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _LEDGER_TABLES}
    assert all(before.values())
    for table in _LEDGER_TABLES:
        with sqlite3.connect(store.path) as db, pytest.raises(sqlite3.IntegrityError, match="never deleted"):
            db.execute(f"DELETE FROM {table}")
    with sqlite3.connect(store.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE regressions SET severity='medium'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM schema_migrations")
        after = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _LEDGER_TABLES}
    assert after == before


def test_record_evaluation_rejects_regressions_outside_the_report(tmp_path):
    store, subject, report = _evaluated_store(tmp_path, FixtureRunner())
    bogus = {
        "benchmark_id": "unknown-task", "suite": "FULL", "split": "REGRESSION", "kind": "reported",
        "severity": "high", "count": 1, "baseline_outcome": {}, "candidate_outcome": {},
    }
    with pytest.raises(ValueError, match="outside the evaluation"):
        store.record_evaluation("eval-2", subject.candidate_id, report, regressions=[bogus])
    assert len(store.candidate_evaluations(subject.candidate_id)) == 1
