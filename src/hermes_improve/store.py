from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .models import Candidate, EvaluationReport, canonical_json, utc_now


class ExperimentStore:
    """SQLite metadata store; large artifacts are stored separately by digest."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
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
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create_experiment(
        self, experiment_id: str, *, lineage_id: str, hypothesis: str,
        candidate_type: str, target: str, risk: str, baseline_commit: str,
        benchmark_version: str, policy_version: str, parent_id: str | None = None,
        source_sessions: Iterable[str] = (),
    ) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO experiments
                (id,parent_id,lineage_id,hypothesis,source_sessions,candidate_type,target,risk,
                 baseline_commit,benchmark_version,policy_version,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (experiment_id,parent_id,lineage_id,hypothesis,canonical_json(list(source_sessions)),
                 candidate_type,target,risk,baseline_commit,benchmark_version,policy_version,utc_now()),
            )

    def add_candidate(self, candidate: Candidate) -> None:
        import hashlib
        diff_hash = hashlib.sha256(candidate.diff.encode()).hexdigest()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO candidates
                (id,experiment_id,candidate_type,target,base_commit,commit_id,diff_hash,content_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (candidate.candidate_id,candidate.experiment_id,candidate.candidate_type.value,
                 candidate.target,candidate.base_commit,candidate.commit,diff_hash,
                 candidate.content_hash,candidate.created_at),
            )

    def record_evaluation(self, evaluation_id: str, candidate_id: str, report: EvaluationReport) -> None:
        with self._lock, self._connect() as db:
            candidate = db.execute("SELECT content_hash FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if candidate is None:
                raise KeyError(f"unknown candidate {candidate_id!r}")
            if report.candidate_id != candidate_id:
                raise ValueError("evaluation candidate id differs from the stored candidate")
            if candidate["content_hash"] != report.candidate_hash:
                raise ValueError("evaluation candidate hash differs from the immutable stored candidate")
            db.execute(
                """INSERT INTO evaluation_runs
                (id,candidate_id,candidate_hash,benchmark_version,policy_version,passed,reasons,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (evaluation_id,candidate_id,report.candidate_hash,report.benchmark_version,
                 report.policy_version,int(report.passed),canonical_json(report.reasons),report.created_at),
            )
            db.executemany(
                "INSERT INTO task_results (evaluation_id,task_id,split,result_json) VALUES (?,?,?,?)",
                [(evaluation_id,row.task_id,row.split.value,canonical_json(row.__dict__)) for row in report.outcomes],
            )
            db.execute(
                "UPDATE experiments SET decision=?,reason=?,finished_at=? WHERE id=(SELECT experiment_id FROM candidates WHERE id=?)",
                ("passed" if report.passed else "rejected", canonical_json(report.reasons), report.created_at, candidate_id),
            )

    def list_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,parent_id,lineage_id,hypothesis,candidate_type,decision,created_at FROM experiments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )]

    def show_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row is None:
                raise KeyError(experiment_id)
            result = dict(row)
            result["source_sessions"] = json.loads(result["source_sessions"])
            result["candidates"] = [dict(item) for item in db.execute(
                "SELECT * FROM candidates WHERE experiment_id=?", (experiment_id,)
            )]
            return result

    def record_artifact(self, digest: str, path: str, media_type: str = "application/octet-stream") -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO artifacts (digest,path,media_type,created_at) VALUES (?,?,?,?)",
                (digest,path,media_type,utc_now()),
            )
