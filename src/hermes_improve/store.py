from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .errors import IntegrityError
from .models import Candidate, EvaluationReport, candidate_digest, canonical_json, utc_now


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
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    signature TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_results (
                    evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(id),
                    task_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (evaluation_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS run_observations (
                    evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(id),
                    task_id TEXT NOT NULL,
                    commit_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    observation_json TEXT NOT NULL,
                    PRIMARY KEY (evaluation_id, task_id, commit_id, seed)
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
                CREATE TABLE IF NOT EXISTS benchmark_candidates (
                    task_fingerprint TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    cluster_id TEXT NOT NULL,
                    eligibility TEXT NOT NULL,
                    reproducibility_score REAL NOT NULL,
                    ground_truth_type TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    source_sessions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(evaluation_runs)")}
            if "metrics_json" not in columns:
                db.execute("ALTER TABLE evaluation_runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")
            if "signature" not in columns:
                db.execute("ALTER TABLE evaluation_runs ADD COLUMN signature TEXT")

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

    def create_experiment_with_candidate(
        self,
        candidate: Candidate,
        *,
        lineage_id: str,
        hypothesis: str,
        risk: str,
        benchmark_version: str,
        policy_version: str,
        source_sessions: Iterable[str] = (),
    ) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            lineage_id, hypothesis, risk, benchmark_version, policy_version,
        )):
            raise ValueError("experiment metadata fields must be non-empty strings")
        if not candidate.diff or not candidate.commit or not candidate.base_commit:
            raise ValueError("candidate requires commit IDs and a non-empty diff")
        import hashlib
        diff_hash = hashlib.sha256(candidate.diff.encode()).hexdigest()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO experiments
                (id,parent_id,lineage_id,hypothesis,source_sessions,candidate_type,target,risk,
                 baseline_commit,benchmark_version,policy_version,created_at)
                VALUES (?,NULL,?,?,?,?,?,?,?,?,?,?)""",
                (candidate.experiment_id, lineage_id, hypothesis, canonical_json(list(source_sessions)),
                 candidate.candidate_type.value, candidate.target, risk, candidate.base_commit,
                 benchmark_version, policy_version, utc_now()),
            )
            db.execute(
                """INSERT INTO candidates
                (id,experiment_id,candidate_type,target,base_commit,commit_id,diff_hash,content_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (candidate.candidate_id, candidate.experiment_id, candidate.candidate_type.value,
                 candidate.target, candidate.base_commit, candidate.commit, diff_hash,
                 candidate.content_hash, candidate.created_at),
            )

    def record_evaluation(
        self, evaluation_id: str, candidate_id: str, report: EvaluationReport,
        signature: str | None = None,
    ) -> None:
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
                (id,candidate_id,candidate_hash,benchmark_version,policy_version,passed,reasons,metrics_json,signature,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (evaluation_id,candidate_id,report.candidate_hash,report.benchmark_version,
                 report.policy_version,int(report.passed),canonical_json(report.reasons),
                 canonical_json(report.metrics),signature,report.created_at),
            )
            db.executemany(
                "INSERT INTO task_results (evaluation_id,task_id,split,result_json) VALUES (?,?,?,?)",
                [(evaluation_id,row.task_id,row.split.value,canonical_json(row.__dict__)) for row in report.outcomes],
            )
            db.executemany(
                """INSERT INTO run_observations
                   (evaluation_id,task_id,commit_id,seed,observation_json) VALUES (?,?,?,?,?)""",
                [
                    (evaluation_id, row.task_id, row.commit, row.seed, canonical_json(row.__dict__))
                    for row in report.run_observations
                ],
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

    def save_benchmark_candidate(self, candidate: Any, session_id: str) -> bool:
        with self._lock, self._connect() as db:
            existing = db.execute(
                "SELECT source_sessions_json FROM benchmark_candidates WHERE task_fingerprint=?",
                (candidate.task_fingerprint,),
            ).fetchone()
            if existing:
                sources = json.loads(existing["source_sessions_json"])
                if session_id not in sources:
                    sources.append(session_id)
                db.execute(
                    "UPDATE benchmark_candidates SET source_sessions_json=? WHERE task_fingerprint=?",
                    (canonical_json(sources), candidate.task_fingerprint),
                )
                return False
            db.execute(
                """INSERT INTO benchmark_candidates
                (task_fingerprint,candidate_id,cluster_id,eligibility,reproducibility_score,
                 ground_truth_type,reasons_json,source_sessions_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (candidate.task_fingerprint,candidate.candidate_id,candidate.cluster_id,
                 candidate.eligibility.value,candidate.reproducibility_score,candidate.ground_truth_type,
                 canonical_json(candidate.reasons),canonical_json([session_id]),utc_now()),
            )
            return True

    def list_benchmark_candidates(self, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM benchmark_candidates ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            result = []
            for row in rows:
                item = dict(row)
                item["reasons"] = json.loads(item.pop("reasons_json"))
                item["source_sessions"] = json.loads(item.pop("source_sessions_json"))
                result.append(item)
            return result

    def promotion_evidence(self, candidate_id: str) -> dict[str, Any]:
        with self._connect() as db:
            candidate = db.execute(
                """SELECT c.*, e.baseline_commit, e.benchmark_version AS experiment_benchmark_version,
                          e.policy_version AS experiment_policy_version, e.decision AS experiment_decision,
                          e.id AS experiment_id
                   FROM candidates c JOIN experiments e ON e.id=c.experiment_id
                   WHERE c.id=?""",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise KeyError(candidate_id)
            expected_hash = candidate_digest(
                candidate["id"], candidate["experiment_id"], candidate["candidate_type"],
                candidate["target"], candidate["base_commit"], candidate["commit_id"],
                candidate["diff_hash"],
            )
            if expected_hash != candidate["content_hash"]:
                raise IntegrityError("candidate metadata or diff hash differs from its immutable content hash")
            if candidate["baseline_commit"] != candidate["base_commit"]:
                raise IntegrityError("experiment baseline differs from the candidate's signed base commit")
            evaluation = db.execute(
                """SELECT * FROM evaluation_runs WHERE candidate_id=?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (candidate_id,),
            ).fetchone()
            if evaluation is None:
                raise ValueError("candidate has no evaluation record")
            results = [
                {
                    **dict(row),
                    "result": json.loads(row["result_json"]),
                }
                for row in db.execute(
                    "SELECT * FROM task_results WHERE evaluation_id=? ORDER BY task_id",
                    (evaluation["id"],),
                )
            ]
            observations = [
                json.loads(row["observation_json"])
                for row in db.execute(
                    """SELECT observation_json FROM run_observations
                       WHERE evaluation_id=? ORDER BY task_id,commit_id,seed""",
                    (evaluation["id"],),
                )
            ]
            return {
                "candidate": dict(candidate),
                "evaluation": dict(evaluation),
                "task_results": results,
                "run_observations": observations,
            }

    def record_release(
        self, release_id: str, experiment_id: str, candidate_hash: str,
        status: str, manifest: dict[str, Any],
    ) -> None:
        if status not in {"PROVISIONAL", "STABLE", "ROLLED_BACK"}:
            raise ValueError("invalid release status")
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO releases (id,experiment_id,candidate_hash,status,manifest_json,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (release_id, experiment_id, candidate_hash, status, canonical_json(manifest), utc_now()),
            )
            db.execute(
                "UPDATE experiments SET decision='promoted',finished_at=? WHERE id=?",
                (utc_now(), experiment_id),
            )

    def update_release_status(self, release_id: str, status: str) -> None:
        if status not in {"PROVISIONAL", "STABLE", "ROLLED_BACK"}:
            raise ValueError("invalid release status")
        with self._lock, self._connect() as db:
            cursor = db.execute("UPDATE releases SET status=? WHERE id=?", (status, release_id))
            if cursor.rowcount != 1:
                raise KeyError(release_id)

    def transition_rollback(self, active_commit: str, target_release_id: str) -> None:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT id,manifest_json FROM releases").fetchall()
            active_ids = []
            for row in rows:
                manifest = json.loads(row["manifest_json"])
                if manifest.get("candidate_commit") == active_commit:
                    active_ids.append(row["id"])
            if len(active_ids) > 1:
                raise ValueError("multiple release records claim the active Git commit")
            db.execute("BEGIN IMMEDIATE")
            if active_ids:
                db.execute("UPDATE releases SET status='ROLLED_BACK' WHERE id=?", (active_ids[0],))
            cursor = db.execute(
                "UPDATE releases SET status='PROVISIONAL' WHERE id=?", (target_release_id,)
            )
            if cursor.rowcount != 1:
                raise KeyError(target_release_id)

    def list_releases(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,experiment_id,candidate_hash,status,created_at FROM releases ORDER BY created_at DESC"
            )]

    def list_candidates(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            return [dict(row) for row in db.execute(
                """SELECT c.id,c.experiment_id,c.candidate_type,c.target,c.base_commit,
                          c.commit_id,c.content_hash,c.created_at,e.decision
                   FROM candidates c JOIN experiments e ON e.id=c.experiment_id
                   ORDER BY c.created_at DESC LIMIT ?""",
                (limit,),
            )]

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            return dict(row)

    def get_release(self, release_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM releases WHERE id=?", (release_id,)).fetchone()
            if row is None:
                raise KeyError(release_id)
            result = dict(row)
            result["manifest"] = json.loads(result.pop("manifest_json"))
            return result
