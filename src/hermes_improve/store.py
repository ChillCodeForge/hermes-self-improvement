from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .errors import IntegrityError
from .ingestion import contains_secret_material
from .miner import BenchmarkCandidate
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
                    created_at TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'PENDING',
                    reviewer TEXT,
                    review_rationale TEXT,
                    reviewed_at TEXT
                );
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
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(evaluation_runs)")}
            if "metrics_json" not in columns:
                db.execute("ALTER TABLE evaluation_runs ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")
            if "signature" not in columns:
                db.execute("ALTER TABLE evaluation_runs ADD COLUMN signature TEXT")
            benchmark_candidate_columns = {
                row[1] for row in db.execute("PRAGMA table_info(benchmark_candidates)")
            }
            for column, definition in (
                ("review_status", "TEXT NOT NULL DEFAULT 'PENDING'"),
                ("reviewer", "TEXT"),
                ("review_rationale", "TEXT"),
                ("reviewed_at", "TEXT"),
            ):
                if column not in benchmark_candidate_columns:
                    db.execute(f"ALTER TABLE benchmark_candidates ADD COLUMN {column} {definition}")

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

    def save_benchmark_candidate(self, candidate: BenchmarkCandidate, session_id: str) -> bool:
        with self._lock, self._connect() as db:
            existing = db.execute(
                """SELECT source_sessions_json, review_status FROM benchmark_candidates
                   WHERE task_fingerprint=?""",
                (candidate.task_fingerprint,),
            ).fetchone()
            if existing:
                if existing["review_status"] != "PENDING":
                    return False
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

    @staticmethod
    def _benchmark_candidate_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["reasons"] = json.loads(item.pop("reasons_json"))
        item["source_sessions"] = json.loads(item.pop("source_sessions_json"))
        return item

    def list_benchmark_candidates(self, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM benchmark_candidates ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [self._benchmark_candidate_record(row) for row in rows]

    def get_benchmark_candidate(self, task_fingerprint: str) -> dict[str, Any]:
        if not isinstance(task_fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", task_fingerprint):
            raise ValueError("task fingerprint must be a lowercase SHA-256 digest")
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM benchmark_candidates WHERE task_fingerprint=?",
                (task_fingerprint,),
            ).fetchone()
        if row is None:
            raise KeyError(task_fingerprint)
        return self._benchmark_candidate_record(row)

    def review_benchmark_candidate(
        self, task_fingerprint: str, *, decision: str, reviewer: str, rationale: str,
    ) -> dict[str, Any]:
        if not isinstance(task_fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", task_fingerprint):
            raise ValueError("task fingerprint must be a lowercase SHA-256 digest")
        if not isinstance(decision, str) or decision not in {"ACCEPTED", "REJECTED"}:
            raise ValueError("benchmark proposal review decision must be ACCEPTED or REJECTED")
        if (
            not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 128
            or not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2048
            or "\x00" in reviewer or "\x00" in rationale
        ):
            raise ValueError("benchmark proposal reviews require a reviewer and rationale within the size limits")
        if contains_secret_material(reviewer) or contains_secret_material(rationale):
            raise ValueError("benchmark proposal review metadata must not contain credential-like material")

        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM benchmark_candidates WHERE task_fingerprint=?",
                (task_fingerprint,),
            ).fetchone()
            if row is None:
                raise KeyError(task_fingerprint)
            if row["review_status"] != "PENDING":
                if (
                    row["review_status"] == decision
                    and row["reviewer"] == reviewer.strip()
                    and row["review_rationale"] == rationale.strip()
                ):
                    return self._benchmark_candidate_record(row)
                raise ValueError("benchmark proposal already has an immutable review decision")
            reviewed_at = utc_now()
            db.execute(
                """UPDATE benchmark_candidates
                   SET review_status=?, reviewer=?, review_rationale=?, reviewed_at=?
                   WHERE task_fingerprint=? AND review_status='PENDING'""",
                (decision, reviewer.strip(), rationale.strip(), reviewed_at, task_fingerprint),
            )
            reviewed = db.execute(
                "SELECT * FROM benchmark_candidates WHERE task_fingerprint=?",
                (task_fingerprint,),
            ).fetchone()
            return self._benchmark_candidate_record(reviewed)

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

    def link_experiment_lineage(self, experiment_id: str, *, parent_id: str, lineage_id: str) -> None:
        """Attach a freshly created experiment to its parent's lineage exactly once."""
        if not all(isinstance(value, str) and value.strip() for value in (experiment_id, parent_id, lineage_id)):
            raise ValueError("experiment lineage identifiers must be non-empty strings")
        if parent_id == experiment_id:
            raise ValueError("an experiment cannot be its own parent")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parent = db.execute(
                "SELECT lineage_id FROM experiments WHERE id=?", (parent_id,)
            ).fetchone()
            if parent is None:
                raise KeyError(parent_id)
            if parent["lineage_id"] != lineage_id:
                raise ValueError("child experiment must join its parent's lineage")
            cursor = db.execute(
                """UPDATE experiments SET parent_id=?, lineage_id=?
                   WHERE id=? AND parent_id IS NULL AND lineage_id=id""",
                (parent_id, lineage_id, experiment_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("experiment lineage is already assigned or the experiment does not exist")

    def create_evolution_run(
        self, run_id: str, *, root_candidate_id: str, lineage_id: str,
        strategy: str, strategy_version: str, config: dict[str, Any],
    ) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            run_id, root_candidate_id, lineage_id, strategy, strategy_version,
        )):
            raise ValueError("evolution run metadata fields must be non-empty strings")
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO evolution_runs
                   (id,root_candidate_id,lineage_id,strategy,strategy_version,config_json,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (run_id, root_candidate_id, lineage_id, strategy, strategy_version,
                 canonical_json(config), utc_now()),
            )

    def finish_evolution_run(
        self, run_id: str, *, status: str, stop_reason: str, report: dict[str, Any],
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("evolution run status must be completed or failed")
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """UPDATE evolution_runs SET status=?, stop_reason=?, report_json=?, finished_at=?
                   WHERE id=? AND status='running'""",
                (status, stop_reason, canonical_json(report), utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("evolution run is unknown or already finished")

    def get_evolution_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM evolution_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        report = result.pop("report_json")
        result["report"] = json.loads(report) if report is not None else None
        return result

    def find_generated_variant(self, target: str, content_sha256: str) -> dict[str, Any] | None:
        _validate_variant_key(target, content_sha256)
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM generated_variants WHERE target=? AND content_sha256=?",
                (target, content_sha256),
            ).fetchone()
        return self._generated_variant_record(row) if row is not None else None

    def record_generated_variant(
        self, *, target: str, content_sha256: str, origin: str, generation: int,
        status: str, parent_candidate_id: str | None = None,
        candidate_id: str | None = None, reason: str = "",
    ) -> None:
        """Record generated content once per target; failed variants are never deleted."""
        _validate_variant_key(target, content_sha256)
        if status not in _VARIANT_STATUSES:
            raise ValueError(f"unknown generated variant status {status!r}")
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("generated variant origin must be non-empty")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generated variant generation must be a non-negative integer")
        now = utc_now()
        with self._lock, self._connect() as db:
            try:
                db.execute(
                    """INSERT INTO generated_variants
                       (target,content_sha256,origin,generation,parent_candidate_id,candidate_id,
                        status,reason,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (target, content_sha256, origin, generation, parent_candidate_id,
                     candidate_id, status, reason[:2000], now, now),
                )
            except sqlite3.IntegrityError as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("generated variant content was already recorded for this target") from exc
                raise

    def update_generated_variant(
        self, target: str, content_sha256: str, *, status: str,
        reason: str = "", scores: dict[str, Any] | None = None,
    ) -> None:
        _validate_variant_key(target, content_sha256)
        if status not in _VARIANT_STATUSES or status == "root":
            raise ValueError(f"invalid generated variant status transition to {status!r}")
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT scores_json FROM generated_variants WHERE target=? AND content_sha256=?",
                (target, content_sha256),
            ).fetchone()
            if row is None:
                raise KeyError(content_sha256)
            merged = json.loads(row["scores_json"]) | dict(scores or {})
            db.execute(
                """UPDATE generated_variants SET status=?, reason=?, scores_json=?, updated_at=?
                   WHERE target=? AND content_sha256=?""",
                (status, reason[:2000], canonical_json(merged), utc_now(), target, content_sha256),
            )

    def list_generated_variants(self, target: str, limit: int = 20) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM generated_variants WHERE target=?
                   ORDER BY updated_at DESC, rowid DESC LIMIT ?""",
                (target, limit),
            ).fetchall()
        return [self._generated_variant_record(row) for row in rows]

    @staticmethod
    def _generated_variant_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["scores"] = json.loads(item.pop("scores_json"))
        return item


_VARIANT_STATUSES = frozenset({
    "root", "created", "rejected", "smoke_rejected", "core_rejected", "core_no_improvement",
    "improved", "full_rejected", "full_passed", "holdout_passed", "holdout_failed",
    "holdout_refused",
})


def _validate_variant_key(target: str, content_sha256: str) -> None:
    if not isinstance(target, str) or not target.strip():
        raise ValueError("generated variant target must be non-empty")
    if not isinstance(content_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", content_sha256):
        raise ValueError("generated variant content hash must be a lowercase SHA-256 digest")
