from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import IntegrityError
from .migrations import apply_migrations
from .ingestion import contains_secret_material
from .miner import BenchmarkCandidate
from .models import Candidate, EvaluationReport, candidate_digest, canonical_json, utc_now
from .models import RELEASE_TRANSITIONS, ReleaseStatus


class ExperimentStore:
    """SQLite metadata store; large artifacts are stored separately by digest."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # The schema is defined only by the versioned migrations package.
        apply_migrations(self.path)

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
            # The first candidate of an experiment names it; a deciding
            # evaluation re-states the candidate it actually judged.
            db.execute(
                """UPDATE experiments SET candidate_commit=?, candidate_hash=?
                   WHERE id=? AND candidate_commit IS NULL""",
                (candidate.commit, candidate.content_hash, candidate.experiment_id),
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
        parent_id: str | None = None,
    ) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            lineage_id, hypothesis, risk, benchmark_version, policy_version,
        )):
            raise ValueError("experiment metadata fields must be non-empty strings")
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip()):
            raise ValueError("parent experiment ID must be a non-empty string")
        if not candidate.diff or not candidate.commit or not candidate.base_commit:
            raise ValueError("candidate requires commit IDs and a non-empty diff")
        diff_hash = hashlib.sha256(candidate.diff.encode()).hexdigest()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO experiments
                (id,parent_id,lineage_id,hypothesis,source_sessions,candidate_type,target,risk,
                 baseline_commit,candidate_commit,candidate_hash,benchmark_version,policy_version,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (candidate.experiment_id, parent_id, lineage_id, hypothesis,
                 canonical_json(list(source_sessions)),
                 candidate.candidate_type.value, candidate.target, risk, candidate.base_commit,
                 candidate.commit, candidate.content_hash,
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
        *,
        decides_experiment: bool = True,
        regressions: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        """Persist an evaluation; staged (SMOKE/CORE) runs pass decides_experiment=False.

        ``regressions`` are the evaluator's detected regressions (see
        ``evaluator.detect_regressions``); each becomes one ``regressions`` row.
        A deciding evaluation also records its execution identity, seeds,
        aggregate metrics and policy gate result on the experiment row.
        """
        regression_rows = _validated_regressions(regressions, report)
        with self._lock, self._connect() as db:
            candidate = db.execute(
                "SELECT experiment_id, commit_id, content_hash FROM candidates WHERE id=?", (candidate_id,),
            ).fetchone()
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
            db.executemany(
                """INSERT INTO regressions
                   (evaluation_run_id,experiment_id,candidate_id,benchmark_id,suite,split,kind,severity,
                    count,baseline_outcome_json,candidate_outcome_json,detected_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (evaluation_id, candidate["experiment_id"], candidate_id, row["benchmark_id"],
                     row["suite"], row["split"], row["kind"], row["severity"], row["count"],
                     canonical_json(row["baseline_outcome"]), canonical_json(row["candidate_outcome"]),
                     report.created_at)
                    for row in regression_rows
                ],
            )
            if not decides_experiment:
                return
            gate = _gate_result(db, candidate["experiment_id"])
            gate["policy"] = {
                "evaluation_id": evaluation_id,
                "candidate_id": candidate_id,
                "suite": report.metrics.get("suite"),
                "policy_version": report.policy_version,
                "benchmark_version": report.benchmark_version,
                "passed": bool(report.passed),
                "reasons": list(report.reasons),
                "regressions": len(regression_rows),
                "decided_at": report.created_at,
            }
            db.execute(
                """UPDATE experiments SET decision=?,reason=?,finished_at=?,
                       candidate_commit=?,candidate_hash=?,model_config_digest=?,environment_digest=?,
                       seeds=?,aggregate_metrics=?,gate_result=?
                   WHERE id=?""",
                ("passed" if report.passed else "rejected", canonical_json(report.reasons), report.created_at,
                 candidate["commit_id"], candidate["content_hash"],
                 _observed_digest(report, "model_config_digest"),
                 _observed_digest(report, "environment_digest"),
                 _observed_seeds(report), canonical_json(dict(report.metrics)), canonical_json(gate),
                 candidate["experiment_id"]),
            )

    def list_regressions(
        self, *, experiment_id: str | None = None, candidate_id: str | None = None,
        evaluation_run_id: str | None = None, severity: str | None = None, limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Recorded regressions, oldest first, optionally filtered."""
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        if severity is not None and severity not in REGRESSION_SEVERITIES:
            raise ValueError(f"unknown regression severity {severity!r}")
        clauses, parameters = [], []
        for column, value in (
            ("experiment_id", experiment_id), ("candidate_id", candidate_id),
            ("evaluation_run_id", evaluation_run_id), ("severity", severity),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        query = "SELECT * FROM regressions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as db:
            rows = db.execute(query + " ORDER BY id LIMIT ?", (*parameters, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["baseline_outcome"] = json.loads(item.pop("baseline_outcome_json"))
            item["candidate_outcome"] = json.loads(item.pop("candidate_outcome_json"))
            result.append(item)
        return result

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
            for column in ("seeds", "aggregate_metrics", "gate_result"):
                if result.get(column) is not None:
                    result[column] = json.loads(result[column])
            result["candidates"] = [dict(item) for item in db.execute(
                "SELECT * FROM candidates WHERE experiment_id=?", (experiment_id,)
            )]
            result["skill_use"] = [
                _skill_use_summary(row["id"], row["candidate_id"], row["metrics_json"])
                for row in db.execute(
                    """SELECT e.id, e.candidate_id, e.metrics_json FROM evaluation_runs e
                       JOIN candidates c ON c.id = e.candidate_id
                       WHERE c.experiment_id=? ORDER BY e.created_at, e.id""",
                    (experiment_id,),
                )
            ]
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
        if status not in {item.value for item in ReleaseStatus}:
            raise ValueError("invalid release status")
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO releases (id,experiment_id,candidate_hash,status,manifest_json,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (release_id, experiment_id, candidate_hash, status, canonical_json(manifest), utc_now()),
            )
            _record_promotion_gate(db, experiment_id, release_id, status, manifest)

    def update_release_status(self, release_id: str, status: str) -> None:
        if status not in {item.value for item in ReleaseStatus}:
            raise ValueError("invalid release status")
        with self._lock, self._connect() as db:
            cursor = db.execute("UPDATE releases SET status=? WHERE id=?", (status, release_id))
            if cursor.rowcount != 1:
                raise KeyError(release_id)

    _RELEASE_TRANSITIONS = {
        "CANDIDATE": frozenset({"PROVISIONAL"}),
        "PROVISIONAL": frozenset({"STABLE", "ROLLED_BACK"}),
        "STABLE": frozenset({"ROLLED_BACK"}),
        "ROLLED_BACK": frozenset(),
    }

    def set_release_status(
        self, release_id: str, status: str, *, reason: str = "post-promotion monitoring",
        actor: str = "monitor",
    ) -> str:
        """Move a release along CANDIDATE -> PROVISIONAL -> STABLE / ROLLED_BACK.

        Returns the previous status. Setting the current status again is an
        idempotent no-op; every real change goes through the audited
        compare-and-set ``transition_release_status``.
        """
        if status not in self._RELEASE_TRANSITIONS:
            raise ValueError("invalid release status")
        with self._connect() as db:
            row = db.execute("SELECT status FROM releases WHERE id=?", (release_id,)).fetchone()
        if row is None:
            raise KeyError(release_id)
        current = row["status"]
        if current == status:
            return current
        self.transition_release_status(
            release_id, status, expected_status=current, reason=reason, actor=actor,
        )
        return current

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

    # --- Promotion evidence, operator approvals and release audit (WP-A) ---

    def candidate_evaluations(self, candidate_id: str) -> list[dict[str, Any]]:
        """All evaluation runs of a candidate, oldest first, with task results
        and run observations (the evidence a promoter must re-verify)."""
        with self._connect() as db:
            if db.execute("SELECT 1 FROM candidates WHERE id=?", (candidate_id,)).fetchone() is None:
                raise KeyError(candidate_id)
            runs = []
            for evaluation in db.execute(
                "SELECT * FROM evaluation_runs WHERE candidate_id=? ORDER BY created_at, rowid",
                (candidate_id,),
            ).fetchall():
                results = [
                    {**dict(row), "result": json.loads(row["result_json"])}
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
                runs.append({
                    "evaluation": dict(evaluation),
                    "task_results": results,
                    "run_observations": observations,
                })
            return runs

    def resolve_promotion_candidate(self, identifier: str) -> str:
        """Resolve an experiment ID (preferred) or a candidate ID to a candidate ID."""
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("promotion requires an experiment or candidate ID")
        with self._connect() as db:
            by_experiment = [
                row["id"] for row in db.execute(
                    "SELECT id FROM candidates WHERE experiment_id=? ORDER BY created_at", (identifier,)
                )
            ]
            experiment = db.execute("SELECT 1 FROM experiments WHERE id=?", (identifier,)).fetchone()
            by_candidate = db.execute("SELECT id FROM candidates WHERE id=?", (identifier,)).fetchone()
        if experiment is not None:
            if len(by_experiment) != 1:
                raise ValueError(
                    f"experiment {identifier!r} has {len(by_experiment)} candidates; promote by candidate ID"
                )
            if by_candidate is not None and by_candidate["id"] != by_experiment[0]:
                raise ValueError(f"identifier {identifier!r} is ambiguous between an experiment and a candidate")
            return by_experiment[0]
        if by_candidate is not None:
            return by_candidate["id"]
        raise KeyError(identifier)

    def record_promotion_approval(self, payload: dict[str, Any], signature: str) -> int:
        """Append a signed operator approval. Approvals are never updated or deleted."""
        required = {
            "schema_version", "purpose", "experiment_id", "candidate_id", "candidate_hash",
            "candidate_type", "policy_version", "operator", "rationale",
            "operator_key_fingerprint", "approved_at",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("promotion approval payload has an invalid field set")
        if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{128}", signature):
            raise ValueError("promotion approval signature must be a hex Ed25519 signature")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            candidate = db.execute(
                "SELECT experiment_id, content_hash, candidate_type FROM candidates WHERE id=?",
                (payload["candidate_id"],),
            ).fetchone()
            if candidate is None:
                raise KeyError(payload["candidate_id"])
            if (
                candidate["experiment_id"] != payload["experiment_id"]
                or candidate["content_hash"] != payload["candidate_hash"]
                or candidate["candidate_type"] != payload["candidate_type"]
            ):
                raise IntegrityError("approval does not match the immutable candidate record")
            cursor = db.execute(
                """INSERT INTO promotion_approvals
                   (experiment_id,candidate_id,candidate_hash,policy_version,operator,
                    key_fingerprint,payload_json,signature,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (payload["experiment_id"], payload["candidate_id"], payload["candidate_hash"],
                 payload["policy_version"], payload["operator"], payload["operator_key_fingerprint"],
                 canonical_json(payload), signature, utc_now()),
            )
            return int(cursor.lastrowid)

    def list_promotion_approvals(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [
                {
                    "id": row["id"],
                    "payload": json.loads(row["payload_json"]),
                    "signature": row["signature"],
                    "created_at": row["created_at"],
                }
                for row in db.execute(
                    "SELECT * FROM promotion_approvals WHERE experiment_id=? ORDER BY id", (experiment_id,)
                )
            ]

    def record_promotion_release(
        self, release_id: str, experiment_id: str, candidate_hash: str,
        manifest: dict[str, Any], *, actor: str = "promoter",
    ) -> None:
        """Record a promoted release as PROVISIONAL, mark the experiment
        promoted, and append the audit event in one transaction."""
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO releases (id,experiment_id,candidate_hash,status,manifest_json,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (release_id, experiment_id, candidate_hash, ReleaseStatus.PROVISIONAL.value,
                 canonical_json(manifest), utc_now()),
            )
            _record_promotion_gate(
                db, experiment_id, release_id, ReleaseStatus.PROVISIONAL.value, manifest,
            )
            self._append_release_event(
                db, release_id, "PROMOTED", None, ReleaseStatus.PROVISIONAL.value,
                {"actor": actor, "experiment_id": experiment_id, "candidate_hash": candidate_hash,
                 "candidate_commit": manifest.get("candidate_commit"),
                 "baseline_commit": manifest.get("baseline_commit")},
            )

    def record_rollback(
        self, active_commit: str, target_release_id: str, detail: dict[str, Any],
    ) -> str | None:
        """Mark the release owning ``active_commit`` ROLLED_BACK and re-activate
        ``target_release_id`` as PROVISIONAL, with audit events, atomically.
        Returns the rolled-back release ID (``None`` if the commit had none)."""
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            target = db.execute("SELECT status FROM releases WHERE id=?", (target_release_id,)).fetchone()
            if target is None:
                raise KeyError(target_release_id)
            active_ids = [
                row["id"] for row in db.execute("SELECT id,manifest_json FROM releases")
                if json.loads(row["manifest_json"]).get("candidate_commit") == active_commit
            ]
            if len(active_ids) > 1:
                raise ValueError("multiple release records claim the active Git commit")
            rolled_back = active_ids[0] if active_ids else None
            if rolled_back is not None:
                previous = db.execute("SELECT status FROM releases WHERE id=?", (rolled_back,)).fetchone()
                db.execute("UPDATE releases SET status='ROLLED_BACK' WHERE id=?", (rolled_back,))
                self._append_release_event(
                    db, rolled_back, "ROLLED_BACK", previous["status"], ReleaseStatus.ROLLED_BACK.value,
                    {**detail, "reactivated_release": target_release_id},
                )
            db.execute("UPDATE releases SET status='PROVISIONAL' WHERE id=?", (target_release_id,))
            self._append_release_event(
                db, target_release_id, "REACTIVATED", target["status"], ReleaseStatus.PROVISIONAL.value,
                {**detail, "rolled_back_release": rolled_back},
            )
            return rolled_back

    def transition_release_status(
        self, release_id: str, new_status: str, *, expected_status: str, reason: str,
        actor: str = "operator",
    ) -> None:
        """Compare-and-set a release status along RELEASE_TRANSITIONS, with an audit event."""
        try:
            new = ReleaseStatus(new_status)
            expected = ReleaseStatus(expected_status)
        except ValueError as exc:
            raise ValueError("invalid release status") from exc
        if new not in RELEASE_TRANSITIONS[expected]:
            raise ValueError(f"release status transition {expected.value} -> {new.value} is not allowed")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2048:
            raise ValueError("release status transitions require a reason up to 2048 characters")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                "UPDATE releases SET status=? WHERE id=? AND status=?",
                (new.value, release_id, expected.value),
            )
            if cursor.rowcount != 1:
                if db.execute("SELECT 1 FROM releases WHERE id=?", (release_id,)).fetchone() is None:
                    raise KeyError(release_id)
                raise ValueError(f"release {release_id!r} is not in status {expected.value}")
            self._append_release_event(
                db, release_id, "STATUS_CHANGED", expected.value, new.value,
                {"actor": actor, "reason": reason.strip()},
            )

    # Audit events that record a decision about a release without changing its status.
    MONITORING_RELEASE_EVENTS = frozenset({"ROLLBACK_BLOCKED"})

    def record_monitoring_event(self, release_id: str, event: str, detail: dict[str, Any]) -> None:
        """Append a status-preserving monitoring event (e.g. a blocked rollback)
        to the release audit, so ``releases events`` and ``status`` show it."""
        if event not in self.MONITORING_RELEASE_EVENTS:
            raise ValueError(f"unsupported monitoring release event: {event!r}")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM releases WHERE id=?", (release_id,)).fetchone()
            if row is None:
                raise KeyError(release_id)
            self._append_release_event(db, release_id, event, row["status"], row["status"], detail)

    def list_release_events(self, release_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM release_events"
        parameters: tuple[Any, ...] = ()
        if release_id is not None:
            query += " WHERE release_id=?"
            parameters = (release_id,)
        with self._connect() as db:
            return [
                {**dict(row), "detail": json.loads(row["detail_json"])}
                for row in db.execute(query + " ORDER BY id", parameters)
            ]

    @staticmethod
    def _append_release_event(
        db: sqlite3.Connection, release_id: str, event: str,
        from_status: str | None, to_status: str | None, detail: dict[str, Any],
    ) -> None:
        db.execute(
            """INSERT INTO release_events (release_id,event,from_status,to_status,detail_json,created_at)
               VALUES (?,?,?,?,?,?)""",
            (release_id, event, from_status, to_status, canonical_json(detail), utc_now()),
        )


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

    def record_benchmark_replay(self, result: dict[str, Any]) -> int:
        """Append immutable replay evidence (a `ReplayResult.to_dict()` payload)."""
        required = {
            "benchmark_id", "benchmark_version", "benchmark_digest", "commit", "runs",
            "reproducible", "passed", "outcome_digest", "replayed_at",
        }
        if not isinstance(result, dict) or not required <= result.keys():
            raise ValueError("benchmark replay evidence is missing required fields")
        if (
            not isinstance(result["runs"], list) or len(result["runs"]) < 2
            or not isinstance(result["reproducible"], bool) or not isinstance(result["passed"], bool)
            or not re.fullmatch(r"[0-9a-f]{64}", str(result["outcome_digest"]))
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(result["commit"]))
        ):
            raise ValueError("benchmark replay evidence is malformed")
        if contains_secret_material(result):
            raise ValueError("benchmark replay evidence must not contain credential-like material")
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """INSERT INTO benchmark_replays
                (benchmark_id,benchmark_version,benchmark_digest,commit_id,runs,reproducible,
                 passed,outcome_digest,result_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (result["benchmark_id"], result["benchmark_version"], result["benchmark_digest"],
                 result["commit"], len(result["runs"]), int(result["reproducible"]),
                 int(result["passed"]), result["outcome_digest"], canonical_json(result),
                 result["replayed_at"]),
            )
            return int(cursor.lastrowid)

    def list_benchmark_replays(self, benchmark_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT result_json FROM benchmark_replays"
        parameters: tuple[str, ...] = ()
        if benchmark_id is not None:
            query += " WHERE benchmark_id=?"
            parameters = (benchmark_id,)
        with self._connect() as db:
            return [json.loads(row["result_json"]) for row in db.execute(query + " ORDER BY id", parameters)]



REGRESSION_KINDS = frozenset({"reported", "task_failure"})
REGRESSION_SEVERITIES = frozenset({"critical", "high", "medium"})
_REGRESSION_FIELDS = frozenset({
    "benchmark_id", "suite", "split", "kind", "severity", "count", "baseline_outcome", "candidate_outcome",
})


def _validated_regressions(
    regressions: Iterable[Mapping[str, Any]], report: EvaluationReport,
) -> list[dict[str, Any]]:
    outcomes = {row.task_id: row for row in report.outcomes}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in regressions:
        if not isinstance(item, Mapping) or set(item) != _REGRESSION_FIELDS:
            raise ValueError("regression records have an invalid field set")
        outcome = outcomes.get(item["benchmark_id"])
        if outcome is None:
            raise ValueError(f"regression names a task outside the evaluation: {item['benchmark_id']!r}")
        if item["suite"] != outcome.suite or item["split"] != outcome.split.value:
            raise ValueError("regression suite or split differs from the evaluated task outcome")
        if item["kind"] not in REGRESSION_KINDS or item["severity"] not in REGRESSION_SEVERITIES:
            raise ValueError("regression kind or severity is unknown")
        count = item["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("regression count must be a positive integer")
        if not isinstance(item["baseline_outcome"], Mapping) or not isinstance(item["candidate_outcome"], Mapping):
            raise ValueError("regression outcome summaries must be objects")
        key = (item["benchmark_id"], item["kind"], item["severity"])
        if key in seen:
            raise ValueError(f"duplicate regression record {key!r}")
        seen.add(key)
        rows.append(dict(item))
    return rows


def _skill_use_summary(evaluation_id: str, candidate_id: str, metrics_json: str | None) -> dict[str, Any]:
    """Per-evaluation skill-use line for ``experiments show`` (lab finding F30).

    Evaluations recorded before skill use was measured show ``None`` values.
    """
    try:
        metrics = json.loads(metrics_json) if metrics_json else {}
    except json.JSONDecodeError:
        metrics = {}
    use = metrics.get("skill_use") if isinstance(metrics, dict) else None
    use = use if isinstance(use, dict) else {}
    baseline = use.get("baseline") if isinstance(use.get("baseline"), dict) else {}
    candidate = use.get("candidate") if isinstance(use.get("candidate"), dict) else {}
    return {
        "evaluation_id": evaluation_id,
        "candidate_id": candidate_id,
        "suite": metrics.get("suite") if isinstance(metrics, dict) else None,
        "target_skill": use.get("target_skill"),
        "baseline_target_skill_loaded_rate": baseline.get("target_skill_loaded_rate"),
        "candidate_target_skill_loaded_rate": candidate.get("target_skill_loaded_rate"),
        "candidate_observed_runs": candidate.get("observed_runs"),
    }


def _gate_result(db: sqlite3.Connection, experiment_id: str) -> dict[str, Any]:
    row = db.execute("SELECT gate_result FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    if row is None:
        raise KeyError(experiment_id)
    return json.loads(row["gate_result"]) if row["gate_result"] is not None else {}


def _observed_digest(report: EvaluationReport, field_name: str) -> str | None:
    """The run digest of an evaluation: the single value all runs shared, or a
    ``composite-sha256:`` digest over the per-task values when tasks differ.
    ``None`` when no run reported one (the per-run values stay in run_observations)."""
    per_task: dict[str, set[str]] = {}
    for observation in report.run_observations:
        value = getattr(observation, field_name)
        if value:
            per_task.setdefault(observation.task_id, set()).add(value)
    distinct = set().union(*per_task.values()) if per_task else set()
    if not distinct:
        return None
    if len(distinct) == 1:
        return next(iter(distinct))
    mapping = {task: sorted(values) for task, values in sorted(per_task.items())}
    return "composite-sha256:" + hashlib.sha256(canonical_json(mapping).encode()).hexdigest()


def _observed_seeds(report: EvaluationReport) -> str | None:
    seeds: dict[str, set[int]] = {}
    for observation in report.run_observations:
        seeds.setdefault(observation.task_id, set()).add(observation.seed)
    if not seeds:
        return None
    return canonical_json({task: sorted(values) for task, values in sorted(seeds.items())})


_PROMOTION_GATE_FIELDS = (
    "promotion_gate", "full_suite_passed", "holdout_passed", "security_passed",
    "policy_version", "benchmark_version", "candidate_commit", "candidate_hash", "manifest_sha256",
)


def _record_promotion_gate(
    db: sqlite3.Connection, experiment_id: str, release_id: str, status: str, manifest: Mapping[str, Any],
) -> None:
    """Mark the experiment promoted and add the promotion gate to gate_result."""
    gate = _gate_result(db, experiment_id)
    promoted_at = utc_now()
    promotion: dict[str, Any] = {"release_id": release_id, "release_status": status, "promoted_at": promoted_at}
    promotion.update({key: manifest[key] for key in _PROMOTION_GATE_FIELDS if key in manifest})
    if "approval" in manifest:
        approval = manifest["approval"]
        promotion["operator_approval"] = (
            {key: approval[key] for key in ("operator", "operator_key_fingerprint") if key in approval}
            if isinstance(approval, Mapping) else None
        )
    gate["promotion"] = promotion
    db.execute(
        "UPDATE experiments SET decision='promoted',finished_at=?,gate_result=? WHERE id=?",
        (promoted_at, canonical_json(gate), experiment_id),
    )


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
