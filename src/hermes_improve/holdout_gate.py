"""Sealed-holdout gating for the stand-alone ``evaluate`` command (spec §16).

Two rules guard the sealed holdout:

* Only finalists get holdout evaluation. A candidate is a finalist when the
  evaluator itself signed SMOKE and CORE runs for exactly this candidate hash,
  benchmark version and policy version, and the newest such run of each suite
  passes the same per-suite evidence gates the promoter applies.
* A holdout generation is spent when its tasks are executed, not when access
  is granted. ``HoldoutSpendingRunner`` durably records the spend (an atomic
  SQLite insert into the spent-holdout registry) immediately before the first
  sealed-holdout task is handed to the runner. A run that crashes or fails
  after that point stays spent; a run that fails before it (infrastructure
  errors, FULL-suite failures) does not burn the holdout. Two concurrent runs
  for the same lineage race on the registry's primary key: exactly one insert
  succeeds, the loser aborts before executing any holdout task.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from cryptography.hazmat.primitives import serialization

from .errors import IntegrityError, PolicyRejected
from .models import Benchmark, RunObservation, Split
from .policy import PromotionPolicy
from .promotion import check_suite_evidence, evidence_suite_of, verify_evaluation_run_signature
from .store import ExperimentStore

FINALIST_EVIDENCE_SUITES = ("SMOKE", "CORE")


def evaluator_public_key(private_key_pem: bytes) -> bytes:
    """PEM public key matching the evaluator's own Ed25519 signing key."""
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def require_holdout_finalist(
    store: ExperimentStore,
    candidate_id: str,
    *,
    candidate_hash: str,
    benchmark_version: str,
    policy: PromotionPolicy,
    public_key: bytes,
) -> dict[str, str]:
    """Refuse sealed-holdout access unless the candidate is a finalist.

    Returns the evaluation IDs used as SMOKE and CORE evidence. Runs for a
    different candidate hash, benchmark version or policy version are ignored;
    a matching run whose signature does not verify is treated as tampering.
    Raises ``PolicyRejected`` when evidence is missing or failing.
    """
    newest: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for run in reversed(store.candidate_evaluations(candidate_id)):
        evaluation = run["evaluation"]
        if (
            evaluation["candidate_hash"] != candidate_hash
            or evaluation["benchmark_version"] != benchmark_version
            or evaluation["policy_version"] != policy.version
        ):
            continue
        try:
            verify_evaluation_run_signature(candidate_id, run, public_key)
        except IntegrityError as exc:
            raise IntegrityError(
                f"evaluation {evaluation['id']} of candidate {candidate_id} failed signature verification; "
                "refusing sealed holdout access"
            ) from exc
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in run["task_results"]:
            grouped.setdefault(evidence_suite_of(item), []).append(item["result"])
        for suite, rows in grouped.items():
            newest.setdefault(suite, (evaluation["id"], rows))
    evidence: dict[str, str] = {}
    for suite in FINALIST_EVIDENCE_SUITES:
        if suite not in newest:
            raise PolicyRejected(
                f"sealed holdout is reserved for finalists: candidate {candidate_id} has no signed {suite} "
                f"evaluation for benchmark version {benchmark_version} and policy version {policy.version}; "
                "run `evaluate --suite SMOKE` and `evaluate --suite CORE` first"
            )
        run_id, rows = newest[suite]
        try:
            check_suite_evidence(suite, rows, policy)
        except PolicyRejected as exc:
            raise PolicyRejected(
                f"sealed holdout is reserved for finalists: newest {suite} evaluation {run_id} "
                f"of candidate {candidate_id} does not pass ({exc})"
            ) from exc
        evidence[suite] = run_id
    return evidence


class HoldoutSpendingRunner:
    """Runner proxy that spends the holdout generation right before first holdout use.

    ``spend`` must durably record the spend and raise when the generation was
    already spent; it is invoked at most once successfully, before the first
    ``run`` call for a ``SEALED_HOLDOUT`` benchmark, so no holdout task executes
    without a committed spent-registry row. All other attributes (for example
    ``llm_influenced``) are delegated to the wrapped runner.
    """

    def __init__(self, runner: Any, spend: Callable[[], object]) -> None:
        self._runner = runner
        self._spend = spend
        self._spent = False
        self._lock = threading.Lock()

    @property
    def holdout_spent(self) -> bool:
        return self._spent

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._runner, name)

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation:
        if benchmark.split is Split.SEALED_HOLDOUT:
            with self._lock:
                if not self._spent:
                    self._spend()
                    self._spent = True
        return self._runner.run(commit, benchmark, seed)

