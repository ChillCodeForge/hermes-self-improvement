"""End to end: staged ``evaluate --suite`` runs feed WP-A's GitPromoter evidence checks."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve import cli
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.candidate_builder import create_skill_candidate
from hermes_improve.config import default_config
from hermes_improve.errors import PolicyRejected
from hermes_improve.models import CandidateType, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.promotion import GitPromoter
from hermes_improve.run_policy import RunPolicy
from hermes_improve.store import ExperimentStore


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


_PROMPTS = {
    "smoke-task": "Repair the broken quoting in the nightly backup shell script",
    "smoke-unrelated": "Translate the onboarding glossary into a sorted YAML mapping",
    "core-task": "Add pagination limits to the audit log query helper",
    "full-task": "Pin every container base image in the deployment manifests",
    "holdout-task": "Rotate the expiring TLS certificate referenced by the proxy unit",
}


def _benchmark(identifier, split, suites, impact_areas=None):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": _PROMPTS[identifier]},
        "taxonomy": {"categories": ["skills"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": ["exit_code"], "subjective": []},
        "ground_truth": {"type": "test", "confidence": 1.0},
        "benchmark": {
            "split": split, "suites": suites, "cluster_id": identifier, "criticality": "normal",
            **({"impact_areas": impact_areas} if impact_areas else {}),
        },
        "security": {"sanitized": True, "contains_secrets": False},
    }


class FakeBenchmarkRunner:
    """Deterministic test double for the BenchmarkRunner protocol."""

    llm_influenced = False

    def __init__(self, baseline_commit):
        self.baseline_commit = baseline_commit
        self.calls = []
        # task_id -> exception raised when the task runs (simulated infrastructure failure).
        self.crash_on: dict[str, Exception] = {}
        # task IDs the candidate fails although the baseline solves them.
        self.regress: set[str] = set()

    def run(self, commit, benchmark, seed):
        if benchmark.id in self.crash_on:
            raise self.crash_on[benchmark.id]
        self.calls.append((commit, benchmark.id, seed))
        candidate = commit != self.baseline_commit
        regressed = candidate and benchmark.id in self.regress
        return RunObservation(
            task_id=benchmark.id, commit=commit, seed=seed,
            environment_digest="sha256:" + "e" * 64,
            success=not regressed, quality=10 if candidate else 5, tokens=100, latency_ms=20,
            deterministic_pass=True, security_pass=True, tool_calls=2,
            verifier_results=({"type": "exit_code", "passed": True},),
        )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    public_root = tmp_path / "benchmarks"
    public_root.mkdir()
    holdout_root = tmp_path / "holdout"
    holdout_root.mkdir()
    for identifier, suites, areas in (
        ("smoke-task", ["SMOKE"], ["skills/demo*"]),
        ("smoke-unrelated", ["SMOKE"], ["skills/unrelated*"]),
        ("core-task", ["CORE"], None),
        ("full-task", ["FULL"], None),
    ):
        (public_root / f"{identifier}.json").write_text(
            json.dumps(_benchmark(identifier, "REGRESSION", suites, areas))
        )
    (holdout_root / "holdout-task.json").write_text(
        json.dumps(_benchmark("holdout-task", "SEALED_HOLDOUT", ["SEALED_HOLDOUT"]))
    )
    benchmark_version = BenchmarkStore(public_root, holdout_root=holdout_root).version

    repository = tmp_path / "brain-repo"
    repository.mkdir()
    _git(repository, "init", "-q", "-b", "main")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    skill = repository / "brain/skills/demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("original\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "baseline")
    baseline = _git(repository, "rev-parse", "HEAD")

    home = tmp_path / "home"
    home.mkdir()
    store = ExperimentStore(home / "experiments.sqlite3")
    config = default_config()
    candidate = create_skill_candidate(
        repository, store, experiment_id="exp-e2e", candidate_id="cand-e2e",
        candidate_type=CandidateType.SKILL_PATCH, target="brain/skills/demo",
        content="---\nname: demo\ndescription: improved\n---\n\nImproved demo skill.\n",
        hypothesis="improve the demo skill", benchmark_version=benchmark_version,
        policy_version=config.promotion.policy_version,
    )

    private = Ed25519PrivateKey.generate()
    key_file = tmp_path / "evaluator.pem"
    key_file.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    runner = FakeBenchmarkRunner(baseline)
    monkeypatch.setattr(cli, "build_runner", lambda args, artifact_store: runner)
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_PROMOTER_UID", raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public_root))
    monkeypatch.setenv("HERMES_EVALUATOR_PRIVATE_KEY", str(key_file))
    monkeypatch.setenv("HERMES_EVALUATOR_UID", str(os.geteuid()))

    def evaluate(suite):
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "evaluate", "cand-e2e", "--repository", str(repository),
            "--holdout-root", str(holdout_root), "--suite", suite,
        ])
        assert cli.main() == 0

    def promoter():
        return GitPromoter(
            store, repository, tmp_path / "releases", tmp_path / "brain-current",
            PromotionPolicy.from_config(config.promotion), public,
        )

    return {
        "evaluate": evaluate, "promoter": promoter, "store": store, "candidate": candidate, "runner": runner,
        "benchmark_store": BenchmarkStore(public_root, holdout_root=holdout_root),
        "public_root": public_root, "ledger": home / "experiments.sqlite3",
    }


def test_staged_cli_evaluations_satisfy_promotion_evidence(setup):
    for suite in ("SMOKE", "CORE"):
        setup["evaluate"](suite)
        assert setup["store"].show_experiment("exp-e2e")["decision"] == "pending"
    setup["evaluate"]("FULL")
    assert setup["store"].show_experiment("exp-e2e")["decision"] == "passed"

    runs = setup["store"].candidate_evaluations("cand-e2e")
    assert [json.loads(run["evaluation"]["metrics_json"])["suite"] for run in runs] == ["SMOKE", "CORE", "FULL"]
    expected = RunPolicy.from_config(default_config().scoring).describe()
    for run in runs:
        assert run["evaluation"]["policy_version"] == default_config().promotion.policy_version
        recorded = json.loads(run["evaluation"]["metrics_json"])["run_policy"]
        assert {key: recorded[key] for key in expected} == expected

    manifest = setup["promoter"]().promote("exp-e2e", release_name="brain-v1", manifest_key=b"m" * 32)
    assert manifest["candidate_commit"] == setup["candidate"].commit


def test_promotion_rejects_cli_evidence_without_core_run(setup):
    setup["evaluate"]("SMOKE")
    # The evaluator itself refuses the sealed holdout to a candidate without CORE evidence.
    with pytest.raises(PolicyRejected, match="no signed CORE evaluation"):
        setup["evaluate"]("FULL")
    # The promoter re-checks independently: CORE evidence that disappears from the
    # ledger after the FULL run still blocks promotion.
    setup["evaluate"]("CORE")
    setup["evaluate"]("FULL")
    core_id = setup["store"].candidate_evaluations("cand-e2e")[1]["evaluation"]["id"]
    with sqlite3.connect(setup["ledger"]) as db:
        # The ledger refuses deletes through SQL; simulate tampering at file level
        # (someone with write access to the database file drops the guards first).
        for table in ("run_observations", "task_results", "evaluation_runs", "regressions"):
            db.execute(f"DROP TRIGGER IF EXISTS {table}_never_delete")
        db.execute("DELETE FROM regressions WHERE evaluation_run_id=?", (core_id,))
        for table in ("run_observations", "task_results"):
            db.execute(f"DELETE FROM {table} WHERE evaluation_id=?", (core_id,))
        db.execute("DELETE FROM evaluation_runs WHERE id=?", (core_id,))
    with pytest.raises(PolicyRejected, match="required CORE evidence is missing"):
        setup["promoter"]().promote("exp-e2e", release_name="brain-v1", manifest_key=b"m" * 32)


def test_run_policy_follows_scoring_config():
    scoring = default_config().scoring
    policy = RunPolicy.from_config(scoring)
    assert (policy.deterministic_runs, policy.smoke_runs, policy.core_runs, policy.full_runs, policy.holdout_runs) == (
        1, 2, 3, 5, 5,
    )
    assert policy == RunPolicy()


def test_evaluate_selects_impact_scoped_suite_and_records_manifest_digests(setup):
    from hermes_improve.suites import ImpactScope, suite_manifest

    for suite in ("SMOKE", "CORE", "FULL"):
        setup["evaluate"](suite)
    runs = setup["store"].candidate_evaluations("cand-e2e")
    smoke, _core, full = (json.loads(run["evaluation"]["metrics_json"])["suite_selection"] for run in runs)
    # brain/skills/demo narrows SMOKE to the task whose impact_areas match skills/demo.
    assert smoke["task_ids"] == ["smoke-task"]
    assert smoke["impact"]["target_areas"] == ["skills/demo"]
    assert {task for _, task, _ in setup["runner"].calls} >= {"smoke-task"}
    assert "smoke-unrelated" not in {task for _, task, _ in setup["runner"].calls}
    assert smoke["holdout_manifest_digest"] is None
    # FULL ignores impact and adds the sealed holdout with its own manifest digest.
    assert full["task_ids"] == ["full-task"] and full["holdout_manifest_digest"].startswith("sha256:")
    store = setup["benchmark_store"]
    selected = [item for item in store.validate() if item.id == "smoke-task"]
    expected = suite_manifest(
        selected, suite="SMOKE", benchmark_version=store.version,
        impact=ImpactScope(target_areas=frozenset({"skills/demo"})),
    )
    assert smoke["suite_manifest_digest"] == expected["digest"]


def test_experiment_ledger_fields_follow_creation_evaluation_and_promotion(setup):
    store, candidate = setup["store"], setup["candidate"]
    created = store.show_experiment("exp-e2e")
    assert (created["candidate_commit"], created["candidate_hash"]) == (candidate.commit, candidate.content_hash)
    evaluation_fields = ("model_config_digest", "environment_digest", "seeds", "aggregate_metrics", "gate_result")
    assert all(created[field] is None for field in evaluation_fields)

    for suite in ("SMOKE", "CORE"):
        setup["evaluate"](suite)
    # Staged runs without a regression do not decide the experiment.
    assert all(store.show_experiment("exp-e2e")[field] is None for field in evaluation_fields)

    setup["evaluate"]("FULL")
    decided = store.show_experiment("exp-e2e")
    full = store.candidate_evaluations("cand-e2e")[-1]
    full_metrics = json.loads(full["evaluation"]["metrics_json"])
    assert decided["decision"] == "passed"
    assert decided["environment_digest"] == "sha256:" + "e" * 64
    assert decided["model_config_digest"] is None  # the fake runner reports no model configuration
    assert decided["seeds"] == {
        task: sorted({row["seed"] for row in full["run_observations"] if row["task_id"] == task})
        for task in ("full-task", "holdout-task")
    }
    assert decided["aggregate_metrics"] == full_metrics
    assert decided["gate_result"]["policy"]["evaluation_id"] == full["evaluation"]["id"]
    assert decided["gate_result"]["policy"]["passed"] is True and decided["gate_result"]["policy"]["suite"] == "FULL"
    assert "promotion" not in decided["gate_result"]

    manifest = setup["promoter"]().promote("exp-e2e", release_name="brain-v1", manifest_key=b"m" * 32)
    promoted = store.show_experiment("exp-e2e")
    assert promoted["decision"] == "promoted"
    assert promoted["gate_result"]["policy"] == decided["gate_result"]["policy"]
    gate = promoted["gate_result"]["promotion"]
    assert gate["release_id"] == "brain-v1" and gate["release_status"] == "PROVISIONAL"
    assert gate["promotion_gate"] == "auto" and gate["operator_approval"] is None
    assert gate["manifest_sha256"] == manifest["manifest_sha256"]
    assert gate["candidate_commit"] == candidate.commit == promoted["candidate_commit"]
    assert (gate["full_suite_passed"], gate["holdout_passed"], gate["security_passed"]) == (True, True, True)

# --- Sealed-holdout finalist gate and spend-on-execution (spec §16) ---


def _spent_rows(setup):
    with sqlite3.connect(setup["ledger"]) as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='spent_holdouts'"
        ).fetchone()
        if not exists:
            return []
        return [row[0] for row in db.execute("SELECT candidate_id FROM spent_holdouts")]


def _holdout_calls(setup):
    return [call for call in setup["runner"].calls if call[1] == "holdout-task"]


def test_full_without_any_screening_evidence_is_refused_before_any_task_runs(setup):
    with pytest.raises(PolicyRejected, match="reserved for finalists.*no signed SMOKE evaluation"):
        setup["evaluate"]("FULL")
    assert setup["runner"].calls == []
    assert _spent_rows(setup) == []
    assert setup["store"].candidate_evaluations("cand-e2e") == []


def test_full_is_refused_when_the_newest_screening_evidence_fails(setup):
    setup["runner"].regress = {"core-task"}
    setup["evaluate"]("SMOKE")
    setup["evaluate"]("CORE")
    calls = len(setup["runner"].calls)
    with pytest.raises(PolicyRejected, match="newest CORE evaluation .* does not pass.*task-success regression"):
        setup["evaluate"]("FULL")
    assert len(setup["runner"].calls) == calls and not _holdout_calls(setup)
    assert _spent_rows(setup) == []


def test_screening_evidence_must_match_the_current_benchmark_version(setup):
    setup["evaluate"]("SMOKE")
    setup["evaluate"]("CORE")
    # A new public task changes the benchmark manifest version: the old SMOKE/CORE
    # runs no longer describe the suite the FULL run would use.
    extra = _benchmark("core-task", "REGRESSION", ["CORE"])
    extra["id"] = "core-extra"
    extra["task"] = {"title": "core-extra", "prompt": "Document the retention window for archived release notes"}
    extra["benchmark"]["cluster_id"] = "core-extra"
    (setup["public_root"] / "core-extra.json").write_text(json.dumps(extra))
    with pytest.raises(PolicyRejected, match="no signed SMOKE evaluation for benchmark version"):
        setup["evaluate"]("FULL")
    assert not _holdout_calls(setup) and _spent_rows(setup) == []


def test_tampered_screening_evidence_is_refused(setup):
    from hermes_improve.errors import IntegrityError

    setup["runner"].regress = {"smoke-task"}
    setup["evaluate"]("SMOKE")
    setup["runner"].regress = set()
    setup["evaluate"]("CORE")
    smoke_id = setup["store"].candidate_evaluations("cand-e2e")[0]["evaluation"]["id"]
    with sqlite3.connect(setup["ledger"]) as db:
        row = db.execute("SELECT rowid, result_json FROM task_results WHERE evaluation_id=?", (smoke_id,)).fetchone()
        forged = {**json.loads(row[1]), "candidate_success": True}
        db.execute("UPDATE task_results SET result_json=? WHERE rowid=?", (json.dumps(forged), row[0]))
    with pytest.raises(IntegrityError, match="failed signature verification"):
        setup["evaluate"]("FULL")
    assert not _holdout_calls(setup) and _spent_rows(setup) == []


def test_finalist_gets_the_holdout_once_and_it_is_spent_after_execution(setup):
    setup["evaluate"]("SMOKE")
    setup["evaluate"]("CORE")
    setup["evaluate"]("FULL")
    assert _holdout_calls(setup)
    lineage = setup["store"].show_experiment("exp-e2e")["lineage_id"]
    assert _spent_rows(setup) == [f"lineage:{lineage}"]
    calls = len(setup["runner"].calls)
    from hermes_improve.errors import AccessDenied

    with pytest.raises(AccessDenied, match="already consumed"):
        setup["evaluate"]("FULL")
    assert len(setup["runner"].calls) == calls  # the refusal happens before any task runs


def test_infrastructure_failure_before_the_holdout_does_not_spend_it(setup):
    setup["evaluate"]("SMOKE")
    setup["evaluate"]("CORE")
    setup["runner"].crash_on = {"full-task": RuntimeError("docker daemon unavailable")}
    with pytest.raises(RuntimeError, match="docker daemon unavailable"):
        setup["evaluate"]("FULL")
    assert not _holdout_calls(setup)
    assert _spent_rows(setup) == []
    # The infrastructure is repaired: the finalist still gets its single holdout run.
    setup["runner"].crash_on = {}
    setup["evaluate"]("FULL")
    assert _holdout_calls(setup)
    assert setup["store"].show_experiment("exp-e2e")["decision"] == "passed"
    assert len(_spent_rows(setup)) == 1


def test_failure_after_holdout_execution_started_still_spends_it(setup):
    from hermes_improve.errors import AccessDenied

    setup["evaluate"]("SMOKE")
    setup["evaluate"]("CORE")
    setup["runner"].crash_on = {"holdout-task": RuntimeError("runner crashed inside the holdout task")}
    with pytest.raises(RuntimeError, match="crashed inside the holdout task"):
        setup["evaluate"]("FULL")
    lineage = setup["store"].show_experiment("exp-e2e")["lineage_id"]
    assert _spent_rows(setup) == [f"lineage:{lineage}"]
    # No signed FULL/holdout evaluation was recorded, and the lineage cannot retry.
    assert [json.loads(run["evaluation"]["metrics_json"])["suite"]
            for run in setup["store"].candidate_evaluations("cand-e2e")] == ["SMOKE", "CORE"]
    setup["runner"].crash_on = {}
    with pytest.raises(AccessDenied, match="already consumed"):
        setup["evaluate"]("FULL")
    assert not _holdout_calls(setup)
