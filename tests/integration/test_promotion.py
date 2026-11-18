from __future__ import annotations

import os
import subprocess

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve.errors import IntegrityError, PolicyRejected
from hermes_improve.evaluation_signing import sign_report
from hermes_improve.models import Candidate, CandidateType, Split, TaskOutcome
from hermes_improve.policy import PromotionPolicy
from hermes_improve.promotion import GitPromoter
from hermes_improve.store import ExperimentStore


def git(repository, *args):
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode().strip()


def _key_pair():
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


ALL_SUITES = ("SMOKE", "CORE", "FULL", "SEALED_HOLDOUT")


def _outcomes(candidate_id, suites, **overrides):
    return tuple(
        TaskOutcome(**{
            "task_id": f"{candidate_id}-{suite.lower()}",
            "split": Split.SEALED_HOLDOUT if suite == "SEALED_HOLDOUT" else Split.REGRESSION,
            "suite": suite, "baseline_success": True, "candidate_success": True,
            "baseline_quality": 5, "candidate_quality": 10,
            "baseline_tokens": 100, "candidate_tokens": 100, "token_usage_reported": True,
            "categories": ("skills",),
            "baseline_latency_ms": 100, "candidate_latency_ms": 100,
            "baseline_tool_calls": 5, "candidate_tool_calls": 5,
            **overrides.get(suite, {}),
        })
        for suite in suites
    )


def _record_run(store, candidate, private, evaluation_id, suites, *, policy=None, **overrides):
    policy = policy or PromotionPolicy()
    report = policy.assess(
        candidate.candidate_id, candidate.content_hash, "bench-v1",
        _outcomes(candidate.candidate_id, suites, **overrides),
    )
    store.record_evaluation(
        evaluation_id, candidate.candidate_id, report, signature=sign_report(report, private),
    )
    return report


def _add_candidate(
    repository, store, experiment_id, candidate_id, parent, content, *,
    approved=True, signing_keys=None, target="brain/skills/demo",
    candidate_type=CandidateType.SKILL_PATCH, path="brain/skills/demo/SKILL.md",
    extra_files=None, suites=ALL_SUITES, policy=None, evaluate=True,
):
    policy = policy or PromotionPolicy()
    git(repository, "checkout", "-q", "-b", f"candidate/{experiment_id}", parent)
    for relative, text in {path: content, **(extra_files or {})}.items():
        file = repository / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
    git(repository, "add", "-A")
    git(repository, "commit", "-m", f"candidate {experiment_id}")
    commit = git(repository, "rev-parse", "HEAD")
    diff = subprocess.run(
        ["git", "-C", str(repository), "diff", "--binary", parent, commit],
        check=True, stdout=subprocess.PIPE,
    ).stdout.decode()
    store.create_experiment(
        experiment_id, lineage_id="lineage-1", hypothesis="improve the skill",
        candidate_type=candidate_type.value, target=target,
        risk="LOW", baseline_commit=parent, benchmark_version="bench-v1",
        policy_version=policy.version,
    )
    candidate = Candidate(
        candidate_id, experiment_id, candidate_type,
        target, parent, commit, diff,
    )
    store.add_candidate(candidate)
    private, public = signing_keys or _key_pair()
    if evaluate:
        outcomes = _outcomes(candidate_id, suites)
        report = policy.assess(candidate_id, candidate.content_hash, "bench-v1", outcomes)
        if not approved:
            report = PromotionPolicy(min_quality_delta=99).assess(
                candidate_id, candidate.content_hash, "bench-v1", outcomes
            )
        assert report.passed is approved
        store.record_evaluation(
            f"eval-{candidate_id}", candidate_id, report,
            signature=sign_report(report, private),
        )
    return candidate, private, public


def _repository(tmp_path, *, sha256=False):
    repository = tmp_path / "git-brain"
    repository.mkdir()
    init_args = ["init", "-b", "main"]
    if sha256:
        init_args.append("--object-format=sha256")
    try:
        git(repository, *init_args)
    except subprocess.CalledProcessError:
        if sha256:
            pytest.skip("installed Git does not support SHA-256 repositories")
        raise
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.invalid")
    skill = repository / "brain" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("original\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "baseline")
    return repository, git(repository, "rev-parse", "HEAD")


def test_promoter_requires_signed_passed_evaluation_and_activates_atomically(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "improved\n")
    release_root = tmp_path / "releases"
    current = tmp_path / "brain-current"
    promoter = GitPromoter(store, repository, release_root, current, PromotionPolicy(), public)
    manifest = promoter.promote("cand-1", release_name="brain-v1", manifest_key=b"m" * 32)
    assert manifest["candidate_commit"] == candidate.commit
    assert git(repository, "rev-parse", "main") == candidate.commit
    assert current.is_symlink()
    assert (current / "brain/skills/demo/SKILL.md").read_text() == "improved\n"
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"
    assert (release_root / "brain-v1" / "brain/skills/demo/SKILL.md").stat().st_mode & 0o222 == 0


def test_promoter_supports_sha256_git_object_format(tmp_path):
    repository, baseline = _repository(tmp_path, sha256=True)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, _, public = _add_candidate(
        repository, store, "exp-sha256", "cand-sha256", baseline, "improved\n",
    )
    promoter = GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current",
        PromotionPolicy(), public,
    )
    result = promoter.promote(
        "cand-sha256", release_name="brain-v1", manifest_key=b"m" * 32,
    )
    assert len(candidate.commit) == 64
    assert result["candidate_commit"] == candidate.commit


def test_promoter_rejects_reserved_candidate_target(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(
        repository, store, "exp-unsafe-target", "cand-unsafe-target", baseline, "improved\n",
        target="brain/skills/.git",
    )
    promoter = GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current",
        PromotionPolicy(), public,
    )

    with pytest.raises(PolicyRejected, match="candidate target is not a safe"):
        promoter.promote(
            "cand-unsafe-target", release_name="brain-v1", manifest_key=b"m" * 32,
        )


def test_unevaluated_or_failed_candidate_cannot_be_promoted(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(
        repository, store, "exp-bad", "cand-bad", baseline, "worse\n", approved=False
    )
    promoter = GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current",
        PromotionPolicy(), public,
    )
    with pytest.raises(PolicyRejected, match="passing"):
        promoter.promote("cand-bad", release_name="brain-v1", manifest_key=b"m" * 32)


def test_promoter_rejects_tampered_or_untrusted_evaluation_signature(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "improved\n")
    with store._connect() as db:
        db.execute("UPDATE evaluation_runs SET signature='00' WHERE candidate_id='cand-1'")
    promoter = GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current",
        PromotionPolicy(), public,
    )
    with pytest.raises(IntegrityError, match="signature"):
        promoter.promote("cand-1", release_name="brain-v1", manifest_key=b"m" * 32)


def test_promoter_rejects_non_skill_change_and_changed_diff(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "improved\n")
    with store._connect() as db:
        db.execute("UPDATE candidates SET diff_hash=? WHERE id='cand-1'", ("0" * 64,))
    promoter = GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current",
        PromotionPolicy(), public,
    )
    with pytest.raises(IntegrityError, match="diff"):
        promoter.promote("cand-1", release_name="brain-v1", manifest_key=b"m" * 32)


def test_candidate_target_metadata_is_bound_by_signature_hash(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "improved\n")
    with store._connect() as db:
        db.execute("UPDATE candidates SET target='brain/skills/other' WHERE id='cand-1'")
    promoter = GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current",
        PromotionPolicy(), public,
    )
    with pytest.raises(IntegrityError, match="content hash"):
        promoter.promote("cand-1", release_name="brain-v1", manifest_key=b"m" * 32)


def test_release_rollback_switches_pointer_and_preserves_tags(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "version one\n")
    release_root = tmp_path / "releases"
    current = tmp_path / "brain-current"
    promoter = GitPromoter(store, repository, release_root, current, PromotionPolicy(), public)
    promoter.promote("cand-1", release_name="brain-v1", manifest_key=b"m" * 32)

    second, _, _ = _add_candidate(
        repository, store, "exp-2", "cand-2", first.commit, "version two\n",
        signing_keys=(private, public),
    )
    promoter.promote("cand-2", release_name="brain-v2", manifest_key=b"m" * 32)
    assert (current / "brain/skills/demo/SKILL.md").read_text() == "version two\n"
    previous_commit = promoter.rollback("brain-v1", manifest_key=b"m" * 32)
    assert previous_commit == second.commit
    assert git(repository, "rev-parse", "main") == first.commit
    assert (current / "brain/skills/demo/SKILL.md").read_text() == "version one\n"
    assert git(repository, "rev-parse", "brain-v1") == first.commit
    assert git(repository, "rev-parse", "brain-v2") == second.commit
    assert store.get_release("brain-v2")["status"] == "ROLLED_BACK"
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"
    events = [(item["release_id"], item["event"], item["to_status"]) for item in store.list_release_events()]
    assert events == [
        ("brain-v1", "PROMOTED", "PROVISIONAL"),
        ("brain-v2", "PROMOTED", "PROVISIONAL"),
        ("brain-v2", "ROLLED_BACK", "ROLLED_BACK"),
        ("brain-v1", "REACTIVATED", "PROVISIONAL"),
    ]
    assert store.list_release_events("brain-v2")[-1]["detail"]["previous_commit"] == second.commit
    assert (release_root / "brain-v2" / "brain/skills/demo/SKILL.md").read_text() == "version two\n"


def test_tampered_promotion_manifest_blocks_rollback(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "improved\n")
    release_root = tmp_path / "releases"
    promoter = GitPromoter(
        store, repository, release_root, tmp_path / "brain-current", PromotionPolicy(), public,
    )
    promoter.promote("cand-1", release_name="brain-v1", manifest_key=b"m" * 32)
    manifest_path = release_root / "brain-v1.manifest.json"
    os.chmod(manifest_path, 0o644)
    manifest_path.write_text('{"candidate_commit":"tampered"}')
    with pytest.raises(IntegrityError, match="manifest"):
        promoter.rollback("brain-v1", manifest_key=b"m" * 32)


def _promoter(store, repository, tmp_path, public, **kwargs):
    policy = kwargs.pop("policy", PromotionPolicy())
    return GitPromoter(
        store, repository, tmp_path / "releases", tmp_path / "brain-current", policy, public, **kwargs,
    )


def test_promote_by_experiment_id_assigns_next_release_and_audits(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    promoter = _promoter(store, repository, tmp_path, public)
    manifest = promoter.promote("exp-1", manifest_key=b"m" * 32)
    assert manifest["release_target"] == "brain-v1"
    assert manifest["promotion_gate"] == "auto" and manifest["approval"] is None
    assert set(manifest["evidence_runs"]) == set(ALL_SUITES)
    _add_candidate(
        repository, store, "exp-2", "cand-2", first.commit, "two\n", signing_keys=(private, public),
    )
    assert promoter.promote("cand-2", manifest_key=b"m" * 32)["release_target"] == "brain-v2"
    assert git(repository, "tag", "--list", "brain-v*").split() == ["brain-v1", "brain-v2"]
    assert [row["event"] for row in store.list_release_events("brain-v1")] == ["PROMOTED"]
    assert store.show_experiment("exp-2")["decision"] == "promoted"


def test_missing_smoke_or_core_evidence_blocks_promotion(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n", suites=("FULL", "SEALED_HOLDOUT"),
    )
    with pytest.raises(PolicyRejected, match="required SMOKE evidence is missing"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    assert git(repository, "rev-parse", "main") == baseline
    relaxed = PromotionPolicy(required_evidence_suites=("FULL", "SEALED_HOLDOUT"))
    manifest = _promoter(store, repository, tmp_path, public, policy=relaxed).promote(
        "exp-1", manifest_key=b"m" * 32,
    )
    assert set(manifest["evidence_runs"]) == {"FULL", "SEALED_HOLDOUT"}


def test_evidence_from_separate_signed_suite_runs(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n", evaluate=False,
    )
    _record_run(store, candidate, private, "eval-smoke", ("SMOKE",))
    _record_run(store, candidate, private, "eval-core", ("CORE",))
    _record_run(store, candidate, private, "eval-final", ("FULL", "SEALED_HOLDOUT"))
    manifest = _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    assert manifest["evidence_runs"] == {
        "SMOKE": "eval-smoke", "CORE": "eval-core", "FULL": "eval-final", "SEALED_HOLDOUT": "eval-final",
    }


def test_tampered_earlier_suite_run_blocks_promotion(tmp_path):
    import json

    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n", evaluate=False,
    )
    _record_run(store, candidate, private, "eval-smoke", ("SMOKE", "CORE"))
    _record_run(store, candidate, private, "eval-final", ("FULL", "SEALED_HOLDOUT"))
    with store._connect() as db:
        row = db.execute(
            "SELECT task_id, result_json FROM task_results WHERE evaluation_id='eval-smoke' LIMIT 1"
        ).fetchone()
        result = json.loads(row["result_json"])
        result["candidate_quality"] = 99
        db.execute(
            "UPDATE task_results SET result_json=? WHERE evaluation_id='eval-smoke' AND task_id=?",
            (json.dumps(result), row["task_id"]),
        )
    with pytest.raises(IntegrityError, match="signature"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)


def test_smoke_success_regression_blocks_promotion(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n", evaluate=False,
    )
    _record_run(
        store, candidate, private, "eval-smoke", ("SMOKE", "CORE"),
        SMOKE={"candidate_success": False},
    )
    _record_run(store, candidate, private, "eval-final", ("FULL", "SEALED_HOLDOUT"))
    with pytest.raises(PolicyRejected, match="SMOKE evidence shows a task-success regression"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)


def test_candidate_changed_after_evaluation_cannot_be_promoted(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    (repository / "brain/skills/demo/SKILL.md").write_text("changed after evaluation\n", encoding="utf-8")
    git(repository, "commit", "-am", "sneaky change")
    with pytest.raises(IntegrityError, match="no longer points to the evaluated commit"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    assert git(repository, "rev-parse", "main") == baseline


def test_candidate_diff_touching_protected_config_is_rejected(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n",
        extra_files={"config/promotion.yaml": "promotion_gates: {TOOL_PATCH: auto}\n"},
    )
    with pytest.raises(PolicyRejected, match="protected paths: config/promotion.yaml"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)


def test_changed_paths_outside_declared_target_are_rejected(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n",
        extra_files={"brain/skills/other/SKILL.md": "smuggled skill\n"},
    )
    with pytest.raises(PolicyRejected, match="do not match the candidate target"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)


def _tool_description_candidate(repository, store, baseline, **kwargs):
    return _add_candidate(
        repository, store, "exp-td", "cand-td", baseline, "Run a shell command safely.\n",
        candidate_type=CandidateType.TOOL_DESCRIPTION,
        target="brain/tool-descriptions/terminal.md",
        path="brain/tool-descriptions/terminal.md",
        **kwargs,
    )


def test_manual_gate_requires_signed_operator_approval(tmp_path):
    from hermes_improve.approvals import approve_promotion

    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, _, evaluator_public = _tool_description_candidate(repository, store, baseline)
    operator_private, operator_public = _key_pair()
    stranger_private, _ = _key_pair()

    with pytest.raises(PolicyRejected, match="requires a signed operator approval"):
        _promoter(store, repository, tmp_path, evaluator_public).promote("exp-td", manifest_key=b"m" * 32)

    approve_promotion(
        store, "exp-td", operator="mallory", rationale="self-approved",
        private_key_pem=stranger_private, policy=PromotionPolicy(),
    )
    promoter = _promoter(
        store, repository, tmp_path, evaluator_public, operator_public_keys=[operator_public],
    )
    with pytest.raises(PolicyRejected, match="requires a signed operator approval"):
        promoter.promote("exp-td", manifest_key=b"m" * 32)
    assert git(repository, "rev-parse", "main") == baseline

    approval = approve_promotion(
        store, "exp-td", operator="alice", rationale="Reviewed the description diff and evidence.",
        private_key_pem=operator_private, policy=PromotionPolicy(),
    )
    assert approval["candidate_hash"] == candidate.content_hash
    manifest = promoter.promote("exp-td", manifest_key=b"m" * 32)
    assert manifest["promotion_gate"] == "manual"
    assert manifest["approval"]["operator"] == "alice"
    assert manifest["approval"]["policy_version"] == "1"
    assert git(repository, "rev-parse", "main") == candidate.commit

    with pytest.raises(ValueError, match="must differ from the evaluator"):
        _promoter(
            store, repository, tmp_path, evaluator_public, operator_public_keys=[evaluator_public],
        )


def test_approvals_are_append_only_and_bound_to_candidate_and_policy(tmp_path):
    import sqlite3

    from hermes_improve.approvals import approve_promotion, load_public_key, verify_promotion_approval

    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, _, _ = _tool_description_candidate(repository, store, baseline)
    operator_private, operator_public = _key_pair()
    approve_promotion(
        store, "exp-td", operator="alice", rationale="ok",
        private_key_pem=operator_private, policy=PromotionPolicy(),
    )
    records = store.list_promotion_approvals("exp-td")
    keys = [load_public_key(operator_public, role="operator")]
    common = dict(
        experiment_id="exp-td", candidate_id="cand-td", candidate_type=CandidateType.TOOL_DESCRIPTION,
        operator_public_keys=keys,
    )
    verify_promotion_approval(records, candidate_hash=candidate.content_hash, policy_version="1", **common)
    with pytest.raises(PolicyRejected):
        verify_promotion_approval(records, candidate_hash=candidate.content_hash, policy_version="2", **common)
    with pytest.raises(PolicyRejected):
        verify_promotion_approval(records, candidate_hash="0" * 64, policy_version="1", **common)
    forged = [{**records[0], "payload": {**records[0]["payload"], "rationale": "edited"}}]
    with pytest.raises(PolicyRejected):
        verify_promotion_approval(forged, candidate_hash=candidate.content_hash, policy_version="1", **common)

    with store._connect() as db, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE promotion_approvals SET policy_version='2'")
    with store._connect() as db, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("DELETE FROM promotion_approvals")
    payload = {**records[0]["payload"], "candidate_hash": "0" * 64}
    with pytest.raises(IntegrityError, match="immutable candidate record"):
        store.record_promotion_approval(payload, records[0]["signature"])


def test_auto_types_need_no_approval_and_unlock_is_config_only(tmp_path):
    import shutil
    from pathlib import Path

    from hermes_improve.approvals import approve_promotion
    from hermes_improve.config import load_config

    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    with pytest.raises(ValueError, match="no approval is needed"):
        approve_promotion(
            store, "exp-1", operator="alice", rationale="not needed",
            private_key_pem=_key_pair()[0], policy=PromotionPolicy(),
        )

    config_dir = tmp_path / "config"
    shutil.copytree(Path(__file__).parents[2] / "config", config_dir)
    for item in [config_dir, *config_dir.iterdir()]:
        os.chmod(item, 0o755 if item.is_dir() else 0o644)
    text = (config_dir / "promotion.yaml").read_text()
    (config_dir / "promotion.yaml").write_text(
        text.replace("TOOL_DESCRIPTION: manual", "TOOL_DESCRIPTION: auto").replace(
            'policy_version: "1"', 'policy_version: "2"'),
    )
    config = load_config(config_dir)
    unlocked = PromotionPolicy.from_config(config.promotion)
    second = tmp_path / "second"
    second.mkdir()
    repository2, baseline2 = _repository(second)
    store2 = ExperimentStore(second / "db.sqlite3")
    _, _, public2 = _tool_description_candidate(repository2, store2, baseline2, policy=unlocked)
    manifest = GitPromoter(
        store2, repository2, second / "releases", second / "current", unlocked, public2,
        config_digest=config.digest,
    ).promote("exp-td", manifest_key=b"m" * 32)
    assert manifest["promotion_gate"] == "auto"
    assert manifest["policy_version"] == "2"
    assert manifest["config_digest"] == config.digest

    _, _, public = _add_candidate(repository, store, "exp-x", "cand-x", baseline, "x\n")
    with pytest.raises(PolicyRejected, match="policy version"):
        _promoter(store, repository, tmp_path, public, policy=unlocked).promote(
            "exp-x", manifest_key=b"m" * 32,
        )


def test_concurrent_promotions_one_wins_and_state_stays_consistent(tmp_path):
    import threading

    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-a", "cand-a", baseline, "a\n")
    second, _, _ = _add_candidate(
        repository, store, "exp-b", "cand-b", baseline, "b\n", signing_keys=(private, public),
    )
    barrier = threading.Barrier(2)
    outcomes = {}

    def run(experiment_id):
        promoter = _promoter(store, repository, tmp_path, public)
        barrier.wait()
        try:
            outcomes[experiment_id] = promoter.promote(experiment_id, manifest_key=b"m" * 32)
        except Exception as exc:  # recorded for the assertions below
            outcomes[experiment_id] = exc

    threads = [threading.Thread(target=run, args=(name,)) for name in ("exp-a", "exp-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = {name: value for name, value in outcomes.items() if isinstance(value, dict)}
    losers = {name: value for name, value in outcomes.items() if isinstance(value, Exception)}
    assert len(winners) == 1 and len(losers) == 1
    (winner, manifest), = winners.items()
    (_, error), = losers.items()
    assert isinstance(error, IntegrityError) and "production Git ref has changed" in str(error)
    winning_commit = first.commit if winner == "exp-a" else second.commit
    assert manifest["release_target"] == "brain-v1"
    assert git(repository, "rev-parse", "main") == winning_commit
    assert git(repository, "tag", "--list", "brain-v*").split() == ["brain-v1"]
    assert [row["id"] for row in store.list_releases()] == ["brain-v1"]
    expected_text = "a\n" if winner == "exp-a" else "b\n"
    assert (tmp_path / "brain-current" / "brain/skills/demo/SKILL.md").read_text() == expected_text
    leftovers = sorted(path.name for path in (tmp_path / "releases").iterdir())
    assert leftovers == [".promotion.lock", "brain-v1", "brain-v1.manifest.json"]


def test_release_status_transitions_are_compare_and_set_and_audited(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    store.transition_release_status(
        "brain-v1", "STABLE", expected_status="PROVISIONAL", reason="enough clean observations",
    )
    assert store.get_release("brain-v1")["status"] == "STABLE"
    with pytest.raises(ValueError, match="not allowed"):
        store.transition_release_status("brain-v1", "PROVISIONAL", expected_status="STABLE", reason="x")
    with pytest.raises(ValueError, match="not in status"):
        store.transition_release_status("brain-v1", "STABLE", expected_status="PROVISIONAL", reason="x")
    with pytest.raises(KeyError):
        store.transition_release_status("brain-v9", "STABLE", expected_status="PROVISIONAL", reason="x")
    last = store.list_release_events("brain-v1")[-1]
    assert (last["event"], last["from_status"], last["to_status"]) == ("STATUS_CHANGED", "PROVISIONAL", "STABLE")
    assert last["detail"]["reason"] == "enough clean observations"


def test_cli_approve_and_promote_by_experiment_id(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from hermes_improve.cli import main

    repository, baseline = _repository(tmp_path)
    home = tmp_path / "state"
    store = ExperimentStore(home / "experiments.sqlite3")
    candidate, _, evaluator_public = _tool_description_candidate(repository, store, baseline)
    operator_private, operator_public = _key_pair()
    (tmp_path / "operator.pem").write_bytes(operator_private)
    (tmp_path / "operator.pub").write_bytes(operator_public)
    (tmp_path / "evaluator.pub").write_bytes(evaluator_public)
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    # Pin the layout explicitly: on a provisioned host the default
    # /srv/hermes-improve layout would otherwise be verified (finding F4).
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "absent-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.setenv("HERMES_OPERATOR_PRIVATE_KEY", str(tmp_path / "operator.pem"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "approve", "exp-td", "--operator", "alice", "--rationale", "reviewed",
    ])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["candidate_hash"] == candidate.content_hash

    monkeypatch.setenv("HERMES_PROMOTER_UID", str(os.geteuid()))
    monkeypatch.delenv("HERMES_EVALUATOR_UID", raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_MANIFEST_KEY", "k" * 32)
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "promote", "exp-td",
        "--repository", str(repository),
        "--release-root", str(tmp_path / "releases"),
        "--deployment-link", str(tmp_path / "brain-current"),
        "--evaluator-public-key", str(tmp_path / "evaluator.pub"),
        "--operator-public-key", str(tmp_path / "operator.pub"),
    ])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["release_target"] == "brain-v1"
    assert result["approval"]["operator"] == "alice"
    assert git(repository, "rev-parse", "brain-v1") == candidate.commit

    monkeypatch.setattr(sys, "argv", ["hermes-improve", "releases", "events"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)[0]["event"] == "PROMOTED"
