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


def _add_candidate(
    repository, store, experiment_id, candidate_id, parent, content, *,
    approved=True, signing_keys=None,
):
    git(repository, "checkout", "-b", f"candidate/{experiment_id}", parent)
    skill = repository / "brain" / "skills" / "demo" / "SKILL.md"
    skill.write_text(content, encoding="utf-8")
    git(repository, "add", "brain/skills/demo/SKILL.md")
    git(repository, "commit", "-m", f"candidate {experiment_id}")
    commit = git(repository, "rev-parse", "HEAD")
    diff = subprocess.run(
        ["git", "-C", str(repository), "diff", "--binary", parent, commit],
        check=True, stdout=subprocess.PIPE,
    ).stdout.decode()
    store.create_experiment(
        experiment_id, lineage_id="lineage-1", hypothesis="improve the skill",
        candidate_type="SKILL_PATCH", target="brain/skills/demo",
        risk="LOW", baseline_commit=parent, benchmark_version="bench-v1",
        policy_version="1",
    )
    candidate = Candidate(
        candidate_id, experiment_id, CandidateType.SKILL_PATCH,
        "brain/skills/demo", parent, commit, diff,
    )
    store.add_candidate(candidate)
    outcomes = (
        TaskOutcome(
            f"{candidate_id}-full", Split.REGRESSION, "FULL", True, True, 5, 10,
            100, 100, categories=("skills",), baseline_latency_ms=100, candidate_latency_ms=100,
            baseline_tool_calls=5, candidate_tool_calls=5,
        ),
        TaskOutcome(
            f"{candidate_id}-holdout", Split.SEALED_HOLDOUT, "SEALED_HOLDOUT", True, True, 5, 10,
            100, 100, categories=("skills",), baseline_latency_ms=100, candidate_latency_ms=100,
            baseline_tool_calls=5, candidate_tool_calls=5,
        ),
    )
    report = PromotionPolicy().assess(candidate_id, candidate.content_hash, "bench-v1", outcomes)
    if not approved:
        report = PromotionPolicy(min_quality_delta=99).assess(
            candidate_id, candidate.content_hash, "bench-v1", outcomes
        )
    assert report.passed is approved
    private, public = signing_keys or _key_pair()
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
