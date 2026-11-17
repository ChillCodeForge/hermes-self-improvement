"""Spec §38 topics that were only weakly covered: precise regression blocking, manifest and Git
state mismatches at promotion time, atomic deployment/rollback on failure, read-only releases."""
from __future__ import annotations

import os
import stat

import pytest

from hermes_improve.errors import IntegrityError, PolicyRejected
from hermes_improve.evaluation_signing import sign_report
from hermes_improve.models import Split, TaskOutcome
from hermes_improve.policy import PromotionPolicy
from hermes_improve.store import ExperimentStore
from test_promotion import _add_candidate, _outcomes, _promoter, _record_run, _repository, git


def _final(**overrides):
    base = dict(
        task_id="full-1", split=Split.REGRESSION, suite="FULL", baseline_success=True, candidate_success=True,
        baseline_quality=5, candidate_quality=10, baseline_tokens=100, candidate_tokens=100, token_usage_reported=True,
        categories=("skills",), baseline_latency_ms=100, candidate_latency_ms=100,
    )
    holdout = dict(base, task_id="holdout-1", split=Split.SEALED_HOLDOUT, suite="SEALED_HOLDOUT")
    return [TaskOutcome(**{**base, **overrides}), TaskOutcome(**holdout)]


def test_clean_final_evidence_passes_so_the_blocking_tests_below_are_meaningful():
    report = PromotionPolicy().assess("c", "h", "v1", _final())
    assert report.passed, report.reasons


@pytest.mark.parametrize("field, reason", [
    ("critical_regressions", "critical regression detected"),
    ("high_regressions", "high regression detected"),
])
def test_single_critical_or_high_regression_blocks_despite_quality_gain(field, reason):
    report = PromotionPolicy().assess("c", "h", "v1", _final(**{field: 1}))
    assert not report.passed
    assert reason in report.reasons
    # With the regression allowance raised only for this severity the same evidence passes:
    # the block comes from the regression gate, not from some other failing gate.
    relaxed = PromotionPolicy(**{f"max_{field}": 1})
    assert relaxed.assess("c", "h", "v1", _final(**{field: 1})).passed


@pytest.mark.parametrize("field, label", [("critical_regressions", "critical"), ("high_regressions", "high")])
def test_promoter_blocks_regression_in_earlier_signed_suite_evidence(tmp_path, field, label):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n", evaluate=False,
    )
    _record_run(store, candidate, private, "eval-smoke", ("SMOKE",))
    _record_run(store, candidate, private, "eval-core", ("CORE",), CORE={field: 1})
    _record_run(store, candidate, private, "eval-final", ("FULL", "SEALED_HOLDOUT"))
    with pytest.raises(PolicyRejected, match=f"{label} regression in CORE evidence"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    assert git(repository, "rev-parse", "main") == baseline
    assert not os.path.lexists(tmp_path / "brain-current")


def test_benchmark_manifest_mismatch_blocks_promotion(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n", evaluate=False,
    )
    # The experiment was created against bench-v1; this evaluation used another manifest.
    report = PromotionPolicy().assess(
        candidate.candidate_id, candidate.content_hash, "bench-v2", _outcomes(candidate.candidate_id, (
            "SMOKE", "CORE", "FULL", "SEALED_HOLDOUT",
        )),
    )
    assert report.passed
    store.record_evaluation("eval-v2", candidate.candidate_id, report, signature=sign_report(report, private))
    with pytest.raises(IntegrityError, match="benchmark version differs from experiment manifest"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    assert git(repository, "rev-parse", "main") == baseline
    assert store.list_releases() == []


def test_git_state_mismatch_production_moved_after_evaluation_blocks_promotion(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _, _, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    git(repository, "checkout", "-q", "main")
    (repository / "brain" / "skills" / "demo" / "NOTES.md").write_text("hotfix\n")
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "out-of-band production change")
    moved = git(repository, "rev-parse", "main")
    with pytest.raises(IntegrityError, match="production Git ref has changed since evaluation"):
        _promoter(store, repository, tmp_path, public).promote("exp-1", manifest_key=b"m" * 32)
    assert git(repository, "rev-parse", "main") == moved
    assert git(repository, "tag", "--list") == ""
    assert store.list_releases() == []


def _release_state(repository, tmp_path):
    link = tmp_path / "brain-current"
    return (
        git(repository, "rev-parse", "main"),
        git(repository, "tag", "--list"),
        os.readlink(link) if link.is_symlink() else None,
        sorted(path.name for path in (tmp_path / "releases").iterdir()),
    )


def test_atomic_deployment_failure_after_switch_restores_everything(tmp_path, monkeypatch):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    promoter = _promoter(store, repository, tmp_path, public)
    promoter.promote("exp-1", manifest_key=b"m" * 32)
    git(repository, "checkout", "-q", "main")
    _add_candidate(repository, store, "exp-2", "cand-2", first.commit, "two\n", signing_keys=(private, public))
    before = _release_state(repository, tmp_path)

    def crash(*args, **kwargs):
        # The deployment pointer has already switched to brain-v2 at this point.
        assert os.readlink(tmp_path / "brain-current").endswith("brain-v2")
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(store, "record_promotion_release", crash)
    with pytest.raises(RuntimeError, match="ledger unavailable"):
        promoter.promote("exp-2", manifest_key=b"m" * 32)
    assert _release_state(repository, tmp_path) == before
    assert (tmp_path / "brain-current" / "brain/skills/demo/SKILL.md").read_text() == "one\n"
    assert [row["id"] for row in store.list_releases()] == ["brain-v1"]


def test_atomic_rollback_failure_restores_main_and_pointer(tmp_path, monkeypatch):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    promoter = _promoter(store, repository, tmp_path, public)
    promoter.promote("exp-1", manifest_key=b"m" * 32)
    git(repository, "checkout", "-q", "main")
    _add_candidate(repository, store, "exp-2", "cand-2", first.commit, "two\n", signing_keys=(private, public))
    promoter.promote("exp-2", manifest_key=b"m" * 32)
    before = _release_state(repository, tmp_path)
    monkeypatch.setattr(store, "record_rollback", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("audit down")))
    with pytest.raises(RuntimeError, match="audit down"):
        promoter.rollback("brain-v1", manifest_key=b"m" * 32)
    assert _release_state(repository, tmp_path) == before
    assert store.get_release("brain-v2")["status"] == "PROVISIONAL"


def test_every_deployed_release_entry_is_read_only(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "one\n",
        extra_files={"brain/skills/demo/refs/notes.md": "nested\n"},
    )
    promoter = _promoter(store, repository, tmp_path, public)
    promoter.promote("exp-1", manifest_key=b"m" * 32)
    git(repository, "checkout", "-q", "main")
    _add_candidate(repository, store, "exp-2", "cand-2", first.commit, "two\n", signing_keys=(private, public))
    promoter.promote("exp-2", manifest_key=b"m" * 32)
    promoter.rollback("brain-v1", manifest_key=b"m" * 32)

    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for release in ("brain-v1", "brain-v2"):
        root = tmp_path / "releases" / release
        entries = [root]
        for directory, subdirs, files in os.walk(root):
            entries += [os.path.join(directory, name) for name in [*subdirs, *files]]
        assert len(entries) > 4
        for entry in entries:
            assert not os.path.islink(entry), entry
            assert os.lstat(entry).st_mode & write_bits == 0, entry
        manifest = tmp_path / "releases" / f"{release}.manifest.json"
        assert manifest.stat().st_mode & write_bits == 0
    assert os.readlink(tmp_path / "brain-current").endswith("brain-v1")
    if os.geteuid() != 0:  # root bypasses mode bits; the OS-isolation tests cover the role users
        with pytest.raises(PermissionError):
            (tmp_path / "brain-current" / "brain/skills/demo/SKILL.md").write_text("tampered\n")
        with pytest.raises(PermissionError):
            (tmp_path / "brain-current" / "brain/skills/demo/new.md").write_text("added\n")
