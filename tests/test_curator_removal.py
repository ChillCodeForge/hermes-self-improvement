"""Curator archive / delete / consolidate become gated removal candidates.

Removal is expressed with existing candidate types: ``consolidate`` (and a
staged ``delete`` with ``absorbed_into`` plus new umbrella content) is a
``SKILL_MERGE``; ``archive`` and every other staged ``delete`` is a
``SKILL_PATCH`` that deletes all tracked files of the skill. Any candidate whose
diff deletes a ``brain/skills/<name>/SKILL.md`` needs a signed operator approval.
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys

import pytest

from hermes_curator_fixture import build_hermes_home, write_pending
from hermes_improve import cli
from hermes_improve.approvals import approve_promotion
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.config import default_config
from hermes_improve.curator import CuratorAdapter, CuratorMutation, CuratorMutationRefused
from hermes_improve.curator_connector import CuratorConnector
from hermes_improve.errors import PolicyRejected
from hermes_improve.models import CandidateType, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.promotion import GitPromoter
from hermes_improve.store import ExperimentStore
from hermes_improve.target_paths import removed_skill_units
from test_evaluate_promote_integration import _benchmark
from test_promotion import ALL_SUITES, _key_pair, _record_run

DEMO = "---\nname: demo\n---\nStep one.\n"
UMBRELLA = "---\nname: umbrella\n---\nUmbrella steps.\n"
MERGED = "---\nname: umbrella\n---\nUmbrella steps.\nDemo steps.\n"


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()


def tree(repository, ref):
    return git(repository, "ls-tree", "-r", "--name-only", ref).split()


@pytest.fixture
def brain(tmp_path):
    repository = tmp_path / "brain-repo"
    repository.mkdir()
    git(repository, "init", "-q", "-b", "main")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.invalid")
    skills = repository / "brain" / "skills"
    (skills / "demo" / "references").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text(DEMO, encoding="utf-8")
    (skills / "demo" / "references" / "notes.md").write_text("Demo notes.\n", encoding="utf-8")
    (skills / "umbrella").mkdir()
    (skills / "umbrella" / "SKILL.md").write_text(UMBRELLA, encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-q", "-m", "baseline")
    return repository


def _adapter(brain, tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    return CuratorAdapter(repository=brain, store=store, benchmark_version="bench-v1"), store


def _production(repository):
    return git(repository, "rev-parse", "main"), git(repository, "status", "--porcelain")


def _pending(tmp_path, pending_id, payload):
    home = tmp_path / "hermes-home"
    if not home.exists():
        build_hermes_home(tmp_path)
    write_pending(home / "pending" / "skills", pending_id, payload)
    return CuratorConnector(home).pending_write(pending_id)


# --- conversion -----------------------------------------------------------------


def test_consolidate_writes_umbrella_and_removes_every_absorbed_file(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    before = _production(brain)
    conversion = adapter.to_candidate(
        CuratorMutation("consolidate", "demo", "overlapping skills", content=MERGED, into="umbrella"),
        experiment_id="exp-merge", candidate_id="cand-merge",
    )
    candidate = conversion.candidate
    assert _production(brain) == before
    assert candidate.candidate_type is CandidateType.SKILL_MERGE
    assert candidate.target == "brain/skills/demo,brain/skills/umbrella"
    assert conversion.unapplied == ()
    assert tree(brain, candidate.commit) == ["brain/skills/umbrella/SKILL.md"]
    assert git(brain, "show", f"{candidate.commit}:brain/skills/umbrella/SKILL.md") + "\n" == MERGED
    experiment = store.show_experiment("exp-merge")
    assert experiment["risk"] == "HIGH"
    assert experiment["hypothesis"].startswith("Curator consolidate demo into umbrella")


def test_consolidate_without_new_umbrella_content_is_a_pure_removal(brain, tmp_path):
    adapter, _ = _adapter(brain, tmp_path)
    for experiment_id, content in (("exp-none", None), ("exp-same", UMBRELLA)):
        conversion = adapter.to_candidate(
            CuratorMutation("consolidate", "demo", "already merged", content=content, into="umbrella"),
            experiment_id=experiment_id, candidate_id=f"cand-{experiment_id}",
        )
        assert conversion.candidate.candidate_type is CandidateType.SKILL_PATCH
        assert conversion.candidate.target == "brain/skills/demo"
        assert tree(brain, conversion.candidate.commit) == ["brain/skills/umbrella/SKILL.md"]


def test_staged_delete_with_absorbed_into_and_content_becomes_a_skill_merge(brain, tmp_path):
    adapter, _ = _adapter(brain, tmp_path)
    pending = _pending(tmp_path, "del00001", {
        "action": "delete", "name": "demo", "absorbed_into": "umbrella", "content": MERGED,
    })
    conversion = adapter.convert_pending_write(pending, experiment_id="exp-del-1", candidate_id="cand-del-1")
    assert conversion.mutation_kind == "consolidate"
    assert conversion.candidate.candidate_type is CandidateType.SKILL_MERGE
    assert conversion.candidate.target == "brain/skills/demo,brain/skills/umbrella"
    assert tree(brain, conversion.candidate.commit) == ["brain/skills/umbrella/SKILL.md"]


def test_staged_delete_with_absorbed_into_keeps_the_production_umbrella(brain, tmp_path):
    adapter, _ = _adapter(brain, tmp_path)
    pending = _pending(tmp_path, "del00002", {"action": "delete", "name": "demo", "absorbed_into": "umbrella"})
    conversion = adapter.convert_pending_write(pending, experiment_id="exp-del-2", candidate_id="cand-del-2")
    commit = conversion.candidate.commit
    assert conversion.candidate.candidate_type is CandidateType.SKILL_PATCH
    assert conversion.candidate.target == "brain/skills/demo"
    assert git(brain, "show", f"{commit}:brain/skills/umbrella/SKILL.md") + "\n" == UMBRELLA
    assert tree(brain, commit) == ["brain/skills/umbrella/SKILL.md"]


def test_staged_delete_without_absorbed_into_archives(brain, tmp_path):
    adapter, _ = _adapter(brain, tmp_path)
    pending = _pending(tmp_path, "del00003", {"action": "delete", "name": "demo"})
    conversion = adapter.convert_pending_write(pending, experiment_id="exp-del-3", candidate_id="cand-del-3")
    assert conversion.mutation_kind == "archive"
    assert conversion.candidate.candidate_type is CandidateType.SKILL_PATCH
    assert tree(brain, conversion.candidate.commit) == ["brain/skills/umbrella/SKILL.md"]


def test_archive_removes_every_tracked_file_of_the_skill(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    before = _production(brain)
    conversion = adapter.to_candidate(
        CuratorMutation("archive", "demo", "idle for 30 days"),
        experiment_id="exp-archive", candidate_id="cand-archive",
    )
    candidate = conversion.candidate
    assert _production(brain) == before
    assert (brain / "brain" / "skills" / "demo" / "SKILL.md").read_text() == DEMO
    assert candidate.candidate_type is CandidateType.SKILL_PATCH
    assert candidate.target == "brain/skills/demo"
    assert git(brain, "diff", "--name-status", "--no-renames", candidate.base_commit, candidate.commit).splitlines() == [
        "D\tbrain/skills/demo/SKILL.md", "D\tbrain/skills/demo/references/notes.md",
    ]
    assert store.show_experiment("exp-archive")["risk"] == "HIGH"
    with pytest.raises(ValueError, match="neither"):
        adapter.to_candidate(
            CuratorMutation("archive", "umbrella", "x", into="demo"), experiment_id="exp-bad", candidate_id="cand-bad",
        )


def test_removal_is_refused_when_the_skill_is_not_on_main(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    attempts = [
        CuratorMutation("archive", "ghost", "x"),
        CuratorMutation("archive", "Demo", "case variant of an existing skill"),
        CuratorMutation("consolidate", "ghost", "x", content=MERGED, into="umbrella"),
        CuratorMutation("consolidate", "demo", "x", content=MERGED, into="ghost"),
    ]
    for index, mutation in enumerate(attempts):
        with pytest.raises(CuratorMutationRefused):
            adapter.to_candidate(mutation, experiment_id=f"exp-r{index}", candidate_id=f"cand-r{index}")
    with pytest.raises(CuratorMutationRefused):
        adapter.convert_pending_write(
            _pending(tmp_path, "del00004", {"action": "delete", "name": "ghost"}),
            experiment_id="exp-r-pending", candidate_id="cand-r-pending",
        )
    with pytest.raises(ValueError, match="consolidated into itself"):
        adapter.to_candidate(
            CuratorMutation("consolidate", "demo", "x", into="demo"), experiment_id="exp-self", candidate_id="cand-self",
        )
    assert store.list_candidates() == []
    assert git(brain, "branch", "--list", "candidate/*") == ""


def test_removal_refuses_skills_with_symlinks_or_unsafe_files(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    skills = brain / "brain" / "skills"
    (skills / "linked").mkdir()
    (skills / "linked" / "SKILL.md").write_text("---\nname: linked\n---\nx\n", encoding="utf-8")
    os.symlink("../../../README", skills / "linked" / "escape.md")
    (skills / "scripted").mkdir()
    (skills / "scripted" / "SKILL.md").write_text("---\nname: scripted\n---\nx\n", encoding="utf-8")
    (skills / "scripted" / "run.sh").write_text("echo hi\n", encoding="utf-8")
    git(brain, "add", ".")
    git(brain, "commit", "-q", "-m", "odd skills")
    with pytest.raises(CuratorMutationRefused, match="non-regular"):
        adapter.to_candidate(CuratorMutation("archive", "linked", "x"), experiment_id="exp-l", candidate_id="cand-l")
    with pytest.raises(ValueError, match="must end in one of"):
        adapter.to_candidate(CuratorMutation("archive", "scripted", "x"), experiment_id="exp-s", candidate_id="cand-s")
    assert store.list_candidates() == []
    assert git(brain, "branch", "--list", "candidate/*") == ""


# --- promotion gate -------------------------------------------------------------


def test_skill_removal_is_never_auto_promotable():
    policy = PromotionPolicy()
    assert CandidateType.SKILL_PATCH in policy.auto_promotable_types
    for kind in CandidateType:
        assert policy.requires_manual_approval(kind, removed_skills=("brain/skills/demo",))
    assert not policy.requires_manual_approval(CandidateType.SKILL_PATCH)
    assert removed_skill_units([
        "brain/skills/demo/SKILL.md", "brain/skills/demo/references/notes.md",
        "brain/skills/other/references/old.md", "brain/skills/odd/skill.MD", "brain/prompts/SKILL.md",
    ]) == ("brain/skills/demo", "brain/skills/odd")


def test_removal_hidden_behind_a_rename_still_needs_approval(brain, tmp_path):
    """A merge whose umbrella text equals the absorbed skill shows up as a rename
    in a default diff; the gate uses --no-renames and still sees the deletion."""
    from hermes_improve.candidate_builder import removed_skills_in_candidate

    git(brain, "checkout", "-q", "-b", "candidate/exp-rename")
    git(brain, "rm", "-q", "-r", "brain/skills/umbrella")
    (brain / "brain" / "skills" / "fresh").mkdir()
    git(brain, "mv", "brain/skills/demo/SKILL.md", "brain/skills/fresh/SKILL.md")
    git(brain, "commit", "-q", "-m", "rename")
    assert "R" in git(brain, "diff", "-M", "--name-status", "main", "HEAD")
    removed = removed_skills_in_candidate(brain, git(brain, "rev-parse", "main"), git(brain, "rev-parse", "HEAD"))
    assert removed == ("brain/skills/demo", "brain/skills/umbrella")


def _evaluated(adapter, store, mutation, experiment_id, signing_key):
    candidate = adapter.to_candidate(
        mutation, experiment_id=experiment_id, candidate_id=f"cand-{experiment_id}",
    ).candidate
    for index, suites in enumerate((("SMOKE",), ("CORE",), ("FULL", "SEALED_HOLDOUT"))):
        _record_run(store, candidate, signing_key, f"eval-{experiment_id}-{index}", suites)
    assert store.show_experiment(experiment_id)["decision"] == "passed"
    return candidate


def test_removal_candidate_needs_operator_approval_and_rollback_restores_the_skill(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    evaluator_private, evaluator_public = _key_pair()
    operator_private, operator_public = _key_pair()
    current = tmp_path / "brain-current"
    promoter = GitPromoter(
        store, brain, tmp_path / "releases", current, PromotionPolicy(), evaluator_public,
        operator_public_keys=[operator_public],
    )

    first = _evaluated(
        adapter, store, CuratorMutation("patch", "umbrella", "clarify", content=MERGED), "exp-v1", evaluator_private,
    )
    with pytest.raises(ValueError, match="no approval is needed"):
        approve_promotion(
            store, "exp-v1", operator="alice", rationale="not a removal",
            private_key_pem=operator_private, policy=PromotionPolicy(), repository=brain,
        )
    manifest = promoter.promote("exp-v1", release_name="brain-v1", manifest_key=b"m" * 32)
    assert manifest["promotion_gate"] == "auto" and manifest["removed_skills"] == []
    assert (current / "brain/skills/demo/SKILL.md").read_text() == DEMO

    archive = _evaluated(
        adapter, store, CuratorMutation("archive", "demo", "idle"), "exp-v2", evaluator_private,
    )
    assert archive.base_commit == first.commit
    with pytest.raises(PolicyRejected, match="requires a signed operator approval"):
        promoter.promote("exp-v2", release_name="brain-v2", manifest_key=b"m" * 32)
    assert git(brain, "rev-parse", "main") == first.commit
    with pytest.raises(ValueError, match="pass the candidate repository"):
        approve_promotion(
            store, "exp-v2", operator="alice", rationale="archive demo",
            private_key_pem=operator_private, policy=PromotionPolicy(),
        )
    approve_promotion(
        store, "exp-v2", operator="alice", rationale="Reviewed: demo is unused and its tasks still pass.",
        private_key_pem=operator_private, policy=PromotionPolicy(), repository=brain,
    )
    manifest = promoter.promote("exp-v2", release_name="brain-v2", manifest_key=b"m" * 32)
    assert manifest["promotion_gate"] == "manual"
    assert manifest["approval"]["operator"] == "alice"
    assert manifest["removed_skills"] == ["brain/skills/demo"]
    assert git(brain, "rev-parse", "main") == archive.commit
    assert not (current / "brain/skills/demo").exists()

    promoter.rollback("brain-v1", manifest_key=b"m" * 32)
    assert git(brain, "rev-parse", "main") == first.commit
    assert (current / "brain/skills/demo/SKILL.md").read_text() == DEMO
    assert (current / "brain/skills/demo/references/notes.md").read_text() == "Demo notes.\n"
    assert store.get_release("brain-v2")["status"] == "ROLLED_BACK"


def test_merge_candidate_is_manually_gated_even_though_skill_merge_is_auto(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    evaluator_private, evaluator_public = _key_pair()
    operator_private, operator_public = _key_pair()
    assert CandidateType.SKILL_MERGE in PromotionPolicy().auto_promotable_types
    _evaluated(
        adapter, store, CuratorMutation("consolidate", "demo", "overlap", content=MERGED, into="umbrella"),
        "exp-merge", evaluator_private,
    )
    promoter = GitPromoter(
        store, brain, tmp_path / "releases", tmp_path / "brain-current", PromotionPolicy(), evaluator_public,
        operator_public_keys=[operator_public],
    )
    with pytest.raises(PolicyRejected, match="requires a signed operator approval"):
        promoter.promote("exp-merge", manifest_key=b"m" * 32)
    approve_promotion(
        store, "exp-merge", operator="alice", rationale="merge reviewed",
        private_key_pem=operator_private, policy=PromotionPolicy(), repository=brain,
    )
    manifest = promoter.promote("exp-merge", manifest_key=b"m" * 32)
    assert manifest["promotion_gate"] == "manual"
    assert manifest["removed_skills"] == ["brain/skills/demo"]


# --- evaluation -----------------------------------------------------------------


class SkillDependentRunner:
    """Fails a task on a commit whose tree lacks a skill the task's impact areas name."""

    llm_influenced = False

    def __init__(self, repository):
        self.repository = repository
        self.calls = []

    def run(self, commit, benchmark, seed):
        self.calls.append((commit, benchmark.id))
        present = set(tree(self.repository, commit))
        areas = benchmark.benchmark.get("impact_areas") or []
        missing = any(
            fnmatch.fnmatchcase(f"skills/{name}", str(pattern)) and f"brain/skills/{name}/SKILL.md" not in present
            for pattern in areas for name in ("demo", "umbrella")
        )
        return RunObservation(
            task_id=benchmark.id, commit=commit, seed=seed, environment_digest="sha256:" + "e" * 64,
            success=not missing, quality=5, tokens=100, latency_ms=20,
            deterministic_pass=not missing, security_pass=True, tool_calls=2,
            verifier_results=({"type": "exit_code", "passed": not missing},),
        )


def test_benchmarks_for_a_removed_skill_still_run_and_surface_as_regressions(brain, tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    public_root = tmp_path / "benchmarks"
    holdout_root = tmp_path / "holdout"
    public_root.mkdir()
    holdout_root.mkdir()
    for identifier, suites, areas in (
        ("smoke-task", ["SMOKE"], ["skills/demo*"]),
        ("smoke-unrelated", ["SMOKE"], ["skills/umbrella*"]),
        ("core-task", ["CORE"], None),
        ("full-task", ["FULL"], ["skills/demo*"]),
    ):
        (public_root / f"{identifier}.json").write_text(json.dumps(_benchmark(identifier, "REGRESSION", suites, areas)))
    (holdout_root / "holdout-task.json").write_text(
        json.dumps(_benchmark("holdout-task", "SEALED_HOLDOUT", ["SEALED_HOLDOUT"]))
    )
    version = BenchmarkStore(public_root, holdout_root=holdout_root).version
    home = tmp_path / "home"
    home.mkdir()
    store = ExperimentStore(home / "experiments.sqlite3")
    adapter = CuratorAdapter(repository=brain, store=store, benchmark_version=version)
    candidate = adapter.to_candidate(
        CuratorMutation("archive", "demo", "idle"), experiment_id="exp-arch", candidate_id="cand-arch",
    ).candidate

    key_file = tmp_path / "evaluator.pem"
    key_file.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    runner = SkillDependentRunner(brain)
    monkeypatch.setattr(cli, "build_runner", lambda args, artifact_store: runner)
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public_root))
    monkeypatch.setenv("HERMES_EVALUATOR_PRIVATE_KEY", str(key_file))
    monkeypatch.setenv("HERMES_EVALUATOR_UID", str(os.geteuid()))
    assert default_config().promotion.policy_version == "1"

    for suite in ("SMOKE", "FULL"):
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "evaluate", "cand-arch", "--repository", str(brain),
            "--holdout-root", str(holdout_root), "--suite", suite,
        ])
        assert cli.main() == 0

    ran_on_candidate = {task for commit, task in runner.calls if commit == candidate.commit}
    assert {"smoke-task", "full-task", "holdout-task"} <= ran_on_candidate
    runs = store.candidate_evaluations("cand-arch")
    assert len(runs) == 2
    for run in runs:
        assert not run["evaluation"]["passed"]
        regressed = {
            item["result"]["task_id"] for item in run["task_results"]
            if item["result"]["baseline_success"] and not item["result"]["candidate_success"]
        }
        assert regressed & {"smoke-task", "full-task"}
    assert store.show_experiment("exp-arch")["decision"] != "passed"


def test_cli_approve_recognises_a_removal_candidate_with_repository(brain, tmp_path, monkeypatch, capsys):
    home = tmp_path / "state"
    store = ExperimentStore(home / "experiments.sqlite3")
    adapter = CuratorAdapter(repository=brain, store=store, benchmark_version="bench-v1")
    evaluator_private, _ = _key_pair()
    operator_private, _ = _key_pair()
    archive = _evaluated(adapter, store, CuratorMutation("archive", "demo", "idle"), "exp-cli", evaluator_private)
    (tmp_path / "operator.pem").write_bytes(operator_private)
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.setenv("HERMES_OPERATOR_PRIVATE_KEY", str(tmp_path / "operator.pem"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "approve", "exp-cli", "--operator", "alice", "--rationale", "archive reviewed",
        "--repository", str(brain),
    ])
    assert cli.main() == 0
    approval = json.loads(capsys.readouterr().out)
    assert approval["candidate_hash"] == archive.content_hash
    assert approval["candidate_type"] == "SKILL_PATCH"


def test_impact_scope_keeps_every_unit_of_a_merge_target():
    from hermes_improve.suites import impact_scope_for_candidate

    scope = impact_scope_for_candidate("SKILL_MERGE", "brain/skills/alpha,brain/skills/git-zeta", ("git",))
    assert scope.target_areas == {"skills/alpha", "skills/git-zeta"}
    assert scope.categories == {"git"}
