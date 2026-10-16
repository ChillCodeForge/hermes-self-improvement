from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_improve.candidate_builder import create_skill_candidate
from hermes_improve.models import CandidateType
from hermes_improve.store import ExperimentStore


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository(tmp_path, *, existing_skill=False):
    repo = tmp_path / "skills-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    if existing_skill:
        skill = repo / "brain/skills/sample"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("original skill\n", encoding="utf-8")
    else:
        (repo / "README.txt").write_text("baseline\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    return repo


def make_candidate(tmp_path, *, candidate_type, existing_skill=False, **kwargs):
    repo = repository(tmp_path, existing_skill=existing_skill)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = create_skill_candidate(
        repo,
        store,
        experiment_id="experiment-1",
        candidate_id="candidate-1",
        candidate_type=candidate_type,
        target="brain/skills/sample",
        content="---\nname: sample\ndescription: test\n---\n\nUse this skill.",
        hypothesis="Add a deterministic sample skill",
        benchmark_version="bench-v1",
        **kwargs,
    )
    return repo, store, candidate


def test_skill_candidate_creates_isolated_commit_and_atomic_store_record(tmp_path):
    repo, store, candidate = make_candidate(
        tmp_path, candidate_type=CandidateType.SKILL_CREATE,
    )
    assert git(repo, "rev-parse", "main") == candidate.base_commit
    assert git(repo, "rev-parse", "candidate/experiment-1") == candidate.commit
    assert candidate.commit != candidate.base_commit
    assert "SKILL.md" in candidate.diff
    assert store.get_candidate("candidate-1")["content_hash"] == candidate.content_hash
    assert store.show_experiment("experiment-1")["candidates"][0]["id"] == "candidate-1"
    assert "candidate/experiment-1" not in git(repo, "worktree", "list", "--porcelain")


def test_skill_candidate_patch_changes_existing_skill(tmp_path):
    repo, _store, candidate = make_candidate(
        tmp_path, candidate_type=CandidateType.SKILL_PATCH, existing_skill=True,
    )
    output = git(repo, "show", f"{candidate.commit}:brain/skills/sample/SKILL.md")
    assert "Use this skill." in output
    assert "original skill" not in output


@pytest.mark.parametrize(
    ("candidate_type", "target", "content", "message"),
    [
        (CandidateType.SKILL_CREATE, "brain/skills/../evaluator", "safe", "unprotected"),
        (CandidateType.SKILL_CREATE, r"brain/skills/sample\..\security", "safe", "unprotected"),
        (CandidateType.SKILL_CREATE, "brain/skills/.git", "safe", "unprotected"),
        (CandidateType.SKILL_CREATE, "brain/skills/SKILL.md", "safe", "unprotected"),
        (CandidateType.SKILL_CREATE, "brain//skills/sample", "safe", "unprotected"),
        (CandidateType.SKILL_CREATE, "brain/skills/sample", "api_key=not-a-real-secret-value", "credential"),
        (CandidateType.SKILL_PATCH, "brain/skills/sample", "content", "does not exist"),
    ],
)
def test_invalid_skill_candidates_leave_repository_unchanged(
    tmp_path, candidate_type, target, content, message,
):
    repo = repository(tmp_path)
    base = git(repo, "rev-parse", "main")
    with pytest.raises(ValueError, match=message):
        create_skill_candidate(
            repo,
            ExperimentStore(tmp_path / "state.sqlite3"),
            experiment_id="invalid-1",
            candidate_id="candidate-invalid",
            candidate_type=candidate_type,
            target=target,
            content=content,
            hypothesis="A test candidate",
            benchmark_version="bench-v1",
        )
    assert git(repo, "rev-parse", "main") == base
    assert subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", "refs/heads/candidate/invalid-1"]
    ).returncode != 0


def test_skill_candidate_requires_explicit_patch_for_existing_skill(tmp_path):
    repo = repository(tmp_path, existing_skill=True)
    with pytest.raises(ValueError, match="already exists"):
        create_skill_candidate(
            repo,
            ExperimentStore(tmp_path / "state.sqlite3"),
            experiment_id="create-existing",
            candidate_id="candidate-existing",
            candidate_type=CandidateType.SKILL_CREATE,
            target="brain/skills/sample",
            content="A replacement is not an implicit create",
            hypothesis="Avoid implicit overwrite",
            benchmark_version="bench-v1",
        )


@pytest.mark.parametrize(
    ("hypothesis", "source_sessions", "message"),
    [
        ("Use ghp_" + "A" * 24, (), "hypothesis"),
        ("A safe hypothesis", ("api_key=not-a-real-secret",), "source session"),
    ],
)
def test_candidate_metadata_rejects_credentials(tmp_path, hypothesis, source_sessions, message):
    repo = repository(tmp_path)
    base_commit = git(repo, "rev-parse", "main")
    with pytest.raises(ValueError, match=message):
        create_skill_candidate(
            repo,
            ExperimentStore(tmp_path / "state.sqlite3"),
            experiment_id="metadata-secret",
            candidate_id="metadata-candidate",
            candidate_type=CandidateType.SKILL_CREATE,
            target="brain/skills/sample",
            content="A new skill",
            hypothesis=hypothesis,
            benchmark_version="bench-v1",
            source_sessions=source_sessions,
        )
    assert git(repo, "rev-parse", "main") == base_commit
    assert subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet",
         "refs/heads/candidate/metadata-secret"]
    ).returncode != 0


def test_store_conflict_rolls_back_candidate_record_and_git_branch(tmp_path):
    repo = repository(tmp_path)
    base_commit = git(repo, "rev-parse", "main")
    store = ExperimentStore(tmp_path / "state.sqlite3")
    store.create_experiment(
        "conflict-experiment",
        lineage_id="existing-lineage",
        hypothesis="Existing experiment",
        candidate_type="SKILL_CREATE",
        target="brain/skills/sample",
        risk="LOW",
        baseline_commit=base_commit,
        benchmark_version="bench-v1",
        policy_version="1",
    )

    with pytest.raises(sqlite3.IntegrityError):
        create_skill_candidate(
            repo,
            store,
            experiment_id="conflict-experiment",
            candidate_id="candidate-conflict",
            candidate_type=CandidateType.SKILL_CREATE,
            target="brain/skills/sample",
            content="A new skill",
            hypothesis="A conflicting candidate",
            benchmark_version="bench-v1",
        )

    with pytest.raises(KeyError):
        store.get_candidate("candidate-conflict")
    assert git(repo, "rev-parse", "main") == base_commit
    assert subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet",
         "refs/heads/candidate/conflict-experiment"]
    ).returncode != 0
    assert "conflict-experiment" in {
        item["id"] for item in store.list_experiments()
    }


def test_candidate_create_cli_materializes_commit_and_reports_hash(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from hermes_improve.cli import main

    repo = repository(tmp_path)
    home = tmp_path / "state"
    content = tmp_path / "SKILL.md"
    content.write_text("---\nname: sample\n---\nA curated skill.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv(
        "HERMES_IMPROVE_BENCHMARKS",
        str(Path(__file__).parents[1] / "benchmarks"),
    )
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "candidate", "create",
        "--repository", str(repo),
        "--experiment-id", "cli-experiment",
        "--candidate-id", "cli-candidate",
        "--type", "SKILL_CREATE",
        "--target", "brain/skills/sample",
        "--content", str(content),
        "--hypothesis", "Create the sample skill",
    ])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "awaiting_evaluation"
    assert result["candidate_hash"]
    assert git(repo, "rev-parse", "main") == result["base_commit"]
