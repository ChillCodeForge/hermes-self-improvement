from __future__ import annotations

import json
import subprocess
import sys

import pytest

from hermes_curator_fixture import build_hermes_home, write_pending
from hermes_improve.cli import main
from hermes_improve.curator import CuratorAdapter, CuratorMutation, CuratorMutationRefused
from hermes_improve.curator_connector import CuratorConnector
from hermes_improve.models import CandidateType
from hermes_improve.store import ExperimentStore

ORIGINAL = "---\nname: demo\n---\nStep one.\nStep two.\n"


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()


@pytest.fixture
def brain(tmp_path):
    repository = tmp_path / "brain-repo"
    repository.mkdir()
    git(repository, "init", "-b", "main")
    git(repository, "config", "user.name", "Test")
    git(repository, "config", "user.email", "test@example.invalid")
    for name in ("demo", "umbrella"):
        skill = repository / "brain" / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(ORIGINAL.replace("demo", name), encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "baseline")
    return repository


def _adapter(brain, tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    return CuratorAdapter(repository=brain, store=store, benchmark_version="bench-v1"), store


def _production_state(repository):
    return (
        git(repository, "rev-parse", "main"),
        (repository / "brain" / "skills" / "demo" / "SKILL.md").read_text(),
        git(repository, "status", "--porcelain"),
    )


def test_curator_patch_becomes_candidate_and_never_touches_production(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    before = _production_state(brain)

    conversion = adapter.to_candidate(
        CuratorMutation("patch", "demo", "clarify step two", old_string="Step two.", new_string="Step two, verified."),
        experiment_id="exp-curator-1", candidate_id="cand-curator-1",
    )

    assert _production_state(brain) == before
    candidate = conversion.candidate
    assert candidate.candidate_type is CandidateType.SKILL_PATCH
    assert candidate.target == "brain/skills/demo"
    assert git(brain, "rev-parse", "candidate/exp-curator-1") == candidate.commit
    assert "Step two, verified." in git(brain, "show", f"{candidate.commit}:brain/skills/demo/SKILL.md")
    assert store.show_experiment("exp-curator-1")["hypothesis"].startswith("Curator patch demo")


def test_curator_direct_write_attempt_is_converted_not_applied(brain, tmp_path):
    """Curator bypass: a write Hermes staged for direct approval becomes a candidate instead."""
    home = build_hermes_home(tmp_path)
    adapter, store = _adapter(brain, tmp_path)
    before = _production_state(brain)
    hermes_before = (home / "pending" / "skills" / "1a2b3c4d.json").read_bytes()

    pending = CuratorConnector(home).pending_write("1a2b3c4d")
    conversion = adapter.convert_pending_write(pending, experiment_id="exp-pending", candidate_id="cand-pending")

    assert _production_state(brain) == before
    assert (home / "pending" / "skills" / "1a2b3c4d.json").read_bytes() == hermes_before
    assert conversion.candidate.candidate_type is CandidateType.SKILL_PATCH
    assert store.list_candidates()[0]["decision"] == "pending"


def test_curator_targeted_patch_requires_unique_match(brain, tmp_path):
    adapter, _ = _adapter(brain, tmp_path)
    with pytest.raises(CuratorMutationRefused, match="occurs 2 times"):
        adapter.to_candidate(
            CuratorMutation("patch", "demo", "x", old_string="Step", new_string="Phase"),
            experiment_id="exp-a", candidate_id="cand-a",
        )
    with pytest.raises(CuratorMutationRefused, match="does not occur"):
        adapter.to_candidate(
            CuratorMutation("patch", "demo", "x", old_string="absent", new_string="Phase"),
            experiment_id="exp-b", candidate_id="cand-b",
        )
    conversion = adapter.to_candidate(
        CuratorMutation("patch", "demo", "rename steps", old_string="Step", new_string="Phase", replace_all=True),
        experiment_id="exp-c", candidate_id="cand-c",
    )
    assert "Step" not in git(brain, "show", f"{conversion.candidate.commit}:brain/skills/demo/SKILL.md")


def test_curator_consolidation_becomes_a_skill_merge(brain, tmp_path):
    adapter, _ = _adapter(brain, tmp_path)
    before = _production_state(brain)
    conversion = adapter.to_candidate(
        CuratorMutation("consolidate", "demo", "same umbrella", content="---\nname: umbrella\n---\nMerged.\n", into="umbrella"),
        experiment_id="exp-merge", candidate_id="cand-merge",
    )
    assert _production_state(brain) == before
    assert conversion.candidate.candidate_type is CandidateType.SKILL_MERGE
    assert conversion.candidate.target == "brain/skills/demo,brain/skills/umbrella"
    assert conversion.unapplied == ()
    assert git(brain, "ls-tree", "-r", "--name-only", conversion.candidate.commit).split() == [
        "brain/skills/umbrella/SKILL.md",
    ]


def test_curator_unsupported_writes_are_refused(brain, tmp_path):
    adapter, store = _adapter(brain, tmp_path)
    with pytest.raises(CuratorMutationRefused, match="directory component"):
        adapter.to_candidate(
            CuratorMutation("create", "plugin:skill", "x", content="text"), experiment_id="exp-y", candidate_id="cand-y",
        )
    home = build_hermes_home(tmp_path)
    pending_dir = home / "pending" / "skills"
    write_pending(pending_dir, "file0001", {"action": "write_file", "name": "demo", "file_path": "refs/a.md", "file_content": "x"})
    connector = CuratorConnector(home)
    with pytest.raises(CuratorMutationRefused):
        adapter.convert_pending_write(
            connector.pending_write("file0001"), experiment_id="exp-file0001", candidate_id="cand-file0001",
        )
    assert store.list_candidates() == []
    assert git(brain, "branch", "--list", "candidate/*") == ""


def test_signal_only_adapter_cannot_convert(tmp_path):
    with pytest.raises(RuntimeError, match="candidate repository"):
        CuratorAdapter(lambda: []).to_candidate(
            CuratorMutation("create", "demo", "x", content="text"), experiment_id="e", candidate_id="c",
        )


def test_cli_curator_propose_and_convert_pending(brain, tmp_path, monkeypatch, capsys):
    home = build_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "benchmarks"))
    content = tmp_path / "new.md"
    content.write_text("---\nname: fresh\n---\nNew skill.\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "curator", "propose", "--kind", "create", "--skill", "fresh",
        "--rationale", "recurring gap", "--content", str(content), "--repository", str(brain),
        "--experiment-id", "exp-cli-1", "--candidate-id", "cand-cli-1",
    ])
    assert main() == 0
    created = json.loads(capsys.readouterr().out)
    assert created["candidate_type"] == "SKILL_CREATE"
    assert created["status"] == "awaiting_evaluation"

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "curator", "convert-pending", "1a2b3c4d", "--hermes-home", str(home),
        "--repository", str(brain), "--experiment-id", "exp-cli-2", "--candidate-id", "cand-cli-2",
    ])
    assert main() == 0
    converted = json.loads(capsys.readouterr().out)
    assert converted["target"] == "brain/skills/demo"
    assert converted["mutation_kind"] == "patch"
