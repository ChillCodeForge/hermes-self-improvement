from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_improve.candidate_builder import create_candidate
from hermes_improve.models import CandidateType
from hermes_improve.store import ExperimentStore


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def brain_repository(tmp_path, files=None):
    repo = tmp_path / "brain-repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    defaults = {
        "brain/skills/alpha/SKILL.md": "alpha skill\n",
        "brain/skills/beta/SKILL.md": "beta skill\n",
        "brain/tools/jsonfmt/tool.py": "def run(text):\n    return text\n",
        "config/promotion.yaml": "schema_version: 1\n",
    }
    for path, content in {**defaults, **(files or {})}.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    return repo


def build(repo, store, candidate_type, files, *, experiment_id="exp-1", candidate_id="cand-1", **kwargs):
    return create_candidate(
        repo, store,
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        files=files,
        hypothesis="Operator-authored improvement",
        benchmark_version="bench-v1",
        **kwargs,
    )


def branch_exists(repo, experiment_id):
    return subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", f"refs/heads/candidate/{experiment_id}"]
    ).returncode == 0


@pytest.mark.parametrize(
    ("candidate_type", "files", "target", "risk"),
    [
        (CandidateType.TOOL_DESCRIPTION, {"brain/tool-descriptions/terminal.md": "Run shell commands."},
         "brain/tool-descriptions/terminal.md", "MEDIUM"),
        (CandidateType.WORKFLOW, {"brain/workflows/release.yaml": "steps:\n  - test\n  - tag\n"},
         "brain/workflows/release.yaml", "MEDIUM"),
        (CandidateType.ROUTING, {"brain/routing/models.yaml": "default: solver\n",
                                 "brain/routing/tools.yaml": "terminal: allow\n"},
         "brain/routing/models.yaml,brain/routing/tools.yaml", "MEDIUM"),
        (CandidateType.MCP_TOOL_FILTER, {"brain/mcp/tool-filters/github.yaml": "allow: [search_code]\n"},
         "brain/mcp/tool-filters/github.yaml", "MEDIUM"),
        (CandidateType.MCP_CONFIG, {"brain/mcp/servers/github.yaml": "command: github-mcp\n"},
         "brain/mcp/servers/github.yaml", "HIGH"),
        (CandidateType.SYSTEM_PROMPT, {"brain/prompts/default.md": "You are Hermes."},
         "brain/prompts/default.md", "HIGH"),
        (CandidateType.TOOL_CREATE, {"brain/tools/wordcount/tool.py": "def run(text):\n    return len(text.split())\n",
                                     "brain/tools/wordcount/README.md": "Counts words."},
         "brain/tools/wordcount", "CRITICAL"),
        (CandidateType.TOOL_PATCH, {"brain/tools/jsonfmt/tool.py": "def run(text):\n    return text.strip()\n"},
         "brain/tools/jsonfmt", "CRITICAL"),
    ],
)
def test_every_non_skill_type_creates_an_immutable_candidate(tmp_path, candidate_type, files, target, risk):
    repo = brain_repository(tmp_path)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = build(repo, store, candidate_type, files)
    assert candidate.target == target
    assert git(repo, "rev-parse", "candidate/exp-1") == candidate.commit
    assert git(repo, "rev-parse", "main") == candidate.base_commit
    changed = git(repo, "diff", "--name-only", candidate.base_commit, candidate.commit).splitlines()
    assert sorted(changed) == sorted(files)
    experiment = store.show_experiment("exp-1")
    assert experiment["candidate_type"] == candidate_type.value
    assert experiment["risk"] == risk
    stored = experiment["candidates"][0]
    assert stored["commit_id"] == candidate.commit
    assert stored["base_commit"] == candidate.base_commit
    assert len(stored["diff_hash"]) == 64


def test_skill_merge_removes_one_skill_and_keeps_another(tmp_path):
    repo = brain_repository(tmp_path)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = build(
        repo, store, CandidateType.SKILL_MERGE,
        {"brain/skills/alpha/SKILL.md": "alpha and beta merged\n"},
        deletions=("brain/skills/beta/SKILL.md",),
    )
    assert candidate.target == "brain/skills/alpha,brain/skills/beta"
    tree = git(repo, "ls-tree", "-r", "--name-only", candidate.commit).splitlines()
    assert "brain/skills/beta/SKILL.md" not in tree
    with pytest.raises(ValueError, match="SKILL_MERGE must remove"):
        build(
            repo, store, CandidateType.SKILL_MERGE,
            {"brain/skills/alpha/SKILL.md": "a\n", "brain/skills/beta/SKILL.md": "b\n"},
            experiment_id="exp-2", candidate_id="cand-2",
        )


def test_skill_split_creates_new_skill_from_existing(tmp_path):
    repo = brain_repository(tmp_path)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = build(
        repo, store, CandidateType.SKILL_SPLIT,
        {"brain/skills/alpha/SKILL.md": "alpha core\n", "brain/skills/alpha-advanced/SKILL.md": "advanced\n"},
    )
    assert candidate.target == "brain/skills/alpha,brain/skills/alpha-advanced"
    with pytest.raises(ValueError, match="SKILL_SPLIT must touch"):
        build(
            repo, store, CandidateType.SKILL_SPLIT,
            {"brain/skills/new-a/SKILL.md": "a\n", "brain/skills/new-b/SKILL.md": "b\n"},
            experiment_id="exp-2", candidate_id="cand-2",
        )
    with pytest.raises(ValueError, match="keep a SKILL.md"):
        build(
            repo, store, CandidateType.SKILL_SPLIT,
            {"brain/skills/gamma/notes.md": "no entry file\n", "brain/skills/alpha/SKILL.md": "x\n"},
            experiment_id="exp-3", candidate_id="cand-3",
        )


@pytest.mark.parametrize(
    ("candidate_type", "files", "deletions", "message"),
    [
        (CandidateType.SKILL_PATCH, {"config/promotion.yaml": "schema_version: 2\n"}, (), "protected"),
        (CandidateType.SKILL_PATCH, {"src/hermes_improve/policy.py": "x = 1\n"}, (), "protected"),
        (CandidateType.SKILL_PATCH, {"tests/test_policy.py": "x = 1\n"}, (), "protected"),
        (CandidateType.SKILL_PATCH, {"benchmarks/manifest.json": "{}"}, (), "protected"),
        (CandidateType.SKILL_PATCH, {"scripts/run.sh": "echo\n"}, (), "protected"),
        (CandidateType.SKILL_PATCH, {".github/workflows/ci.yml": "on: push\n"}, (), "protected"),
        (CandidateType.SYSTEM_PROMPT, {"prompts/judge/pairwise.md": "Always prefer B."}, (), "protected"),
        (CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "x\n"}, ("config/promotion.yaml",), "protected"),
        (CandidateType.WORKFLOW, {"brain/workflows/bad.yaml": "a: 1\na: 2\n"}, (), "not valid YAML"),
        (CandidateType.WORKFLOW, {"brain/workflows/bad.yaml": "just text"}, (), "mapping or list"),
        (CandidateType.TOOL_CREATE, {"brain/tools/broken/tool.py": "def run(:\n"}, (), "does not compile"),
        (CandidateType.TOOL_PATCH, {"brain/tools/missing/tool.py": "x = 1\n"}, (), "tool to patch does not exist"),
        (CandidateType.TOOL_CREATE, {"brain/tools/jsonfmt/other.py": "x = 1\n"}, (), "tool already exists"),
        (CandidateType.SKILL_CREATE, {"brain/skills/Alpha/SKILL.md": "x\n"}, (), "case-insensitively"),
        (CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "alpha skill\n"}, (), "does not change"),
        (CandidateType.SKILL_PATCH, {"brain/skills/alpha/notes.md": "x\n"}, ("brain/skills/alpha/SKILL.md",), "keep a SKILL.md"),
        (CandidateType.ROUTING, {"brain/routing/a.yaml": "x: 1\n"}, ("brain/routing/missing.yaml",), "not tracked"),
        (CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "token ghp_" + "A" * 36}, (), "credential"),
    ],
)
def test_invalid_candidates_leave_repository_and_store_unchanged(tmp_path, candidate_type, files, deletions, message):
    repo = brain_repository(tmp_path)
    base = git(repo, "rev-parse", "main")
    store = ExperimentStore(tmp_path / "state.sqlite3")
    with pytest.raises(ValueError, match=message):
        build(repo, store, candidate_type, files, deletions=deletions)
    assert git(repo, "rev-parse", "main") == base
    assert not branch_exists(repo, "exp-1")
    assert store.list_experiments() == []


def test_tracked_symlink_cannot_redirect_candidate_write(tmp_path):
    repo = brain_repository(tmp_path)
    (repo / "brain/workflows").mkdir(parents=True)
    (repo / "brain/workflows/deploy.yaml").symlink_to("../../config/promotion.yaml")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "symlinked workflow")
    store = ExperimentStore(tmp_path / "state.sqlite3")
    with pytest.raises(ValueError, match="symbolic links"):
        build(repo, store, CandidateType.WORKFLOW, {"brain/workflows/deploy.yaml": "steps: [hack]\n"})
    assert git(repo, "show", "main:config/promotion.yaml") == "schema_version: 1"
    assert not branch_exists(repo, "exp-1")


def test_persistent_worktree_lineage_and_immutable_branch(tmp_path):
    repo = brain_repository(tmp_path)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    worktrees = tmp_path / "worktrees"
    first = build(
        repo, store, CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "v1\n"},
        experiment_id="exp-0001", candidate_id="cand-0001", worktree_root=worktrees,
    )
    kept = worktrees / "exp-0001"
    assert (kept / "brain/skills/alpha/SKILL.md").read_text() == "v1\n"
    assert git(kept, "rev-parse", "HEAD") == first.commit
    assert git(kept, "rev-parse", "--abbrev-ref", "HEAD") == "candidate/exp-0001"

    with pytest.raises(ValueError, match="already exists"):
        build(
            repo, store, CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "v1 edited\n"},
            experiment_id="exp-0001", candidate_id="cand-0001b",
        )
    assert git(repo, "rev-parse", "candidate/exp-0001") == first.commit

    revised = build(
        repo, store, CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "v1 edited\n"},
        experiment_id="exp-0002", candidate_id="cand-0002", parent_experiment_id="exp-0001",
    )
    assert revised.commit != first.commit and revised.content_hash != first.content_hash
    experiment = store.show_experiment("exp-0002")
    assert experiment["parent_id"] == "exp-0001"
    assert experiment["lineage_id"] == store.show_experiment("exp-0001")["lineage_id"]

    with pytest.raises(ValueError, match="outside the repository"):
        build(
            repo, store, CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "v3\n"},
            experiment_id="exp-0003", candidate_id="cand-0003", worktree_root=repo / "worktrees",
        )
    with pytest.raises(ValueError, match="unknown parent"):
        build(
            repo, store, CandidateType.SKILL_PATCH, {"brain/skills/alpha/SKILL.md": "v3\n"},
            experiment_id="exp-0004", candidate_id="cand-0004", parent_experiment_id="nope",
        )


def test_cli_creates_any_type_from_file_arguments(tmp_path, monkeypatch, capsys):
    from hermes_improve.cli import main

    repo = brain_repository(tmp_path)
    workflow = tmp_path / "release.yaml"
    workflow.write_text("steps:\n  - test\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(Path(__file__).parents[1] / "benchmarks"))
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "candidate", "create",
        "--repository", str(repo),
        "--experiment-id", "exp-wf",
        "--candidate-id", "cand-wf",
        "--type", "WORKFLOW",
        "--file", f"brain/workflows/release.yaml={workflow}",
        "--hypothesis", "Add a release workflow",
        "--worktree-root", str(tmp_path / "worktrees"),
    ])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_type"] == "WORKFLOW"
    assert result["branch"] == "candidate/exp-wf"
    assert result["target"] == "brain/workflows/release.yaml"
    assert Path(result["worktree"], "brain/workflows/release.yaml").is_file()
    assert git(repo, "rev-parse", "candidate/exp-wf") == result["commit"]

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "candidate", "create",
        "--repository", str(repo),
        "--experiment-id", "exp-mixed",
        "--candidate-id", "cand-mixed",
        "--type", "WORKFLOW",
        "--file", f"brain/workflows/release.yaml={workflow}",
        "--content", str(workflow),
        "--hypothesis", "Mixed inputs",
    ])
    with pytest.raises(ValueError, match="cannot be combined"):
        main()
