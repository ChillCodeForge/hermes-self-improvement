from __future__ import annotations

import json
import os
import sys

import pytest

from evolution_doubles import TARGET, ScriptedOptimizerModel, SkillMarkerRunner, evolution_fixture, git
from hermes_improve import cli
from hermes_improve.errors import AccessDenied


def _wire(monkeypatch, tmp_path, model):
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "benchmarks"))
    # Development mode without a provisioned OS layout; the agent boundary
    # itself is covered below and in tests/security/test_os_isolation.py.
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "absent-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setattr(cli.QwenConfig, "from_env", classmethod(lambda cls: None))
    monkeypatch.setattr(cli, "OpenAICompatibleQwen", lambda config: model)
    runners = []
    monkeypatch.setattr(
        cli, "DockerCommandRunner",
        lambda repository, artifacts, benchmark_root: runners.append(SkillMarkerRunner(repository)) or runners[-1],
    )
    return runners


def test_evolve_cli_reports_generations_and_finalists_without_holdout_or_promotion(
    tmp_path, monkeypatch, capsys,
):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["cli variant quality=3", "cli flat quality=1"], ["cli best quality=6"]])
    runners = _wire(monkeypatch, tmp_path, model)
    settings = tmp_path / "evolution.json"
    settings.write_text(json.dumps({"stop_after_no_improvement_generations": 5, "smoke_seeds": [4]}))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(fixture.repo),
        "--settings", str(settings), "--max-generations", "3", "--candidates-per-generation", "2",
    ])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert [row["generation"] for row in output["generations"]] == [1, 2, 3]
    assert output["generations"][0]["created"] == 2 and output["generations"][1]["improved"] == 1
    assert output["stop_reason"] == "optimizer_exhausted"
    assert {row["holdout"] for row in output["finalists"]} == {"awaiting_evaluator"}
    assert output["next_steps"][0].startswith(f"hermes-improve evaluate {output['best_candidate_id']}")
    assert "not performed" in output["promotion"]
    # SMOKE uses the explicit settings seed; CORE/FULL derive theirs from the run policy.
    assert {seed for _, task, seed in runners[0].calls if task == "train-task"} == {4, 0}
    run = fixture.store.get_evolution_run(output["run_id"])
    assert run["config"]["smoke_seeds"] == [4] and run["config"]["max_generations"] == 3
    assert fixture.store.list_releases() == []


def test_evolve_cli_rejects_unknown_settings_before_any_model_call(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([])
    _wire(monkeypatch, tmp_path, model)
    settings = tmp_path / "evolution.json"
    settings.write_text(json.dumps({"promote_automatically": True}))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(fixture.repo), "--settings", str(settings),
    ])
    with pytest.raises(ValueError, match="unknown evolution settings"):
        cli.main()
    assert model.calls == []


def test_evolve_cli_gepa_strategy_requires_pinned_package(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    _wire(monkeypatch, tmp_path, ScriptedOptimizerModel([]))
    monkeypatch.setattr(cli.GepaOptimizeStrategy, "__init__", _raise_version_error)
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(fixture.repo), "--strategy", "gepa",
    ])
    with pytest.raises(RuntimeError, match="pinned"):
        cli.main()


def _raise_version_error(self, model):
    from hermes_improve.gepa_adapter import GepaVersionError

    raise GepaVersionError("GEPA version '0.0.1' is not supported; this adapter is pinned")


def test_candidate_generate_cli_creates_candidates_from_selected_proposals(tmp_path, monkeypatch, capsys):
    fixture = evolution_fixture(tmp_path, with_holdout=False)

    class AuthorModel:
        def complete(self, *, role, system_prompt, user_payload):
            assert role == "qwen_optimizer" and "UNTRUSTED EVIDENCE" in system_prompt
            return json.dumps({"content": "---\nname: generated\n---\nPrefer pinned images."})

    _wire(monkeypatch, tmp_path, AuthorModel())
    proposals = tmp_path / "deep.json"
    proposals.write_text(json.dumps({"candidates": [
        {"summary": "a", "candidate_type": "SKILL_PATCH", "target": TARGET, "hypothesis": "pin images"},
        {"summary": "b", "candidate_type": "SKILL_CREATE", "target": "brain/skills/new", "hypothesis": "new"},
    ]}))
    main_before = git(fixture.repo, "rev-parse", "main")
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "candidate", "generate", "--repository", str(fixture.repo),
        "--proposal-file", str(proposals), "--proposal-index", "1", "--proposal-index", "0",
    ])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    rows = output["candidates"]
    assert [(row["proposal_index"], row["status"]) for row in rows] == [(1, "created"), (0, "created")]
    assert fixture.store.get_candidate(rows[0]["candidate_id"])["target"] == "brain/skills/new"
    assert git(fixture.repo, "rev-parse", "main") == main_before

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "candidate", "generate", "--repository", str(fixture.repo),
        "--proposal-file", str(proposals), *sum((["--proposal-index", "0"] for _ in range(2)), []),
    ])
    with pytest.raises(ValueError, match="distinct"):
        cli.main()


def test_candidate_subcommands_coexist_and_dispatch_without_foreign_attributes(tmp_path, monkeypatch, capsys):
    parser = cli.build_parser()
    generate = parser.parse_args([
        "candidate", "generate", "--repository", "r", "--proposal-file", "p.json", "--proposal-index", "0",
    ])
    assert generate.candidate_command == "generate" and not hasattr(generate, "file")
    create = parser.parse_args([
        "candidate", "create", "--repository", "r", "--experiment-id", "e", "--candidate-id", "c",
    ])
    assert create.candidate_command == "create" and create.file == []
    for argv in (
        ["candidate", "export", "c", "--repository", "r", "--out-dir", "o"],
        ["candidate", "import", "c", "--handoff-dir", "h", "--mirror", "m"],
        ["candidate", "sync-baseline", "--repository", "r", "--production", "p"],
    ):
        assert parser.parse_args(argv).candidate_command == argv[1]

    # `candidate generate` must reach its own branch, never the create branch
    # (which reads --file/--delete that the generate subparser does not define).
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    proposals = tmp_path / "deep.json"
    proposals.write_text(json.dumps({"candidates": [{
        "candidate_type": "SKILL_PATCH", "target": TARGET, "hypothesis": "tighten the sample skill",
    }]}))
    model = ScriptedOptimizerModel([])
    model.complete = lambda **kwargs: json.dumps({"content": "---\nname: sample\ndescription: d\n---\n\nBetter.\n"})
    _wire(monkeypatch, tmp_path, model)
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "candidate", "generate", "--repository", str(fixture.repo),
        "--proposal-file", str(proposals), "--proposal-index", "0",
    ])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert [row["status"] for row in output["candidates"]] == ["created"]
    experiment = fixture.store.show_experiment(output["candidates"][0]["experiment_id"])
    assert experiment["decision"] == "pending"
    assert git(fixture.repo, "branch", "--list", f"candidate/{experiment['id']}")


def _agent_layout(tmp_path):
    from hermes_improve.isolation import load_layout

    path = tmp_path / "layout.json"
    path.write_text(json.dumps({
        "schema_version": 1, "root": str(tmp_path / "srv"),
        "users": {"agent": "hsi-a", "evaluator": "hsi-e", "promoter": "hsi-p"},
        "ledger_group": "hsi-ledger", "cli": "/usr/bin/hermes-improve",
    }))
    path.chmod(0o644)
    return load_layout(path, required_owner=os.getuid())


def test_evolve_cli_fails_closed_without_an_isolation_layout(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    _wire(monkeypatch, tmp_path, model)
    monkeypatch.delenv("HERMES_IMPROVE_ISOLATION")
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(fixture.repo),
    ])
    with pytest.raises(PermissionError, match="no provisioned isolation layout"):
        cli.main()
    assert model.calls == []
    assert git(fixture.repo, "branch", "--list", "candidate/*").split() == ["candidate/root-exp"]


def test_evolve_cli_runs_as_agent_and_refuses_paths_outside_the_agent_boundary(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    _wire(monkeypatch, tmp_path, model)
    layout = _agent_layout(tmp_path)
    roles = []
    monkeypatch.setattr(cli, "enforce_isolation", lambda role: roles.append(role) or layout)
    root = layout.root
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(root / "agent"))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root / "benchmarks" / "public"))
    for repository in (
        fixture.repo, root / "brain" / "production.git", root / "releases" / "brain-v1",
        root / "holdout", root / "workspaces",
    ):
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "evolve", fixture.root_id, "--repository", str(repository),
        ])
        with pytest.raises(PermissionError, match="agent repository must be"):
            cli.main()
    # Benchmarks and agent state must be the provisioned public and agent paths.
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(root / "brain" / "candidates.git"),
    ])
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "benchmarks"))
    with pytest.raises(PermissionError, match="benchmarks_public must be the provisioned path"):
        cli.main()
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root / "benchmarks" / "public"))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(root / "ledger"))
    with pytest.raises(PermissionError, match="agent_state must be the provisioned path"):
        cli.main()
    assert set(roles) == {"agent"} and model.calls == []


def test_require_agent_paths_accepts_only_the_candidate_repository_and_workspaces(tmp_path):
    from hermes_improve.isolation import require_agent_paths

    layout = _agent_layout(tmp_path)
    root = layout.root
    common = {"benchmark_root": root / "benchmarks" / "public", "state_dir": root / "agent"}
    require_agent_paths(layout, repository=root / "brain" / "candidates.git", **common)
    require_agent_paths(layout, repository=root / "workspaces" / "exp-0001", **common)
    for forbidden in (
        root / "brain" / "production.git", root / "deploy" / "current", root / "holdout",
        root / "ledger", root / "workspaces" / ".." / "holdout",
    ):
        with pytest.raises(PermissionError, match="agent repository must be"):
            require_agent_paths(layout, repository=forbidden, **common)


def test_evolve_cli_refuses_a_gateway_that_could_reach_the_holdout(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    _wire(monkeypatch, tmp_path, model)
    monkeypatch.setattr(cli.EvaluationGateway, "holdout_available", property(lambda self: True))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(fixture.repo),
    ])
    with pytest.raises(AccessDenied, match="never see sealed holdout"):
        cli.main()
    assert model.calls == []
