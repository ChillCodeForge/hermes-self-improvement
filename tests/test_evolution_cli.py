from __future__ import annotations

import json
import sys

import pytest

from evolution_doubles import TARGET, ScriptedOptimizerModel, SkillMarkerRunner, evolution_fixture, git
from hermes_improve import cli


def _wire(monkeypatch, tmp_path, model):
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "benchmarks"))
    monkeypatch.setattr(cli.QwenConfig, "from_env", classmethod(lambda cls: None))
    monkeypatch.setattr(cli, "OpenAICompatibleQwen", lambda config: model)
    runners = []
    monkeypatch.setattr(
        cli, "DockerCommandRunner",
        lambda repository, artifacts: runners.append(SkillMarkerRunner(repository)) or runners[-1],
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
    assert {seed for _, task, seed in runners[0].calls if task == "train-task"} >= {4, 1, 2, 3}
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
    monkeypatch.setattr(cli.GepaStrategy, "__init__", _raise_version_error)
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
