"""`evolve` selects its screening runner exactly like `evaluate` (docker | local | hermes)."""
from __future__ import annotations

import json
import sys

import pytest

from evolution_doubles import ScriptedOptimizerModel, SkillMarkerRunner, evolution_fixture
from hermes_improve import cli
from hermes_improve.errors import AccessDenied
from hermes_improve.run_policy import RunPolicy

_ORIGINAL_BUILD_RUNNER = cli.build_runner

# Every option that shapes the sandboxed runner; evaluate and evolve must expose the same set.
RUNNER_OPTIONS = {
    "--runner", "--acknowledge-no-container-isolation", "--timeout", "--max-turns", "--max-tool-calls",
    "--network", "--network-allow", "--verifier-http-allow", "--hermes-agent-dir", "--hermes-python",
    "--model", "--model-base-url", "--model-api-key-env", "--fixtures-root",
}


class LlmSkillMarkerRunner(SkillMarkerRunner):
    """TEST DOUBLE for HermesBatchRunner: same outcomes, but declares a model in the loop."""

    llm_influenced = True


def _subparser(name):
    parser = cli.build_parser()
    action = next(item for item in parser._actions if item.dest == "command")
    return action.choices[name]


def _options(name):
    return {option for action in _subparser(name)._actions for option in action.option_strings}


def _wire(monkeypatch, tmp_path, model, runner_factory=SkillMarkerRunner):
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "benchmarks"))
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "absent-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setattr(cli.QwenConfig, "from_env", classmethod(lambda cls: None))
    monkeypatch.setattr(cli, "OpenAICompatibleQwen", lambda config: model)
    built = []

    def build_runner(args, artifact_store):
        built.append({
            "runner": args.runner,
            "benchmark_root": getattr(args, "benchmark_root", None),
            "holdout_root": getattr(args, "holdout_root", None),
            "args": args,
        })
        built[-1]["instance"] = runner_factory(args.repository)
        return built[-1]["instance"]

    monkeypatch.setattr(cli, "build_runner", build_runner)
    return built


def _evolve(monkeypatch, fixture, *extra):
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evolve", fixture.root_id, "--repository", str(fixture.repo),
        "--max-generations", "1", "--candidates-per-generation", "1", *extra,
    ])


def test_evolve_exposes_the_same_runner_options_as_evaluate_with_backward_compatible_defaults():
    assert RUNNER_OPTIONS <= _options("evaluate")
    assert RUNNER_OPTIONS <= _options("evolve")
    # evolve is the generator: it must never be able to name the sealed holdout.
    assert "--holdout-root" in _options("evaluate") and "--holdout-root" not in _options("evolve")
    args = cli.build_parser().parse_args(["evolve", "cand-1", "--repository", "."])
    assert args.runner == "docker" and args.timeout == 120 and args.max_turns == 30
    assert args.max_tool_calls == 200 and args.network is None and args.network_allow == []
    assert args.verifier_http_allow == [] and args.acknowledge_no_container_isolation is False
    assert not hasattr(args, "holdout_root")
    hermes = cli.build_parser().parse_args([
        "evolve", "cand-1", "--repository", ".", "--runner", "hermes", "--hermes-agent-dir", "/opt/hermes",
        "--model", "qwen", "--model-base-url", "http://127.0.0.1:1/v1", "--fixtures-root", "/fx",
    ])
    assert hermes.runner == "hermes" and str(hermes.hermes_agent_dir).replace("\\", "/") == "/opt/hermes"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evolve", "cand-1", "--repository", ".", "--runner", "podman"])


@pytest.mark.parametrize("runner", ["docker", "hermes"])
def test_evolve_screens_through_the_selected_runner_without_holdout_access(
    tmp_path, monkeypatch, capsys, runner,
):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["variant quality=3"]])
    built = _wire(monkeypatch, tmp_path, model)
    _evolve(monkeypatch, fixture, "--runner", runner)
    assert cli.main() == 0
    json.loads(capsys.readouterr().out)
    assert len(built) == 1
    assert built[0]["runner"] == runner
    assert built[0]["benchmark_root"] == tmp_path / "benchmarks"
    assert built[0]["holdout_root"] is None
    # The gateway screened with exactly the runner build_runner returned.
    assert built[0]["instance"].calls


def test_evolve_default_runner_is_docker_without_holdout_root(tmp_path, monkeypatch, capsys):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    _wire(monkeypatch, tmp_path, ScriptedOptimizerModel([["variant quality=3"]]))
    # Exercise the real build_runner; only the Docker-backed runner class is replaced.
    monkeypatch.setattr(cli, "build_runner", _ORIGINAL_BUILD_RUNNER)
    constructed = []

    def docker(repository, artifacts, **kwargs):
        constructed.append(kwargs)
        return SkillMarkerRunner(repository)

    monkeypatch.setattr(cli, "DockerCommandRunner", docker)
    _evolve(monkeypatch, fixture)
    assert cli.main() == 0
    capsys.readouterr()
    assert len(constructed) == 1
    assert constructed[0]["benchmark_root"] == tmp_path / "benchmarks"
    assert constructed[0]["holdout_root"] is None
    assert constructed[0]["default_timeout"] == 120
    assert constructed[0]["network"].mode == "disabled"


def test_evolve_uses_the_llm_run_policy_for_an_llm_influenced_runner(tmp_path, monkeypatch, capsys):
    policy = RunPolicy.from_config(cli.config_from_environment().scoring)
    seen = {}
    for name, factory in (("deterministic", SkillMarkerRunner), ("hermes", LlmSkillMarkerRunner)):
        root = tmp_path / name
        root.mkdir()
        fixture = evolution_fixture(root, with_holdout=False)
        built = _wire(monkeypatch, root, ScriptedOptimizerModel([["variant quality=3"]]), factory)
        _evolve(monkeypatch, fixture, "--runner", "docker" if name == "deterministic" else "hermes")
        assert cli.main() == 0
        capsys.readouterr()
        calls = built[0]["instance"].calls
        # validation-task belongs to CORE only, so its seeds are exactly the CORE run count.
        seen[name] = {seed for commit, task, seed in calls if task == "validation-task"}
    assert seen["deterministic"] == {0}
    assert seen["hermes"] == set(policy.seeds(policy.core_runs))
    assert len(seen["hermes"]) >= 3


def test_evolve_refuses_the_holdout_with_any_runner(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    _wire(monkeypatch, tmp_path, model)
    monkeypatch.setattr(cli.EvaluationGateway, "holdout_available", property(lambda self: True))
    _evolve(monkeypatch, fixture, "--runner", "hermes")
    with pytest.raises(AccessDenied, match="never see sealed holdout"):
        cli.main()
    assert model.calls == []


def test_evolve_refuses_a_holdout_root_smuggled_onto_the_arguments(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    built = _wire(monkeypatch, tmp_path, model)
    original = cli.build_parser

    def parser_with_holdout():
        parser = original()
        parser.set_defaults(holdout_root=tmp_path / "holdout")
        return parser

    monkeypatch.setattr(cli, "build_parser", parser_with_holdout)
    _evolve(monkeypatch, fixture)
    with pytest.raises(AccessDenied, match="holdout"):
        cli.main()
    assert built == [] and model.calls == []


def test_evolve_local_runner_requires_the_isolation_acknowledgement(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    _wire(monkeypatch, tmp_path, model)
    monkeypatch.setattr(cli, "build_runner", _ORIGINAL_BUILD_RUNNER)
    _evolve(monkeypatch, fixture, "--runner", "local")
    with pytest.raises(ValueError, match="--acknowledge-no-container-isolation"):
        cli.main()
    assert model.calls == []


def test_evolve_hermes_runner_validates_its_options_like_evaluate(tmp_path, monkeypatch):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([["never used quality=9"]])
    _wire(monkeypatch, tmp_path, model)
    monkeypatch.setattr(cli, "build_runner", _ORIGINAL_BUILD_RUNNER)
    _evolve(
        monkeypatch, fixture, "--runner", "hermes", "--hermes-agent-dir", str(tmp_path),
        "--model", "m", "--model-base-url", "http://127.0.0.1:1", "--network", "n", "--network-allow", "a:1",
    )
    with pytest.raises(ValueError, match="networking disabled"):
        cli.main()
    assert model.calls == []
