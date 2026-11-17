"""Spec §20: no hardcoded model, and the four logical Qwen roles keep separate config, prompts and logs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType

import pytest

from openai_stub import OpenAIStub
from hermes_improve.config import ModelRoleConfig, ModelsConfig
from hermes_improve.dreaming import DreamingEngine
from hermes_improve.judge import PairwiseJudge
from hermes_improve.qwen import QWEN_ROLES, OpenAICompatibleQwen, QwenConfig

REPO = Path(__file__).resolve().parents[1]
# Families and version spellings of real model names; none may appear in code or shipped config.
MODEL_NAME = re.compile(
    r"(?i)\b(qwen[\d.]+[\w.-]*|qwen-\d[\w.-]*|gpt-[\d][\w.-]*|gpt-4o|claude-[\w.-]+|llama[-\d][\w.-]*"
    r"|mistral-[\w.-]+|mixtral[\w.-]*|gemma[-\d][\w.-]*|deepseek[-\w.]*|phi-\d[\w.-]*|\d+(\.\d+)?b-instruct)\b"
)


def _shipped_files():
    for root in ("src", "config", "prompts"):
        for path in (REPO / root).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".json", ".toml"}:
                yield path


def test_no_model_name_is_hardcoded_in_code_config_or_prompts():
    hits = []
    for path in _shipped_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MODEL_NAME.search(line):
                hits.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert hits == []


def _models_config():
    return ModelsConfig(roles=MappingProxyType({
        "qwen_solver": ModelRoleConfig("QWEN_SOLVER_MODEL", "QWEN_MODEL"),
        "qwen_optimizer": ModelRoleConfig("QWEN_OPTIMIZER_MODEL", "QWEN_MODEL"),
        "qwen_dreamer": ModelRoleConfig("QWEN_DREAMER_MODEL", "QWEN_MODEL"),
        "qwen_judge": ModelRoleConfig("QWEN_JUDGE_MODEL", "QWEN_MODEL"),
    }))


def test_each_role_uses_its_configured_model_and_its_own_log(tmp_path):
    with OpenAIStub(lambda body: '{"ok": true}') as stub:
        environ = {
            "QWEN_BASE_URL": stub.base_url, "QWEN_MODEL": "shared-model",
            "QWEN_OPTIMIZER_MODEL": "optimizer-model", "QWEN_DREAMER_MODEL": "dreamer-model",
            "QWEN_JUDGE_MODEL": "judge-model", "QWEN_LOG_DIR": str(tmp_path / "logs"),
        }
        config = QwenConfig.from_env(_models_config(), environ)
        assert dict(config.role_models) == {
            "qwen_solver": "shared-model", "qwen_optimizer": "optimizer-model",
            "qwen_dreamer": "dreamer-model", "qwen_judge": "judge-model",
        }
        model = OpenAICompatibleQwen(config)
        for role in sorted(QWEN_ROLES):
            model.complete(role=role, system_prompt=f"prompt for {role}", user_payload="{}")
        sent = {
            body["messages"][0]["content"].split("]", 1)[0].removeprefix("[logical_role="): body["model"]
            for body in stub.requests
        }
    assert sent == dict(config.role_models)
    for role in QWEN_ROLES:
        records = [json.loads(line) for line in (tmp_path / "logs" / f"{role}.jsonl").read_text().splitlines()]
        assert [(item["role"], item["model"]) for item in records] == [(role, config.role_models[role])]
        assert model.stats[role]["requests"] == 1
    with pytest.raises(ValueError, match="unknown Qwen logical role"):
        model.complete(role="qwen_admin", system_prompt="x", user_payload="{}")


def test_shipped_models_config_resolves_roles_from_environment_only():
    environ = {"QWEN_BASE_URL": "http://127.0.0.1:9", "QWEN_MODEL": "shared", "QWEN_JUDGE_MODEL": "judge"}
    config = QwenConfig.from_env(ModelsConfig(), environ)
    assert config.model_for("qwen_judge") == "judge"
    assert {config.model_for(role) for role in QWEN_ROLES - {"qwen_judge"}} == {"shared"}
    with pytest.raises(ValueError, match="non-empty model name"):
        QwenConfig("http://127.0.0.1:9", "m", role_models=MappingProxyType({"qwen_admin": "x"})).validate_endpoint()


class _Recorder:
    """TEST DOUBLE for the Qwen client: records role and system prompt per call."""

    def __init__(self):
        self.calls = []

    def complete(self, *, role, system_prompt, user_payload):
        self.calls.append((role, system_prompt))
        if role == "qwen_dreamer":
            return json.dumps({"hypotheses": [{"summary": "s", "issue": "i", "evidence_session_ids": ["s1"]}]})
        if role == "qwen_judge":
            return json.dumps({"winner": "A", "confidence": 0.9, "uncertainty": 0.1, "rationale": "r"})
        return json.dumps({"candidates": []})


def test_code_paths_use_their_own_role_and_disjoint_prompts():
    recorder = _Recorder()
    engine = DreamingEngine(recorder)
    engine.rem([{"session_id": "s1", "task": {"prompt": "p"}, "errors": ["timeout"]}])
    engine.deep([{"summary": "s", "issue": "i"}])
    PairwiseJudge(recorder).compare(
        task_prompt="t", criteria=["c"], baseline_output="a", candidate_output="b",
    )
    prompts: dict[str, set[str]] = {}
    for role, prompt in recorder.calls:
        prompts.setdefault(role, set()).add(prompt)
    assert set(prompts) == {"qwen_dreamer", "qwen_optimizer", "qwen_judge"}
    roles = list(prompts)
    for index, first in enumerate(roles):
        for second in roles[index + 1:]:
            assert not prompts[first] & prompts[second], (first, second)
    assert all("qwen_dreamer" in prompt for prompt in prompts["qwen_dreamer"])
    assert all("qwen_optimizer" in prompt for prompt in prompts["qwen_optimizer"])


def test_hermes_solver_defaults_to_the_qwen_solver_role(monkeypatch):
    from hermes_improve import cli

    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.setenv("QWEN_BASE_URL", "http://127.0.0.1:8000/")
    monkeypatch.setenv("QWEN_MODEL", "solver-model")
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    assert cli._solver_role_defaults() == ("solver-model", "http://127.0.0.1:8000/v1", "QWEN_API_KEY")
    monkeypatch.delenv("QWEN_API_KEY")
    monkeypatch.delenv("QWEN_MODEL")
    monkeypatch.delenv("QWEN_BASE_URL")
    assert cli._solver_role_defaults() == (None, None, None)


def test_shared_model_variable_comes_from_the_models_config_not_a_hardcoded_name():
    local = ModelRoleConfig("HERMES_LOCAL_MODEL")
    models = ModelsConfig(roles=MappingProxyType({
        "qwen_solver": local, "qwen_optimizer": local, "qwen_dreamer": local,
        "qwen_judge": ModelRoleConfig("HERMES_LOCAL_JUDGE", "HERMES_LOCAL_MODEL"),
    }))
    environ = {"QWEN_BASE_URL": "http://127.0.0.1:9", "HERMES_LOCAL_MODEL": "local", "HERMES_LOCAL_JUDGE": "lj"}
    config = QwenConfig.from_env(models, environ)
    assert config.model == "local" and config.judge_model == "lj"
    assert config.model_for("qwen_solver") == "local" and config.model_for("qwen_judge") == "lj"
    # QWEN_MODEL is not part of this configuration, so it neither satisfies nor is demanded.
    with pytest.raises(ValueError, match="HERMES_LOCAL_MODEL must be set"):
        QwenConfig.from_env(models, {"QWEN_BASE_URL": "http://127.0.0.1:9", "QWEN_MODEL": "ignored"})
    # Roles without a common fallback use the solver role's own variable as the shared model.
    solo = ModelsConfig(roles=MappingProxyType({
        "qwen_solver": ModelRoleConfig("QWEN_SOLVER_ONLY"),
        "qwen_optimizer": ModelRoleConfig("QWEN_OPT_ONLY"),
        "qwen_dreamer": ModelRoleConfig("QWEN_DREAM_ONLY"),
        "qwen_judge": ModelRoleConfig("QWEN_JUDGE_ONLY"),
    }))
    assert QwenConfig.from_env(solo, {"QWEN_BASE_URL": "http://127.0.0.1:9", "QWEN_SOLVER_ONLY": "s"}).model == "s"
