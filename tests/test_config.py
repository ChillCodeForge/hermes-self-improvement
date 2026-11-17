from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from hermes_improve.config import (
    ConfigError,
    config_from_environment,
    default_config,
    load_config,
)
from hermes_improve.models import CandidateType
from hermes_improve.policy import PromotionPolicy


SHIPPED = Path(__file__).parents[1] / "config"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(SHIPPED, target)
    for path in [target, *target.iterdir()]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    return target


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_shipped_config_is_identical_to_builtin_defaults(tmp_path):
    loaded = load_config(_copy(tmp_path))
    defaults = default_config()
    assert loaded.promotion == defaults.promotion
    assert loaded.scoring == defaults.scoring
    assert loaded.taxonomy == defaults.taxonomy
    assert loaded.models == defaults.models
    assert loaded.digest == defaults.digest
    assert PromotionPolicy.from_config(loaded.promotion) == PromotionPolicy()


def test_default_gates_follow_candidate_type_risk_defaults():
    promotion = default_config().promotion
    assert promotion.auto_promotable_types == {
        CandidateType.SKILL_CREATE, CandidateType.SKILL_PATCH,
        CandidateType.SKILL_MERGE, CandidateType.SKILL_SPLIT,
    }
    for kind in (
        CandidateType.TOOL_DESCRIPTION, CandidateType.WORKFLOW, CandidateType.ROUTING,
        CandidateType.MCP_TOOL_FILTER, CandidateType.MCP_CONFIG, CandidateType.SYSTEM_PROMPT,
        CandidateType.TOOL_CREATE, CandidateType.TOOL_PATCH,
    ):
        assert PromotionPolicy.from_config(promotion).requires_manual_approval(kind)
    scoring = default_config().scoring
    assert scoring.deterministic_runs == 1
    assert (scoring.llm_runs["SMOKE"].min, scoring.llm_runs["SMOKE"].max) == (1, 2)
    assert scoring.llm_runs["CORE"].min == 3 and scoring.llm_runs["FULL"].min == 5
    assert scoring.evolution.max_generations == 10
    assert scoring.evolution.minimum_meaningful_delta == 0.5


def test_type_unlock_is_only_a_protected_config_change(tmp_path):
    root = _copy(tmp_path)
    _replace(root / "promotion.yaml", "WORKFLOW: manual", "WORKFLOW: auto")
    _replace(root / "promotion.yaml", 'policy_version: "1"', 'policy_version: "2"')
    config = load_config(root)
    policy = PromotionPolicy.from_config(config.promotion)
    assert policy.version == "2"
    assert not policy.requires_manual_approval(CandidateType.WORKFLOW)
    assert policy.requires_manual_approval(CandidateType.SYSTEM_PROMPT)
    assert config.digest != default_config().digest


def test_thresholds_flow_into_policy(tmp_path):
    root = _copy(tmp_path)
    _replace(root / "promotion.yaml", "min_quality_delta: 3.0", "min_quality_delta: 7.5")
    policy = PromotionPolicy.from_config(load_config(root).promotion)
    assert policy.min_quality_delta == 7.5


def test_generalization_token_and_exception_settings_flow_into_policy(tmp_path):
    root = _copy(tmp_path)
    _replace(root / "promotion.yaml", "min_holdout_quality_delta: 0.0", "min_holdout_quality_delta: 1.5")
    _replace(root / "promotion.yaml", "require_token_measurement: true", "require_token_measurement: false")
    _replace(root / "promotion.yaml", "token_exception_min_quality_ci_lower: 3.0",
             "token_exception_min_quality_ci_lower: null")
    _replace(root / "promotion.yaml", "latency_exception_min_quality_ci_lower: 3.0",
             "latency_exception_min_quality_ci_lower: 8")
    policy = PromotionPolicy.from_config(load_config(root).promotion)
    assert policy.min_holdout_quality_delta == 1.5
    assert policy.require_token_measurement is False
    assert policy.token_exception_min_quality_ci_lower is None
    assert policy.latency_exception_min_quality_ci_lower == 8.0


def test_omitted_optional_policy_keys_fall_back_to_strict_defaults(tmp_path):
    root = _copy(tmp_path)
    for line in (
        "  min_holdout_quality_delta: 0.0\n", "  token_exception_min_quality_ci_lower: 3.0\n",
        "  latency_exception_min_quality_ci_lower: 3.0\n", "require_token_measurement: true\n",
    ):
        _replace(root / "promotion.yaml", line, "")
    policy = PromotionPolicy.from_config(load_config(root).promotion)
    assert policy == PromotionPolicy()
    assert policy.require_token_measurement is True


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("require_token_measurement: true", "require_token_measurement: 0", "true or false"),
        ("require_token_measurement: true", "require_token_measurement: \"no\"", "true or false"),
        ("min_holdout_quality_delta: 0.0", "min_holdout_quality_delta: .inf", "finite"),
        ("min_holdout_quality_delta: 0.0", "min_holdout_quality_delta: null", "finite"),
        ("token_exception_min_quality_ci_lower: 3.0", "token_exception_min_quality_ci_lower: yes", "finite"),
    ],
)
def test_invalid_generalization_and_token_settings_are_rejected(tmp_path, old, new, message):
    root = _copy(tmp_path)
    _replace(root / "promotion.yaml", old, new)
    with pytest.raises(ConfigError, match=message):
        load_config(root)


@pytest.mark.parametrize(
    ("filename", "old", "new", "message"),
    [
        ("promotion.yaml", "max_high_regressions: 0", "max_high_regressions: 0\n  max_high_regressions: 1", "duplicate key"),
        ("promotion.yaml", "schema_version: 1", "schema_version: 1\nextra: true", "unknown extra"),
        ("promotion.yaml", "required_evidence_suites: [SMOKE, CORE, FULL, SEALED_HOLDOUT]",
         "required_evidence_suites: [SMOKE, CORE, FULL]", "FULL and SEALED_HOLDOUT"),
        ("promotion.yaml", "TOOL_PATCH: manual", "TOOL_PATCH: sometimes", "must be one of"),
        ("promotion.yaml", "  TOOL_PATCH: manual\n", "", "every candidate type"),
        ("promotion.yaml", "max_medium_regression_rate: 0.01", "max_medium_regression_rate: .nan", "finite"),
        ("promotion.yaml", "max_critical_regressions: 0", "max_critical_regressions: true", "non-negative integer"),
        ("scoring.yaml", "dashboard_only: true", "dashboard_only: false", "dashboard_only"),
        ("scoring.yaml", "SMOKE: {min: 1, max: 2}", "SMOKE: {min: 3, max: 2}", "min <= max"),
        ("models.yaml", "qwen_solver: {model_env: QWEN_MODEL}", "qwen_solver: {model_env: qwen2.5-coder}", "not model names"),
        ("taxonomy.yaml", "SKILL_CREATE: LOW", "SKILL_CREATE: TRIVIAL", "must be one of"),
        ("promotion.yaml", "schema_version: 1", "schema_version: 1\nhack: !!python/object/apply:os.system [id]", "YAML"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, filename, old, new, message):
    root = _copy(tmp_path)
    _replace(root / filename, old, new)
    with pytest.raises(ConfigError, match=message):
        load_config(root)


def test_missing_file_symlink_and_writable_config_are_rejected(tmp_path):
    root = _copy(tmp_path)
    (root / "models.yaml").unlink()
    with pytest.raises(ConfigError, match="models.yaml"):
        load_config(root)

    root = _copy(tmp_path / "symlink")
    real = tmp_path / "elsewhere.yaml"
    shutil.copy(root / "scoring.yaml", real)
    (root / "scoring.yaml").unlink()
    (root / "scoring.yaml").symlink_to(real)
    with pytest.raises(ConfigError, match="scoring.yaml"):
        load_config(root)

    root = _copy(tmp_path / "writable")
    os.chmod(root / "promotion.yaml", 0o666)
    with pytest.raises(ConfigError, match="group/world-writable"):
        load_config(root)


def test_environment_selects_config_and_ignores_working_directory(tmp_path, monkeypatch):
    planted = _copy(tmp_path / "cwd")
    _replace(planted / "promotion.yaml", "TOOL_PATCH: manual", "TOOL_PATCH: auto")
    monkeypatch.chdir(planted.parent)
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    assert config_from_environment().promotion.promotion_gates[CandidateType.TOOL_PATCH] == "manual"
    monkeypatch.setenv("HERMES_IMPROVE_CONFIG", str(planted))
    assert config_from_environment().promotion.promotion_gates[CandidateType.TOOL_PATCH] == "auto"


def test_policy_cannot_drop_mandatory_evidence_suites():
    with pytest.raises(ValueError, match="FULL and SEALED_HOLDOUT"):
        PromotionPolicy(required_evidence_suites=("SMOKE", "FULL"))


def test_taxonomy_categories_match_the_benchmark_schema(tmp_path):
    from hermes_improve.benchmark_schema import DIFFICULTIES, TAXONOMY_CATEGORIES

    loaded = load_config(_copy(tmp_path))
    assert loaded.taxonomy.categories == TAXONOMY_CATEGORIES
    assert default_config().taxonomy.categories == TAXONOMY_CATEGORIES
    assert loaded.taxonomy.difficulties == DIFFICULTIES


def test_taxonomy_categories_are_required_and_validated(tmp_path):
    directory = _copy(tmp_path)
    _replace(directory / "taxonomy.yaml", "categories: [coding,", "categories: [Coding,")
    with pytest.raises(ConfigError, match="categories"):
        load_config(directory)
