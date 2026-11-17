"""Protected, operator-owned configuration loaded from ``config/*.yaml``.

The configuration directory is read from the protected checkout (or an
explicit operator path) and never from a candidate worktree. Candidate diffs
that touch ``config/`` are rejected by the candidate builder and the promoter
(see :mod:`hermes_improve.target_paths`). The built-in defaults below are
identical to the historical :class:`~hermes_improve.policy.PromotionPolicy`
constants, so a missing configuration never weakens a gate.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .models import CandidateType, canonical_json


CONFIG_FILES = ("promotion.yaml", "scoring.yaml", "taxonomy.yaml", "models.yaml")
PROMOTION_GATES = ("auto", "manual")
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
EVIDENCE_SUITES = ("SMOKE", "CORE", "FULL", "SEALED_HOLDOUT")
MANDATORY_EVIDENCE_SUITES = frozenset({"FULL", "SEALED_HOLDOUT"})
_MAX_CONFIG_BYTES = 256 * 1024
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_NAME = re.compile(r"^QWEN_[A-Z][A-Z0-9_]{0,63}$")
_QWEN_ROLES = ("qwen_solver", "qwen_optimizer", "qwen_dreamer", "qwen_judge")

DEFAULT_PROMOTION_GATES: Mapping[CandidateType, str] = MappingProxyType({
    CandidateType.SKILL_CREATE: "auto",
    CandidateType.SKILL_PATCH: "auto",
    CandidateType.SKILL_MERGE: "auto",
    CandidateType.SKILL_SPLIT: "auto",
    CandidateType.TOOL_DESCRIPTION: "manual",
    CandidateType.WORKFLOW: "manual",
    CandidateType.ROUTING: "manual",
    CandidateType.MCP_TOOL_FILTER: "manual",
    CandidateType.MCP_CONFIG: "manual",
    CandidateType.SYSTEM_PROMPT: "manual",
    CandidateType.TOOL_CREATE: "manual",
    CandidateType.TOOL_PATCH: "manual",
})

DEFAULT_CANDIDATE_RISK: Mapping[CandidateType, str] = MappingProxyType({
    CandidateType.SKILL_CREATE: "LOW",
    CandidateType.SKILL_PATCH: "LOW",
    CandidateType.SKILL_MERGE: "LOW",
    CandidateType.SKILL_SPLIT: "LOW",
    CandidateType.TOOL_DESCRIPTION: "MEDIUM",
    CandidateType.WORKFLOW: "MEDIUM",
    CandidateType.ROUTING: "MEDIUM",
    CandidateType.MCP_TOOL_FILTER: "MEDIUM",
    CandidateType.MCP_CONFIG: "HIGH",
    CandidateType.SYSTEM_PROMPT: "HIGH",
    CandidateType.TOOL_CREATE: "CRITICAL",
    CandidateType.TOOL_PATCH: "CRITICAL",
})


class ConfigError(ValueError):
    """Raised when protected configuration is missing, unsafe, or invalid."""


@dataclass(frozen=True)
class PromotionConfig:
    policy_version: str = "1"
    max_critical_regressions: int = 0
    max_high_regressions: int = 0
    max_medium_regression_rate: float = 0.01
    min_quality_delta: float = 3.0
    max_category_drop: float = 2.0
    max_token_increase_ratio: float = 0.10
    max_latency_increase_ratio: float = 0.15
    min_holdout_quality_delta: float = 0.0
    require_token_measurement: bool = True
    token_exception_min_quality_ci_lower: float | None = 3.0
    latency_exception_min_quality_ci_lower: float | None = 3.0
    promotion_gates: Mapping[CandidateType, str] = DEFAULT_PROMOTION_GATES
    required_evidence_suites: tuple[str, ...] = EVIDENCE_SUITES

    @property
    def auto_promotable_types(self) -> frozenset[CandidateType]:
        return frozenset(kind for kind, gate in self.promotion_gates.items() if gate == "auto")


@dataclass(frozen=True)
class RunCountRange:
    min: int
    max: int


@dataclass(frozen=True)
class EvolutionConfig:
    max_generations: int = 10
    candidates_per_generation: int = 3
    max_candidates_total: int = 30
    stop_after_no_improvement_candidates: int = 8
    stop_after_no_improvement_generations: int = 3
    minimum_meaningful_delta: float = 0.5


@dataclass(frozen=True)
class ClusteringConfig:
    # Cosine-similarity threshold of the semantic clustering layer (layer 5).
    # Only used when an embedding endpoint is configured (models.yaml embedding).
    semantic_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class ScoringConfig:
    overall_score_dashboard_only: bool = True
    deterministic_runs: int = 1
    llm_runs: Mapping[str, RunCountRange] = field(default_factory=lambda: MappingProxyType({
        "SMOKE": RunCountRange(1, 2),
        "CORE": RunCountRange(3, 3),
        "FULL": RunCountRange(5, 5),
        "SEALED_HOLDOUT": RunCountRange(5, 5),
    }))
    evolution: EvolutionConfig = EvolutionConfig()
    clustering: ClusteringConfig = ClusteringConfig()


@dataclass(frozen=True)
class TaxonomyConfig:
    candidate_risk: Mapping[CandidateType, str] = DEFAULT_CANDIDATE_RISK
    difficulties: tuple[str, ...] = ("easy", "medium", "hard")
    # Benchmark task categories; must equal benchmark_schema.TAXONOMY_CATEGORIES.
    categories: tuple[str, ...] = (
        "coding", "git", "shell", "data-transform", "docs", "devops", "research", "mcp",
    )


@dataclass(frozen=True)
class ModelRoleConfig:
    model_env: str
    fallback_model_env: str | None = None


@dataclass(frozen=True)
class EmbeddingEnvConfig:
    """Environment-variable names of the optional embedding endpoint (layer 5).

    There is deliberately no fallback to the chat endpoint and no default URL:
    the semantic layer is enabled only when both URL and model are set.
    """

    base_url_env: str = "QWEN_EMBEDDING_BASE_URL"
    model_env: str = "QWEN_EMBEDDING_MODEL"
    api_key_env: str = "QWEN_EMBEDDING_API_KEY"
    timeout_env: str = "QWEN_EMBEDDING_TIMEOUT"
    allow_remote_env: str = "QWEN_EMBEDDING_ALLOW_REMOTE"


@dataclass(frozen=True)
class ModelsConfig:
    endpoint_env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({
        "base_url": "QWEN_BASE_URL",
        "api_key": "QWEN_API_KEY",
        "timeout": "QWEN_TIMEOUT",
        "max_concurrency": "QWEN_MAX_CONCURRENCY",
    }))
    roles: Mapping[str, ModelRoleConfig] = field(default_factory=lambda: MappingProxyType({
        "qwen_solver": ModelRoleConfig("QWEN_MODEL"),
        "qwen_optimizer": ModelRoleConfig("QWEN_MODEL"),
        "qwen_dreamer": ModelRoleConfig("QWEN_MODEL"),
        "qwen_judge": ModelRoleConfig("QWEN_JUDGE_MODEL", "QWEN_MODEL"),
    }))
    embedding: EmbeddingEnvConfig = EmbeddingEnvConfig()


@dataclass(frozen=True)
class HermesConfig:
    promotion: PromotionConfig = PromotionConfig()
    scoring: ScoringConfig = ScoringConfig()
    taxonomy: TaxonomyConfig = TaxonomyConfig()
    models: ModelsConfig = ModelsConfig()
    source: str = "built-in-defaults"
    digest: str = ""


def default_config() -> HermesConfig:
    """Built-in defaults; identical to the shipped ``config/*.yaml`` files."""
    return HermesConfig(digest=_digest_documents(None))


def load_config(directory: Path | None) -> HermesConfig:
    """Load and strictly validate the four protected configuration files.

    ``None`` selects the built-in defaults. A supplied directory must contain
    all four files; partial configuration is rejected rather than silently
    mixed with defaults. Symlinks and group/world-writable files are refused.
    """
    if directory is None:
        return default_config()
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ConfigError(f"configuration directory must be a real directory: {root}")
    _require_protected_mode(root)
    documents: dict[str, Any] = {}
    for name in CONFIG_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ConfigError(f"configuration file is missing or not a regular file: {name}")
        _require_protected_mode(path)
        with path.open("rb") as source:
            raw = source.read(_MAX_CONFIG_BYTES + 1)
        if len(raw) > _MAX_CONFIG_BYTES:
            raise ConfigError(f"configuration file exceeds 256 KiB: {name}")
        try:
            document = yaml.load(raw.decode("utf-8"), Loader=_StrictLoader)
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ConfigError(f"configuration file {name} is not valid UTF-8 YAML: {exc}") from exc
        if not isinstance(document, dict):
            raise ConfigError(f"configuration file {name} must contain a mapping")
        documents[name] = document
    return HermesConfig(
        promotion=_parse_promotion(documents["promotion.yaml"]),
        scoring=_parse_scoring(documents["scoring.yaml"]),
        taxonomy=_parse_taxonomy(documents["taxonomy.yaml"]),
        models=_parse_models(documents["models.yaml"]),
        source=str(root.resolve()),
        digest=_digest_documents(documents),
    )


def config_from_environment() -> HermesConfig:
    """Load ``HERMES_IMPROVE_CONFIG`` if set, otherwise the built-in defaults.

    The current working directory is deliberately not searched: a command
    started inside a candidate worktree must not pick up candidate files.
    """
    value = os.environ.get("HERMES_IMPROVE_CONFIG")
    return load_config(Path(value)) if value else default_config()


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _value in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping,
)


def strict_yaml_load(text: str) -> Any:
    """Parse YAML with the safe loader and duplicate-key rejection."""
    return yaml.load(text, Loader=_StrictLoader)


def _require_protected_mode(path: Path) -> None:
    mode = path.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ConfigError(f"configuration path must not be group/world-writable: {path}")


def _exact_keys(section: Mapping[str, Any], name: str, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(section, dict):
        raise ConfigError(f"{name} must be a mapping")
    keys = set(section)
    missing = required - keys
    unknown = keys - required - set(optional)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unknown {', '.join(sorted(str(item) for item in unknown))}")
        raise ConfigError(f"{name}: {'; '.join(detail)}")


def _schema(document: Mapping[str, Any], name: str) -> None:
    version = document.get("schema_version")
    if isinstance(version, bool) or version != 1:
        raise ConfigError(f"{name} requires schema_version: 1")


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _ratio(value: Any, name: str, *, upper: float = 1.0) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not 0 <= value <= upper
    ):
        raise ConfigError(f"{name} must be a finite number between 0 and {upper}")
    return float(value)


def _non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
    ):
        raise ConfigError(f"{name} must be a finite non-negative number")
    return float(value)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    return float(value)


def _optional_finite_number(value: Any, name: str) -> float | None:
    """A finite number, or ``null`` to disable the rule it configures."""
    return None if value is None else _finite_number(value, name)


def _candidate_type_mapping(section: Any, name: str, allowed: tuple[str, ...]) -> Mapping[CandidateType, str]:
    if not isinstance(section, dict):
        raise ConfigError(f"{name} must be a mapping of every candidate type")
    expected = {kind.value for kind in CandidateType}
    keys = {str(key) for key in section}
    if keys != expected or len(section) != len(expected):
        missing = expected - keys
        unknown = keys - expected
        raise ConfigError(
            f"{name} must list every candidate type exactly once "
            f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})"
        )
    result = {}
    for key, value in section.items():
        if value not in allowed:
            raise ConfigError(f"{name}.{key} must be one of {', '.join(allowed)}")
        result[CandidateType(key)] = value
    return MappingProxyType(result)


def _parse_promotion(document: Mapping[str, Any]) -> PromotionConfig:
    _exact_keys(document, "promotion.yaml", {
        "schema_version", "policy_version", "thresholds", "promotion_gates",
        "required_evidence_suites",
    }, {"require_token_measurement"})
    _schema(document, "promotion.yaml")
    version = document["policy_version"]
    if not isinstance(version, str) or not _SAFE_VERSION.fullmatch(version):
        raise ConfigError("promotion.yaml policy_version must be a safe string identifier")
    thresholds = document["thresholds"]
    # The generalization and exception keys are optional; when absent the
    # strict built-in defaults apply (they never relax the policy).
    _exact_keys(thresholds, "promotion.yaml thresholds", {
        "max_critical_regressions", "max_high_regressions", "max_medium_regression_rate",
        "min_quality_delta", "max_category_drop", "max_token_increase_ratio",
        "max_latency_increase_ratio",
    }, {
        "min_holdout_quality_delta", "token_exception_min_quality_ci_lower",
        "latency_exception_min_quality_ci_lower",
    })
    defaults = PromotionConfig()
    require_tokens = document.get("require_token_measurement", defaults.require_token_measurement)
    if not isinstance(require_tokens, bool):
        raise ConfigError("promotion.yaml require_token_measurement must be true or false")
    suites = document["required_evidence_suites"]
    if (
        not isinstance(suites, list) or not suites
        or any(item not in EVIDENCE_SUITES for item in suites)
        or len(set(suites)) != len(suites)
    ):
        raise ConfigError(
            "promotion.yaml required_evidence_suites must be unique entries of "
            + ", ".join(EVIDENCE_SUITES)
        )
    if not MANDATORY_EVIDENCE_SUITES <= set(suites):
        raise ConfigError("promotion.yaml required_evidence_suites must include FULL and SEALED_HOLDOUT")
    return PromotionConfig(
        policy_version=version,
        max_critical_regressions=_non_negative_int(
            thresholds["max_critical_regressions"], "max_critical_regressions"),
        max_high_regressions=_non_negative_int(
            thresholds["max_high_regressions"], "max_high_regressions"),
        max_medium_regression_rate=_ratio(
            thresholds["max_medium_regression_rate"], "max_medium_regression_rate"),
        min_quality_delta=_non_negative_number(thresholds["min_quality_delta"], "min_quality_delta"),
        max_category_drop=_non_negative_number(thresholds["max_category_drop"], "max_category_drop"),
        max_token_increase_ratio=_ratio(
            thresholds["max_token_increase_ratio"], "max_token_increase_ratio", upper=10.0),
        max_latency_increase_ratio=_ratio(
            thresholds["max_latency_increase_ratio"], "max_latency_increase_ratio", upper=10.0),
        min_holdout_quality_delta=_finite_number(
            thresholds.get("min_holdout_quality_delta", defaults.min_holdout_quality_delta),
            "min_holdout_quality_delta"),
        require_token_measurement=require_tokens,
        token_exception_min_quality_ci_lower=_optional_finite_number(
            thresholds.get("token_exception_min_quality_ci_lower", defaults.token_exception_min_quality_ci_lower),
            "token_exception_min_quality_ci_lower"),
        latency_exception_min_quality_ci_lower=_optional_finite_number(
            thresholds.get(
                "latency_exception_min_quality_ci_lower", defaults.latency_exception_min_quality_ci_lower),
            "latency_exception_min_quality_ci_lower"),
        promotion_gates=_candidate_type_mapping(
            document["promotion_gates"], "promotion.yaml promotion_gates", PROMOTION_GATES),
        required_evidence_suites=tuple(item for item in EVIDENCE_SUITES if item in suites),
    )


def _parse_scoring(document: Mapping[str, Any]) -> ScoringConfig:
    _exact_keys(
        document, "scoring.yaml", {"schema_version", "overall_score", "run_counts", "evolution"},
        {"clustering"},
    )
    _schema(document, "scoring.yaml")
    overall = document["overall_score"]
    _exact_keys(overall, "scoring.yaml overall_score", {"dashboard_only"})
    if overall["dashboard_only"] is not True:
        raise ConfigError("scoring.yaml overall_score.dashboard_only must be true; no score decides promotion")
    run_counts = document["run_counts"]
    _exact_keys(run_counts, "scoring.yaml run_counts", {"deterministic", "llm"})
    deterministic = _positive_int(run_counts["deterministic"], "run_counts.deterministic")
    llm = run_counts["llm"]
    _exact_keys(llm, "scoring.yaml run_counts.llm", set(EVIDENCE_SUITES))
    ranges = {}
    for suite in EVIDENCE_SUITES:
        entry = llm[suite]
        _exact_keys(entry, f"scoring.yaml run_counts.llm.{suite}", {"min", "max"})
        low = _positive_int(entry["min"], f"run_counts.llm.{suite}.min")
        high = _positive_int(entry["max"], f"run_counts.llm.{suite}.max")
        if low > high or high > 50:
            raise ConfigError(f"run_counts.llm.{suite} requires 1 <= min <= max <= 50")
        ranges[suite] = RunCountRange(low, high)
    evolution = document["evolution"]
    names = {
        "max_generations", "candidates_per_generation", "max_candidates_total",
        "stop_after_no_improvement_candidates", "stop_after_no_improvement_generations",
        "minimum_meaningful_delta",
    }
    _exact_keys(evolution, "scoring.yaml evolution", names)
    integers = {
        name: _positive_int(evolution[name], f"evolution.{name}")
        for name in names - {"minimum_meaningful_delta"}
    }
    if integers["candidates_per_generation"] > integers["max_candidates_total"]:
        raise ConfigError("evolution.candidates_per_generation cannot exceed max_candidates_total")
    clustering = ClusteringConfig()
    if "clustering" in document:
        section = document["clustering"]
        _exact_keys(section, "scoring.yaml clustering", {"semantic_similarity_threshold"})
        threshold = section["semantic_similarity_threshold"]
        if (
            isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold <= 1
        ):
            raise ConfigError(
                "scoring.yaml clustering.semantic_similarity_threshold must be a finite number in (0, 1]"
            )
        clustering = ClusteringConfig(float(threshold))
    return ScoringConfig(
        overall_score_dashboard_only=True,
        deterministic_runs=deterministic,
        llm_runs=MappingProxyType(ranges),
        evolution=EvolutionConfig(
            **integers,
            minimum_meaningful_delta=_non_negative_number(
                evolution["minimum_meaningful_delta"], "evolution.minimum_meaningful_delta"),
        ),
        clustering=clustering,
    )


def _parse_taxonomy(document: Mapping[str, Any]) -> TaxonomyConfig:
    _exact_keys(document, "taxonomy.yaml", {"schema_version", "candidate_risk", "difficulties", "categories"})
    _schema(document, "taxonomy.yaml")
    difficulties = document["difficulties"]
    if (
        not isinstance(difficulties, list) or not difficulties
        or any(not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item) for item in difficulties)
        or len(set(difficulties)) != len(difficulties)
    ):
        raise ConfigError("taxonomy.yaml difficulties must be unique lowercase identifiers")
    categories = document["categories"]
    if (
        not isinstance(categories, list) or not categories
        or any(not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item) for item in categories)
        or len(set(categories)) != len(categories)
    ):
        raise ConfigError("taxonomy.yaml categories must be unique lowercase identifiers")
    return TaxonomyConfig(
        candidate_risk=_candidate_type_mapping(
            document["candidate_risk"], "taxonomy.yaml candidate_risk", RISK_LEVELS),
        difficulties=tuple(difficulties),
        categories=tuple(categories),
    )


def _parse_models(document: Mapping[str, Any]) -> ModelsConfig:
    _exact_keys(document, "models.yaml", {"schema_version", "endpoint_env", "roles"}, {"embedding"})
    _schema(document, "models.yaml")
    endpoint = document["endpoint_env"]
    _exact_keys(endpoint, "models.yaml endpoint_env", {"base_url", "api_key", "timeout", "max_concurrency"})
    for key, value in endpoint.items():
        if not isinstance(value, str) or not _ENV_NAME.fullmatch(value):
            raise ConfigError(f"models.yaml endpoint_env.{key} must name a QWEN_* environment variable")
    roles = document["roles"]
    _exact_keys(roles, "models.yaml roles", set(_QWEN_ROLES))
    parsed = {}
    for role in _QWEN_ROLES:
        entry = roles[role]
        _exact_keys(entry, f"models.yaml roles.{role}", {"model_env"}, {"fallback_model_env"})
        values = [entry["model_env"], entry.get("fallback_model_env", "QWEN_MODEL")]
        if any(not isinstance(value, str) or not _ENV_NAME.fullmatch(value) for value in values):
            raise ConfigError(
                f"models.yaml roles.{role} must reference QWEN_* environment variables, not model names"
            )
        parsed[role] = ModelRoleConfig(entry["model_env"], entry.get("fallback_model_env"))
    embedding = EmbeddingEnvConfig()
    if "embedding" in document:
        section = document["embedding"]
        fields = {
            "base_url_env", "model_env", "api_key_env", "timeout_env", "allow_remote_env",
        }
        _exact_keys(section, "models.yaml embedding", fields)
        for key, value in section.items():
            if not isinstance(value, str) or not _ENV_NAME.fullmatch(value):
                raise ConfigError(
                    f"models.yaml embedding.{key} must name a QWEN_* environment variable, "
                    "not a model name or URL"
                )
        embedding = EmbeddingEnvConfig(**section)
    return ModelsConfig(
        endpoint_env=MappingProxyType(dict(endpoint)), roles=MappingProxyType(parsed),
        embedding=embedding,
    )


def _digest_documents(documents: Mapping[str, Any] | None) -> str:
    if documents is None:
        documents = _default_documents()
    return hashlib.sha256(canonical_json(documents).encode("utf-8")).hexdigest()


def _default_documents() -> dict[str, Any]:
    """Default configuration expressed in the on-disk document format."""
    promotion = PromotionConfig()
    scoring = ScoringConfig()
    taxonomy = TaxonomyConfig()
    models = ModelsConfig()
    return {
        "promotion.yaml": {
            "schema_version": 1,
            "policy_version": promotion.policy_version,
            "thresholds": {
                "max_critical_regressions": promotion.max_critical_regressions,
                "max_high_regressions": promotion.max_high_regressions,
                "max_medium_regression_rate": promotion.max_medium_regression_rate,
                "min_quality_delta": promotion.min_quality_delta,
                "max_category_drop": promotion.max_category_drop,
                "max_token_increase_ratio": promotion.max_token_increase_ratio,
                "max_latency_increase_ratio": promotion.max_latency_increase_ratio,
                "min_holdout_quality_delta": promotion.min_holdout_quality_delta,
                "token_exception_min_quality_ci_lower": promotion.token_exception_min_quality_ci_lower,
                "latency_exception_min_quality_ci_lower": promotion.latency_exception_min_quality_ci_lower,
            },
            "require_token_measurement": promotion.require_token_measurement,
            "promotion_gates": {kind.value: gate for kind, gate in promotion.promotion_gates.items()},
            "required_evidence_suites": list(promotion.required_evidence_suites),
        },
        "scoring.yaml": {
            "schema_version": 1,
            "overall_score": {"dashboard_only": True},
            "run_counts": {
                "deterministic": scoring.deterministic_runs,
                "llm": {
                    suite: {"min": value.min, "max": value.max}
                    for suite, value in scoring.llm_runs.items()
                },
            },
            "evolution": {
                "max_generations": scoring.evolution.max_generations,
                "candidates_per_generation": scoring.evolution.candidates_per_generation,
                "max_candidates_total": scoring.evolution.max_candidates_total,
                "stop_after_no_improvement_candidates":
                    scoring.evolution.stop_after_no_improvement_candidates,
                "stop_after_no_improvement_generations":
                    scoring.evolution.stop_after_no_improvement_generations,
                "minimum_meaningful_delta": scoring.evolution.minimum_meaningful_delta,
            },
            "clustering": {
                "semantic_similarity_threshold": scoring.clustering.semantic_similarity_threshold,
            },
        },
        "taxonomy.yaml": {
            "schema_version": 1,
            "candidate_risk": {kind.value: risk for kind, risk in taxonomy.candidate_risk.items()},
            "difficulties": list(taxonomy.difficulties),
            "categories": list(taxonomy.categories),
        },
        "models.yaml": {
            "schema_version": 1,
            "endpoint_env": dict(models.endpoint_env),
            "roles": {
                role: (
                    {"model_env": entry.model_env, "fallback_model_env": entry.fallback_model_env}
                    if entry.fallback_model_env else {"model_env": entry.model_env}
                )
                for role, entry in models.roles.items()
            },
            "embedding": {
                "base_url_env": models.embedding.base_url_env,
                "model_env": models.embedding.model_env,
                "api_key_env": models.embedding.api_key_env,
                "timeout_env": models.embedding.timeout_env,
                "allow_remote_env": models.embedding.allow_remote_env,
            },
        },
    }
