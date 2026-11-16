"""Published Pydantic schema and JSON/YAML loading for benchmark records.

The Pydantic model is the published, exportable *structural* schema (types,
required sections, enumerations). `Benchmark.from_dict` remains the
authoritative *semantic* validator (hard-benchmark thresholds, secret scanning,
provenance, label hygiene) and always runs after structural parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import Benchmark, Eligibility, Split

SCHEMA_VERSION = 1

# Category taxonomy shared with config/taxonomy.yaml. Corpus validation
# (`hermes-improve benchmarks validate`) rejects records outside this list.
TAXONOMY_CATEGORIES: tuple[str, ...] = (
    "coding",
    "git",
    "shell",
    "data-transform",
    "docs",
    "devops",
    "research",
    "mcp",
)
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
CRITICALITIES: tuple[str, ...] = ("low", "normal", "high", "critical")
BENCHMARK_FILE_SUFFIXES: frozenset[str] = frozenset({".json", ".yaml", ".yml"})
MAX_BENCHMARK_FILE_BYTES = 1024 * 1024
_SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_SHA256_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Section(BaseModel):
    """Sections allow documented runner-specific extras (for example `command`)."""

    model_config = ConfigDict(strict=True, extra="allow", populate_by_name=True)


class SourceSection(_Section):
    type: str = Field(min_length=1, description="Origin kind, e.g. manual, session, regression.")
    provenance: str = Field(min_length=1, description="Descriptive origin metadata (not an attestation).")
    session_id: str | None = Field(default=None, min_length=1)
    verified_at: str | None = Field(default=None, min_length=1, description="ISO-8601 timestamp.")


class TaskSection(_Section):
    title: str
    prompt: str


class TaxonomySection(_Section):
    categories: list[str]
    difficulty: str = Field(min_length=1)
    required_capabilities: list[str]


class EnvironmentSection(_Section):
    type: str | None = None
    repository_fixture: str | None = Field(
        default=None, min_length=1,
        description="Repository-relative fixture directory evaluated at the replayed commit.",
    )
    git_commit: str | None = Field(default=None, pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    dependency_digest: str | None = Field(default=None, min_length=1)
    container_image: str | None = Field(default=None, min_length=1)
    network_policy: str | None = Field(default=None, min_length=1)
    input_data: str | None = Field(
        default=None, min_length=1,
        description="Digest of the task input (for corpus fixtures: sha256 over the fixture tree).",
    )
    command: list[str] | None = None
    cwd: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


class ReplayEvidence(_Section):
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    runs: int = Field(ge=2)
    reproducible: bool
    passed: bool
    outcome_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    replayed_at: str = Field(min_length=1)


class EligibilitySection(_Section):
    class_: Eligibility = Field(alias="class", strict=False)
    reproducibility_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    replay_verified: bool
    replay_evidence: ReplayEvidence | None = None


class VerificationSection(_Section):
    deterministic: list[str | dict[str, Any]]
    subjective: list[Any]


class GroundTruthSection(_Section):
    type: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    expected_exit_code: int | None = None
    structured_result: dict[str, Any] | None = None


class BenchmarkSection(_Section):
    split: Split = Field(strict=False)
    suites: list[str]
    cluster_id: str = Field(min_length=1)
    criticality: Literal["low", "normal", "high", "critical"] | None = None
    impact_areas: list[str] | None = Field(
        default=None,
        description="Glob patterns over candidate target paths used for impact-aware suites.",
    )


class SecuritySection(_Section):
    sanitized: bool
    contains_secrets: bool


class BenchmarkRecord(BaseModel):
    """Structural schema of a benchmark record (schema_version 1)."""

    model_config = ConfigDict(
        strict=True, extra="forbid", populate_by_name=True,
        title="HermesBenchmarkRecord",
    )

    schema_version: Literal[1]
    id: str = Field(pattern=_SAFE_ID_PATTERN)
    version: int = Field(ge=1)
    source: SourceSection
    task: TaskSection
    taxonomy: TaxonomySection
    environment: EnvironmentSection
    eligibility: EligibilitySection
    verification: VerificationSection
    ground_truth: GroundTruthSection
    benchmark: BenchmarkSection
    security: SecuritySection


def benchmark_json_schema() -> dict[str, Any]:
    """Return the published JSON Schema for benchmark records."""
    schema = BenchmarkRecord.model_json_schema(by_alias=True, mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/ChillCodeForge/hermes-self-improvement/benchmark-record.schema.json"
    return schema


def parse_benchmark_document(data: object, *, source: str = "<memory>") -> Benchmark:
    """Structurally validate with Pydantic, then apply authoritative semantic validation."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: benchmark document must be an object")
    try:
        BenchmarkRecord.model_validate(dict(data))
    except ValidationError as exc:
        raise ValueError(f"{source}: benchmark schema validation failed: {_summarize(exc)}") from None
    return Benchmark.from_dict(data)


def load_benchmark_document(path: Path) -> dict[str, Any]:
    """Read one JSON or YAML benchmark document from disk."""
    path = Path(path)
    data = path.read_bytes()
    return decode_benchmark_document(path.name, data)


def decode_benchmark_document(name: str, data: bytes) -> dict[str, Any]:
    """Decode JSON (`.json`) or YAML (`.yaml`/`.yml`) bytes into a plain object.

    Duplicate keys are rejected in both formats; YAML is restricted to the safe
    loader and to JSON-compatible scalar types.
    """
    suffix = Path(name).suffix.lower()
    if suffix not in BENCHMARK_FILE_SUFFIXES:
        raise ValueError(f"{name}: benchmark files must use .json, .yaml or .yml")
    if len(data) > MAX_BENCHMARK_FILE_BYTES:
        raise ValueError(f"{name}: benchmark file exceeds {MAX_BENCHMARK_FILE_BYTES} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: benchmark file must be UTF-8") from exc
    if suffix == ".json":
        try:
            document = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}: invalid JSON benchmark document: {exc.msg}") from None
    else:
        try:
            document = yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506 - strict safe loader
        except yaml.YAMLError as exc:
            raise ValueError(f"{name}: invalid YAML benchmark document: {exc}") from None
        _require_json_compatible(name, document)
    if not isinstance(document, dict):
        raise ValueError(f"{name}: benchmark document must be an object")
    return document


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                None, None, "benchmark YAML mapping keys must be strings", key_node.start_mark,
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate YAML key {key!r}", key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping,
)


def _require_json_compatible(name: str, value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{name}: benchmark YAML must not contain NaN or infinite numbers")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_compatible(name, item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_json_compatible(name, item)
        return
    raise ValueError(
        f"{name}: benchmark YAML contains a non-JSON value of type {type(value).__name__}; "
        "quote dates and timestamps"
    )


def _summarize(exc: ValidationError) -> str:
    # Never echo input values: holdout content must not leak into error output.
    parts = []
    for error in exc.errors(include_input=False, include_url=False)[:10]:
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    if exc.error_count() > 10:
        parts.append(f"... {exc.error_count() - 10} more")
    return "; ".join(parts)
