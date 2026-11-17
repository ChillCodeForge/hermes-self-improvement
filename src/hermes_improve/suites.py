"""Benchmark suite selection and manifests (spec §17).

Suite membership is declared per record in `benchmark.suites`. Selection rules:

* SMOKE / CORE: public, non-shadow records that declare the suite. An
  optional `ImpactScope` narrows them to records whose categories,
  required capabilities or `benchmark.impact_areas` globs touch the
  candidate. Critical REGRESSION records declared in the suite are always
  kept as a safety net. SMOKE is capped at `SMOKE_LIMIT` records.
* FULL: every public, non-shadow record that declares FULL. Impact is
  ignored: FULL is the complete regression pool.
* SEALED_HOLDOUT: every SEALED_HOLDOUT record supplied by the caller (only the
  evaluator can load them). Impact is ignored so finalists cannot be
  evaluated on a cherry-picked holdout subset.

Selections are sorted (criticality, then id/version) and therefore
deterministic; the manifest digest pins the exact task set.
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence

from .models import Benchmark, Eligibility, Split, canonical_json

SMOKE_LIMIT = 50
_CRITICALITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "low": 3}


class Suite(StrEnum):
    SMOKE = "SMOKE"
    CORE = "CORE"
    FULL = "FULL"
    SEALED_HOLDOUT = "SEALED_HOLDOUT"


@dataclass(frozen=True)
class ImpactScope:
    """What a candidate can affect; empty scope means "no impact narrowing"."""

    categories: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    target_areas: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in ("categories", "capabilities", "target_areas"):
            values = getattr(self, name)
            if not isinstance(values, frozenset) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"impact {name} must be a frozenset of non-empty strings")
        for area in self.target_areas:
            if area.startswith("/") or ".." in area.split("/"):
                raise ValueError("impact target areas must be relative candidate paths")

    @property
    def empty(self) -> bool:
        return not (self.categories or self.capabilities or self.target_areas)

    @classmethod
    def parse(cls, specs: Iterable[str]) -> "ImpactScope":
        """Parse `category:NAME`, `capability:NAME` and `target:PATH` specs."""
        buckets: dict[str, set[str]] = {"category": set(), "capability": set(), "target": set()}
        for spec in specs:
            kind, separator, value = spec.partition(":")
            if not separator or kind not in buckets or not value.strip():
                raise ValueError(
                    f"invalid impact spec {spec!r}; use category:NAME, capability:NAME or target:PATH"
                )
            if kind == "target" and value.strip().startswith("/"):
                raise ValueError("impact target areas must be relative candidate paths")
            buckets[kind].add(value.strip())
        return cls(
            frozenset(buckets["category"]),
            frozenset(buckets["capability"]),
            frozenset(_normalize_target(item) for item in buckets["target"]),
        )

    def matches(self, benchmark: Benchmark) -> bool:
        if self.empty:
            return True
        categories = {str(item) for item in benchmark.taxonomy.get("categories", [])}
        capabilities = {str(item) for item in benchmark.taxonomy.get("required_capabilities", [])}
        if categories & self.categories or capabilities & self.capabilities:
            return True
        areas = benchmark.benchmark.get("impact_areas") or []
        return any(
            fnmatch.fnmatchcase(target, str(pattern))
            for target in self.target_areas
            for pattern in areas
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "categories": sorted(self.categories),
            "capabilities": sorted(self.capabilities),
            "target_areas": sorted(self.target_areas),
        }


def _normalize_target(target: str) -> str:
    normalized = target.strip().strip("/")
    if normalized.startswith("brain/"):
        normalized = normalized[len("brain/"):]
    return normalized


def declared_suites(benchmark: Benchmark) -> frozenset[str]:
    return frozenset(str(item).upper() for item in benchmark.benchmark.get("suites", []))


def select_suite(
    benchmarks: Iterable[Benchmark],
    suite: Suite | str,
    *,
    impact: ImpactScope | None = None,
    hard_only: bool = True,
    smoke_limit: int = SMOKE_LIMIT,
) -> tuple[Benchmark, ...]:
    """Pure, deterministic suite selection. See module docstring for the rules."""
    suite = Suite(str(suite).upper())
    if smoke_limit < 1:
        raise ValueError("smoke_limit must be positive")
    rows = list(benchmarks)
    identities = [(item.id, item.version) for item in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("suite selection requires unique benchmark id/version pairs")
    rows = _latest_versions(rows)
    if hard_only:
        rows = [
            item for item in rows
            if Eligibility(item.eligibility["class"]) is Eligibility.HARD_BENCHMARK
        ]
    if suite is Suite.SEALED_HOLDOUT:
        return tuple(sorted((item for item in rows if item.split is Split.SEALED_HOLDOUT), key=_order))
    pool = [
        item for item in rows
        if item.split not in {Split.SEALED_HOLDOUT, Split.SHADOW} and suite.value in declared_suites(item)
    ]
    if suite is not Suite.FULL and impact is not None and not impact.empty:
        pool = [
            item for item in pool
            if impact.matches(item) or (
                item.split is Split.REGRESSION and item.benchmark.get("criticality") == "critical"
            )
        ]
    selected = sorted(pool, key=_order)
    if suite is Suite.SMOKE:
        selected = selected[:smoke_limit]
    return tuple(selected)


def _latest_versions(rows: Sequence[Benchmark]) -> list[Benchmark]:
    latest: dict[str, Benchmark] = {}
    for item in rows:
        if item.id not in latest or item.version > latest[item.id].version:
            latest[item.id] = item
    return list(latest.values())


def _order(benchmark: Benchmark) -> tuple[int, str, int]:
    criticality = str(benchmark.benchmark.get("criticality") or "normal")
    return (_CRITICALITY_ORDER.get(criticality, 2), benchmark.id, benchmark.version)


def suite_manifest(
    selected: Sequence[Benchmark],
    *,
    suite: Suite | str,
    benchmark_version: str,
    impact: ImpactScope | None = None,
    include_task_details: bool = True,
) -> dict[str, Any]:
    """Manifest pinning a suite selection to the validated corpus version.

    `digest` is SHA-256 over the canonical manifest body (everything except
    `digest`). Holdout manifests are built with `include_task_details=False`
    so only ids/versions/digests are present.
    """
    suite = Suite(str(suite).upper())
    if not isinstance(benchmark_version, str) or not benchmark_version.startswith("sha256:"):
        raise ValueError("benchmark_version must be the validated corpus manifest version")
    tasks = []
    for item in selected:
        entry: dict[str, Any] = {"id": item.id, "version": item.version, "digest": item.digest}
        if include_task_details:
            entry |= {
                "split": item.split.value,
                "cluster_id": item.cluster_id,
                "categories": list(item.taxonomy.get("categories", [])),
            }
        tasks.append(entry)
    body = {
        "schema_version": 1,
        "suite": suite.value,
        "benchmark_version": benchmark_version,
        "impact": (impact or ImpactScope()).to_dict(),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    return body | {"digest": manifest_body_digest(body)}


def corpus_manifest(benchmarks: Sequence[Benchmark], *, benchmark_version: str) -> dict[str, Any]:
    """Public corpus manifest: version digest, per-record digests and distributions."""
    rows = sorted(benchmarks, key=lambda item: (item.id, item.version))
    if any(item.split is Split.SEALED_HOLDOUT for item in rows):
        raise ValueError("the public corpus manifest must not describe sealed holdout records")
    by_split = Counter(item.split.value for item in rows)
    by_category = Counter(
        str(category) for item in rows for category in item.taxonomy.get("categories", [])
    )
    by_difficulty = Counter(str(item.taxonomy.get("difficulty")) for item in rows)
    by_suite = Counter(suite for item in rows for suite in declared_suites(item))
    by_eligibility = Counter(str(item.eligibility["class"]) for item in rows)
    body = {
        "schema_version": 1,
        "benchmark_version": benchmark_version,
        "record_count": len(rows),
        "counts": {
            "split": dict(sorted(by_split.items())),
            "category": dict(sorted(by_category.items())),
            "difficulty": dict(sorted(by_difficulty.items())),
            "suite": dict(sorted(by_suite.items())),
            "eligibility": dict(sorted(by_eligibility.items())),
        },
        "records": [
            {
                "id": item.id,
                "version": item.version,
                "digest": item.digest,
                "split": item.split.value,
                "cluster_id": item.cluster_id,
                "suites": sorted(declared_suites(item)),
            }
            for item in rows
        ],
    }
    return body | {"digest": manifest_body_digest(body)}


def manifest_body_digest(body: dict[str, Any]) -> str:
    if "digest" in body:
        raise ValueError("manifest body must not already contain a digest")
    return "sha256:" + hashlib.sha256(canonical_json(body).encode()).hexdigest()


def verify_manifest(manifest: dict[str, Any]) -> None:
    """Raise if a manifest's digest does not match its body."""
    if not isinstance(manifest, dict) or "digest" not in manifest:
        raise ValueError("manifest must be an object with a digest")
    body = {key: value for key, value in manifest.items() if key != "digest"}
    if manifest_body_digest(body) != manifest["digest"]:
        raise ValueError("benchmark manifest digest does not match its contents")


_NARROWING_CANDIDATE_TYPES = frozenset({"SKILL_CREATE", "SKILL_PATCH", "SKILL_MERGE", "SKILL_SPLIT"})


def impact_scope_for_candidate(
    candidate_type: str, target: str, categories: Iterable[str],
) -> ImpactScope:
    """Impact scope derived from a candidate's brain target.

    Skill candidates narrow SMOKE/CORE to their brain area (`skills/<name>`,
    matched against `benchmark.impact_areas`) and to taxonomy categories named
    by the skill (`coding`, `git-helper` -> `git`, ...). Every other candidate
    type (prompts, routing, tools, MCP, workflows) can affect any task, so it
    gets the empty scope: no narrowing.
    """
    kind = str(getattr(candidate_type, "value", candidate_type))
    if kind not in _NARROWING_CANDIDATE_TYPES:
        return ImpactScope()
    categories = tuple(categories)
    areas: set[str] = set()
    matched: set[str] = set()
    # Merge/split targets list several units ("brain/skills/a,brain/skills/b");
    # every unit, including a removed one, keeps its own tasks in scope.
    for unit in target.split(","):
        area = _normalize_target(unit)
        if not area or area.startswith("/") or ".." in area.split("/"):
            raise ValueError("candidate target must be a relative brain path")
        name = area.rsplit("/", 1)[-1]
        tokens = name.split("-")
        areas.add(area)
        matched.update(
            category for category in categories
            if name == category or name.startswith(category + "-") or name.endswith("-" + category)
            or category in tokens
        )
    return ImpactScope(categories=frozenset(matched), target_areas=frozenset(areas))


def select_evaluation_suite(
    benchmarks: Iterable[Benchmark], suite: Suite | str, *, impact: ImpactScope | None = None,
) -> tuple[tuple[Benchmark, ...], tuple[Benchmark, ...], ImpactScope]:
    """Tasks for one staged evaluation run: (suite tasks, holdout tasks, applied impact).

    FULL additionally returns every sealed holdout task. When impact narrowing
    leaves SMOKE/CORE empty, the un-narrowed suite is used (more evidence,
    never less) and the applied scope is reported as empty.
    """
    suite = Suite(str(suite).upper())
    if suite is Suite.SEALED_HOLDOUT:
        raise ValueError("SEALED_HOLDOUT is evaluated only as part of FULL")
    rows = list(benchmarks)
    applied = impact or ImpactScope()
    selected = select_suite(rows, suite, impact=applied)
    if not selected and not applied.empty:
        applied = ImpactScope()
        selected = select_suite(rows, suite)
    holdout = select_suite(rows, Suite.SEALED_HOLDOUT) if suite is Suite.FULL else ()
    if not selected:
        raise ValueError(f"no HARD_BENCHMARK tasks are assigned to suite {suite.value}")
    if suite is Suite.FULL and not holdout:
        raise ValueError("promotion evaluation requires at least one FULL and one sealed holdout benchmark")
    return selected, holdout, applied


SUITE_SELECTION_RULES_VERSION = "suite-selection-rules-1"


def suite_selection_decision_fields(
    *,
    suite: Suite | str,
    candidate_type: str,
    requested_impact: ImpactScope,
    applied_impact: ImpactScope,
    selected: Sequence[Benchmark],
    holdout: Sequence[Benchmark],
    manifest_digest: str,
) -> dict[str, Any]:
    """Decision-history fields (spec §36) for one impact-aware suite selection.

    Records whether the selection was narrowed to the candidate's impact scope
    or ran on the un-narrowed suite (no scope, FULL, or the empty-narrowing
    fallback). Pure and recorded after ``select_evaluation_suite`` returned, so
    it cannot change the selection. Sealed holdout tasks are counted, never
    named. The rules are deterministic and carry no calibrated confidence.
    """
    suite = Suite(str(suite).upper())
    return {
        "decision_type": "suite_selection",
        "input_features": {
            "suite": suite.value,
            "candidate_type": str(getattr(candidate_type, "value", candidate_type)),
            "requested_impact": requested_impact.to_dict(),
            "applied_impact": applied_impact.to_dict(),
            "narrowing_fell_back": not requested_impact.empty and applied_impact.empty,
            "selected_tasks": len(selected),
            "holdout_tasks": len(holdout),
            "suite_manifest_digest": manifest_digest,
        },
        "available_options": ("impact_narrowed", "unnarrowed"),
        "selected_option": (
            "unnarrowed" if applied_impact.empty or suite is Suite.FULL else "impact_narrowed"
        ),
        "confidence": None,
        "source": "suite-selection-rules",
        "model_version": SUITE_SELECTION_RULES_VERSION,
    }
