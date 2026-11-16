"""Shared benchmark record factory for WP-B tests."""

from __future__ import annotations

import copy
import hashlib

_WORDS = (
    "archive", "bracket", "cobalt", "delta", "ember", "falcon", "glacier", "harbor", "indigo",
    "juniper", "kettle", "lantern", "meadow", "nectar", "orchid", "pepper", "quartz", "raven",
    "saffron", "tundra", "umber", "velvet", "willow", "xenon", "yonder", "zephyr",
)


def distinct_prompt(identifier: str) -> str:
    """Deterministic prompt with vocabulary unlikely to collide with other identifiers."""
    digest = hashlib.sha256(identifier.encode()).digest()
    words = [_WORDS[byte % len(_WORDS)] + str(byte) for byte in digest[:12]]
    return "Handle " + " ".join(words)


def record(
    identifier: str,
    split: str = "TRAIN",
    cluster: str | None = None,
    *,
    prompt: str | None = None,
    categories: list[str] | None = None,
    capabilities: list[str] | None = None,
    suites: list[str] | None = None,
    criticality: str = "normal",
    eligibility: str = "HARD_BENCHMARK",
    impact_areas: list[str] | None = None,
) -> dict:
    data = {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "manual", "provenance": "test-fixture"},
        "task": {
            "title": identifier,
            "prompt": prompt if prompt is not None else distinct_prompt(identifier),
        },
        "taxonomy": {
            "categories": categories if categories is not None else ["coding"],
            "difficulty": "easy",
            "required_capabilities": capabilities if capabilities is not None else [],
        },
        "environment": {"type": "fixture", "network_policy": "disabled"},
        "eligibility": {
            "class": eligibility, "reproducibility_score": 0.9, "replay_verified": True,
        },
        "verification": {"deterministic": [{"type": "exit_code"}], "subjective": []},
        "ground_truth": {"type": "deterministic_tests", "confidence": 1.0},
        "benchmark": {
            "split": split,
            "suites": suites if suites is not None else ["SMOKE", "CORE", "FULL"],
            "cluster_id": cluster or f"cluster-{identifier}",
            "criticality": criticality,
        },
        "security": {"sanitized": True, "contains_secrets": False},
    }
    if impact_areas is not None:
        data["benchmark"]["impact_areas"] = impact_areas
    return copy.deepcopy(data)
