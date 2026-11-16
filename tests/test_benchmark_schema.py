from __future__ import annotations

import json
import sys

import pytest
import yaml

from benchmark_factory import record
from hermes_improve.benchmark_schema import (
    BenchmarkRecord,
    TAXONOMY_CATEGORIES,
    benchmark_json_schema,
    decode_benchmark_document,
    parse_benchmark_document,
)
from hermes_improve.benchmarks import BenchmarkStore


def test_json_schema_export_covers_every_spec_section():
    schema = benchmark_json_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert set(schema["required"]) == {
        "schema_version", "id", "version", "source", "task", "taxonomy", "environment",
        "eligibility", "verification", "ground_truth", "benchmark", "security",
    }
    definitions = schema["$defs"]
    assert {"session_id", "verified_at", "provenance"} <= set(definitions["SourceSection"]["properties"])
    assert {
        "repository_fixture", "git_commit", "dependency_digest", "container_image",
        "network_policy", "input_data",
    } <= set(definitions["EnvironmentSection"]["properties"])
    assert "class" in definitions["EligibilitySection"]["properties"]
    assert {"split", "suites", "cluster_id", "criticality"} <= set(
        definitions["BenchmarkSection"]["properties"]
    )


def test_cli_prints_the_schema(monkeypatch, capsys, tmp_path):
    from hermes_improve.cli import main

    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "benchmarks", "schema"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["title"] == "HermesBenchmarkRecord"


def test_yaml_and_json_documents_parse_to_identical_benchmarks():
    data = record("yaml-task")
    from_json = parse_benchmark_document(decode_benchmark_document("a.json", json.dumps(data).encode()))
    from_yaml = parse_benchmark_document(
        decode_benchmark_document("a.yaml", yaml.safe_dump(data, sort_keys=False).encode())
    )
    assert from_json.digest == from_yaml.digest


def test_store_loads_yaml_records_and_skips_fixture_and_manifest_trees(tmp_path):
    root = tmp_path / "public"
    (root / "train").mkdir(parents=True)
    (root / "train" / "a.yaml").write_text(yaml.safe_dump(record("yaml-a")), encoding="utf-8")
    (root / "train" / "b.json").write_text(json.dumps(record("json-b")), encoding="utf-8")
    (root / "fixtures" / "task").mkdir(parents=True)
    (root / "fixtures" / "task" / "input.json").write_text('{"not": "a benchmark"}', encoding="utf-8")
    (root / "manifests").mkdir()
    (root / "manifests" / "corpus.json").write_text("{}", encoding="utf-8")
    assert sorted(item.id for item in BenchmarkStore(root).validate()) == ["json-b", "yaml-a"]


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("dup.json", b'{"id": "a", "id": "b"}', "duplicate JSON key"),
        ("dup.yaml", b"id: a\nid: b\n", "duplicate YAML key"),
        ("date.yaml", b"verified_at: 2024-01-01\n", "non-JSON value"),
        ("list.yaml", b"- a\n", "must be an object"),
        ("bad.txt", b"{}", ".json, .yaml or .yml"),
        ("tags.yaml", b"x: !!python/object:os.system {}\n", "invalid YAML"),
    ],
)
def test_document_decoding_rejects_ambiguous_or_unsafe_input(name, payload, message):
    with pytest.raises(ValueError, match=message):
        decode_benchmark_document(name, payload)


@pytest.mark.parametrize(
    ("mutate", "location"),
    [
        (lambda data: data.__setitem__("version", "1"), "version"),
        (lambda data: data["security"].__setitem__("sanitized", "yes"), "security.sanitized"),
        (lambda data: data["benchmark"].__setitem__("split", "PUBLIC"), "benchmark.split"),
        (lambda data: data.__setitem__("unexpected", 1), "unexpected"),
        (lambda data: data["taxonomy"].pop("categories"), "taxonomy.categories"),
        (lambda data: data["eligibility"].__setitem__("reproducibility_score", 1.5), "reproducibility_score"),
    ],
)
def test_structural_schema_rejects_wrong_types_and_unknown_fields(mutate, location):
    data = record("strict-task")
    mutate(data)
    with pytest.raises(ValueError, match="schema validation failed") as error:
        parse_benchmark_document(data, source="record.json")
    assert location in str(error.value)


def test_schema_errors_never_echo_input_values():
    data = record("secretive-task")
    data["task"]["prompt"] = ["SEALED-HOLDOUT-ANSWER-TEXT"]
    with pytest.raises(ValueError) as error:
        parse_benchmark_document(data)
    assert "SEALED-HOLDOUT-ANSWER-TEXT" not in str(error.value)


def test_semantic_validation_still_runs_after_structural_parsing():
    data = record("weak-hard-task")
    data["eligibility"]["reproducibility_score"] = 0.5
    BenchmarkRecord.model_validate(data)
    with pytest.raises(ValueError, match="at least 0.8"):
        parse_benchmark_document(data)


def test_taxonomy_matches_shared_category_list():
    assert TAXONOMY_CATEGORIES == (
        "coding", "git", "shell", "data-transform", "docs", "devops", "research", "mcp",
    )
