from __future__ import annotations

from pathlib import Path

import pytest

from hermes_improve.candidates import Generator
from hermes_improve.curator import CuratorAdapter, ImprovementSignal
from hermes_improve.models import CandidateType
from hermes_improve.store import ExperimentStore
from hermes_improve.verifiers import (
    ExitCodeVerifier,
    FileExistsVerifier,
    FileContentVerifier,
    JsonSchemaVerifier,
    RegexVerifier,
    VerifierRegistry,
)


def test_generator_persists_a_proposal_without_writing_production(tmp_path):
    experiments = ExperimentStore(tmp_path / "state.sqlite3")
    experiments.create_experiment(
        "exp-1", lineage_id="lineage-1", hypothesis="improve",
        candidate_type="SKILL_PATCH", target="brain/skills/example",
        risk="LOW", baseline_commit="base", benchmark_version="v1",
        policy_version="1",
    )
    generator = Generator(experiments)
    signal = ImprovementSignal("curator", "usage signal", "review skill", 0.7)
    candidate = generator.create_candidate(
        "exp-1", signal, candidate_type=CandidateType.SKILL_PATCH,
        target="brain/skills/example", base_commit="base",
        candidate_commit="candidate-commit", diff="+ improved proposal\n",
    )
    assert candidate.content_hash
    assert experiments.show_experiment("exp-1")["candidates"][0]["id"] == candidate.candidate_id
    assert not (tmp_path / "brain").exists()


def test_curator_adapter_emits_signals_without_mutation():
    adapter = CuratorAdapter(lambda: [
        {"name": "unused", "use_count": 0, "patch_count": 0, "activity_count": 0},
        {"name": "used", "use_count": 4, "patch_count": 1, "activity_count": 7},
    ])
    signals = adapter.signals()
    assert len(signals) == 1
    assert signals[0].source == "curator"
    assert signals[0].evidence["name"] == "used"


def test_file_verifiers_reject_path_escape(tmp_path):
    assert not FileExistsVerifier("../outside").verify(tmp_path).passed
    assert not FileContentVerifier("../outside", "x").verify(tmp_path).passed


def test_file_json_regex_and_command_verifiers(tmp_path):
    (tmp_path / "settings.json").write_text('{"enabled": true}', encoding="utf-8")
    assert FileExistsVerifier("settings.json").verify(tmp_path).passed
    assert FileContentVerifier("settings.json", "enabled").verify(tmp_path).passed
    assert JsonSchemaVerifier(("enabled",)).verify('{"enabled": true}').passed
    assert RegexVerifier(r"^release-\d+$").verify("release-4").passed
    assert ExitCodeVerifier(("python", "-c", "raise SystemExit(0)")).verify(tmp_path).passed


def test_registry_rejects_duplicate_verifier_names():
    registry = VerifierRegistry()
    verifier = RegexVerifier(".*")
    registry.register("match", verifier)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("match", verifier)
