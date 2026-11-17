from __future__ import annotations

from pathlib import Path

import pytest

from hermes_improve.candidates import Generator
from hermes_improve.curator import CuratorAdapter, ImprovementSignal
from hermes_improve.dreaming import DreamingEngine
from hermes_improve.models import CandidateType
from hermes_improve.store import ExperimentStore


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


def test_light_dreaming_includes_caller_supplied_read_only_curator_signals():
    adapter = CuratorAdapter(lambda: [
        {"name": "sample-skill", "use_count": 5, "patch_count": 1, "activity_count": 3},
    ])

    report = DreamingEngine(curator=adapter).light([])

    assert report.sessions_processed == 0
    assert report.clusters["curator"] == 1
    assert report.signals[0].source == "curator"
    assert report.signals[0].evidence["name"] == "sample-skill"


@pytest.mark.parametrize(
    "row",
    [
        {"name": "sample", "use_count": True},
        {"name": "sample", "use_count": -1},
        {"name": "sample", "use_count": 1.5},
        {"name": "sample", "use_count": 1 << 64},
        {"name": "  ", "use_count": 1},
        {"name": "Bearer abcd1234", "use_count": 1},
        None,
    ],
)
def test_curator_adapter_rejects_malformed_or_unsafe_rows(row):
    with pytest.raises(ValueError):
        CuratorAdapter(lambda: [row]).signals()
