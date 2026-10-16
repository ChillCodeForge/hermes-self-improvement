from __future__ import annotations

import pytest
import json

from hermes_improve.qwen import OpenAICompatibleQwen, QwenConfig, _parse_judgment, judge_pairwise
from hermes_improve.self_evolution import EvolutionLimits, SelfEvolutionAdapter, evolve_candidates
from hermes_improve.security import AccessDenied, SpentHoldoutRegistry
from hermes_improve.statistics import paired_mean_confidence_interval


def test_qwen_config_defaults_to_loopback_only(monkeypatch):
    monkeypatch.delenv("QWEN_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="localhost"):
        QwenConfig("https://example.com", "model").validate_endpoint()
    QwenConfig("http://127.0.0.1:8000", "local-model").validate_endpoint()


def test_qwen_uses_judge_model_and_tracks_role_metrics(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size=-1):
            return json.dumps({
                "choices": [{"message": {"content": "{\"winner\":\"A\",\"confidence\":0.9}"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            }).encode()

    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    model = OpenAICompatibleQwen(QwenConfig(
        "http://127.0.0.1:8000", "solver-model", judge_model="judge-model",
        api_key="test-api-key",
    ))
    content = model.complete(role="qwen_judge", system_prompt="judge", user_payload="evidence")
    assert content.startswith("{")
    body = json.loads(requests[0][0].data)
    assert body["model"] == "judge-model"
    assert requests[0][0].get_header("Authorization") == "Bearer test-api-key"
    assert model.stats["qwen_judge"]["requests"] == 1
    assert model.stats["qwen_judge"]["input_tokens"] == 12
    assert model.stats["qwen_judge"]["output_tokens"] == 5


def test_invalid_qwen_judge_json_is_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_judgment("nope")
    with pytest.raises(ValueError, match="confidence"):
        _parse_judgment('{"winner":"A","confidence":2}')


def test_pairwise_judge_swaps_labels_and_abstains_on_position_bias():
    class Judge:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return '{"winner":"B","confidence":0.9}'

    judge = Judge()
    assert judge_pairwise(judge, "baseline", "candidate") == "unreliable"
    assert judge.calls == 2


def test_pairwise_judge_maps_consistent_candidate_preference():
    class Judge:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return '{"winner":"B","confidence":0.9}' if self.calls == 1 else '{"winner":"A","confidence":0.9}'

    assert judge_pairwise(Judge(), "baseline", "candidate") == "candidate"


def test_self_evolution_adapter_pins_external_strategy_version():
    class Strategy:
        version = "pinned-1"

        def evolve(self, candidate, feedback, generation):
            return [candidate + "-a", candidate + "-b"]

    adapter = SelfEvolutionAdapter("pinned-1", Strategy())
    assert evolve_candidates(
        adapter, "base", "feedback",
        limits=EvolutionLimits(max_generations=2, candidates_per_generation=1, max_candidates_total=2),
    ) == [["base-a"], ["base-a-a"]]
    with pytest.raises(RuntimeError, match="version mismatch"):
        SelfEvolutionAdapter("pinned-2", Strategy())


def test_paired_confidence_interval_is_repeatable():
    low, high = paired_mean_confidence_interval([3, 4, 5])
    assert low < 4 < high
    assert paired_mean_confidence_interval([4]) == (4, 4)


def test_spent_holdout_feedback_is_persisted_and_not_reusable(tmp_path):
    path = tmp_path / "holdout.sqlite3"
    first = SpentHoldoutRegistry(str(path))
    first.mark_spent("holdout-v1", "candidate-1")
    assert SpentHoldoutRegistry(str(path)).is_spent("holdout-v1", "candidate-1")
    with pytest.raises(AccessDenied, match="already consumed"):
        SpentHoldoutRegistry(str(path)).mark_spent("holdout-v1", "candidate-1")
