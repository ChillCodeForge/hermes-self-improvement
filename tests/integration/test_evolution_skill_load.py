"""Evolution screening reports skill variants the agent never loaded (lab finding F30)."""
from __future__ import annotations

import json
import re

from evolution_doubles import TARGET, ScriptedOptimizerModel, evolution_fixture, skill_text
from hermes_improve.evolution_loop import NEVER_LOADED_REASON, EvolutionConfig, EvolutionLoop
from hermes_improve.optimizer import ModelMutationStrategy


def _loop(fixture, model, **overrides):
    return EvolutionLoop(
        repository=fixture.repo, store=fixture.store, gateway=fixture.gateway(persist=False),
        strategy=ModelMutationStrategy(model), expected_strategy_version="1",
        config=EvolutionConfig.from_mapping({"max_generations": 2, **overrides}),
    )


def _evidence(call):
    payload = call["user_payload"]
    nonce = re.match(r"<<<UNTRUSTED EVIDENCE ([0-9a-f]{32}) BEGIN>>>\n", payload)
    body = payload[nonce.end():].split(f"\n<<<UNTRUSTED EVIDENCE {nonce.group(1)} END>>>")[0]
    return json.loads(body)["untrusted_evidence"]


def _variant(fixture, marker):
    rows = fixture.store.list_generated_variants(TARGET, 50)
    return next(row for row in rows if row["status"] != "root" and marker in _content_of(fixture, row))


def _content_of(fixture, row):
    from evolution_doubles import git

    candidate = fixture.store.get_candidate(row["candidate_id"])
    return git(fixture.repo, "show", f"{candidate['commit_id']}:{TARGET}/SKILL.md")


def test_never_loaded_variant_gets_an_explicit_reason_and_the_optimizer_sees_the_rate(tmp_path):
    fixture = evolution_fixture(tmp_path, root_content=skill_text("root quality=1 SKILL_READ"))
    model = ScriptedOptimizerModel([
        ["variant unread quality=0 SKILL_UNREAD", "variant same quality=1 SKILL_UNREAD alt"],
        [],
    ])
    _loop(fixture, model).run(fixture.root_id)

    smoke_rejected = _variant(fixture, "variant unread")
    assert smoke_rejected["status"] == "smoke_rejected"
    assert smoke_rejected["reason"].startswith(f"{NEVER_LOADED_REASON} in SMOKE")
    assert "smoke score below parent" in smoke_rejected["reason"]
    assert smoke_rejected["scores"]["smoke_skill_loaded_rate"] == 0.0

    no_gain = _variant(fixture, "variant same")
    assert no_gain["status"] == "core_no_improvement"
    assert no_gain["reason"].startswith(f"{NEVER_LOADED_REASON} in CORE")
    assert no_gain["scores"]["core_skill_loaded_rate"] == 0.0

    first, second = (_evidence(call) for call in model.calls)
    root_feedback = first["feedback"][0]
    assert root_feedback["skill_loaded_rate"] == 1.0
    assert all(task["skill_loaded_rate"] == 1.0 for task in root_feedback["train"])
    unread = [item for item in second["feedback"] if item["skill_loaded_rate"] == 0.0]
    assert unread, "the optimizer must see that a variant was never loaded"
    assert all(task["skill_loaded_rate"] == 0.0 for item in unread for task in item["train"])
    assert all(
        aggregate["candidate_skill_loaded_rate"] == 0.0 for item in unread for aggregate in item["aggregates"]
    )
    attempts = {attempt["status"]: attempt["scores"] for attempt in second["prior_attempts"]}
    assert attempts["smoke_rejected"]["smoke_skill_loaded_rate"] == 0.0


def test_loaded_or_unobserved_skill_use_adds_no_reason(tmp_path):
    fixture = evolution_fixture(tmp_path)  # no markers: skill use unobserved (command runner)
    model = ScriptedOptimizerModel([["variant low quality=0"], ["variant read quality=0 SKILL_READ"]])
    _loop(fixture, model, stop_after_no_improvement_generations=5).run(fixture.root_id)

    for marker in ("variant low", "variant read"):
        row = _variant(fixture, marker)
        assert row["status"] == "smoke_rejected"
        assert NEVER_LOADED_REASON not in row["reason"]
    feedback = _evidence(model.calls[1])["feedback"]
    assert feedback[0]["skill_loaded_rate"] is None
    assert any(item["skill_loaded_rate"] is None for item in feedback[1:])
