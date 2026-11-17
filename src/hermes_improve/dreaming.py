from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .curator import CuratorAdapter
from .models import CandidateType, ImprovementSignal
from .prompting import load_prompt, wrap_untrusted_evidence
from .skill_paths import validate_skill_target


_SKILL_CANDIDATE_TYPES = {
    CandidateType.SKILL_CREATE.value,
    CandidateType.SKILL_PATCH.value,
    CandidateType.SKILL_MERGE.value,
    CandidateType.SKILL_SPLIT.value,
}


class DreamingModel(Protocol):
    def complete(self, *, role: str, system_prompt: str, user_payload: str, json_object: bool = False) -> str: ...


@dataclass(frozen=True)
class DreamingReport:
    sessions_processed: int
    clusters: dict[str, int]
    signals: tuple[ImprovementSignal, ...]
    hypotheses: tuple[dict[str, Any], ...] = ()


class DreamingEngine:
    """LIGHT works without an LLM; model-backed stages only produce proposals."""

    def __init__(
        self, model: DreamingModel | None = None, curator: CuratorAdapter | None = None,
    ) -> None:
        self.model = model
        self.curator = curator

    def light(self, sessions: Sequence[Mapping[str, Any]]) -> DreamingReport:
        errors: Counter[str] = Counter()
        tools: Counter[str] = Counter()
        for session in sessions:
            for error in session.get("errors", []) if isinstance(session.get("errors", []), list) else []:
                if isinstance(error, str):
                    errors[error.strip().casefold()[:180]] += 1
            tool_stats = session.get("tool_stats")
            if isinstance(tool_stats, Mapping):
                for name, stats in tool_stats.items():
                    if not isinstance(name, str) or not isinstance(stats, Mapping):
                        continue
                    count = stats.get("count", 0)
                    failures = stats.get("failure", 0)
                    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                        tools[name] += count
                    if isinstance(failures, int) and not isinstance(failures, bool) and failures > 0:
                        errors[f"tool {name} failed"] += failures
            else:
                for tool in session.get("tool_calls", []) if isinstance(session.get("tool_calls", []), list) else []:
                    if isinstance(tool, Mapping) and isinstance(tool.get("name"), str):
                        tools[tool["name"]] += 1
        signals = []
        for error, count in errors.most_common(10):
            if count > 1:
                signals.append(ImprovementSignal(
                    source="dreaming-light", summary=f"Recurring error observed {count} times",
                    issue=error, impact_score=min(1.0, count / max(1, len(sessions))),
                    evidence={"occurrences": count},
                ))
        curator_signals = self.curator.signals() if self.curator is not None else []
        signals.extend(curator_signals)
        clusters = {"errors": len(errors), "tools": len(tools)}
        if self.curator is not None:
            clusters["curator"] = len(curator_signals)
        return DreamingReport(
            sessions_processed=len(sessions),
            clusters=clusters,
            signals=tuple(signals),
        )

    def rem(self, sessions: Sequence[Mapping[str, Any]]) -> DreamingReport:
        if self.model is None:
            raise RuntimeError("REM dreaming requires a configured local model")
        light = self.light(sessions)
        system_prompt = load_prompt("dreaming/rem.md")
        payload = wrap_untrusted_evidence(
            {"sessions": [_untrusted_projection(session) for session in sessions]}
        )
        response = self.model.complete(
            role="qwen_dreamer",
            system_prompt=system_prompt,
            user_payload=payload,
            json_object=True,
        )
        parsed = _parse_hypotheses(response)
        return DreamingReport(light.sessions_processed, light.clusters, light.signals, tuple(parsed))

    def deep(self, hypotheses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("DEEP dreaming requires a configured local model")
        bounded = list(hypotheses[:5])
        system_prompt = load_prompt("dreaming/deep.md")
        payload = wrap_untrusted_evidence(
            {"hypotheses": [_untrusted_projection(item) for item in bounded]}
        )
        response = self.model.complete(
            role="qwen_optimizer",
            system_prompt=system_prompt,
            user_payload=payload,
            json_object=True,
        )
        value = json.loads(response)
        if not isinstance(value, dict):
            raise ValueError("dreaming response must be a JSON object")
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("dreaming response must contain a candidates array")
        safe = []
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary")
            candidate_type = item.get("candidate_type")
            target = item.get("target", "")
            hypothesis = item.get("hypothesis")
            if (
                candidate_type not in _SKILL_CANDIDATE_TYPES
                or not isinstance(target, str)
                or not isinstance(summary, str) or not isinstance(hypothesis, str)
                or not summary.strip() or not hypothesis.strip()
                or len(summary) > 500 or len(hypothesis) > 2000
            ):
                continue
            try:
                validate_skill_target(target)
            except ValueError:
                continue
            safe.append({
                "summary": summary,
                "candidate_type": candidate_type,
                "target": target,
                "hypothesis": hypothesis,
            })
        return safe


def _untrusted_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Only pass bounded data fields; all content remains explicitly evidence."""
    allowed = {
        "session_id", "task", "errors", "tool_calls", "tool_stats", "final_status",
        "hypothesis", "summary", "issue",
    }
    projected = {key: value[key] for key in allowed if key in value}
    return {"untrusted_evidence": projected}


def _parse_hypotheses(response: str) -> list[dict[str, Any]]:
    value = json.loads(response)
    if not isinstance(value, dict):
        raise ValueError("dreaming response must be a JSON object")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("dreaming response must contain a hypotheses array")
    result = []
    for item in hypotheses:
        if not isinstance(item, dict) or not isinstance(item.get("summary"), str) or not isinstance(item.get("issue"), str):
            continue
        ids = item.get("evidence_session_ids", [])
        if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
            continue
        result.append({
            "summary": item["summary"][:500],
            "issue": item["issue"][:1000],
            "evidence_session_ids": ids[:20],
        })
    return result[:20]
