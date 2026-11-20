from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .clustering import cluster_texts, task_fingerprint
from .curator import CuratorAdapter
from .ingestion import OUTCOME_LABELS, OUTCOME_SOURCES
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


_MAX_ERROR_LENGTH = 180
_MAX_ERROR_SIGNALS = 10
# Volatile tokens masked before clustering, applied in this order. Patterns
# run on case-folded text.
_VOLATILE_TOKENS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<id>"),
    (re.compile(r"(?:\b[a-z]:)?[\w.~-]*(?:[\\/][\w.~-]+)+[\\/]?"), "<path>"),
    (re.compile(r"\b0x[0-9a-f]+\b|\b(?=[0-9a-f]*\d)[0-9a-f]{8,}\b"), "<id>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<n>"),
)


def normalize_error(text: str) -> str:
    """Case-fold an error message and mask paths, ids and numbers.

    Errors that differ only in such volatile tokens normalize to the same
    string and therefore always share a LIGHT cluster.
    """
    normalized = text.strip().casefold()
    for pattern, placeholder in _VOLATILE_TOKENS:
        normalized = pattern.sub(placeholder, normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True)
class ErrorCluster:
    """One group of similar LIGHT errors.

    ``cluster_id`` is ``err-`` plus 16 hex digits and depends only on the set
    of normalized member texts; ``representative`` is the most frequent raw
    (case-folded, truncated) variant, ties broken lexicographically.
    """

    cluster_id: str
    representative: str
    count: int
    variants: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "representative": self.representative,
            "count": self.count,
            "variants": self.variants,
        }


@dataclass(frozen=True)
class DreamingReport:
    sessions_processed: int
    clusters: dict[str, int]
    signals: tuple[ImprovementSignal, ...]
    hypotheses: tuple[dict[str, Any], ...] = ()
    error_clusters: tuple[ErrorCluster, ...] = ()
    rejected_hypotheses: tuple[dict[str, Any], ...] = ()


def cluster_errors(
    errors: Mapping[str, int], tool_failures: Mapping[str, int] | None = None,
) -> tuple[ErrorCluster, ...]:
    """Group raw error strings into deterministic clusters without a model.

    Free-text errors are normalized (:func:`normalize_error`, exact match) and
    the distinct normalized texts are clustered lexically with
    :func:`clustering.cluster_texts` (fingerprint + MinHash/LSH shingle
    Jaccard, no semantic layer). Tool failures are keyed by tool name only and
    never merged with each other. The result is sorted by descending count,
    then cluster id, and does not depend on input order.
    """
    by_normalized: dict[str, Counter[str]] = defaultdict(Counter)
    for raw, count in errors.items():
        by_normalized[normalize_error(raw)][raw] += count
    assignment = cluster_texts({text: text for text in by_normalized})
    members: dict[str, Counter[str]] = defaultdict(Counter)
    for text, variants in by_normalized.items():
        members["err-" + assignment[text].removeprefix("sem-")].update(variants)
    for name, count in (tool_failures or {}).items():
        raw = f"tool {name} failed"
        members["err-" + task_fingerprint(raw)[:16]][raw] += count
    clusters = [
        ErrorCluster(
            cluster_id=cluster_id,
            representative=min(variants.items(), key=lambda item: (-item[1], item[0]))[0],
            count=sum(variants.values()),
            variants=len(variants),
        )
        for cluster_id, variants in members.items()
    ]
    clusters.sort(key=lambda cluster: (-cluster.count, cluster.cluster_id))
    return tuple(clusters)


_MAX_SIGNAL_SESSION_IDS = 20


def _labelled_failure_signal(sessions: Sequence[Mapping[str, Any]]) -> ImprovementSignal | None:
    """One LIGHT signal for sessions an operator labelled ``incorrect``/``partial``.

    ``final_status`` only says that the conversation ended; a labelled failure
    is the only correctness evidence LIGHT has (finding F11).
    """
    labelled = 0
    failures: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    session_ids: list[str] = []
    for session in sessions:
        outcome = _outcome_label(session.get("outcome"))
        if outcome is None:
            continue
        labelled += 1
        if outcome["label"] == "correct":
            continue
        failures[outcome["label"]] += 1
        sources[outcome["source"]] += 1
        session_id = session.get("session_id")
        if isinstance(session_id, str) and len(session_ids) < _MAX_SIGNAL_SESSION_IDS:
            session_ids.append(session_id)
    total = sum(failures.values())
    if not total:
        return None
    return ImprovementSignal(
        source="dreaming-light",
        summary=f"{total} of {labelled} labelled sessions were incorrect or partial",
        issue="sessions ended (final_status) but their outcome was labelled incorrect or partial",
        impact_score=min(1.0, total / max(1, len(sessions))),
        evidence={
            "kind": "labelled_outcome",
            "incorrect": failures["incorrect"],
            "partial": failures["partial"],
            "labelled": labelled,
            "sources": dict(sorted(sources.items())),
            "session_ids": session_ids,
        },
    )


class DreamingEngine:
    """LIGHT works without an LLM; model-backed stages only produce proposals."""

    def __init__(
        self, model: DreamingModel | None = None, curator: CuratorAdapter | None = None,
    ) -> None:
        self.model = model
        self.curator = curator

    def light(self, sessions: Sequence[Mapping[str, Any]]) -> DreamingReport:
        errors: Counter[str] = Counter()
        tool_failures: Counter[str] = Counter()
        tools: Counter[str] = Counter()
        for session in sessions:
            for error in session.get("errors", []) if isinstance(session.get("errors", []), list) else []:
                if isinstance(error, str) and error.strip():
                    errors[error.strip().casefold()[:_MAX_ERROR_LENGTH]] += 1
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
                        tool_failures[name] += failures
            else:
                for tool in session.get("tool_calls", []) if isinstance(session.get("tool_calls", []), list) else []:
                    if isinstance(tool, Mapping) and isinstance(tool.get("name"), str):
                        tools[tool["name"]] += 1
        error_clusters = cluster_errors(errors, tool_failures)
        signals = []
        for cluster in error_clusters[:_MAX_ERROR_SIGNALS]:
            if cluster.count > 1:
                signals.append(ImprovementSignal(
                    source="dreaming-light", summary=f"Recurring error observed {cluster.count} times",
                    issue=cluster.representative,
                    impact_score=min(1.0, cluster.count / max(1, len(sessions))),
                    evidence={
                        "occurrences": cluster.count,
                        "cluster_id": cluster.cluster_id,
                        "variants": cluster.variants,
                    },
                ))
        outcome_signal = _labelled_failure_signal(sessions)
        if outcome_signal is not None:
            signals.append(outcome_signal)
        curator_signals = self.curator.signals() if self.curator is not None else []
        signals.extend(curator_signals)
        clusters = {"errors": len(error_clusters), "tools": len(tools)}
        if outcome_signal is not None:
            clusters["labelled_failures"] = outcome_signal.evidence["incorrect"] + outcome_signal.evidence["partial"]
        if self.curator is not None:
            clusters["curator"] = len(curator_signals)
        return DreamingReport(
            sessions_processed=len(sessions),
            clusters=clusters,
            signals=tuple(signals),
            error_clusters=error_clusters,
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
        kept, rejected = _validate_hypotheses(_parse_hypotheses(response), sessions)
        clusters = dict(light.clusters)
        if rejected:
            clusters["rejected_hypotheses"] = len(rejected)
        return DreamingReport(
            light.sessions_processed, clusters, light.signals, tuple(kept), light.error_clusters,
            tuple(rejected),
        )

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
        "hypothesis", "summary", "issue", "kind",
    }
    projected = {key: value[key] for key in allowed if key in value}
    outcome = _outcome_label(value.get("outcome"))
    if outcome is not None:
        projected["outcome"] = {
            key: outcome[key] for key in ("label", "source", "note")
            if isinstance(outcome.get(key), str) and (key != "note" or outcome[key])
        }
    return {"untrusted_evidence": projected}


def _outcome_label(value: Any) -> Mapping[str, Any] | None:
    """The session's outcome label if it is well formed, else ``None``."""
    if (
        isinstance(value, Mapping)
        and value.get("label") in OUTCOME_LABELS
        and value.get("source") in OUTCOME_SOURCES
    ):
        return value
    return None


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


_HYPOTHESIS_KIND = re.compile(r"\s*(correctness|efficiency)\s*:", re.IGNORECASE)
_FAILED_LABELS = {"incorrect", "partial"}


def _hypothesis_kind(summary: str) -> str:
    """``correctness``/``efficiency`` from the summary prefix, else ``unknown``."""
    match = _HYPOTHESIS_KIND.match(summary)
    return match.group(1).casefold() if match else "unknown"


def _validate_hypotheses(
    hypotheses: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check REM hypotheses against the sessions REM actually received (F31).

    Every hypothesis gets a ``kind`` from its summary prefix. Evidence ids that
    are not among ``sessions`` are removed (and listed in
    ``unknown_evidence_session_ids``); a hypothesis left without evidence is
    rejected. An ``efficiency`` hypothesis is rejected as soon as one of its
    evidence sessions is labelled ``incorrect`` or ``partial``: efficiency may
    only rest on sessions that were right. Unlabelled evidence cannot be
    checked, so such an efficiency hypothesis is kept with
    ``outcome_verified: false`` (``true`` once a ``correct`` session backs it).
    ``correctness`` and ``unknown`` hypotheses are kept.

    Returns ``(kept, rejected)``; each rejected item carries a ``reason``.
    """
    labels: dict[str, str | None] = {}
    for session in sessions:
        session_id = session.get("session_id")
        if isinstance(session_id, str):
            outcome = _outcome_label(session.get("outcome"))
            labels[session_id] = outcome["label"] if outcome is not None else None
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in hypotheses:
        cited = list(dict.fromkeys(item["evidence_session_ids"]))
        known = [session_id for session_id in cited if session_id in labels]
        unknown = [session_id for session_id in cited if session_id not in labels]
        hypothesis = {**item, "kind": _hypothesis_kind(item["summary"]), "evidence_session_ids": known}
        if unknown:
            hypothesis["unknown_evidence_session_ids"] = unknown
        if not known:
            reason = "unknown_evidence" if unknown else "no_evidence"
            rejected.append({**hypothesis, "reason": reason})
            continue
        if hypothesis["kind"] == "efficiency":
            failed = [session_id for session_id in known if labels[session_id] in _FAILED_LABELS]
            if failed:
                rejected.append({**hypothesis, "reason": "efficiency_on_failed_sessions", "failed_session_ids": failed})
                continue
            hypothesis["outcome_verified"] = any(labels[session_id] == "correct" for session_id in known)
        kept.append(hypothesis)
    return kept, rejected
