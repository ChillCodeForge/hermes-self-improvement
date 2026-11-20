"""Blind pairwise LLM judge for subjective benchmarks (spec §22).

The judge is only consulted when a benchmark has no deterministic verifiers
and declares subjective criteria. It sees each pair twice with swapped
positions; position-dependent answers, low confidence, high uncertainty and
invalid JSON (after a bounded number of retries) all yield ``unreliable``.
A judge verdict can block a candidate but never promote one on its own.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


JUDGE_ROLE = "qwen_judge"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "judge"
_SYSTEM_PROMPT_FILE = "pairwise_system.md"
_RESULTS = ("candidate", "baseline", "tie", "unreliable")
# Bounds for the per-order judgments kept in an aggregated (signed) verdict.
MAX_AGGREGATED_PRESENTATIONS = 20
MAX_PRESENTATION_RATIONALE_CHARS = 500


class ChatModel(Protocol):
    def complete(self, *, role: str, system_prompt: str, user_payload: str, json_object: bool = False) -> str: ...


@dataclass(frozen=True)
class JudgeConfig:
    prompt_dir: Path = DEFAULT_PROMPT_DIR
    max_attempts: int = 2
    min_confidence: float = 0.5
    max_uncertainty: float = 0.5
    max_response_chars: int = 20_000

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or not 1 <= self.max_attempts <= 5:
            raise ValueError("judge max_attempts must be between 1 and 5")
        for name in ("min_confidence", "max_uncertainty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"judge {name} must be between 0 and 1")
        if isinstance(self.max_response_chars, bool) or not isinstance(self.max_response_chars, int) \
                or self.max_response_chars < 100:
            raise ValueError("judge max_response_chars must be an integer >= 100")


@dataclass(frozen=True)
class Judgment:
    winner: str
    confidence: float
    uncertainty: float
    rationale: str


@dataclass(frozen=True)
class JudgeVerdict:
    result: str
    confidence: float | None
    uncertainty: float | None
    reasons: tuple[str, ...] = ()
    attempts: int = 0
    prompt_sha256: str = ""
    presentations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    presentations_omitted: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["presentations"] = list(self.presentations)
        return data


def parse_judgment(response: str) -> Judgment:
    """Strictly parse one judge answer; any deviation raises ``ValueError``."""
    if not isinstance(response, str):
        raise ValueError("judge response must be text")
    try:
        value = json.loads(response.strip(), parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("judge response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    unknown = set(value) - {"winner", "confidence", "uncertainty", "rationale"}
    if unknown:
        raise ValueError(f"judge response has unexpected fields: {sorted(unknown)}")
    if value.get("winner") not in {"A", "B", "TIE"}:
        raise ValueError("judge response has invalid winner")
    numbers = {}
    for key in ("confidence", "uncertainty"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) \
                or not 0 <= number <= 1:
            raise ValueError(f"judge {key} must be between zero and one")
        numbers[key] = float(number)
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > 2000:
        raise ValueError("judge rationale must be a string of at most 2000 characters")
    return Judgment(value["winner"], numbers["confidence"], numbers["uncertainty"], rationale)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


class PairwiseJudge:
    def __init__(self, model: ChatModel, config: JudgeConfig | None = None) -> None:
        self.model = model
        self.config = config or JudgeConfig()
        prompt_path = Path(self.config.prompt_dir) / _SYSTEM_PROMPT_FILE
        if not prompt_path.is_file():
            raise FileNotFoundError(f"judge system prompt not found: {prompt_path}")
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        if not self.system_prompt.strip():
            raise ValueError("judge system prompt is empty")
        self.prompt_sha256 = hashlib.sha256(self.system_prompt.encode()).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "role": JUDGE_ROLE,
            "prompt_sha256": self.prompt_sha256,
            "max_attempts": self.config.max_attempts,
            "min_confidence": self.config.min_confidence,
            "max_uncertainty": self.config.max_uncertainty,
        }

    def compare(
        self,
        *,
        task_prompt: str,
        criteria: Sequence[str],
        baseline_output: str,
        candidate_output: str,
        baseline_files: Sequence[Mapping[str, Any]] = (),
        candidate_files: Sequence[Mapping[str, Any]] = (),
    ) -> JudgeVerdict:
        """Judge one pair twice with swapped positions.

        ``*_files`` are the deliverable files the benchmark's criteria declare
        (``deliverables.collect_deliverables`` rows); they reach the judge as
        ``files_A``/``files_B`` next to the final answers, framed as untrusted.
        """
        if not isinstance(task_prompt, str) or not all(isinstance(item, str) for item in criteria):
            raise TypeError("judge task and criteria must be strings")
        baseline = (baseline_output, tuple(baseline_files))
        candidate = (candidate_output, tuple(candidate_files))
        first, first_attempts, first_error = self._ask(task_prompt, criteria, baseline, candidate)
        second, second_attempts, second_error = self._ask(task_prompt, criteria, candidate, baseline)
        attempts = first_attempts + second_attempts
        presentations = (
            {"order": "A=baseline,B=candidate", "judgment": asdict(first) if first else None, "error": first_error},
            {"order": "A=candidate,B=baseline", "judgment": asdict(second) if second else None, "error": second_error},
        )
        if first is None or second is None:
            return JudgeVerdict(
                "unreliable", None, None,
                (f"judge returned no valid judgment after {self.config.max_attempts} attempts",),
                attempts, self.prompt_sha256, presentations,
            )
        preference_first = {"A": "baseline", "B": "candidate", "TIE": "tie"}[first.winner]
        preference_second = {"A": "candidate", "B": "baseline", "TIE": "tie"}[second.winner]
        confidence = min(first.confidence, second.confidence)
        uncertainty = max(first.uncertainty, second.uncertainty)
        if preference_first != preference_second:
            return JudgeVerdict(
                "unreliable", confidence, uncertainty,
                (f"position-dependent contradiction: {preference_first} vs {preference_second}",),
                attempts, self.prompt_sha256, presentations,
            )
        reasons = []
        if confidence < self.config.min_confidence:
            reasons.append(f"judge confidence {confidence:.2f} below {self.config.min_confidence:.2f}")
        if uncertainty > self.config.max_uncertainty:
            reasons.append(f"judge uncertainty {uncertainty:.2f} above {self.config.max_uncertainty:.2f}")
        if reasons:
            return JudgeVerdict(
                "unreliable", confidence, uncertainty, tuple(reasons), attempts, self.prompt_sha256, presentations
            )
        return JudgeVerdict(preference_first, confidence, uncertainty, (), attempts, self.prompt_sha256, presentations)

    def _ask(
        self, task_prompt: str, criteria: Sequence[str],
        arm_a: tuple[str, tuple[Mapping[str, Any], ...]], arm_b: tuple[str, tuple[Mapping[str, Any], ...]],
    ) -> tuple[Judgment | None, int, str | None]:
        limit = self.config.max_response_chars
        body: dict[str, Any] = {
            "task": task_prompt[:limit],
            "criteria": list(criteria),
            "response_A": arm_a[0][-limit:],
            "response_B": arm_b[0][-limit:],
        }
        if arm_a[1] or arm_b[1]:
            body["files_A"] = _render_files(arm_a[1], limit)
            body["files_B"] = _render_files(arm_b[1], limit)
        payload = json.dumps(body, ensure_ascii=False)
        error = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                answer = self.model.complete(
                    role=JUDGE_ROLE, system_prompt=self.system_prompt, user_payload=payload, json_object=True,
                )
                return parse_judgment(answer), attempt, None
            except (ValueError, RuntimeError) as exc:
                error = str(exc)[:500]
        return None, self.config.max_attempts, error


def _render_files(files: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, str]]:
    """Judge view of deliverable files: path, capture status and (bounded) content only."""
    return [
        {"path": str(item.get("path", "")), "status": str(item.get("status", "")),
         "content": str(item.get("content", ""))[:limit]}
        for item in files
    ]


def _aggregate_presentations(
    verdicts: Sequence[JudgeVerdict], seeds: Sequence[int] | None,
) -> tuple[tuple[dict[str, Any], ...], int]:
    """Per-run, per-order judgments for auditing position bias, bounded in count and size."""
    entries = []
    for index, verdict in enumerate(verdicts):
        for presentation in verdict.presentations:
            entry: dict[str, Any] = {"run": index}
            if seeds is not None:
                entry["seed"] = seeds[index]
            entry["order"] = presentation.get("order")
            judgment = presentation.get("judgment")
            if isinstance(judgment, dict):
                judgment = dict(judgment)
                rationale = judgment.get("rationale")
                if isinstance(rationale, str):
                    judgment["rationale"] = rationale[:MAX_PRESENTATION_RATIONALE_CHARS]
            entry["judgment"] = judgment
            error = presentation.get("error")
            entry["error"] = error[:MAX_PRESENTATION_RATIONALE_CHARS] if isinstance(error, str) else error
            entries.append(entry)
    kept = tuple(entries[:MAX_AGGREGATED_PRESENTATIONS])
    return kept, len(entries) - len(kept)


def aggregate_verdicts(verdicts: Sequence[JudgeVerdict], seeds: Sequence[int] | None = None) -> JudgeVerdict:
    """Combine per-seed verdicts; any unreliable or disagreeing run is unreliable.

    The per-seed, per-order presentations are kept (at most
    ``MAX_AGGREGATED_PRESENTATIONS`` entries, rationales and errors cut to
    ``MAX_PRESENTATION_RATIONALE_CHARS``) so position bias stays auditable in
    the signed evaluation; ``presentations_omitted`` counts what the cap dropped.
    """
    if not verdicts:
        raise ValueError("at least one judge verdict is required")
    if seeds is not None and len(seeds) != len(verdicts):
        raise ValueError("judge seeds must match the verdicts one to one")
    presentations, omitted = _aggregate_presentations(verdicts, seeds)
    audit = {"presentations": presentations, "presentations_omitted": omitted}
    results = {verdict.result for verdict in verdicts}
    confidences = [verdict.confidence for verdict in verdicts if verdict.confidence is not None]
    uncertainties = [verdict.uncertainty for verdict in verdicts if verdict.uncertainty is not None]
    confidence = min(confidences) if confidences else None
    uncertainty = max(uncertainties) if uncertainties else None
    attempts = sum(verdict.attempts for verdict in verdicts)
    reasons = tuple(reason for verdict in verdicts for reason in verdict.reasons)
    if "unreliable" in results:
        return JudgeVerdict(
            "unreliable", confidence, uncertainty, reasons, attempts, verdicts[0].prompt_sha256, **audit,
        )
    if len(results) > 1:
        return JudgeVerdict(
            "unreliable", confidence, uncertainty,
            reasons + (f"judge verdicts disagree across runs: {sorted(results)}",),
            attempts, verdicts[0].prompt_sha256, **audit,
        )
    result = results.pop()
    if result not in _RESULTS:
        raise ValueError(f"unknown judge result: {result!r}")
    return JudgeVerdict(result, confidence, uncertainty, reasons, attempts, verdicts[0].prompt_sha256, **audit)
