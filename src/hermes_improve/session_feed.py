"""Post-promotion monitoring evidence derived from ingested Hermes sessions.

Sessions reach the local queue through the existing ingestion paths (the
``on_session_finalize`` plugin hook, ``hermes-improve ingest`` of a session
export). This module turns those recorded sessions into
:class:`~hermes_improve.monitoring.ProductionObservation` rows for a
PROVISIONAL release:

* **Attribution.** A session counts for the release that was active while it
  ran. Activation times come from the append-only ``release_events`` audit
  (``PROMOTED`` and ``REACTIVATED`` events). A session whose time span crosses
  an activation, lies within ``switch_margin_seconds`` of one, has no usable
  timestamps, or is not a production session is excluded and counted by
  reason, never guessed.
* **Derivation.** Success, tool calls, tool errors, retries, agent latency,
  tokens (only when the export reports them, otherwise unknown) and user
  corrections are computed by the deterministic rules documented on each
  helper. Correction signals are heuristics and are recorded as such in the
  observation's ``derivation``.

Session content is untrusted evidence (spec §28): it is only parsed, counted
and pattern-matched. Nothing in it is executed, followed, or used to choose a
release, an observation ID, a source label or a status; the session queue is
never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from .ingestion import SessionQueue
from .monitoring import ACTIVATION_EVENTS, ProductionObservation, ReleaseMonitor
from .store import ExperimentStore


HEURISTICS_VERSION = "session-feed-1"
DEFAULT_SWITCH_MARGIN_SECONDS = 60.0
_MAX_SESSIONS = 10_000
_MAX_CORRECTION_TEXT = 200
_MAX_TOOL_ARGUMENT_CHARS = 10_000
_MAX_SKILLS = 32
_EARLIEST_TIMESTAMP = 946_684_800.0  # 2000-01-01T00:00:00Z
_FUTURE_TOLERANCE_SECONDS = 300.0
_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

# Hermes writes these exact prefixes for tool results it synthesised itself
# (agent/tool_executor.py ``_unfinished_tool_result`` at v0.22.2).
_TOOL_CANCELLED_PREFIX = "[Tool execution cancelled"
_TOOL_ERROR_PREFIX = "Error"
_FAILED_FINISH_REASONS = frozenset({"length", "content_filter", "error", "tool_calls"})

# Correction heuristics. Each pattern is anchored at the start of a user
# message that directly answers an assistant turn (see ``classify_correction``).
_UNDO_RETRY = re.compile(
    r"^(?:/(?:undo|retry|redo|rollback)\b"
    r"|(?:undo|revert|redo|retry)\b"
    r"|roll ?(?:it |that )?back\b"
    r"|try (?:it |that )?again\b"
    r"|(?:mach(?:e)? (?:das |es )?)?rückgängig\b"
    r"|(?:noch ?(?:ein)?mal|versuch(?:'?s| es) (?:noch ?(?:ein)?mal|erneut))\b)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"^(?:(?:no|nope|nah)\b(?!\s+(?:problem|worries|rush)\b)"
    r"|(?:wrong|incorrect)\b"
    r"|(?:that'?s|that is|this is|it'?s|it is)\s+(?:not\s+(?:right|correct|what)|wrong|incorrect)\b"
    r"|not\s+(?:what|like that|quite|right|correct)\b"
    r"|(?:that|this|it)\s+(?:didn'?t|did not|doesn'?t|does not)\s+work\b"
    r"|you\s+(?:broke|missed|forgot|misunderstood|didn'?t|did not)\b"
    r"|stop\b"
    r"|(?:nein|falsch|stopp?)\b"
    r"|(?:das|es)\s+(?:ist|war)\s+(?:falsch|nicht\s+(?:richtig|korrekt|das))\b"
    r"|du\s+hast\s+(?:vergessen|übersehen|falsch)\b)",
    re.IGNORECASE,
)
_LEADING_NOISE = re.compile(r"^[\s\"'`*>_~.,:;!?\-—–]+")


@dataclass(frozen=True)
class ReleaseInterval:
    release_id: str
    start: float
    end: float | None  # ``None``: still active


@dataclass(frozen=True)
class SessionMetrics:
    success: bool
    success_rule: str
    tool_calls: int
    tool_errors: int
    retries: int
    tokens: int | None
    latency_ms: float
    user_corrections: int
    corrections: Mapping[str, int]
    skills: tuple[str, ...]


@dataclass
class FeedReport:
    release_id: str
    heuristics_version: str = HEURISTICS_VERSION
    scanned: int = 0
    attributed: int = 0
    recorded: int = 0
    duplicates: int = 0
    conflicts: int = 0
    other_release: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    tokens_unknown: int = 0

    def exclude(self, reason: str) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "excluded": dict(sorted(self.excluded.items()))}


class SessionFeed:
    """Builds monitoring observations for a release from the session queue (read-only)."""

    def __init__(
        self,
        sessions: SessionQueue,
        monitor: ReleaseMonitor,
        *,
        switch_margin_seconds: float = DEFAULT_SWITCH_MARGIN_SECONDS,
        now: float | None = None,
    ) -> None:
        if (
            isinstance(switch_margin_seconds, bool) or not isinstance(switch_margin_seconds, (int, float))
            or not math.isfinite(switch_margin_seconds) or not 0 <= switch_margin_seconds <= 86_400
        ):
            raise ValueError("switch margin must be between 0 and 86400 seconds")
        self.sessions = sessions
        self.monitor = monitor
        self.switch_margin = float(switch_margin_seconds)
        self.now = now

    def collect(self, release_id: str, *, limit: int = _MAX_SESSIONS) -> FeedReport:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_SESSIONS:
            raise ValueError("session limit must be between 1 and 10000")
        release = self.monitor.store.get_release(release_id)
        if release["status"] != "PROVISIONAL":
            raise ValueError("session evidence is only collected for provisional releases")
        timeline = activation_timeline(self.monitor.store)
        if not any(interval.release_id == release_id for interval in timeline):
            raise ValueError(
                f"{release_id} has no recorded activation event; sessions cannot be attributed to it"
            )
        current = [interval for interval in timeline if interval.release_id == release_id][-1]
        now = time.time() if self.now is None else self.now
        report = FeedReport(release_id)
        for session_id in self.sessions.session_ids(limit):
            report.scanned += 1
            try:
                session = self.sessions.get(session_id)
            except (KeyError, ValueError):
                report.exclude("unreadable")
                continue
            if _is_batch_session(session):
                report.exclude("not_production_session")
                continue
            span = session_time_span(session, now=now)
            if isinstance(span, str):
                report.exclude(span)
                continue
            owner = attribute_interval(span, timeline, margin=self.switch_margin)
            if isinstance(owner, str):
                report.exclude(owner.removeprefix("excluded:"))
                continue
            if owner.release_id != release_id:
                report.other_release += 1
                continue
            if owner.start != current.start:
                # The release ran then, but in an activation that has since ended
                # (it was re-activated by a rollback): that evidence describes a
                # finished deployment and is not counted for the current one.
                report.exclude("earlier_activation")
                continue
            try:
                metrics = derive_session_metrics(session)
            except ValueError:
                report.exclude("malformed_transcript")
                continue
            report.attributed += 1
            if metrics.tokens is None:
                report.tokens_unknown += 1
            observation = ProductionObservation(
                release_id=release_id,
                task_id=f"session:{session_id}"[:200],
                success=metrics.success,
                tool_errors=metrics.tool_errors,
                tokens=metrics.tokens,
                tool_calls=metrics.tool_calls,
                latency_ms=metrics.latency_ms,
                user_corrections=metrics.user_corrections,
                retries=metrics.retries,
                critical_regression=False,
                observation_id="hermes_session:" + hashlib.sha256(
                    f"{release_id}\0{session_id}".encode()
                ).hexdigest()[:48],
                skills=metrics.skills,
                source="hermes_session",
                derivation={
                    "heuristics_version": HEURISTICS_VERSION,
                    "success_rule": metrics.success_rule,
                    "tokens": "unknown" if metrics.tokens is None else "reported",
                    "latency": "agent_active_time",
                    "user_corrections_heuristics": dict(sorted(metrics.corrections.items())),
                    "evidence_trust": "untrusted",
                },
            )
            try:
                duplicate, _count = self.monitor.record(observation)
            except ValueError:
                # Same session already recorded with different derived values
                # (e.g. an older heuristics version): keep the first record.
                report.conflicts += 1
                continue
            if duplicate:
                report.duplicates += 1
            else:
                report.recorded += 1
        return report


# --- Release attribution -------------------------------------------------


def activation_timeline(store: ExperimentStore) -> list[ReleaseInterval]:
    """Ordered activation intervals of releases from the release audit events.

    ``PROMOTED`` and ``REACTIVATED`` each make their release the active one from
    the event time until the next activation. Status changes (STABLE) do not
    switch the deployed brain and are ignored.
    """
    activations: list[tuple[float, str]] = []
    for event in store.list_release_events():
        if event["event"] not in ACTIVATION_EVENTS:
            continue
        moment = parse_timestamp(event["created_at"])
        if moment is None:
            raise ValueError(f"release event {event['id']} has an unparseable timestamp")
        activations.append((moment, event["release_id"]))
    activations.sort(key=lambda item: item[0])
    return [
        ReleaseInterval(release, start, activations[index + 1][0] if index + 1 < len(activations) else None)
        for index, (start, release) in enumerate(activations)
    ]


def attribute_interval(
    span: tuple[float, float], timeline: Iterable[ReleaseInterval], *, margin: float,
) -> ReleaseInterval | str:
    """Activation interval covering the whole span, or ``excluded:<reason>``."""
    start, end = span
    intervals = list(timeline)
    if not intervals:
        return "excluded:no_recorded_release"
    for interval in intervals:
        switch = interval.start
        if start < switch <= end:
            return "excluded:spans_release_switch"
        if switch - margin <= end and start <= switch + margin:
            return "excluded:within_switch_margin"
    for interval in intervals:
        if interval.start <= start and (interval.end is None or end < interval.end):
            return interval
    return "excluded:before_first_recorded_release"


def session_time_span(session: Mapping[str, Any], *, now: float) -> tuple[float, float] | str:
    """``(start, end)`` epoch seconds of a session, or an exclusion reason.

    Message ``timestamp`` values (Hermes persists one per message, REAL epoch
    seconds) are required; session-row ``started_at``/``ended_at`` widen the
    span when present. A session export whose ``ended_at`` is present but null
    has not ended. Naive ISO strings, non-finite or implausible values make the
    time unknown: nothing is guessed.
    """
    messages = session.get("messages")
    if not isinstance(messages, list) or not messages:
        return "no_transcript"
    points: list[float] = []
    for message in messages:
        if not isinstance(message, dict):
            return "unknown_time"
        moment = parse_timestamp(message.get("timestamp"))
        if moment is None:
            return "unknown_time"
        points.append(moment)
    if "ended_at" in session and session["ended_at"] is None:
        return "session_not_ended"
    for key in ("started_at", "ended_at"):
        if key in session:
            moment = parse_timestamp(session[key])
            if moment is None:
                return "unknown_time"
            points.append(moment)
    start, end = min(points), max(points)
    if start < _EARLIEST_TIMESTAMP or end > now + _FUTURE_TOLERANCE_SECONDS:
        return "implausible_time"
    return start, end


def parse_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        moment = float(value)
        return moment if math.isfinite(moment) else None
    if isinstance(value, str) and 0 < len(value) <= 64:
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        return parsed.timestamp()
    return None


def _is_batch_session(session: Mapping[str, Any]) -> bool:
    source = session.get("source")
    return isinstance(source, dict) and source.get("type") == "hermes_batch_runner"


# --- Per-session derivation ----------------------------------------------


def derive_session_metrics(session: Mapping[str, Any]) -> SessionMetrics:
    """Deterministic per-session metrics from an OpenAI-format Hermes transcript.

    * ``tool_calls``: entries in assistant ``tool_calls`` lists.
    * ``tool_errors``: tool results Hermes marks as failed: content starting
      with ``Error``, or a JSON object with a non-zero ``exit_code``,
      ``success: false`` or a non-empty ``error`` (the persisted subset of
      Hermes' own ``_detect_tool_failure``). Cancelled calls are interruptions,
      not errors.
    * ``retries``: tool calls repeating the exact name and arguments of an
      earlier call in the session whose result was an error.
    * ``tokens``: ``input_tokens + output_tokens`` from the session row (or a
      ``usage`` object) when both are integers and the sum is positive; Hermes
      defaults both columns to 0, so 0 means "not reported" and becomes unknown.
    * ``latency_ms``: agent-active time, the sum over user turns of the time
      from the user message to the last assistant/tool message before the next
      user message (human think time is excluded).
    * ``success``: see ``_success``.
    """
    messages = session.get("messages")
    if not isinstance(messages, list):
        raise ValueError("session has no message list")
    tool_calls = tool_errors = retries = 0
    calls_by_id: dict[str, tuple[str, str]] = {}
    failed_calls: set[tuple[str, str]] = set()
    skills: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("session message must be an object")
        role = message.get("role")
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                name, arguments = _tool_call_signature(call)
                tool_calls += 1
                if (name, arguments) in failed_calls:
                    retries += 1
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    calls_by_id[call_id] = (name, arguments)
                skill = _skill_reference(name, arguments)
                if skill and skill not in skills and len(skills) < _MAX_SKILLS:
                    skills.append(skill)
        elif role == "tool":
            if _is_tool_error(message):
                tool_errors += 1
                signature = calls_by_id.get(message.get("tool_call_id")) if isinstance(
                    message.get("tool_call_id"), str) else None
                if signature is not None:
                    failed_calls.add(signature)
    corrections = correction_signals(session)
    success, rule = _success(session, corrections)
    return SessionMetrics(
        success=success,
        success_rule=rule,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        retries=retries,
        tokens=_reported_tokens(session),
        latency_ms=_agent_latency_ms(messages),
        user_corrections=sum(corrections.values()),
        corrections=corrections,
        skills=tuple(sorted(skills)),
    )


def correction_signals(session: Mapping[str, Any]) -> dict[str, int]:
    """Heuristic user-correction counts, by heuristic name.

    * ``rewind``: the session row's ``rewind_count`` (Hermes increments it when
      the user rewinds the transcript to an earlier message, i.e. an explicit undo).
    * ``explicit_undo_retry``: a user message answering an assistant turn that
      starts with an undo/retry request (``/undo``, ``undo``, ``revert``,
      ``try again``, ``rückgängig``, ``nochmal`` ...).
    * ``negation_after_assistant``: a user message answering an assistant turn
      that starts with a negation or correction (``no``, ``wrong``, ``that's not
      right``, ``you forgot``, ``nein``, ``falsch`` ...).
    * ``interrupted_tool_call``: tool results Hermes wrote because the user
      interrupted the run (``[Tool execution cancelled``), or an ``interrupted``
      final status when no such marker exists.
    """
    signals = {"rewind": 0, "explicit_undo_retry": 0, "negation_after_assistant": 0, "interrupted_tool_call": 0}
    rewinds = session.get("rewind_count")
    if isinstance(rewinds, int) and not isinstance(rewinds, bool) and 0 < rewinds < (1 << 31):
        signals["rewind"] = rewinds
    previous_role: str | None = None
    for message in session.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            if _message_text(message.get("content")).startswith(_TOOL_CANCELLED_PREFIX):
                signals["interrupted_tool_call"] += 1
            continue
        if role == "user" and previous_role == "assistant":
            kind = classify_correction(_message_text(message.get("content")))
            if kind is not None:
                signals[kind] += 1
        if role in {"user", "assistant"}:
            previous_role = role
    final = session.get("final_status")
    if not signals["interrupted_tool_call"] and isinstance(final, dict) and final.get("interrupted") is True:
        signals["interrupted_tool_call"] = 1
    return {name: count for name, count in signals.items() if count}


def classify_correction(text: str) -> str | None:
    """Classify the start of one user message; ``None`` when it is not a correction."""
    if not isinstance(text, str):
        return None
    head = _LEADING_NOISE.sub("", text[:_MAX_CORRECTION_TEXT]).casefold()
    head = head.replace("’", "'")
    if _UNDO_RETRY.match(head):
        return "explicit_undo_retry"
    if _NEGATION.match(head):
        return "negation_after_assistant"
    return None


def _success(session: Mapping[str, Any], corrections: Mapping[str, int]) -> tuple[bool, str]:
    """Session success, explicit status first, then the final-answer rule.

    1. ``final_status`` booleans from the Hermes hook: ``failed`` or
       ``interrupted`` true means failure, ``completed`` true means success.
       String statuses (``completed`` / ``partial`` / ``incomplete``) likewise.
    2. Otherwise (``final_answer`` rule): the last user/assistant/tool message
       is an assistant answer with text, no pending tool calls, a finish reason
       that is not a truncation or error, and the user did not interrupt.
    """
    final = session.get("final_status")
    if isinstance(final, dict):
        if final.get("failed") is True or final.get("interrupted") is True:
            return False, "final_status"
        if final.get("completed") is True:
            return True, "final_status"
    elif isinstance(final, str) and final in {"completed", "partial", "incomplete"}:
        return final == "completed", "final_status"
    if corrections.get("interrupted_tool_call"):
        return False, "final_answer"
    last = None
    for message in session.get("messages") or []:
        if isinstance(message, dict) and message.get("role") in {"user", "assistant", "tool"}:
            last = message
    if last is None or last.get("role") != "assistant":
        return False, "final_answer"
    if last.get("tool_calls"):
        return False, "final_answer"
    if last.get("finish_reason") in _FAILED_FINISH_REASONS:
        return False, "final_answer"
    return bool(_message_text(last.get("content")).strip()), "final_answer"


def _reported_tokens(session: Mapping[str, Any]) -> int | None:
    for container in (session, session.get("usage")):
        if not isinstance(container, dict):
            continue
        values = [container.get("input_tokens"), container.get("output_tokens")]
        if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
            total = values[0] + values[1]
            if 0 < total < (1 << 62):
                return total
    return None


def _agent_latency_ms(messages: list[Any]) -> float:
    total = 0.0
    turn_start: float | None = None
    turn_last: float | None = None
    for message in messages:
        moment = parse_timestamp(message.get("timestamp")) if isinstance(message, dict) else None
        if moment is None:
            continue
        if message.get("role") == "user":
            if turn_start is not None and turn_last is not None:
                total += max(0.0, turn_last - turn_start)
            turn_start, turn_last = moment, None
        elif message.get("role") in {"assistant", "tool"} and turn_start is not None:
            turn_last = moment
    if turn_start is not None and turn_last is not None:
        total += max(0.0, turn_last - turn_start)
    return round(total * 1000.0, 3)


def _tool_call_signature(call: Mapping[str, Any]) -> tuple[str, str]:
    function = call.get("function")
    source = function if isinstance(function, dict) else call
    name = source.get("name")
    arguments = source.get("arguments")
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, sort_keys=True)
        except (TypeError, ValueError):
            arguments = ""
    return (name[:200] if isinstance(name, str) else ""), arguments[:_MAX_TOOL_ARGUMENT_CHARS]


def _skill_reference(name: str, arguments: str) -> str | None:
    """Brain skill directory named by a ``skill_view`` call, if it is a valid name."""
    if name != "skill_view":
        return None
    try:
        parsed = json.loads(arguments) if arguments else None
    except json.JSONDecodeError:
        return None
    skill = parsed.get("name") if isinstance(parsed, dict) else None
    if not isinstance(skill, str) or ":" in skill:
        return None
    skill = skill.rstrip("/").rsplit("/", 1)[-1]
    return skill if _SKILL_NAME.fullmatch(skill) else None


def _is_tool_error(message: Mapping[str, Any]) -> bool:
    text = _message_text(message.get("content"))
    if text.startswith(_TOOL_CANCELLED_PREFIX):
        return False
    if text.startswith(_TOOL_ERROR_PREFIX):
        return True
    stripped = text.lstrip()
    if not stripped.startswith("{") or len(stripped) > 1_000_000:
        return False
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    exit_code = data.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
        return True
    return data.get("success") is False or bool(data.get("error"))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""
