"""Read-only shadow replay of recorded tasks against a provisional release.

Prior sessions that exercised a skill changed by the release are replayed
through an injected runner against the release's immutable deployment
checkout. Benchmarks of the SHADOW split (spec §11: post-promotion evidence)
can be replayed through the same runner. Replay results become post-promotion
evidence (source ``shadow_replay``) for
:class:`hermes_improve.monitoring.ReleaseMonitor`.
Session content stays untrusted evidence: only the task prompt is passed to
the runner, marked untrusted, and the session queue is never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .ingestion import SessionQueue
from .models import Benchmark, Split, canonical_json
from .monitoring import ACTIVATION_EVENTS, ProductionObservation, ReleaseMonitor, release_affected_skills


_MAX_RUNNER_OUTPUT_BYTES = 1024 * 1024
_MAX_PROMPT_CHARS = 20_000
_MAX_REPLAYS = 500


@dataclass(frozen=True)
class ReplayTask:
    session_id: str
    prompt: str
    skills: tuple[str, ...]
    evidence_trust: str = "untrusted"
    benchmark_id: str | None = None


@dataclass(frozen=True)
class ReplayOutcome:
    success: bool
    tool_errors: int
    tokens: int
    tool_calls: int
    latency_ms: float
    retries: int = 0
    critical_regression: bool = False


class ReplayRunner(Protocol):
    def run(self, task: ReplayTask, release_checkout: Path) -> ReplayOutcome: ...


@dataclass
class ReplayReport:
    release_id: str
    selected: int = 0
    recorded: int = 0
    duplicates: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    last_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ShadowReplayer:
    def __init__(self, sessions: SessionQueue, monitor: ReleaseMonitor, runner: ReplayRunner) -> None:
        self.sessions = sessions
        self.monitor = monitor
        self.runner = runner

    def _affected_patterns(self, release_id: str, limit: int) -> dict[str, re.Pattern[str]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_REPLAYS:
            raise ValueError("replay limit must be between 1 and 500")
        release = self.monitor.store.get_release(release_id)
        affected = sorted(release_affected_skills(release["manifest"]))
        return {
            name: re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])") for name in affected
        }

    def select_benchmark_tasks(
        self, release_id: str, benchmarks: Iterable[Benchmark], *, limit: int = 20,
    ) -> list[ReplayTask]:
        """SHADOW-split benchmarks touching a skill the release changed (all, if it changed none)."""
        patterns = self._affected_patterns(release_id, limit)
        tasks: list[ReplayTask] = []
        for benchmark in sorted(benchmarks, key=lambda item: (item.id, item.version)):
            if benchmark.split is not Split.SHADOW:
                continue
            prompt = benchmark.task.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if patterns:
                text = canonical_json({"task": benchmark.task, "taxonomy": benchmark.taxonomy})
                skills = tuple(name for name, pattern in patterns.items() if pattern.search(text))
                if not skills:
                    continue
            else:
                skills = ()
            identifier = f"{benchmark.id}@v{benchmark.version}"
            tasks.append(ReplayTask(
                f"shadow-benchmark:{identifier}", prompt[:_MAX_PROMPT_CHARS], skills, benchmark_id=identifier,
            ))
            if len(tasks) >= limit:
                break
        return tasks

    def select_tasks(self, release_id: str, *, limit: int = 20) -> list[ReplayTask]:
        patterns = self._affected_patterns(release_id, limit)
        tasks: list[ReplayTask] = []
        for session_id in self.sessions.session_ids(10_000):
            session = self.sessions.get(session_id)
            task = session.get("task")
            prompt = task.get("prompt") if isinstance(task, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if patterns:
                text = canonical_json(session)
                skills = tuple(name for name, pattern in patterns.items() if pattern.search(text))
                if not skills:
                    continue
            else:
                skills = ()
            tasks.append(ReplayTask(session_id, prompt[:_MAX_PROMPT_CHARS], skills))
            if len(tasks) >= limit:
                break
        return tasks

    def replay(self, release_id: str, *, limit: int = 20) -> ReplayReport:
        checkout = self._provisional_checkout(release_id)
        return self._replay_tasks(release_id, checkout, self.select_tasks(release_id, limit=limit))

    def replay_benchmarks(
        self, release_id: str, benchmarks: Iterable[Benchmark], *, limit: int = 20,
    ) -> ReplayReport:
        """Replay SHADOW-split benchmarks against the provisional release as extra evidence."""
        checkout = self._provisional_checkout(release_id)
        return self._replay_tasks(
            release_id, checkout, self.select_benchmark_tasks(release_id, benchmarks, limit=limit),
        )

    def _provisional_checkout(self, release_id: str) -> Path:
        release = self.monitor.store.get_release(release_id)
        if release["status"] != "PROVISIONAL":
            raise ValueError("shadow replay is only collected for provisional releases")
        return read_only_checkout(release["manifest"].get("deployment_path"))

    def _replay_tasks(self, release_id: str, checkout: Path, tasks: list[ReplayTask]) -> ReplayReport:
        report = ReplayReport(release_id)
        report.selected = len(tasks)
        epoch = _activation_key(self.monitor, release_id)
        for task in tasks:
            try:
                outcome = self.runner.run(task, checkout)
                _validate_outcome(outcome)
            except Exception as exc:  # runner/infrastructure failure is not release evidence
                report.failures.append({"session_id": task.session_id, "error": f"{type(exc).__name__}: {exc}"[:500]})
                continue
            # Session and benchmark replays use separate hash domains so a
            # session ID can never collide with a benchmark replay record.
            key = (
                f"{release_id}\0{task.session_id}" if task.benchmark_id is None
                else f"{release_id}\0benchmark\0{task.benchmark_id}"
            ) + epoch
            observation_id = "shadow_replay:" + hashlib.sha256(key.encode()).hexdigest()[:48]
            decision = self.monitor.observe(ProductionObservation(
                release_id=release_id,
                task_id=f"replay:{task.session_id}"[:200],
                success=outcome.success,
                tool_errors=outcome.tool_errors,
                tokens=outcome.tokens,
                tool_calls=outcome.tool_calls,
                latency_ms=float(outcome.latency_ms),
                retries=outcome.retries,
                critical_regression=outcome.critical_regression,
                observation_id=observation_id,
                skills=task.skills,
                source="shadow_replay",
            ))
            if decision.duplicate:
                report.duplicates += 1
            else:
                report.recorded += 1
            report.last_status = decision.status
            if decision.status == "STABLE":
                break
        return report


class CommandReplayRunner:
    """Runs an operator-provided replay command (for example a Hermes wrapper).

    The command receives one JSON object on stdin (``session_id``, ``prompt``,
    ``release_checkout``, ``evidence_trust``, plus ``benchmark_id`` for SHADOW
    benchmark replays) and must print one JSON object
    with ``success``, ``tool_errors``, ``tokens``, ``tool_calls`` and
    ``latency_ms`` (optionally ``retries`` and ``critical_regression``). No
    shell is involved and the environment is reduced to PATH/HOME plus
    ``HERMES_BRAIN_CHECKOUT``.
    """

    def __init__(self, argv: Sequence[str], *, timeout: float = 600.0) -> None:
        if (
            isinstance(argv, (str, bytes)) or not argv
            or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
        ):
            raise ValueError("replay runner command must be a non-empty list of arguments")
        if not Path(argv[0]).is_absolute():
            raise ValueError("replay runner executable must be an absolute path")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
            raise ValueError("replay runner timeout must be between 0 and 3600 seconds")
        self.argv = list(argv)
        self.timeout = float(timeout)

    def run(self, task: ReplayTask, release_checkout: Path) -> ReplayOutcome:
        payload = {
            "session_id": task.session_id,
            "prompt": task.prompt,
            "release_checkout": str(release_checkout),
            "evidence_trust": task.evidence_trust,
        }
        if task.benchmark_id is not None:
            payload["benchmark_id"] = task.benchmark_id
        request = json.dumps(payload).encode("utf-8")
        result = subprocess.run(
            self.argv, input=request, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=self.timeout, check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "HERMES_BRAIN_CHECKOUT": str(release_checkout),
            },
        )
        if result.returncode:
            raise RuntimeError(f"replay runner exited with status {result.returncode}")
        if len(result.stdout) > _MAX_RUNNER_OUTPUT_BYTES:
            raise ValueError("replay runner output exceeds the 1 MiB limit")
        try:
            data = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("replay runner must print one JSON object") from exc
        if not isinstance(data, dict):
            raise ValueError("replay runner must print one JSON object")
        allowed = {"success", "tool_errors", "tokens", "tool_calls", "latency_ms", "retries", "critical_regression"}
        missing = {"success", "tool_errors", "tokens", "tool_calls", "latency_ms"} - data.keys()
        if missing or data.keys() - allowed:
            raise ValueError("replay runner output has missing or unknown fields")
        outcome = ReplayOutcome(**data)
        _validate_outcome(outcome)
        return outcome


def read_only_checkout(path: Any) -> Path:
    """Return the release checkout if it is a real directory without write permission bits."""
    if not isinstance(path, str) or not path:
        raise ValueError("release manifest has no deployment_path for replay")
    checkout = Path(path)
    try:
        mode = os.lstat(checkout).st_mode
    except FileNotFoundError as exc:
        raise ValueError("release deployment checkout does not exist") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError("release deployment checkout must be a real directory, not a link")
    if stat.S_IMODE(mode) & 0o222:
        raise PermissionError("shadow replay requires a read-only release checkout")
    return checkout


def _validate_outcome(outcome: ReplayOutcome) -> None:
    if not isinstance(outcome, ReplayOutcome):
        raise TypeError("replay runner must return a ReplayOutcome")
    if not isinstance(outcome.success, bool) or not isinstance(outcome.critical_regression, bool):
        raise ValueError("replay success and critical_regression must be boolean")
    for value in (outcome.tool_errors, outcome.tokens, outcome.tool_calls, outcome.retries):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 63):
            raise ValueError("replay counters must be non-negative integers")
    latency = outcome.latency_ms
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError("replay latency must be finite and non-negative")


def _activation_key(monitor: ReleaseMonitor, release_id: str) -> str:
    """Observation-ID suffix that separates replays of a re-activated release.

    Replay observation IDs are deterministic per release and task. A release
    re-activated by a rollback starts a new monitoring epoch, so its replays
    are new evidence and get an epoch-specific ID; the first activation keeps
    the original ID format so existing records stay deduplicated.
    """
    events = [
        event for event in monitor.store.list_release_events(release_id)
        if event["event"] in ACTIVATION_EVENTS
    ]
    if not events or events[-1]["event"] != "REACTIVATED":
        return ""
    return f"\0activation\0{events[-1]['id']}"
