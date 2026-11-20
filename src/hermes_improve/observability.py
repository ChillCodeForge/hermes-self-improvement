"""Operator observability report (spec section 35).

Every number is computed from persisted state: the experiment store, the
session queue, the monitoring database, the public benchmark tree, and the
observability log written by the CLI after Dreaming and every Qwen-backed
command (``dream``, ``candidate generate``, ``evolve``, ``evaluate --judge``).
Missing sources are reported as ``"no data"``; a metric whose source is not
configured is reported as ``"unavailable"`` together with the reason. Nothing
is estimated. Reading never creates or modifies a database.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

from .benchmarks import BenchmarkStore
from .models import canonical_json, utc_now


NO_DATA = "no data"
UNAVAILABLE = "unavailable"
_QWEN_ROLES = frozenset({"qwen_solver", "qwen_optimizer", "qwen_dreamer", "qwen_judge"})


class ObservabilityLog:
    """Append-only log of Dreaming runs, Qwen usage, and proposal conversions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dreaming_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    sessions_processed INTEGER NOT NULL,
                    clusters_json TEXT NOT NULL,
                    signals INTEGER NOT NULL,
                    hypotheses INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS qwen_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    requests INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS proposal_conversions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gpu_runtime_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    seconds REAL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def record_dreaming_run(
        self, *, stage: str, sessions_processed: int, clusters: Mapping[str, int],
        signals: int, hypotheses: int,
    ) -> None:
        if stage not in {"light", "rem", "deep"}:
            raise ValueError("dreaming stage must be light, rem, or deep")
        for value in (sessions_processed, signals, hypotheses, *clusters.values()):
            _require_counter(value)
        if not all(isinstance(key, str) for key in clusters):
            raise ValueError("cluster names must be strings")
        with sqlite3.connect(self.path) as db:
            db.execute(
                """INSERT INTO dreaming_runs (stage,sessions_processed,clusters_json,signals,hypotheses,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (stage, sessions_processed, canonical_json(dict(clusters)), signals, hypotheses, utc_now()),
            )

    def record_qwen_stats(self, stats: Mapping[str, Mapping[str, Any]]) -> None:
        rows = []
        for role, values in stats.items():
            if role not in _QWEN_ROLES:
                raise ValueError(f"unknown Qwen role {role!r}")
            requests, input_tokens, output_tokens = (
                values.get("requests"), values.get("input_tokens"), values.get("output_tokens"),
            )
            latency = values.get("latency_ms")
            for value in (requests, input_tokens, output_tokens):
                _require_counter(value)
            if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
                raise ValueError("Qwen latency must be finite and non-negative")
            if requests:
                rows.append((role, requests, input_tokens, output_tokens, float(latency), utc_now()))
        if rows:
            with sqlite3.connect(self.path) as db:
                db.executemany(
                    """INSERT INTO qwen_usage (role,requests,input_tokens,output_tokens,latency_ms,created_at)
                       VALUES (?,?,?,?,?,?)""",
                    rows,
                )

    def record_proposal_conversion(self, candidate_id: str) -> None:
        if not isinstance(candidate_id, str) or not candidate_id.strip() or len(candidate_id) > 200:
            raise ValueError("candidate_id must be a non-empty string")
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO proposal_conversions (candidate_id,created_at) VALUES (?,?)",
                (candidate_id, utc_now()),
            )

    def record_gpu_runtime(
        self, *, command: str, metrics: tuple[str, ...], seconds: float | None, status: str, detail: str = "",
    ) -> None:
        """Store one GPU-runtime sample: a counter delta (``status='ok'``) or why there is none."""
        if command not in _QWEN_COMMANDS:
            raise ValueError(f"unknown Qwen-backed command {command!r}")
        if status not in {"ok", "scrape_failed", "counter_reset", "metrics_missing"}:
            raise ValueError("unknown GPU runtime sample status")
        if (status == "ok") != (seconds is not None):
            raise ValueError("only an ok GPU runtime sample carries seconds")
        if seconds is not None and (
            isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0
        ):
            raise ValueError("GPU runtime seconds must be finite and non-negative")
        with sqlite3.connect(self.path) as db:
            db.execute(
                """INSERT INTO gpu_runtime_samples (command,metrics_json,seconds,status,detail,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (command, canonical_json(list(metrics)), None if seconds is None else float(seconds),
                 status, str(detail)[:500], utc_now()),
            )


_QWEN_COMMANDS = frozenset({"dream-rem", "dream-deep", "candidate-generate", "evolve", "evaluate-judge"})
_METRIC_NAME = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*$")
_MAX_METRICS_BYTES = 4 * 1024 * 1024
GPU_METRICS_URL_ENV = "HERMES_GPU_METRICS_URL"
GPU_RUNTIME_METRICS_ENV = "HERMES_GPU_RUNTIME_METRICS"
GPU_METRICS_ALLOW_REMOTE_ENV = "HERMES_GPU_METRICS_ALLOW_REMOTE"
GPU_RUNTIME_UNCONFIGURED = (
    f"no GPU runtime samples recorded; set {GPU_METRICS_URL_ENV} (model server Prometheus /metrics URL) and "
    f"{GPU_RUNTIME_METRICS_ENV} (comma-separated cumulative seconds counters) to measure it"
)


class GpuRuntimeProbe:
    """Reads operator-named cumulative seconds counters from a model server's Prometheus endpoint.

    Configured only through the environment: ``HERMES_GPU_METRICS_URL`` (for
    example the vLLM or llama.cpp ``/metrics`` URL) and
    ``HERMES_GPU_RUNTIME_METRICS`` (comma-separated counter names, each a
    cumulative number of seconds, e.g. llama.cpp's
    ``llamacpp:prompt_seconds_total,llamacpp:tokens_predicted_seconds_total``).
    Every series of a named metric is summed. The runtime of a command is the
    difference of that sum after and before the command, so it includes any
    other traffic the server handled in that window. The URL must be loopback
    unless ``HERMES_GPU_METRICS_ALLOW_REMOTE=true``.
    """

    def __init__(self, url: str, metrics: tuple[str, ...], *, timeout: float = 5.0, allow_remote: bool = False) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{GPU_METRICS_URL_ENV} must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError(f"{GPU_METRICS_URL_ENV} must not contain credentials or a fragment")
        hostname = parsed.hostname.lower()
        try:
            local = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            local = hostname == "localhost"
        if not local and not allow_remote:
            raise ValueError(
                f"{GPU_METRICS_URL_ENV} must be loopback unless {GPU_METRICS_ALLOW_REMOTE_ENV}=true is set"
            )
        if not metrics or any(not _METRIC_NAME.fullmatch(name) for name in metrics) or len(set(metrics)) != len(metrics):
            raise ValueError(f"{GPU_RUNTIME_METRICS_ENV} must list distinct Prometheus metric names")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("GPU metrics timeout must be positive")
        self.url = url
        self.metrics = tuple(metrics)
        self.timeout = timeout

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "GpuRuntimeProbe | None":
        """The configured probe, ``None`` when no metrics URL is set; partial configuration is an error."""
        environ = os.environ if environ is None else environ
        url = environ.get(GPU_METRICS_URL_ENV, "").strip()
        names = tuple(item.strip() for item in environ.get(GPU_RUNTIME_METRICS_ENV, "").split(",") if item.strip())
        if not url and not names:
            return None
        if not url or not names:
            raise ValueError(f"{GPU_METRICS_URL_ENV} and {GPU_RUNTIME_METRICS_ENV} must be set together")
        return cls(url, names, allow_remote=environ.get(GPU_METRICS_ALLOW_REMOTE_ENV, "").lower() == "true")

    def read_seconds(self) -> float:
        """Sum of all series of the configured counters; raises when one is absent."""
        request = urllib.request.Request(self.url, headers={"Accept": "text/plain"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(_MAX_METRICS_BYTES + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"GPU metrics scrape failed: {exc}") from exc
        if len(raw) > _MAX_METRICS_BYTES:
            raise RuntimeError("GPU metrics response exceeds the 4 MiB limit")
        return sum_prometheus_counters(raw.decode("utf-8", errors="replace"), self.metrics)


class MetricsMissing(ValueError):
    """A configured counter is absent from the scraped exposition."""


def sum_prometheus_counters(text: str, metrics: tuple[str, ...]) -> float:
    """Sum every series of each named metric in Prometheus text exposition format."""
    totals: dict[str, float] = {}
    wanted = set(metrics)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_end = len(line)
        for index, char in enumerate(line):
            if char in "{ \t":
                name_end = index
                break
        name = line[:name_end]
        if name not in wanted:
            continue
        rest = line[name_end:]
        if rest.startswith("{"):
            closing = rest.rfind("}")
            if closing < 0:
                raise ValueError(f"malformed Prometheus series for {name}")
            rest = rest[closing + 1:]
        fields = rest.split()
        if not fields:
            raise ValueError(f"Prometheus series for {name} has no value")
        value = float(fields[0])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Prometheus counter {name} must be finite and non-negative")
        totals[name] = totals.get(name, 0.0) + value
    missing = [name for name in metrics if name not in totals]
    if missing:
        raise MetricsMissing(f"metrics endpoint does not expose: {', '.join(missing)}")
    return sum(totals.values())


@contextmanager
def record_qwen_usage(
    state_dir: Path, model: Any, *, command: str, environ: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Record the model's per-role Qwen usage and, if configured, GPU runtime for one command.

    Both are recorded when the wrapped block finishes, including when it
    raises, so failed runs still account for the requests they made. A GPU
    scrape that fails is stored as a failed sample with its reason; it never
    changes the outcome of the command. A misconfigured probe fails before any
    Qwen request is made.
    """
    if command not in _QWEN_COMMANDS:
        raise ValueError(f"unknown Qwen-backed command {command!r}")
    log = ObservabilityLog(Path(state_dir) / "observability.sqlite3")
    probe = GpuRuntimeProbe.from_environment(environ)
    before: float | None = None
    before_error: tuple[str, str] | None = None
    if probe is not None:
        try:
            before = probe.read_seconds()
        except MetricsMissing as exc:
            before_error = ("metrics_missing", str(exc))
        except (RuntimeError, ValueError) as exc:
            before_error = ("scrape_failed", str(exc))
    try:
        yield
    finally:
        log.record_qwen_stats(model.stats)
        if probe is not None:
            _record_gpu_sample(log, probe, command, before, before_error)


def _record_gpu_sample(
    log: ObservabilityLog, probe: GpuRuntimeProbe, command: str,
    before: float | None, before_error: tuple[str, str] | None,
) -> None:
    if before_error is not None:
        log.record_gpu_runtime(
            command=command, metrics=probe.metrics, seconds=None, status=before_error[0],
            detail=f"before command: {before_error[1]}",
        )
        return
    try:
        after = probe.read_seconds()
    except MetricsMissing as exc:
        log.record_gpu_runtime(
            command=command, metrics=probe.metrics, seconds=None, status="metrics_missing",
            detail=f"after command: {exc}",
        )
        return
    except (RuntimeError, ValueError) as exc:
        log.record_gpu_runtime(
            command=command, metrics=probe.metrics, seconds=None, status="scrape_failed",
            detail=f"after command: {exc}",
        )
        return
    assert before is not None
    if after < before:
        log.record_gpu_runtime(
            command=command, metrics=probe.metrics, seconds=None, status="counter_reset",
            detail="counters decreased during the command (model server restarted)",
        )
        return
    log.record_gpu_runtime(command=command, metrics=probe.metrics, seconds=after - before, status="ok")


def build_report(home: Path, benchmark_root: Path) -> dict[str, Any]:
    home = Path(home)
    experiments = _open(home / "experiments.sqlite3")
    sessions = _open(home / "sessions.sqlite3")
    monitoring = _open(home / "monitoring.sqlite3")
    log = _open(home / "observability.sqlite3")
    try:
        return {
            "generated_at": utc_now(),
            "hermes": _hermes(monitoring, sessions),
            "improvement": _improvement(experiments),
            "benchmarks": _benchmarks(benchmark_root, experiments),
            "dreaming": _dreaming(log, experiments),
            "qwen": _qwen(log, experiments),
            "production": _production(experiments, monitoring),
        }
    finally:
        for db in (experiments, sessions, monitoring, log):
            if db is not None:
                db.close()


def render_text(report: Mapping[str, Any]) -> str:
    lines = [f"Hermes improvement report ({report['generated_at']})"]
    for section in ("hermes", "improvement", "benchmarks", "dreaming", "qwen", "production"):
        lines.append("")
        lines.append(section.capitalize())
        for key, value in report[section].items():
            rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            lines.append(f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"


_OBSERVATION_METRICS = ("success", "tokens", "tool_calls", "tool_errors", "retries", "critical_regression")


def _hermes(monitoring: sqlite3.Connection | None, sessions: sqlite3.Connection | None) -> dict[str, Any]:
    """Hermes task metrics.

    The headline metrics count only ``source='production'`` observations (real
    traffic). ``observations_by_source`` breaks every recorded observation down
    by source (production, hermes_session, shadow_replay, historical_replay) so
    the section reconciles with ``production.post_promotion_regressions``, which
    counts critical observations of every source.
    """
    result: dict[str, Any] = {
        "observed_tasks": NO_DATA, "success_rate": NO_DATA, "tokens_per_success": NO_DATA,
        "tool_calls_per_success": NO_DATA, "failed_tool_calls": NO_DATA, "retry_rate": NO_DATA,
        "queued_sessions": NO_DATA, "observations_by_source": NO_DATA,
    }
    if sessions is not None and _has_table(sessions, "sessions"):
        result["queued_sessions"] = dict(sessions.execute(
            "SELECT status, COUNT(*) FROM sessions GROUP BY status ORDER BY status"
        ).fetchall()) or NO_DATA
    if monitoring is None or not _has_table(monitoring, "production_observations"):
        return result
    by_source: dict[str, list[tuple[Any, ...]]] = {}
    for row in monitoring.execute(
        f"SELECT {_source_column(monitoring)},{','.join(_OBSERVATION_METRICS)} FROM production_observations"
    ):
        by_source.setdefault(row[0], []).append(row[1:])
    if not by_source:
        return result
    result["observations_by_source"] = {
        source: {**_observation_metrics(rows), "critical_observations": sum(row[5] for row in rows)}
        for source, rows in sorted(by_source.items())
    }
    production = by_source.get("production")
    if production:
        result.update(_observation_metrics(production))
    return result


def _observation_metrics(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    successes = sum(row[0] for row in rows)
    return {
        "observed_tasks": len(rows),
        "success_rate": successes / len(rows),
        "tokens_per_success": sum(row[1] for row in rows) / successes if successes else NO_DATA,
        "tool_calls_per_success": sum(row[2] for row in rows) / successes if successes else NO_DATA,
        "failed_tool_calls": sum(row[3] for row in rows),
        "retry_rate": mean(bool(row[4]) for row in rows),
    }


def _source_column(monitoring: sqlite3.Connection) -> str:
    # Databases from before the ``source`` column only held production observations.
    return "source" if _has_column(monitoring, "production_observations", "source") else "'production'"


def _improvement(db: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidates_generated": NO_DATA, "candidates_rejected": NO_DATA, "candidates_promoted": NO_DATA,
        "experiments_by_decision": NO_DATA, "generations": NO_DATA, "average_quality_delta": NO_DATA,
        "evaluations": NO_DATA, "regressions_caught": NO_DATA, "skill_load": NO_DATA,
    }
    if db is None or not _has_table(db, "experiments"):
        return result
    decisions = dict(db.execute("SELECT decision, COUNT(*) FROM experiments GROUP BY decision").fetchall())
    candidates = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    parents = dict(db.execute("SELECT id, parent_id FROM experiments").fetchall())
    depth_cache: dict[str, int] = {}

    def depth(experiment_id: str) -> int:
        seen, current, steps = set(), experiment_id, 0
        while current in parents and parents[current] and current not in seen:
            seen.add(current)
            current, steps = parents[current], steps + 1
        return steps

    for experiment_id in parents:
        depth_cache[experiment_id] = depth(experiment_id)
    evaluations = db.execute("SELECT passed, reasons, metrics_json FROM evaluation_runs").fetchall()
    deltas = []
    regressions = 0
    skill_rates = []
    for passed, reasons, metrics in evaluations:
        rate = _candidate_skill_loaded_rate(json.loads(metrics))
        if rate is not None:
            skill_rates.append(rate)
        value = json.loads(metrics).get("quality_delta_mean")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            deltas.append(value)
        if not passed and any("regression" in str(reason).lower() for reason in json.loads(reasons)):
            regressions += 1
    result.update({
        "candidates_generated": candidates,
        "candidates_rejected": decisions.get("rejected", 0),
        "candidates_promoted": decisions.get("promoted", 0),
        "experiments_by_decision": decisions or NO_DATA,
        "generations": (max(depth_cache.values()) + 1) if depth_cache else NO_DATA,
        "average_quality_delta": mean(deltas) if deltas else NO_DATA,
        "evaluations": len(evaluations) if evaluations else NO_DATA,
        "regressions_caught": regressions if evaluations else NO_DATA,
        # Lab finding F30: a skill candidate whose skill the agent never opened shows
        # delta 0 whatever its content; this counts such evaluations.
        "skill_load": {
            "evaluations_with_target_skill": len(skill_rates),
            "candidate_skill_never_loaded": sum(rate == 0 for rate in skill_rates),
            "candidate_target_skill_loaded_rate_mean": mean(skill_rates),
        } if skill_rates else NO_DATA,
    })
    return result


def _candidate_skill_loaded_rate(metrics: Any) -> float | None:
    use = metrics.get("skill_use") if isinstance(metrics, dict) else None
    candidate = use.get("candidate") if isinstance(use, dict) else None
    rate = candidate.get("target_skill_loaded_rate") if isinstance(candidate, dict) else None
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
        return None
    return float(rate)


def _benchmarks(root: Path, experiments: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": NO_DATA, "categories": NO_DATA, "difficulty": NO_DATA, "ground_truth_types": NO_DATA,
        "reproducibility": NO_DATA, "split_distribution": NO_DATA, "spent_holdouts": NO_DATA,
    }
    if experiments is not None and _has_table(experiments, "spent_holdouts"):
        result["spent_holdouts"] = experiments.execute("SELECT COUNT(*) FROM spent_holdouts").fetchone()[0]
    if not Path(root).is_dir():
        return result
    benchmarks = BenchmarkStore(Path(root)).load_public()
    if not benchmarks:
        return result
    categories: Counter[str] = Counter()
    for item in benchmarks:
        for category in item.taxonomy.get("categories", []):
            categories[str(category)] += 1
    scores = [
        float(item.eligibility["reproducibility_score"]) for item in benchmarks
        if isinstance(item.eligibility.get("reproducibility_score"), (int, float))
    ]
    result.update({
        "count": len(benchmarks),
        "categories": dict(sorted(categories.items())),
        "difficulty": dict(sorted(Counter(str(item.taxonomy.get("difficulty", "unspecified")) for item in benchmarks).items())),
        "ground_truth_types": dict(sorted(Counter(str(item.ground_truth.get("type", "unspecified")) for item in benchmarks).items())),
        "reproducibility": {"mean": mean(scores), "min": min(scores), "max": max(scores)} if scores else NO_DATA,
        "split_distribution": dict(sorted(Counter(item.split.value for item in benchmarks).items())),
    })
    return result


def _dreaming(log: sqlite3.Connection | None, experiments: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runs": NO_DATA, "sessions_processed": NO_DATA, "clusters": NO_DATA,
        "hypotheses": NO_DATA, "candidate_conversion_rate": NO_DATA,
    }
    if log is None or not _has_table(log, "dreaming_runs"):
        return result
    rows = log.execute("SELECT stage, sessions_processed, clusters_json, hypotheses FROM dreaming_runs").fetchall()
    if not rows:
        return result
    clusters: Counter[str] = Counter()
    for _stage, _sessions, clusters_json, _hypotheses in rows:
        clusters.update(json.loads(clusters_json))
    hypotheses = sum(row[3] for row in rows)
    conversions = log.execute("SELECT COUNT(*) FROM proposal_conversions").fetchone()[0]
    result.update({
        "runs": dict(Counter(row[0] for row in rows)),
        "sessions_processed": sum(row[1] for row in rows),
        "clusters": dict(sorted(clusters.items())),
        "hypotheses": hypotheses,
        "candidate_conversion_rate": conversions / hypotheses if hypotheses else NO_DATA,
    })
    return result


def _qwen(log: sqlite3.Connection | None, experiments: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requests": NO_DATA, "input_tokens": NO_DATA, "output_tokens": NO_DATA,
        "average_latency_ms": NO_DATA, "by_role": NO_DATA,
        "gpu_runtime": UNAVAILABLE, "gpu_runtime_reason": GPU_RUNTIME_UNCONFIGURED,
        "judge_uncertainty": _judge_uncertainty(experiments),
    }
    if log is None:
        return result
    if _has_table(log, "gpu_runtime_samples"):
        samples = log.execute(
            "SELECT command, seconds, status FROM gpu_runtime_samples ORDER BY id"
        ).fetchall()
        if samples:
            measured = [row for row in samples if row[2] == "ok"]
            by_command: dict[str, float] = {}
            for command, seconds, _status in measured:
                by_command[command] = by_command.get(command, 0.0) + seconds
            result["gpu_runtime"] = {
                "seconds": sum(row[1] for row in measured),
                "measured_commands": len(measured),
                "unmeasured_commands": dict(sorted(Counter(row[2] for row in samples if row[2] != "ok").items())),
                "seconds_by_command": dict(sorted(by_command.items())),
            }
            del result["gpu_runtime_reason"]
    if not _has_table(log, "qwen_usage"):
        return result
    rows = log.execute(
        "SELECT role, SUM(requests), SUM(input_tokens), SUM(output_tokens), SUM(latency_ms) FROM qwen_usage GROUP BY role"
    ).fetchall()
    if not rows:
        return result
    requests = sum(row[1] for row in rows)
    result.update({
        "requests": requests,
        "input_tokens": sum(row[2] for row in rows),
        "output_tokens": sum(row[3] for row in rows),
        "average_latency_ms": sum(row[4] for row in rows) / requests if requests else NO_DATA,
        "by_role": {row[0]: {"requests": row[1], "input_tokens": row[2], "output_tokens": row[3]} for row in rows},
    })
    return result


def _judge_uncertainty(experiments: sqlite3.Connection | None) -> Any:
    """Aggregate of the judge uncertainty stored per judged task in ``task_results``."""
    if experiments is None or not _has_table(experiments, "task_results"):
        return NO_DATA
    uncertainties: list[float] = []
    unreliable = 0
    for (result_json,) in experiments.execute("SELECT result_json FROM task_results"):
        row = json.loads(result_json)
        value = row.get("judge_uncertainty")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("task_results contains a non-numeric judge_uncertainty")
        uncertainties.append(float(value))
        if row.get("judge_result") == "unreliable":
            unreliable += 1
    if not uncertainties:
        return NO_DATA
    return {
        "judged_tasks": len(uncertainties),
        "mean": mean(uncertainties),
        "max": max(uncertainties),
        "unreliable_tasks": unreliable,
    }


def _production(experiments: sqlite3.Connection | None, monitoring: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "brain_version": NO_DATA, "release_status": NO_DATA, "releases_by_status": NO_DATA,
        "post_promotion_regressions": NO_DATA, "rollback_count": NO_DATA,
    }
    if experiments is not None and _has_table(experiments, "releases"):
        releases = experiments.execute("SELECT id, status FROM releases ORDER BY created_at DESC, id DESC").fetchall()
        if releases:
            active = next((row for row in releases if row[1] in {"PROVISIONAL", "STABLE"}), None)
            statuses = Counter(row[1] for row in releases)
            result.update({
                "brain_version": active[0] if active else NO_DATA,
                "release_status": active[1] if active else NO_DATA,
                "releases_by_status": dict(sorted(statuses.items())),
                "rollback_count": statuses.get("ROLLED_BACK", 0),
            })
    if monitoring is not None and _has_table(monitoring, "production_observations"):
        critical_by_source = dict(monitoring.execute(
            f"SELECT {_source_column(monitoring)}, COALESCE(SUM(critical_regression=1), 0) "
            "FROM production_observations GROUP BY 1 ORDER BY 1"
        ).fetchall())
        actions = blocked = 0
        if _has_table(monitoring, "monitoring_actions"):
            actions = monitoring.execute(
                "SELECT COUNT(DISTINCT release_id) FROM monitoring_actions WHERE action LIKE 'ROLLBACK%'"
            ).fetchone()[0]
            blocked = monitoring.execute(
                "SELECT COUNT(*) FROM monitoring_actions WHERE action='ROLLBACK_BLOCKED'"
            ).fetchone()[0]
        result["post_promotion_regressions"] = {
            # All sources; the per-source split matches hermes.observations_by_source.
            "critical_observations": sum(critical_by_source.values()),
            "critical_observations_by_source": critical_by_source or NO_DATA,
            "releases_with_rollback_decision": actions,
            # Regressions the monitor could not roll back; each needs an operator.
            "blocked_rollbacks": blocked,
        }
    return result


def _open(path: Path) -> sqlite3.Connection | None:
    if not path.is_file():
        return None
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)


def _has_table(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _has_column(db: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in db.execute(f"PRAGMA table_info({table})"))


def _require_counter(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 63):
        raise ValueError("observability counters must be non-negative integers")


# Release audit events that start a release's monitoring epoch (as in monitoring.py).
_ACTIVATION_EVENTS = ("PROMOTED", "REACTIVATED")


def _current_rollback_block(experiments: sqlite3.Connection, release_id: str) -> dict[str, Any] | None:
    """Latest ROLLBACK_BLOCKED event of the release's current activation, if any.

    A block recorded during an earlier activation of the release concerned a
    deployment that has ended and is not reported as current.
    """
    if not _has_table(experiments, "release_events"):
        return None
    latest = None
    for event, detail_json, created_at in experiments.execute(
        "SELECT event, detail_json, created_at FROM release_events WHERE release_id=? ORDER BY id",
        (release_id,),
    ):
        if event in _ACTIVATION_EVENTS:
            latest = None
        elif event == "ROLLBACK_BLOCKED":
            detail = json.loads(detail_json)
            latest = {
                "at": created_at,
                "reason": detail.get("reason"),
                "assessment_reasons": detail.get("assessment_reasons", []),
                "target_release": detail.get("target_release"),
                "action": "operator action required",
            }
    return latest


def build_status(
    home: Path, benchmark_root: Path, *, deployment_link: Path | None = None,
) -> dict[str, Any]:
    """Current state for ``hermes-improve status`` (read-only, never creates a database).

    The active brain version comes from the deployment pointer when one is
    known (that is what Hermes reads) and otherwise from the release ledger;
    when both exist and disagree the mismatch is reported, not resolved.
    """
    home = Path(home)
    result: dict[str, Any] = {
        "brain_version": NO_DATA,
        "brain_version_source": NO_DATA,
        "active_release": NO_DATA,
        "deployment": NO_DATA,
        "counts": {
            "experiments_by_decision": NO_DATA, "candidates": NO_DATA, "evaluation_runs": NO_DATA,
            "releases_by_status": NO_DATA, "rollbacks": NO_DATA, "spent_holdouts": NO_DATA,
            "sessions_by_status": NO_DATA, "public_benchmarks": NO_DATA, "blocked_rollbacks": NO_DATA,
        },
    }
    counts = result["counts"]
    ledger_release: dict[str, Any] | None = None
    try:
        experiments = _open(home / "experiments.sqlite3")
    except sqlite3.Error:
        experiments = None
        counts["experiments_by_decision"] = UNAVAILABLE
    if experiments is not None:
        try:
            if _has_table(experiments, "experiments"):
                counts["experiments_by_decision"] = dict(sorted(experiments.execute(
                    "SELECT decision, COUNT(*) FROM experiments GROUP BY decision"
                ).fetchall()))
                counts["candidates"] = experiments.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
                counts["evaluation_runs"] = experiments.execute(
                    "SELECT COUNT(*) FROM evaluation_runs"
                ).fetchone()[0]
            if _has_table(experiments, "spent_holdouts"):
                counts["spent_holdouts"] = experiments.execute("SELECT COUNT(*) FROM spent_holdouts").fetchone()[0]
            if _has_table(experiments, "releases"):
                releases = experiments.execute(
                    "SELECT id, status, created_at FROM releases ORDER BY created_at DESC, id DESC"
                ).fetchall()
                statuses = Counter(row[1] for row in releases)
                counts["releases_by_status"] = dict(sorted(statuses.items()))
                counts["rollbacks"] = statuses.get("ROLLED_BACK", 0)
                if _has_table(experiments, "release_events"):
                    counts["blocked_rollbacks"] = experiments.execute(
                        "SELECT COUNT(*) FROM release_events WHERE event='ROLLBACK_BLOCKED'"
                    ).fetchone()[0]
                active = next((row for row in releases if row[1] in {"PROVISIONAL", "STABLE"}), None)
                if active is not None:
                    ledger_release = {"id": active[0], "status": active[1], "created_at": active[2]}
                    blocked = _current_rollback_block(experiments, active[0])
                    if blocked is not None:
                        ledger_release["rollback_blocked"] = blocked
                    result["active_release"] = ledger_release
        except sqlite3.Error:
            counts["experiments_by_decision"] = UNAVAILABLE
        finally:
            experiments.close()
    try:
        sessions = _open(home / "sessions.sqlite3")
    except sqlite3.Error:
        sessions = None
    if sessions is not None:
        try:
            if _has_table(sessions, "sessions"):
                counts["sessions_by_status"] = dict(sessions.execute(
                    "SELECT status, COUNT(*) FROM sessions GROUP BY status ORDER BY status"
                ).fetchall())
        except sqlite3.Error:
            counts["sessions_by_status"] = UNAVAILABLE
        finally:
            sessions.close()
    if Path(benchmark_root).is_dir():
        try:
            counts["public_benchmarks"] = len(BenchmarkStore(Path(benchmark_root)).load_public())
        except (OSError, ValueError) as exc:
            counts["public_benchmarks"] = f"invalid: {str(exc)[:200]}"
    if ledger_release is not None:
        result["brain_version"] = ledger_release["id"]
        result["brain_version_source"] = "release ledger"
    if deployment_link is not None:
        link = Path(deployment_link)
        if link.is_symlink():
            deployed = Path(os.readlink(link)).name
            result["deployment"] = {"link": str(link), "release": deployed}
            result["brain_version"] = deployed
            result["brain_version_source"] = "deployment link"
            if ledger_release is not None and ledger_release["id"] != deployed:
                result["deployment"]["mismatch"] = (
                    f"deployment points to {deployed} but the ledger's active release is {ledger_release['id']}"
                )
        elif os.path.lexists(link):
            result["deployment"] = {"link": str(link), "error": "deployment pointer is not a symlink"}
        else:
            result["deployment"] = {"link": str(link), "release": None}
    return result
