"""Operator observability report (spec section 35).

Every number is computed from persisted state: the experiment store, the
session queue, the monitoring database, the public benchmark tree, and the
observability log written by the CLI after Dreaming and Qwen-backed runs.
Missing sources are reported as ``"no data"``; a metric without any possible
source in this package is reported as ``"unavailable"``. Nothing is estimated.
Reading never creates or modifies a database.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

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
            "qwen": _qwen(log),
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


def _hermes(monitoring: sqlite3.Connection | None, sessions: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "observed_tasks": NO_DATA, "success_rate": NO_DATA, "tokens_per_success": NO_DATA,
        "tool_calls_per_success": NO_DATA, "failed_tool_calls": NO_DATA, "retry_rate": NO_DATA,
        "queued_sessions": NO_DATA,
    }
    if sessions is not None and _has_table(sessions, "sessions"):
        result["queued_sessions"] = dict(sessions.execute(
            "SELECT status, COUNT(*) FROM sessions GROUP BY status ORDER BY status"
        ).fetchall()) or NO_DATA
    if monitoring is None or not _has_table(monitoring, "production_observations"):
        return result
    source_filter = "WHERE source='production'" if _has_column(monitoring, "production_observations", "source") else ""
    rows = monitoring.execute(
        f"SELECT success,tokens,tool_calls,tool_errors,retries FROM production_observations {source_filter}"
    ).fetchall()
    if not rows:
        return result
    successes = sum(row[0] for row in rows)
    result.update({
        "observed_tasks": len(rows),
        "success_rate": successes / len(rows),
        "tokens_per_success": sum(row[1] for row in rows) / successes if successes else NO_DATA,
        "tool_calls_per_success": sum(row[2] for row in rows) / successes if successes else NO_DATA,
        "failed_tool_calls": sum(row[3] for row in rows),
        "retry_rate": mean(bool(row[4]) for row in rows),
    })
    return result


def _improvement(db: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidates_generated": NO_DATA, "candidates_rejected": NO_DATA, "candidates_promoted": NO_DATA,
        "experiments_by_decision": NO_DATA, "generations": NO_DATA, "average_quality_delta": NO_DATA,
        "evaluations": NO_DATA, "regressions_caught": NO_DATA,
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
    for passed, reasons, metrics in evaluations:
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
    })
    return result


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


def _qwen(log: sqlite3.Connection | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requests": NO_DATA, "input_tokens": NO_DATA, "output_tokens": NO_DATA,
        "average_latency_ms": NO_DATA, "by_role": NO_DATA,
        "gpu_runtime": UNAVAILABLE, "judge_uncertainty": NO_DATA,
    }
    if log is None or not _has_table(log, "qwen_usage"):
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
        critical = monitoring.execute(
            "SELECT COUNT(*) FROM production_observations WHERE critical_regression=1"
        ).fetchone()[0]
        actions = 0
        if _has_table(monitoring, "monitoring_actions"):
            actions = monitoring.execute(
                "SELECT COUNT(DISTINCT release_id) FROM monitoring_actions WHERE action LIKE 'ROLLBACK%'"
            ).fetchone()[0]
        result["post_promotion_regressions"] = {
            "critical_observations": critical, "releases_with_rollback_decision": actions,
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
