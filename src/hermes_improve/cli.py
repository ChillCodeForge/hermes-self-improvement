from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from . import __version__
from .artifacts import ArtifactStore
from .batch_ingestion import load_batch_runner_export
from .benchmarks import BenchmarkStore
from .candidate_builder import create_skill_candidate
from .docker_runner import DockerCommandRunner
from .dreaming import DreamingEngine
from .ingestion import SessionQueue
from .miner import mine_session
from .monitoring import ProductionObservation, ReleaseMonitor
from .evaluation_signing import report_payload
from .evaluator import PairedEvaluator
from .models import Candidate, CandidateType, Role, Split
from .policy import PromotionPolicy
from .promotion import GitPromoter
from .qwen import OpenAICompatibleQwen, QwenConfig
from .security import SpentHoldoutRegistry, consume_holdout_generation
from .store import ExperimentStore


_MAX_SQLITE_INTEGER = (1 << 63) - 1
_MAX_MONITOR_FILE_BYTES = 10 * 1024 * 1024


def _paths() -> tuple[Path, Path]:
    home = Path(os.environ.get("HERMES_IMPROVE_HOME", Path.home() / ".hermes-improve"))
    return home / "experiments.sqlite3", Path(os.environ.get("HERMES_IMPROVE_BENCHMARKS", "benchmarks"))


def _require_role_uid(role: str) -> None:
    variable = f"HERMES_{role.upper()}_UID"
    other_role = "PROMOTER" if role == "evaluator" else "EVALUATOR"
    expected = os.environ.get(variable)
    if expected is None:
        raise RuntimeError(f"{variable} must identify the dedicated {role} Unix user")
    if not expected.isdigit():
        raise ValueError(f"{variable} must be a numeric Unix UID")
    other = os.environ.get(f"HERMES_{other_role}_UID")
    if other is not None:
        if not other.isdigit():
            raise ValueError(f"HERMES_{other_role}_UID must be a numeric Unix UID")
        if int(other) == int(expected):
            raise PermissionError("evaluator and promoter must use distinct Unix UIDs")
    if os.geteuid() != int(expected):
        raise PermissionError(f"{role} commands must run as {variable}")


def _load_observations(path: Path) -> list[ProductionObservation]:
    with path.open("rb") as source:
        raw = source.read(_MAX_MONITOR_FILE_BYTES + 1)
    if len(raw) > _MAX_MONITOR_FILE_BYTES:
        raise ValueError("monitoring observation file exceeds the 10 MiB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("monitoring observation file must be UTF-8") from exc
    allowed = {
        "observation_id", "release_id", "task_id", "success", "tool_errors", "tokens",
        "tool_calls", "latency_ms", "user_corrections", "retries", "critical_regression",
    }
    required = {
        "observation_id", "release_id", "task_id", "success", "tool_errors",
        "tokens", "tool_calls", "latency_ms",
    }
    observations = []
    seen_ids = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if len(observations) >= 10000:
            raise ValueError("monitoring observation file exceeds the 10,000-record limit")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"monitoring observation line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"monitoring observation line {line_number} must be a JSON object")
        missing = required - item.keys()
        extra = item.keys() - allowed
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing {', '.join(sorted(missing))}")
            if extra:
                detail.append(f"unknown {', '.join(sorted(extra))}")
            raise ValueError(f"monitoring observation line {line_number}: {'; '.join(detail)}")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in (
            "observation_id", "release_id", "task_id",
        )):
            raise ValueError(f"monitoring observation line {line_number} requires non-empty string identifiers")
        if item["observation_id"] in seen_ids:
            raise ValueError(f"duplicate observation_id on monitoring line {line_number}")
        seen_ids.add(item["observation_id"])
        if not isinstance(item["success"], bool):
            raise ValueError(f"monitoring observation line {line_number}: success must be boolean")
        for key in ("tool_errors", "tokens", "tool_calls", "user_corrections", "retries"):
            value = item.get(key, 0)
            if (
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value > _MAX_SQLITE_INTEGER
            ):
                raise ValueError(f"monitoring observation line {line_number}: {key} must be a non-negative integer")
        latency = item["latency_ms"]
        try:
            valid_latency = (
                not isinstance(latency, bool) and isinstance(latency, (int, float))
                and math.isfinite(latency) and latency >= 0
            )
        except OverflowError:
            valid_latency = False
        if not valid_latency:
            raise ValueError(f"monitoring observation line {line_number}: latency_ms must be finite and non-negative")
        critical = item.get("critical_regression", False)
        if not isinstance(critical, bool):
            raise ValueError(f"monitoring observation line {line_number}: critical_regression must be boolean")
        observations.append(ProductionObservation(
            release_id=item["release_id"],
            task_id=item["task_id"],
            success=item["success"],
            tool_errors=item["tool_errors"],
            tokens=item["tokens"],
            tool_calls=item["tool_calls"],
            latency_ms=latency,
            user_corrections=item.get("user_corrections", 0),
            retries=item.get("retries", 0),
            critical_regression=critical,
            observation_id=item["observation_id"],
        ))
    if not observations:
        raise ValueError("monitoring observation file contains no records")
    return observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-improve", description="Safety-gated Hermes improvement foundations")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("export", type=Path)
    batch_ingest = sub.add_parser("ingest-batch")
    batch_ingest.add_argument("trajectories", type=Path)
    dream = sub.add_parser("dream")
    dream.add_argument("--stage", choices=["light", "rem", "deep"], default="light")
    dream.add_argument("--limit", type=int, default=100)
    dream.add_argument("--hypotheses", type=Path)
    benchmarks = sub.add_parser("benchmarks")
    benchmark_sub = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    benchmark_sub.add_parser("validate")
    mine = benchmark_sub.add_parser("mine")
    mine.add_argument("session_id")
    benchmark_sub.add_parser("candidates")
    listing = benchmark_sub.add_parser("list")
    listing.add_argument("--role", choices=[role.value for role in Role], default=Role.OPERATOR.value)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("candidate_id")
    evaluate.add_argument("--repository", type=Path, required=True)
    evaluate.add_argument("--holdout-root", type=Path, required=True)
    evaluate.add_argument("--seeds", default="1")
    experiments = sub.add_parser("experiments")
    experiment_sub = experiments.add_subparsers(dest="experiment_command", required=True)
    listing = experiment_sub.add_parser("list")
    listing.add_argument("--limit", type=int, default=100)
    show = experiment_sub.add_parser("show")
    show.add_argument("experiment_id")
    candidates = sub.add_parser("candidates")
    candidates.add_argument("candidate_command", choices=["list"])
    candidates.add_argument("--limit", type=int, default=100)
    candidate = sub.add_parser("candidate")
    candidate_sub = candidate.add_subparsers(dest="candidate_command", required=True)
    create = candidate_sub.add_parser("create")
    create.add_argument("--repository", type=Path, required=True)
    create.add_argument("--experiment-id", required=True)
    create.add_argument("--candidate-id", required=True)
    create.add_argument("--type", choices=["SKILL_CREATE", "SKILL_PATCH"], required=True)
    create.add_argument("--target", required=True, help="skill directory under brain/skills/")
    create.add_argument("--content", type=Path, required=True, help="file containing the new SKILL.md text")
    create.add_argument("--hypothesis", required=True)
    create.add_argument("--source-session", action="append", default=[])
    releases = sub.add_parser("releases")
    releases.add_argument("release_command", choices=["list"])
    monitor = sub.add_parser("monitor")
    monitor_sub = monitor.add_subparsers(dest="monitor_command", required=True)
    observe = monitor_sub.add_parser("ingest")
    observe.add_argument("observations", type=Path)
    promote = sub.add_parser("promote")
    promote.add_argument("candidate_id")
    promote.add_argument("--release", required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("release")
    for command in (promote, rollback):
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--release-root", type=Path, required=True)
        command.add_argument("--deployment-link", type=Path, required=True)
        command.add_argument("--evaluator-public-key", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    database, benchmark_root = _paths()
    if args.command == "status":
        print(json.dumps({"version": __version__, "database": str(database), "benchmarks": str(benchmark_root), "phase": "evaluation-foundation"}, indent=2))
    elif args.command == "ingest":
        try:
            export = json.loads(args.export.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"session export is invalid JSON: {exc}") from exc
        if not isinstance(export, dict):
            raise ValueError("session export must be a JSON object")
        digest = SessionQueue(database.parent / "sessions.sqlite3").ingest(export)
        print(json.dumps({"session_id": export["session_id"], "sha256": digest, "status": "queued"}, indent=2))
    elif args.command == "ingest-batch":
        exports = load_batch_runner_export(args.trajectories)
        SessionQueue(database.parent / "sessions.sqlite3").ingest_many(exports)
        print(json.dumps({
            "imported": len(exports),
            "session_ids": [item["session_id"] for item in exports],
            "status": "queued_as_untrusted_evidence",
        }, indent=2))
    elif args.command == "dream":
        if args.stage == "deep":
            if args.hypotheses is None:
                raise ValueError("DEEP dreaming requires --hypotheses JSON input")
            hypotheses = json.loads(args.hypotheses.read_text(encoding="utf-8"))
            if not isinstance(hypotheses, list) or not all(isinstance(row, dict) for row in hypotheses):
                raise ValueError("--hypotheses must contain a JSON array of objects")
            result = DreamingEngine(OpenAICompatibleQwen(QwenConfig.from_env())).deep(hypotheses)
            print(json.dumps({"candidates": result}, indent=2))
            return 0
        queue = SessionQueue(database.parent / "sessions.sqlite3")
        session_ids = queue.queued(args.limit)
        sessions = [queue.get(session_id) for session_id in session_ids]
        if args.stage == "light":
            report = DreamingEngine().light(sessions)
        else:
            report = DreamingEngine(OpenAICompatibleQwen(QwenConfig.from_env())).rem(sessions)
        for session_id in session_ids:
            queue.set_status(session_id, "processed")
        print(json.dumps({
            "sessions_processed": report.sessions_processed,
            "clusters": report.clusters,
            "signals": [signal.__dict__ for signal in report.signals],
            "hypotheses": report.hypotheses,
        }, indent=2))
    elif args.command == "benchmarks":
        store = BenchmarkStore(benchmark_root)
        if args.benchmark_command == "validate":
            rows = store.validate()
            print(json.dumps({"valid_benchmarks": len(rows), "version": store.version}))
        elif args.benchmark_command == "mine":
            session = SessionQueue(database.parent / "sessions.sqlite3").get(args.session_id)
            candidate = mine_session(session)
            added = ExperimentStore(database).save_benchmark_candidate(candidate, args.session_id)
            print(json.dumps(candidate.__dict__ | {
                "eligibility": candidate.eligibility.value,
                "status": "created" if added else "deduplicated",
            }, indent=2))
        elif args.benchmark_command == "candidates":
            print(json.dumps(ExperimentStore(database).list_benchmark_candidates(), indent=2))
        else:
            print(json.dumps(store.list(Role(args.role)), indent=2))
    elif args.command == "experiments":
        store = ExperimentStore(database)
        if args.experiment_command == "list":
            print(json.dumps(store.list_experiments(args.limit), indent=2))
        else:
            print(json.dumps(store.show_experiment(args.experiment_id), indent=2))
    elif args.command == "candidates":
        print(json.dumps(ExperimentStore(database).list_candidates(args.limit), indent=2))
    elif args.command == "candidate":
        with args.content.open("rb") as source:
            content_bytes = source.read(256 * 1024 + 1)
        if len(content_bytes) > 256 * 1024:
            raise ValueError("skill candidate content exceeds the 256 KiB limit")
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("skill candidate content must be UTF-8") from exc
        benchmark_version = BenchmarkStore(benchmark_root).version
        result = create_skill_candidate(
            args.repository,
            ExperimentStore(database),
            experiment_id=args.experiment_id,
            candidate_id=args.candidate_id,
            candidate_type=CandidateType(args.type),
            target=args.target,
            content=content,
            hypothesis=args.hypothesis,
            benchmark_version=benchmark_version,
            source_sessions=tuple(args.source_session),
        )
        print(json.dumps({
            "candidate_id": result.candidate_id,
            "experiment_id": result.experiment_id,
            "candidate_hash": result.content_hash,
            "base_commit": result.base_commit,
            "commit": result.commit,
            "target": result.target,
            "status": "awaiting_evaluation",
        }, indent=2))
    elif args.command == "releases":
        print(json.dumps(ExperimentStore(database).list_releases(), indent=2))
    elif args.command == "monitor":
        observations = _load_observations(args.observations)
        monitor = ReleaseMonitor(database.parent / "monitoring.sqlite3", ExperimentStore(database))
        results = []
        for observation in observations:
            decision = monitor.observe(observation)
            results.append({
                "observation_id": observation.observation_id,
                "release_id": observation.release_id,
                "task_id": observation.task_id,
                "decision": decision.status,
                "sample_count": decision.sample_count,
                "reasons": decision.reasons,
                "duplicate": decision.duplicate,
            })
        print(json.dumps({"observations": results}, indent=2))
    elif args.command == "evaluate":
        _require_role_uid("evaluator")
        private_key_path = os.environ.get("HERMES_EVALUATOR_PRIVATE_KEY")
        if not private_key_path:
            raise RuntimeError("HERMES_EVALUATOR_PRIVATE_KEY must name the evaluator-only Ed25519 private key")
        holdout_root = args.holdout_root.resolve()
        public_root = benchmark_root.resolve()
        if holdout_root == public_root or holdout_root.is_relative_to(public_root) or public_root.is_relative_to(holdout_root):
            raise ValueError("sealed holdout root must be outside and separate from the public benchmark tree")
        store = ExperimentStore(database)
        stored_candidate = store.get_candidate(args.candidate_id)
        diff = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(args.repository),
             "diff", "--no-ext-diff", "--no-textconv", "--binary",
             stored_candidate["base_commit"], stored_candidate["commit_id"]],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
        if diff.returncode:
            raise RuntimeError(f"cannot read immutable candidate diff: {diff.stderr.decode(errors='replace')}")
        diff_hash = hashlib.sha256(diff.stdout).hexdigest()
        if diff_hash != stored_candidate["diff_hash"]:
            raise ValueError("Git candidate diff does not match its stored immutable hash")
        candidate = Candidate(
            stored_candidate["id"],
            stored_candidate["experiment_id"],
            CandidateType(stored_candidate["candidate_type"]),
            stored_candidate["target"],
            stored_candidate["base_commit"],
            stored_candidate["commit_id"],
            diff.stdout.decode("utf-8"),
            stored_candidate["created_at"],
        )
        if candidate.content_hash != stored_candidate["content_hash"]:
            raise ValueError("candidate metadata differs from its stored immutable content hash")
        access_store = BenchmarkStore(
            benchmark_root,
            holdout_root=holdout_root,
        )
        benchmarks = access_store.validate()
        full_ids = [
            item.id for item in benchmarks
            if item.split is not Split.SEALED_HOLDOUT
            and "FULL" in {str(suite).upper() for suite in item.benchmark.get("suites", [])}
        ]
        holdout_ids = [item.id for item in benchmarks if item.split is Split.SEALED_HOLDOUT]
        if not full_ids or not holdout_ids:
            raise ValueError("promotion evaluation requires at least one FULL and one sealed holdout benchmark")
        try:
            seeds = [int(value) for value in args.seeds.split(",")]
        except ValueError as exc:
            raise ValueError("--seeds must be comma-separated integers") from exc
        if not seeds or any(value < 0 for value in seeds):
            raise ValueError("--seeds must contain non-negative integers")
        security_access = access_store.access
        grant = security_access.evaluator_holdout_grant(Role.EVALUATOR)
        artifact_store = ArtifactStore(database.parent / "artifacts")
        runner = DockerCommandRunner(args.repository, artifact_store)
        evaluator = PairedEvaluator(
            access_store,
            runner,
            PromotionPolicy(),
            experiment_store=store,
            evaluator_signing_key=Path(private_key_path).read_bytes(),
            artifact_store=artifact_store,
        )
        experiment = store.show_experiment(stored_candidate["experiment_id"])
        consume_holdout_generation(
            SpentHoldoutRegistry(str(database)),
            [item for item in benchmarks if item.split is Split.SEALED_HOLDOUT],
            experiment["lineage_id"],
        )
        report = evaluator.evaluate(
            candidate,
            baseline_commit=stored_candidate["base_commit"],
            benchmark_ids=full_ids + holdout_ids,
            seeds=seeds,
            suite="FULL",
            benchmark_version=access_store.version,
            holdout_grant=grant,
            evaluation_id=f"eval-{args.candidate_id}-{__import__('uuid').uuid4().hex}",
        )
        print(json.dumps(report_payload(report), indent=2))
    elif args.command in {"promote", "rollback"}:
        _require_role_uid("promoter")
        manifest_key = os.environ.get("HERMES_IMPROVE_MANIFEST_KEY")
        if manifest_key is None:
            raise RuntimeError("HERMES_IMPROVE_MANIFEST_KEY is required")
        promoter = GitPromoter(
            ExperimentStore(database),
            args.repository,
            args.release_root,
            args.deployment_link,
            PromotionPolicy(),
            args.evaluator_public_key.read_bytes(),
        )
        if args.command == "promote":
            result = promoter.promote(
                args.candidate_id, release_name=args.release, manifest_key=manifest_key.encode()
            )
        else:
            result = {"previous_commit": promoter.rollback(
                args.release, manifest_key=manifest_key.encode()
            )}
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
