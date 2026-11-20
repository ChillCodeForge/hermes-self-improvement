from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import __version__
from .artifacts import ArtifactStore
from .batch_ingestion import load_batch_runner_export
from .benchmark_schema import benchmark_json_schema, load_benchmark_document, parse_benchmark_document
from .benchmarks import BenchmarkStore, lint_public_corpus
from .curator import CuratorAdapter, CuratorMutation
from .curator_connector import CuratorConnector
from .decisions import DECISION_TYPES, DecisionRecorder
from .candidate_builder import create_candidate, create_skill_candidate
from .clustering import NOT_CONFIGURED, SemanticReport
from .config import config_from_environment
from .embeddings import semantic_layer_from_environment
from .approvals import approve_promotion
from .errors import AccessDenied, ImprovementError
from .docker_runner import DockerCommandRunner, LocalSkillProcedureRunner
from .hermes_plugin import HookSignalStore
from .observability import ObservabilityLog, build_report, build_status, render_text
from .shadow_replay import CommandReplayRunner, ShadowReplayer
from .dreaming import DreamingEngine
from .ingestion import SessionQueue, contains_secret_material
from .holdout_rotation import rotate_holdout
from .isolation import (
    ROLES as ISOLATION_ROLES, check_isolation, enforce_isolation, load_layout, require_agent_paths, resolve_layout_path,
)
from .miner import corpus_similarity_index, mine_session
from .replay import is_command_benchmark, replay_benchmark, replay_kind
from .monitoring import ProductionObservation, ReleaseMonitor, validate_observation
from .session_feed import DEFAULT_SWITCH_MARGIN_SECONDS, SessionFeed
from .evaluation_signing import report_payload
from .evaluator import PairedEvaluator
from .hermes_runner import HermesBatchRunner, HermesModelConfig
from .judge import PairwiseJudge
from .evolution_loop import EvaluationGateway, EvolutionConfig, EvolutionLoop
from .gepa_adapter import SUPPORTED_GEPA_VERSION, GepaOptimizeStrategy, GepaStrategy
from .models import Benchmark, Candidate, CandidateType, Eligibility, Role, Split
from .optimizer import ModelMutationStrategy
from .policy import PromotionPolicy
from .promotion import GitPromoter
from .proposal_generator import generate_candidates_from_proposals
from .qwen import OpenAICompatibleQwen, QwenConfig, resolve_role_models
from .role_handoff import export_candidate, import_bundle_ref, import_candidate, sync_production_baseline
from .run_policy import RunPolicy
from .sandbox import DockerSandboxExecutor, NetworkPolicy, SandboxLimits
from .holdout_gate import HoldoutSpendingRunner, evaluator_public_key, require_holdout_finalist
from .security import SpentHoldoutRegistry, consume_holdout_generation, holdout_generation_spent
from .target_paths import assert_evaluable_candidate_paths
from .store import ExperimentStore
from .suites import (
    ImpactScope, Suite, corpus_manifest, impact_scope_for_candidate, select_evaluation_suite, select_suite,
    smoke_size_warning, suite_manifest,
)
# Observability and decision-history helpers (spec §35/§36).
from contextlib import nullcontext
from .eligibility import assess_benchmark, eligibility_decision_fields
from .evaluator import judge_route, needs_qwen_decision_fields
from .observability import record_qwen_usage
from .suites import suite_selection_decision_fields


_MAX_SQLITE_INTEGER = (1 << 63) - 1
_MAX_MONITOR_FILE_BYTES = 10 * 1024 * 1024
_MAX_CURATOR_REPORT_BYTES = 1024 * 1024
_MAX_CURATOR_REPORT_RECORDS = 10_000
_MAX_DREAMING_PROPOSAL_BYTES = 1024 * 1024
_MAX_DREAMING_PROPOSALS = 5
_MAX_CANDIDATE_FILE_BYTES = 256 * 1024
_MAX_EVOLUTION_SETTINGS_BYTES = 64 * 1024


def _paths() -> tuple[Path, Path]:
    home = Path(os.environ.get("HERMES_IMPROVE_HOME", Path.home() / ".hermes-improve"))
    return home / "experiments.sqlite3", Path(os.environ.get("HERMES_IMPROVE_BENCHMARKS", "benchmarks"))


def _report_semantic_layer(report: SemanticReport) -> dict[str, object]:
    """Layer-5 status for CLI output; a missing or partial layer is also warned on stderr."""
    warning = report.warning()
    if warning is not None:
        print(f"warning: {warning}", file=sys.stderr)
    return report.to_dict()


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
        "skills", "source",
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
            if key == "tokens" and value is None:
                continue  # tokens not reported: recorded as unknown, never as 0
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
        skills = item.get("skills", [])
        if not isinstance(skills, list):
            raise ValueError(f"monitoring observation line {line_number}: skills must be a list of skill names")
        observation = ProductionObservation(
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
            skills=tuple(skills),
            source=item.get("source", "production"),
        )
        try:
            validate_observation(observation)
        except ValueError as exc:
            raise ValueError(f"monitoring observation line {line_number}: {exc}") from exc
        observations.append(observation)
    if not observations:
        raise ValueError("monitoring observation file contains no records")
    return observations


def _bounded_json_preview(value: object, limit: int) -> dict[str, object]:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return {"json": serialized[:limit], "truncated": len(serialized) > limit}


def _load_curator_usage_report(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    if len(raw) > _MAX_CURATOR_REPORT_BYTES:
        raise ValueError("Curator usage report exceeds the 1 MiB limit")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Curator usage report must be valid UTF-8 JSON") from exc
    if not isinstance(data, list) or len(data) > _MAX_CURATOR_REPORT_RECORDS:
        raise ValueError("Curator usage report must be an array of at most 10,000 rows")
    allowed = {"name", "use_count", "patch_count", "activity_count"}
    for index, row in enumerate(data, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Curator usage report row {index} must be an object")
        extra = row.keys() - allowed
        if extra:
            raise ValueError(
                f"Curator usage report row {index} has unsupported fields: "
                f"{', '.join(sorted(extra))}"
            )
    return data


def _load_dreaming_proposal(
    path: Path, index: int,
) -> tuple[CandidateType, str, str]:
    if isinstance(index, bool) or index < 0:
        raise ValueError("Dreaming proposal index must be a non-negative integer")
    with path.open("rb") as source:
        raw = source.read(_MAX_DREAMING_PROPOSAL_BYTES + 1)
    if len(raw) > _MAX_DREAMING_PROPOSAL_BYTES:
        raise ValueError("Dreaming proposal file exceeds the 1 MiB limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Dreaming proposal file must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("candidates"), list):
        raise ValueError("Dreaming proposal file must contain a candidates array")
    candidates = value["candidates"]
    if len(candidates) > _MAX_DREAMING_PROPOSALS:
        raise ValueError("Dreaming proposal file exceeds the five-candidate limit")
    if index >= len(candidates):
        raise ValueError("Dreaming proposal index is out of range")
    proposal = candidates[index]
    if not isinstance(proposal, dict):
        raise ValueError("selected Dreaming proposal must be an object")
    candidate_type = proposal.get("candidate_type")
    target = proposal.get("target")
    hypothesis = proposal.get("hypothesis")
    if not isinstance(candidate_type, str) or candidate_type not in {
        "SKILL_CREATE", "SKILL_PATCH",
    }:
        raise ValueError("Dreaming proposals only support skill create/patch candidates")
    if (
        not isinstance(target, str) or not target.strip()
        or not isinstance(hypothesis, str) or not hypothesis.strip()
    ):
        raise ValueError("selected Dreaming proposal requires a target and hypothesis")
    return CandidateType(candidate_type), target, hypothesis


def _load_candidate_files(specs: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for spec in specs:
        target, separator, local = spec.partition("=")
        if not separator or not target or not local:
            raise ValueError("--file expects BRAIN_PATH=LOCAL_FILE")
        if target in files:
            raise ValueError(f"--file names the same brain path twice: {target}")
        with Path(local).open("rb") as source:
            raw = source.read(_MAX_CANDIDATE_FILE_BYTES + 1)
        if len(raw) > _MAX_CANDIDATE_FILE_BYTES:
            raise ValueError(f"candidate file exceeds the 256 KiB limit: {local}")
        try:
            files[target] = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"candidate file must be UTF-8: {local}") from exc
    return files


def _candidate_summary(candidate: Candidate, worktree_root: Path | None) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "experiment_id": candidate.experiment_id,
        "candidate_type": candidate.candidate_type.value,
        "candidate_hash": candidate.content_hash,
        "base_commit": candidate.base_commit,
        "commit": candidate.commit,
        "diff_hash": hashlib.sha256(candidate.diff.encode()).hexdigest(),
        "branch": f"candidate/{candidate.experiment_id}",
        "worktree": (
            str(Path(worktree_root).resolve() / candidate.experiment_id)
            if worktree_root is not None else None
        ),
        "target": candidate.target,
        "status": "awaiting_evaluation",
    }


def select_suite_benchmarks(
    benchmarks: list[Benchmark], suite: str, *, impact: ImpactScope | None = None,
) -> tuple[list[str], list[str]]:
    """Task IDs of a staged evaluation via WP-B's `select_suite`; FULL adds every sealed holdout task."""
    if str(suite).upper() not in {"SMOKE", "CORE", "FULL"}:
        raise ValueError(f"unsupported evaluation suite: {suite!r}")
    selected, holdout, _applied = select_evaluation_suite(benchmarks, suite, impact=impact)
    return [item.id for item in selected], [item.id for item in holdout]


def _solver_role_defaults() -> tuple[str | None, str | None, str | None]:
    """Model, OpenAI-compatible base URL and API-key variable name of the qwen_solver role."""
    models = config_from_environment().models
    model = resolve_role_models(models, os.environ).get("qwen_solver")
    base_url = os.environ.get(models.endpoint_env["base_url"], "").rstrip("/")
    key_env = models.endpoint_env["api_key"]
    return model, f"{base_url}/v1" if base_url else None, key_env if os.environ.get(key_env) else None


def build_runner(args: argparse.Namespace, artifact_store: ArtifactStore):
    """Construct the sandboxed benchmark runner selected on the command line."""
    limits = SandboxLimits(
        timeout_seconds=args.timeout, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
    )
    network = NetworkPolicy()
    if args.network is not None or args.network_allow:
        network = NetworkPolicy("allowlist", args.network, frozenset(args.network_allow))
    http_allowlist = frozenset(args.verifier_http_allow)
    benchmark_root = getattr(args, "benchmark_root", None)
    holdout_root = getattr(args, "holdout_root", None)
    if args.runner == "docker":
        return DockerCommandRunner(
            args.repository, artifact_store, default_timeout=args.timeout, network=network,
            verifier_http_allowlist=http_allowlist, benchmark_root=benchmark_root, holdout_root=holdout_root,
        )
    if args.runner == "local":
        if not getattr(args, "acknowledge_no_container_isolation", False):
            raise ValueError("--runner local requires --acknowledge-no-container-isolation")
        if network.mode != "disabled" or http_allowlist:
            raise ValueError("--runner local supports disabled networking only")
        if benchmark_root is None:
            raise ValueError("--runner local needs the evaluator benchmark root")
        return LocalSkillProcedureRunner(
            args.repository, artifact_store, benchmark_root=benchmark_root, holdout_root=holdout_root,
            default_timeout=args.timeout, acknowledge_no_isolation=True,
        )
    model_name, model_base_url, model_api_key_env = args.model, args.model_base_url, args.model_api_key_env
    if not model_name or not model_base_url:
        # Default to the qwen_solver logical role from config/models.yaml.
        solver_model, solver_base_url, solver_key_env = _solver_role_defaults()
        model_name = model_name or solver_model
        model_base_url = model_base_url or solver_base_url
        model_api_key_env = model_api_key_env or solver_key_env
    if args.hermes_agent_dir is None or not model_name or not model_base_url:
        raise ValueError(
            "--runner hermes requires --hermes-agent-dir and a solver model "
            "(--model/--model-base-url or the qwen_solver role environment)"
        )
    if network.mode != "disabled":
        raise ValueError("--runner hermes executes agent tools with networking disabled only")

    def executor_factory(benchmark: Benchmark) -> DockerSandboxExecutor:
        return DockerSandboxExecutor(str(benchmark.environment.get("container_image")), limits)

    return HermesBatchRunner(
        args.repository, artifact_store,
        hermes_agent_dir=args.hermes_agent_dir,
        model=HermesModelConfig(
            model=model_name, base_url=model_base_url, api_key_env=model_api_key_env,
        ),
        executor_factory=executor_factory,
        python_executable=args.hermes_python,
        limits=limits,
        fixtures_root=args.fixtures_root,
        verifier_http_allowlist=http_allowlist,
    )


def add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Runner selection and sandbox options shared by ``evaluate`` and ``evolve`` (read by build_runner)."""
    parser.add_argument(
        "--runner", choices=["docker", "hermes", "local"], default="docker",
        help="docker: command benchmarks; hermes: hermes_agent benchmarks with the real Hermes agent; "
             "local: skill_procedure benchmarks only, in a process sandbox without container isolation",
    )
    parser.add_argument(
        "--acknowledge-no-container-isolation", action="store_true",
        help="required with --runner local (development hosts and tests without Docker)",
    )
    parser.add_argument("--timeout", type=float, default=120, help="per-run sandbox timeout in seconds")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=200)
    parser.add_argument("--network", help="dedicated internal Docker network for allowlisted benchmarks")
    parser.add_argument("--network-allow", action="append", default=[], metavar="HOST:PORT")
    parser.add_argument("--verifier-http-allow", action="append", default=[], metavar="HOST:PORT")
    parser.add_argument(
        "--hermes-agent-dir", type=Path,
        help="Hermes source checkout (run_agent.py, batch_runner.py) used by the evaluation driver",
    )
    parser.add_argument("--hermes-python", default=sys.executable)
    parser.add_argument("--model")
    parser.add_argument("--model-base-url")
    parser.add_argument("--model-api-key-env")
    parser.add_argument("--fixtures-root", type=Path)


#: Benchmarks each `--runner` replays, and their name in error messages.
_REPLAY_SELECTION: dict[str, tuple[str, Callable[[Benchmark], bool]]] = {
    "docker": ("command", is_command_benchmark),
    "local": ("skill_procedure", lambda item: replay_kind(item) == "skill_procedure"),
    "hermes": ("hermes_agent", lambda item: replay_kind(item) == "hermes_agent"),
}


def _replay_benchmarks(args: argparse.Namespace, store: BenchmarkStore, database: Path) -> int:
    label, selected = _REPLAY_SELECTION[args.runner]
    # `store.validate()` reads the public tree only; replay_benchmark refuses SEALED_HOLDOUT.
    benchmarks = [item for item in store.validate() if selected(item)]
    if args.benchmark_id is not None:
        benchmarks = [item for item in benchmarks if item.id == args.benchmark_id]
        if not benchmarks:
            raise ValueError(f"no public {label} benchmark with id {args.benchmark_id!r}")
        benchmarks = [max(benchmarks, key=lambda item: item.version)]
    if not benchmarks:
        raise ValueError(f"no public {label} benchmarks to replay")
    artifact_store = ArtifactStore(database.parent / "artifacts")
    # Same runner selection as `evaluate`, but never with the sealed holdout root.
    args.benchmark_root = store.root
    args.holdout_root = None
    runner = build_runner(args, artifact_store)
    model = getattr(getattr(runner, "model", None), "model", None)
    experiment_store = ExperimentStore(database)
    results = []
    for benchmark in sorted(benchmarks, key=lambda item: (item.id, item.version)):
        result = replay_benchmark(
            runner, benchmark, commit=args.commit, runs=args.runs, runner_name=args.runner,
            model=model if isinstance(model, str) else None,
        )
        payload = result.to_dict()
        experiment_store.record_benchmark_replay(payload)
        results.append(payload)
    print(json.dumps({
        "replayed": len(results),
        "replay_verified": sum(1 for item in results if item["replay_verified"]),
        "results": results,
    }, indent=2))
    return 0 if all(item["replay_verified"] for item in results) else 1


def _read_label_lines(path: Path) -> list[object]:
    """Parse an outcome-label JSONL file (bounded like a session batch)."""
    with path.open("rb") as source:
        raw = source.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("outcome label file exceeds the 10 MiB limit")
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"outcome label line {number} is invalid JSON: {exc}") from exc
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-improve", description="Safety-gated Hermes improvement foundations")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="current brain version, active release and ledger counts (read-only)")
    status.add_argument(
        "--deployment-link", type=Path,
        help="deployment symlink Hermes reads (default: the provisioned layout's deploy/current, if any)",
    )
    ingest = sub.add_parser("ingest")
    ingest.add_argument("export", type=Path)
    batch_ingest = sub.add_parser("ingest-batch")
    batch_ingest.add_argument("trajectories", type=Path)
    sessions = sub.add_parser("sessions", help="attach outcome evidence to queued sessions")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    label = sessions_sub.add_parser("label", help="label one queued session's outcome")
    label.add_argument("session_id")
    label.add_argument("--outcome", choices=["correct", "incorrect", "partial"], required=True)
    label.add_argument("--source", choices=["verifier", "user", "monitor"], required=True)
    label.add_argument("--note", default="", help="short redacted note, e.g. the expected value")
    import_labels = sessions_sub.add_parser(
        "import-labels", help="JSONL of {session_id, outcome, source, note?}; all or nothing",
    )
    import_labels.add_argument("labels", type=Path)
    dream = sub.add_parser("dream")
    dream.add_argument("--stage", choices=["light", "rem", "deep"], default="light")
    dream.add_argument("--limit", type=int, default=100)
    dream.add_argument("--hypotheses", type=Path)
    dream.add_argument("--curator-report", type=Path)
    dream.add_argument(
        "--curator-hermes-home", type=Path,
        help="read live Curator usage read-only from this Hermes home",
    )
    curator = sub.add_parser("curator")
    curator_sub = curator.add_subparsers(dest="curator_command", required=True)
    curator_status = curator_sub.add_parser("status")
    curator_propose = curator_sub.add_parser("propose")
    curator_propose.add_argument("--kind", choices=["create", "patch", "consolidate", "archive"], required=True)
    curator_propose.add_argument("--skill", required=True)
    curator_propose.add_argument("--into")
    curator_propose.add_argument("--rationale", required=True)
    curator_propose.add_argument("--content", type=Path, help="complete new SKILL.md text")
    curator_propose.add_argument("--old-string-file", type=Path)
    curator_propose.add_argument("--new-string-file", type=Path)
    curator_propose.add_argument("--replace-all", action="store_true")
    curator_convert = curator_sub.add_parser("convert-pending")
    curator_convert.add_argument("pending_id")
    for command in (curator_status, curator_convert):
        command.add_argument("--hermes-home", type=Path)
    for command in (curator_propose, curator_convert):
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--experiment-id", required=True)
        command.add_argument("--candidate-id", required=True)
    report = sub.add_parser("report")
    report.add_argument("--json", action="store_true")
    decisions = sub.add_parser("decisions")
    decisions_sub = decisions.add_subparsers(dest="decisions_command", required=True)
    decisions_export = decisions_sub.add_parser("export")
    decisions_export.add_argument("--output", type=Path, help="new JSONL file (default: stdout)")
    decisions_export.add_argument("--type", choices=sorted(DECISION_TYPES))
    decisions_export.add_argument("--labeled-only", action="store_true")
    decisions_backfill = decisions_sub.add_parser("backfill")
    decisions_backfill.add_argument("decision_id")
    decisions_backfill.add_argument("--ground-truth", required=True)
    decisions_backfill.add_argument("--source", required=True)
    benchmarks = sub.add_parser("benchmarks")
    benchmark_sub = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    validate_benchmarks = benchmark_sub.add_parser("validate")
    validate_benchmarks.add_argument(
        "--holdout-root", type=Path,
        help="also check public/holdout leakage against this sealed holdout directory",
    )
    mine = benchmark_sub.add_parser("mine")
    mine.add_argument("session_id")
    benchmark_sub.add_parser("candidates")
    inspect_candidate = benchmark_sub.add_parser("inspect-candidate")
    inspect_candidate.add_argument("task_fingerprint")
    review_candidate = benchmark_sub.add_parser("review-candidate")
    review_candidate.add_argument("task_fingerprint")
    review_candidate.add_argument("--decision", choices=["accept", "reject"], required=True)
    review_candidate.add_argument("--reviewer", required=True)
    review_candidate.add_argument("--rationale", required=True)
    listing = benchmark_sub.add_parser("list")
    listing.add_argument("--role", choices=[role.value for role in Role], default=Role.OPERATOR.value)
    listing.add_argument(
        "--suite", choices=[suite.value for suite in Suite if suite is not Suite.SEALED_HOLDOUT],
        help="select a suite and print its manifest",
    )
    listing.add_argument(
        "--impact", action="append", default=[],
        help="impact scope: category:NAME, capability:NAME or target:PATH (repeatable)",
    )
    benchmark_sub.add_parser("schema", help="print the benchmark record JSON Schema")
    benchmark_sub.add_parser("manifest", help="print the public corpus manifest")
    replay = benchmark_sub.add_parser(
        "replay",
        help="re-run public benchmarks with the selected runner and check reproducibility "
             "(docker: command, local: skill_procedure, hermes: hermes_agent benchmarks)",
    )
    replay.add_argument(
        "--id", dest="benchmark_id", help="replay one benchmark (default: all benchmarks of the runner's kind)",
    )
    replay.add_argument("--commit", required=True, help="full Git object ID of the repository state")
    replay.add_argument("--repository", type=Path, required=True)
    replay.add_argument("--runs", type=int, default=3)
    add_runner_arguments(replay)
    holdout = benchmark_sub.add_parser("holdout", help="operator-managed sealed holdout generations")
    holdout_sub = holdout.add_subparsers(dest="holdout_command", required=True)
    rotate = holdout_sub.add_parser("rotate", help="validate STAGING and activate it as a new generation")
    rotate.add_argument("staging", type=Path)
    rotate.add_argument("--store-root", type=Path, required=True)
    add_benchmark = benchmark_sub.add_parser("add")
    add_benchmark.add_argument("file", type=Path, help="operator-authored public benchmark JSON or YAML")
    add_benchmark.add_argument(
        "--holdout-root", type=Path, required=True,
        help="sealed holdout benchmark directory used for leakage validation",
    )
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("candidate_id")
    evaluate.add_argument("--repository", type=Path, required=True)
    evaluate.add_argument("--holdout-root", type=Path, required=True)
    evaluate.add_argument(
        "--suite", choices=["SMOKE", "CORE", "FULL"], default="FULL",
        help="FULL also evaluates the sealed holdout; SMOKE/CORE are staged, non-deciding runs",
    )
    evaluate.add_argument("--seeds", help="explicit comma-separated seeds; default derives them from the run policy")
    add_runner_arguments(evaluate)
    evaluate.add_argument("--judge", action="store_true", help="enable the qwen_judge role (QWEN_* environment)")
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
    create.add_argument("--type", choices=[kind.value for kind in CandidateType])
    create.add_argument("--target", help="skill directory under brain/skills/")
    create.add_argument("--content", type=Path, help="file containing the new SKILL.md text")
    create.add_argument(
        "--file", action="append", default=[], metavar="BRAIN_PATH=LOCAL_FILE",
        help="write LOCAL_FILE to BRAIN_PATH in the candidate (any candidate type; repeatable)",
    )
    create.add_argument(
        "--delete", action="append", default=[], metavar="BRAIN_PATH",
        help="remove a tracked brain file in the candidate (repeatable)",
    )
    create.add_argument(
        "--worktree-root", type=Path,
        help="keep the candidate worktree at WORKTREE_ROOT/<experiment-id> (default: temporary)",
    )
    create.add_argument("--parent-experiment", help="experiment this candidate revises (lineage)")
    create.add_argument("--hypothesis")
    create.add_argument(
        "--proposal-file", type=Path,
        help="DEEP Dreaming JSON file to select candidate metadata from",
    )
    create.add_argument(
        "--proposal-index", type=int,
        help="zero-based candidate index in --proposal-file",
    )
    create.add_argument("--source-session", action="append", default=[])
    export = candidate_sub.add_parser("export", help="agent: write a candidate bundle into the handoff directory")
    export.add_argument("candidate_id")
    export.add_argument("--repository", type=Path, required=True, help="agent-owned candidate repository")
    export.add_argument("--out-dir", type=Path, required=True, help="handoff directory")
    import_parser = candidate_sub.add_parser("import", help="evaluator: verify a handoff bundle into the ledger")
    import_parser.add_argument("candidate_id")
    import_parser.add_argument("--handoff-dir", type=Path, required=True)
    import_parser.add_argument("--mirror", type=Path, required=True, help="evaluator-owned bare mirror repository")
    sync = candidate_sub.add_parser("sync-baseline", help="agent: copy production main into the candidate repository")
    sync.add_argument("--repository", type=Path, required=True)
    sync.add_argument("--production", type=Path, required=True)
    security = sub.add_parser("security")
    security_sub = security.add_subparsers(dest="security_command", required=True)
    security_check = security_sub.add_parser("check", help="verify OS isolation of the provisioned layout")
    security_check.add_argument("--role", choices=list(ISOLATION_ROLES))
    security_check.add_argument("--layout", type=Path, help="layout.json (default: $HERMES_IMPROVE_LAYOUT)")
    security_check.add_argument(
        "--production", type=Path, action="append", default=[],
        help="additional Hermes deployment checkout that must be read-only",
    )
    generate = candidate_sub.add_parser(
        "generate", help="author candidate content for selected DEEP proposals with the qwen_optimizer role",
    )
    generate.add_argument("--repository", type=Path, required=True)
    generate.add_argument("--proposal-file", type=Path, required=True)
    generate.add_argument(
        "--proposal-index", type=int, action="append", required=True,
        help="zero-based proposal index; repeat to select up to five proposals",
    )
    generate.add_argument("--source-session", action="append", default=[])
    evolve = sub.add_parser("evolve", help="iteratively evolve a skill candidate (generator + screening only)")
    evolve.add_argument("candidate_id")
    evolve.add_argument("--repository", type=Path, required=True)
    evolve.add_argument("--strategy", choices=["qwen", "gepa", "gepa-proposer"], default="qwen")
    evolve.add_argument("--settings", type=Path, help="JSON object with evolution setting overrides")
    evolve.add_argument("--max-generations", type=int)
    evolve.add_argument("--candidates-per-generation", type=int)
    evolve.add_argument("--max-candidates-total", type=int)
    # Same runner selection as `evaluate`, so agent benchmarks can be screened
    # with the real Hermes agent. There is deliberately no --holdout-root.
    add_runner_arguments(evolve)
    releases = sub.add_parser("releases")
    releases.add_argument("release_command", choices=["list", "events"])
    releases.add_argument("--release", dest="release_filter", help="limit events to one release")
    monitor = sub.add_parser("monitor")
    monitor_sub = monitor.add_subparsers(dest="monitor_command", required=True)
    observe = monitor_sub.add_parser("ingest")
    observe.add_argument("observations", type=Path)
    monitor_evaluate = monitor_sub.add_parser("evaluate")
    monitor_evaluate.add_argument("release")
    monitor_evaluate.add_argument(
        "--apply", action="store_true",
        help="execute the outcome: mark STABLE or roll back through the promoter (promoter UID required)",
    )
    monitor_replay = monitor_sub.add_parser("replay")
    monitor_replay.add_argument("release")
    monitor_replay.add_argument(
        "--runner-command", required=True,
        help='JSON array argv of the replay command, e.g. ["/opt/hermes-replay"]',
    )
    monitor_replay.add_argument("--limit", type=int, default=20)
    monitor_replay.add_argument("--timeout", type=float, default=600.0)
    monitor_replay.add_argument(
        "--source", choices=["sessions", "shadow-benchmarks"], default="sessions",
        help="replay recorded sessions (default) or SHADOW-split benchmarks",
    )
    monitor_collect = monitor_sub.add_parser(
        "collect", help="derive observations for a provisional release from ingested Hermes sessions",
    )
    monitor_collect.add_argument("release")
    monitor_collect.add_argument("--limit", type=int, default=10_000, help="sessions to scan (max 10000)")
    monitor_collect.add_argument(
        "--switch-margin", type=float, default=DEFAULT_SWITCH_MARGIN_SECONDS,
        help="exclude sessions within this many seconds of a release switch",
    )
    monitor_collect.add_argument(
        "--apply", action="store_true",
        help="execute the outcome: mark STABLE or roll back through the promoter (promoter UID required)",
    )
    promote = sub.add_parser("promote")
    promote.add_argument("candidate_id", metavar="experiment_id", help="experiment ID (or candidate ID)")
    promote.add_argument("--release", help="release tag brain-vN (default: next free number)")
    promote.add_argument(
        "--operator-public-key", type=Path, action="append", default=[],
        help="trusted operator Ed25519 public key for manual-gate approvals (repeatable)",
    )
    rollback = sub.add_parser("rollback")
    rollback.add_argument("release")
    for command in (promote, rollback):
        command.add_argument("--repository", type=Path, required=True)
        command.add_argument("--release-root", type=Path, required=True)
        command.add_argument("--deployment-link", type=Path, required=True)
        command.add_argument("--evaluator-public-key", type=Path, required=True)
    for option in ("--repository", "--release-root", "--deployment-link", "--evaluator-public-key"):
        monitor_evaluate.add_argument(option, type=Path)
        monitor_collect.add_argument(option, type=Path)
    approve = sub.add_parser("approve", help="sign an operator approval for a manually gated experiment")
    approve.add_argument("experiment_id")
    approve.add_argument("--operator", required=True)
    approve.add_argument("--rationale", required=True)
    approve.add_argument(
        "--repository", type=Path,
        help="candidate repository; needed to approve a skill-removal candidate of an auto-gated type",
    )
    promote.add_argument(
        "--candidate-bundle", type=Path,
        help="handoff bundle to fetch the evaluated candidate branch from before promotion",
    )
    return parser


def _hermes_home(configured: Path | None) -> Path:
    if configured is not None:
        return configured
    environment = os.environ.get("HERMES_HOME", "").strip()
    return Path(environment) if environment else Path.home() / ".hermes"


def _read_text_file(path: Path, label: str, limit: int = 256 * 1024) -> str:
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{label} exceeds the {limit // 1024} KiB limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc


def _curator_status(database: Path, hermes_home: Path) -> dict[str, object]:
    connector = CuratorConnector(hermes_home)
    snapshot = connector.snapshot()
    recorder = DecisionRecorder(database)
    signals = []
    for signal in connector.signals(snapshot):
        decision = recorder.record(
            decision_type="curator_signal",
            input_features={"summary": signal.summary, "evidence": dict(signal.evidence)},
            available_options=("no_action", "propose_candidate"),
            selected_option="propose_candidate",
            confidence=signal.impact_score,
            source="curator-connector",
            model_version="curator-signal-rules-1",
        )
        signals.append(signal.__dict__ | {"evidence": dict(signal.evidence), "decision_id": decision.id})
    hook_path = database.parent / "hook_signals.sqlite3"
    return {
        "curator": connector.status(),
        "signals": signals,
        "hook_signals": HookSignalStore(hook_path).summary() if hook_path.is_file() else "no data",
        "production_writes": "none; Curator mutations are converted into candidates",
    }


def _record_risk_decision(database: Path, config, candidate_type: CandidateType, args, result) -> None:
    """Decision history (spec §36) for the configured candidate-type risk lookup; recorded after use."""
    from .config import RISK_LEVELS

    DecisionRecorder(database).record(
        decision_type="risk",
        input_features={
            "candidate_type": candidate_type.value,
            "target": result.target,
            "changed_files": len(args.file),
            "deleted_files": len(args.delete),
            "experiment_id": result.experiment_id,
        },
        available_options=RISK_LEVELS,
        selected_option=config.taxonomy.candidate_risk[candidate_type],
        confidence=None,
        source="taxonomy-candidate-risk",
        model_version=f"taxonomy-{config.digest[:16]}",
    )


def _conversion_payload(conversion) -> dict[str, object]:
    candidate = conversion.candidate
    return {
        "candidate_id": candidate.candidate_id,
        "experiment_id": candidate.experiment_id,
        "candidate_type": candidate.candidate_type.value,
        "candidate_hash": candidate.content_hash,
        "target": candidate.target,
        "commit": candidate.commit,
        "mutation_kind": conversion.mutation_kind,
        "unapplied": list(conversion.unapplied),
        "status": "awaiting_evaluation",
    }


def _apply_monitoring(args: argparse.Namespace, database: Path, monitor: ReleaseMonitor) -> dict[str, object]:
    """Execute a monitoring outcome (STABLE or rollback) as the promoter."""
    _require_role_uid("promoter")
    manifest_key = os.environ.get("HERMES_IMPROVE_MANIFEST_KEY")
    if manifest_key is None:
        raise RuntimeError("HERMES_IMPROVE_MANIFEST_KEY is required")
    missing = [
        option for option, value in (
            ("--repository", args.repository), ("--release-root", args.release_root),
            ("--deployment-link", args.deployment_link),
            ("--evaluator-public-key", args.evaluator_public_key),
        ) if value is None
    ]
    if missing:
        raise ValueError(f"monitor {args.monitor_command} --apply requires {', '.join(missing)}")
    promoter = GitPromoter(
        ExperimentStore(database), args.repository, args.release_root, args.deployment_link,
        PromotionPolicy.from_config(config_from_environment().promotion),
        args.evaluator_public_key.read_bytes(),
    )
    return monitor.apply(
        args.release, promoter=promoter, manifest_key=manifest_key.encode(),
        actor=f"uid:{os.geteuid()}",
    )


def run(argv: list[str] | None = None) -> int:
    """Execute one CLI command; refusals propagate as exceptions (see :func:`main`)."""
    args = build_parser().parse_args(argv)
    database, benchmark_root = _paths()
    if args.command == "status":
        deployment_link = args.deployment_link
        if deployment_link is None:
            layout_path = resolve_layout_path()
            if layout_path.is_file():
                try:
                    deployment_link = load_layout(layout_path).deployment_link
                except (OSError, ValueError):
                    deployment_link = None
        print(json.dumps({
            "version": __version__, "database": str(database), "benchmarks": str(benchmark_root),
            **build_status(database.parent, benchmark_root, deployment_link=deployment_link),
        }, indent=2))
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
    elif args.command == "sessions":
        queue = SessionQueue(database.parent / "sessions.sqlite3")
        if args.sessions_command == "label":
            outcome = queue.label_outcome(
                args.session_id, outcome=args.outcome, source=args.source, note=args.note,
            )
            print(json.dumps({"session_id": args.session_id, "outcome": outcome}, indent=2))
        else:
            print(json.dumps({"labelled": queue.label_outcomes_many(_read_label_lines(args.labels))}))
    elif args.command == "dream":
        if args.stage == "deep":
            if args.curator_report is not None or args.curator_hermes_home is not None:
                raise ValueError("--curator-report is supported only by LIGHT and REM dreaming")
            if args.hypotheses is None:
                raise ValueError("DEEP dreaming requires --hypotheses JSON input")
            hypotheses = json.loads(args.hypotheses.read_text(encoding="utf-8"))
            if not isinstance(hypotheses, list) or not all(isinstance(row, dict) for row in hypotheses):
                raise ValueError("--hypotheses must contain a JSON array of objects")
            model = OpenAICompatibleQwen(QwenConfig.from_env())
            with record_qwen_usage(database.parent, model, command="dream-deep"):
                result = DreamingEngine(model).deep(hypotheses)
            log = ObservabilityLog(database.parent / "observability.sqlite3")
            log.record_dreaming_run(
                stage="deep", sessions_processed=0, clusters={}, signals=0, hypotheses=len(result),
            )
            print(json.dumps({"candidates": result}, indent=2))
            return 0
        if args.curator_report is not None and args.curator_hermes_home is not None:
            raise ValueError("use either --curator-report or --curator-hermes-home, not both")
        if args.curator_report is not None:
            curator = CuratorAdapter(lambda: _load_curator_usage_report(args.curator_report))
        elif args.curator_hermes_home is not None:
            curator = CuratorAdapter(CuratorConnector(args.curator_hermes_home).usage_rows)
        else:
            curator = None
        queue = SessionQueue(database.parent / "sessions.sqlite3")
        # LIGHT reads sessions it has not seen and only marks its own stage;
        # REM reads every still-queued session and completes it (finding F7).
        session_ids = queue.pending_for_stage(args.stage, args.limit)
        sessions = [queue.get(session_id) for session_id in session_ids]
        log = ObservabilityLog(database.parent / "observability.sqlite3")
        if args.stage == "light":
            report = DreamingEngine(curator=curator).light(sessions)
        elif not sessions:
            # Nothing queued: no Qwen request, an empty REM report.
            report = DreamingEngine(curator=curator).light(sessions)
        else:
            model = OpenAICompatibleQwen(QwenConfig.from_env())
            with record_qwen_usage(database.parent, model, command="dream-rem"):
                report = DreamingEngine(model, curator=curator).rem(sessions)
        for session_id in session_ids:
            queue.mark_stage(session_id, args.stage)
            if args.stage == "rem":
                queue.set_status(session_id, "processed")
        log.record_dreaming_run(
            stage=args.stage, sessions_processed=report.sessions_processed, clusters=report.clusters,
            signals=len(report.signals), hypotheses=len(report.hypotheses),
        )
        print(json.dumps({
            "sessions_processed": report.sessions_processed,
            "clusters": report.clusters,
            "error_clusters": [cluster.to_dict() for cluster in report.error_clusters],
            "signals": [signal.__dict__ for signal in report.signals],
            "hypotheses": report.hypotheses,
            "rejected_hypotheses": report.rejected_hypotheses,
        }, indent=2))
    elif args.command == "benchmarks":
        # Leakage-checking commands use the semantic layer (layer 5) when an
        # embedding endpoint is configured and always report whether it ran.
        semantic = (
            semantic_layer_from_environment()
            if args.benchmark_command in {"validate", "add", "mine", "holdout"}
            else None
        )
        store = (
            BenchmarkStore(
                benchmark_root, holdout_root=getattr(args, "holdout_root", None), semantic=semantic,
            )
            if args.benchmark_command in {"add", "validate"}
            else BenchmarkStore(benchmark_root)
        )
        if args.benchmark_command == "validate":
            rows = store.validate()
            lint_public_corpus([row for row in rows if row.split is not Split.SEALED_HOLDOUT])
            print(json.dumps({
                "valid_benchmarks": len(rows), "version": store.version,
                "semantic_layer": _report_semantic_layer(store.last_leakage_report.semantic),
            }))
        elif args.benchmark_command == "mine":
            session = SessionQueue(database.parent / "sessions.sqlite3").get(args.session_id)
            candidate = mine_session(
                session,
                cluster_index=corpus_similarity_index(store.load_public(), semantic=semantic),
            )
            added = ExperimentStore(database).save_benchmark_candidate(candidate, args.session_id)
            DecisionRecorder(database).record(**eligibility_decision_fields(
                eligibility=candidate.eligibility,
                reproducibility_score=candidate.reproducibility_score,
                ground_truth_type=candidate.ground_truth_type,
                reasons=candidate.reasons,
                subject={"origin": "mined_session", "task_fingerprint": candidate.task_fingerprint},
                source="benchmark-miner",
            ))
            print(json.dumps(candidate.__dict__ | {
                "eligibility": candidate.eligibility.value,
                "status": "created" if added else "deduplicated",
                "semantic_layer": _report_semantic_layer(
                    NOT_CONFIGURED if semantic is None
                    else SemanticReport(
                        "applied", model_id=semantic.model_id, threshold=float(semantic.threshold),
                        cache_hits=semantic.cache_hits, cache_misses=semantic.cache_misses,
                    )
                ),
            }, indent=2))
        elif args.benchmark_command == "candidates":
            print(json.dumps(ExperimentStore(database).list_benchmark_candidates(), indent=2))
        elif args.benchmark_command == "inspect-candidate":
            experiment_store = ExperimentStore(database)
            proposal = experiment_store.get_benchmark_candidate(args.task_fingerprint)
            source_sessions = proposal.pop("source_sessions")
            if (
                not isinstance(source_sessions, list)
                or not all(isinstance(session_id, str) for session_id in source_sessions)
            ):
                raise ValueError("benchmark proposal has invalid source-session metadata")
            if not source_sessions:
                raise ValueError("benchmark proposal has no source sessions to inspect")
            preview = []
            session_queue = SessionQueue(database.parent / "sessions.sqlite3")
            for session_id in source_sessions[:3]:
                session = session_queue.get(session_id)
                if contains_secret_material(session):
                    raise ValueError(
                        f"source session {session_id!r} contains credential-like material; "
                        "refusing to display it"
                    )
                preview.append({
                    "session_id": session_id,
                    "task": _bounded_json_preview(session.get("task", {}), 8192),
                    "environment": _bounded_json_preview(session.get("environment", {}), 2048),
                    "verification": _bounded_json_preview(session.get("verification", {}), 4096),
                    "ground_truth": _bounded_json_preview(session.get("ground_truth", {}), 2048),
                    "security": _bounded_json_preview(session.get("security", {}), 2048),
                    "replay_verified": session.get("replay_verified"),
                })
            print(json.dumps({
                "proposal": proposal,
                "source_session_count": len(source_sessions),
                "source_preview_truncated": len(source_sessions) > len(preview),
                "source_previews": preview,
            }, indent=2))
        elif args.benchmark_command == "review-candidate":
            result = ExperimentStore(database).review_benchmark_candidate(
                args.task_fingerprint,
                decision={"accept": "ACCEPTED", "reject": "REJECTED"}[args.decision],
                reviewer=args.reviewer,
                rationale=args.rationale,
            )
            print(json.dumps({
                "task_fingerprint": result["task_fingerprint"],
                "review_status": result["review_status"],
                "reviewer": result["reviewer"],
                "review_rationale": result["review_rationale"],
                "reviewed_at": result["reviewed_at"],
                "benchmark_admitted": False,
            }, indent=2))
        elif args.benchmark_command == "add":
            document = load_benchmark_document(args.file)
            benchmark = parse_benchmark_document(document, source=args.file.name)
            path = store.admit_public(benchmark)
            leakage = store.last_leakage_report
            assessed = assess_benchmark(benchmark)
            DecisionRecorder(database).record(**eligibility_decision_fields(
                eligibility=assessed.eligibility,
                reproducibility_score=assessed.reproducibility_score,
                ground_truth_type=assessed.ground_truth.value,
                reasons=assessed.reasons,
                subject={
                    "origin": "admitted_benchmark", "benchmark_id": benchmark.id,
                    "benchmark_version": benchmark.version,
                    "declared_eligibility": str(benchmark.eligibility["class"]),
                },
                source="benchmark-admission",
            ))
            print(json.dumps({
                "status": "admitted",
                "benchmark_id": benchmark.id,
                "version": benchmark.version,
                "path": str(path),
                "manifest_version": store.version,
                "semantic_layer": _report_semantic_layer(leakage.semantic),
            }, indent=2))
        elif args.benchmark_command == "schema":
            print(json.dumps(benchmark_json_schema(), indent=2, sort_keys=True))
        elif args.benchmark_command == "manifest":
            rows = store.validate()
            print(json.dumps(corpus_manifest(rows, benchmark_version=store.version), indent=2))
        elif args.benchmark_command == "replay":
            return _replay_benchmarks(args, store, database)
        elif args.benchmark_command == "holdout":
            result = rotate_holdout(
                args.staging, args.store_root, public_root=benchmark_root, semantic=semantic,
            )
            if result.semantic_layer is not None:
                _report_semantic_layer(result.semantic_layer)
            print(json.dumps(result.to_dict(), indent=2))
        elif args.suite is not None:
            role = Role(args.role)
            visible = []
            for item in store.load_public() if role is Role.GENERATOR else store.validate():
                try:
                    store.access.authorize_benchmark_read(role, item.split)
                except AccessDenied:
                    continue
                visible.append(item)
            impact = ImpactScope.parse(args.impact)
            selected = select_suite(visible, args.suite, impact=impact)
            print(json.dumps(suite_manifest(
                selected, suite=args.suite, benchmark_version=store.version, impact=impact,
            ), indent=2))
        else:
            if args.impact:
                raise ValueError("--impact requires --suite")
            print(json.dumps(store.list(Role(args.role)), indent=2))
    elif args.command == "experiments":
        store = ExperimentStore(database)
        if args.experiment_command == "list":
            print(json.dumps(store.list_experiments(args.limit), indent=2))
        else:
            print(json.dumps(store.show_experiment(args.experiment_id), indent=2))
    elif args.command == "candidates":
        print(json.dumps(ExperimentStore(database).list_candidates(args.limit), indent=2))
    elif args.command == "candidate" and args.candidate_command == "generate":
        indexes = args.proposal_index
        if len(indexes) > _MAX_DREAMING_PROPOSALS or len(set(indexes)) != len(indexes):
            raise ValueError("select between one and five distinct --proposal-index values")
        proposals = []
        for index in indexes:
            candidate_type, target, hypothesis = _load_dreaming_proposal(args.proposal_file, index)
            proposals.append({
                "candidate_type": candidate_type.value, "target": target, "hypothesis": hypothesis,
            })
        author_model = OpenAICompatibleQwen(QwenConfig.from_env())
        with record_qwen_usage(database.parent, author_model, command="candidate-generate"):
            outcomes = generate_candidates_from_proposals(
                proposals,
                model=author_model,
                repository=args.repository,
                store=ExperimentStore(database),
                benchmark_version=BenchmarkStore(benchmark_root).version,
                source_sessions=tuple(args.source_session),
                policy_version=config_from_environment().promotion.policy_version,
            )
        print(json.dumps({
            "candidates": [
                outcome.to_dict() | {"proposal_index": indexes[outcome.proposal_index]}
                for outcome in outcomes
            ],
            "promotion": "not performed; created candidates await evaluation",
        }, indent=2))
    elif args.command == "evolve":
        # The optimizer is the generator: it runs as the agent role, which must
        # hold no holdout access and no production write (spec §2, §32).
        layout = enforce_isolation("agent")
        if layout is not None:
            require_agent_paths(
                layout, repository=args.repository, benchmark_root=benchmark_root, state_dir=database.parent,
            )
        overrides: dict[str, object] = {}
        if args.settings is not None:
            with args.settings.open("rb") as source:
                raw = source.read(_MAX_EVOLUTION_SETTINGS_BYTES + 1)
            if len(raw) > _MAX_EVOLUTION_SETTINGS_BYTES:
                raise ValueError("evolution settings file exceeds the 64 KiB limit")
            try:
                loaded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("evolution settings file must be valid UTF-8 JSON") from exc
            if not isinstance(loaded, dict):
                raise ValueError("evolution settings file must contain a JSON object")
            overrides.update(loaded)
        for key in ("max_generations", "candidates_per_generation", "max_candidates_total"):
            if getattr(args, key) is not None:
                overrides[key] = getattr(args, key)
        hermes_config = config_from_environment()
        config = EvolutionConfig.from_config(hermes_config.scoring.evolution, overrides)
        model = OpenAICompatibleQwen(QwenConfig.from_env())
        if args.strategy in {"gepa", "gepa-proposer"}:
            strategy = GepaOptimizeStrategy(model) if args.strategy == "gepa" else GepaStrategy(model)
            expected_version = f"gepa-{SUPPORTED_GEPA_VERSION}"
        else:
            strategy = ModelMutationStrategy(model)
            expected_version = ModelMutationStrategy.version
        # The screening runner is selected like `evaluate` (--runner docker|local|hermes),
        # but as the generator it must never be pointed at the sealed holdout.
        if getattr(args, "holdout_root", None) is not None:
            raise AccessDenied("evolve runs as the generator and must never receive a sealed holdout root")
        args.benchmark_root = benchmark_root
        artifact_store = ArtifactStore(database.parent / "artifacts")
        runner = build_runner(args, artifact_store)
        store = ExperimentStore(database)
        # PairedEvaluator reads runner.llm_influenced: Hermes screens with the LLM
        # run counts (SMOKE/CORE/FULL), docker and local with one deterministic run.
        gateway = EvaluationGateway(
            BenchmarkStore(benchmark_root),
            runner,
            PromotionPolicy.from_config(hermes_config.promotion),
            generator_store=BenchmarkStore(benchmark_root),
            run_policy=RunPolicy.from_config(hermes_config.scoring),
        )
        if gateway.holdout_available:
            raise AccessDenied("evolve runs as the generator and must never see sealed holdout tasks")
        for warning in gateway.screening_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        loop = EvolutionLoop(
            repository=args.repository, store=store, gateway=gateway, strategy=strategy,
            expected_strategy_version=expected_version, config=config,
            decisions=DecisionRecorder(database),
        )
        with record_qwen_usage(database.parent, model, command="evolve"):
            report = loop.run(args.candidate_id)
        print(json.dumps(report.to_dict() | {
            "next_steps": [
                f"hermes-improve evaluate {item.candidate_id} --suite SMOKE, then CORE, then FULL "
                "(evaluator identity; signed SMOKE+CORE evidence gates the sealed holdout)"
                for item in report.finalists if item.full_passed
            ],
        }, indent=2))
    elif args.command == "security":
        layout = load_layout(args.layout or resolve_layout_path())
        report = check_isolation(layout, args.role, production=args.production)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1
    elif args.command == "candidate" and args.candidate_command == "export":
        print(json.dumps(export_candidate(
            ExperimentStore(database), args.repository, args.candidate_id, args.out_dir,
        ), indent=2))
    elif args.command == "candidate" and args.candidate_command == "sync-baseline":
        commit = sync_production_baseline(args.repository, args.production)
        print(json.dumps({"main": commit, "status": "synchronized"}, indent=2))
    elif args.command == "candidate" and args.candidate_command == "import":
        layout = enforce_isolation("evaluator")
        _require_role_uid("evaluator")
        if layout is not None:
            layout.require_paths({
                "handoff": args.handoff_dir, "ledger": database.parent,
            })
            if args.mirror.resolve() != layout.evaluator_mirror:
                raise PermissionError(f"--mirror must be the provisioned evaluator mirror {layout.evaluator_mirror}")
        imported = import_candidate(ExperimentStore(database), args.handoff_dir, args.candidate_id, args.mirror)
        print(json.dumps({
            "candidate_id": imported.candidate_id,
            "experiment_id": imported.experiment_id,
            "candidate_hash": imported.content_hash,
            "commit": imported.commit,
            "status": "imported_awaiting_evaluation",
        }, indent=2))
    elif args.command == "candidate" and args.candidate_command == "create" and (args.file or args.delete):
        if any(value is not None for value in (
            args.content, args.target, args.proposal_file, args.proposal_index,
        )):
            raise ValueError("--file/--delete cannot be combined with --content, --target, or --proposal-file")
        if args.type is None or args.hypothesis is None:
            raise ValueError("--file/--delete require --type and --hypothesis")
        config = config_from_environment()
        candidate_type = CandidateType(args.type)
        risk = config.taxonomy.candidate_risk[candidate_type]
        result = create_candidate(
            args.repository,
            ExperimentStore(database),
            experiment_id=args.experiment_id,
            candidate_id=args.candidate_id,
            candidate_type=candidate_type,
            files=_load_candidate_files(args.file),
            deletions=tuple(args.delete),
            hypothesis=args.hypothesis,
            benchmark_version=BenchmarkStore(benchmark_root).version,
            source_sessions=tuple(args.source_session),
            policy_version=config.promotion.policy_version,
            risk=risk,
            worktree_root=args.worktree_root,
            parent_experiment_id=args.parent_experiment,
        )
        _record_risk_decision(database, config, candidate_type, args, result)
        print(json.dumps(_candidate_summary(result, args.worktree_root), indent=2))
    elif args.command == "candidate" and args.candidate_command == "create":
        if args.proposal_file is not None:
            if args.proposal_index is None:
                raise ValueError("--proposal-index is required with --proposal-file")
            if any(value is not None for value in (args.type, args.target, args.hypothesis)):
                raise ValueError(
                    "--proposal-file cannot be combined with --type, --target, or --hypothesis"
                )
            candidate_type, target, hypothesis = _load_dreaming_proposal(
                args.proposal_file, args.proposal_index,
            )
        else:
            if args.proposal_index is not None:
                raise ValueError("--proposal-index requires --proposal-file")
            if any(value is None for value in (args.type, args.target, args.hypothesis)):
                raise ValueError(
                    "provide --proposal-file and --proposal-index or all of "
                    "--type, --target, and --hypothesis"
                )
            candidate_type = CandidateType(args.type)
            target = args.target
            hypothesis = args.hypothesis
        if args.content is None:
            raise ValueError("--content is required unless --file/--delete are used")
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
            candidate_type=candidate_type,
            target=target,
            content=content,
            hypothesis=hypothesis,
            benchmark_version=benchmark_version,
            source_sessions=tuple(args.source_session),
            policy_version=config_from_environment().promotion.policy_version,
            worktree_root=args.worktree_root,
            parent_experiment_id=args.parent_experiment,
        )
        if args.proposal_file is not None:
            ObservabilityLog(database.parent / "observability.sqlite3").record_proposal_conversion(
                result.candidate_id
            )
        print(json.dumps(_candidate_summary(result, args.worktree_root), indent=2))
    elif args.command == "releases" and args.release_command == "events":
        print(json.dumps(ExperimentStore(database).list_release_events(args.release_filter), indent=2))
    elif args.command == "releases":
        print(json.dumps(ExperimentStore(database).list_releases(), indent=2))
    elif args.command == "approve":
        key_path = os.environ.get("HERMES_OPERATOR_PRIVATE_KEY")
        if not key_path:
            raise RuntimeError("HERMES_OPERATOR_PRIVATE_KEY must name the operator's Ed25519 private key")
        print(json.dumps(approve_promotion(
            ExperimentStore(database),
            args.experiment_id,
            operator=args.operator,
            rationale=args.rationale,
            private_key_pem=Path(key_path).read_bytes(),
            policy=PromotionPolicy.from_config(config_from_environment().promotion),
            repository=args.repository,
        ), indent=2))
    elif args.command == "curator":
        if args.curator_command == "status":
            print(json.dumps(_curator_status(database, _hermes_home(args.hermes_home)), indent=2))
            return 0
        adapter = CuratorAdapter(
            repository=args.repository,
            store=ExperimentStore(database),
            benchmark_version=BenchmarkStore(benchmark_root).version,
        )
        if args.curator_command == "convert-pending":
            pending = CuratorConnector(_hermes_home(args.hermes_home)).pending_write(args.pending_id)
            conversion = adapter.convert_pending_write(
                pending, experiment_id=args.experiment_id, candidate_id=args.candidate_id,
            )
        else:
            targeted = args.old_string_file is not None or args.new_string_file is not None
            if targeted and (args.old_string_file is None or args.new_string_file is None):
                raise ValueError("--old-string-file and --new-string-file must be supplied together")
            mutation = CuratorMutation(
                kind=args.kind,
                skill=args.skill,
                rationale=args.rationale,
                content=_read_text_file(args.content, "SKILL.md content") if args.content else None,
                old_string=_read_text_file(args.old_string_file, "old string") if targeted else None,
                new_string=_read_text_file(args.new_string_file, "new string") if targeted else None,
                replace_all=args.replace_all,
                into=args.into,
            )
            conversion = adapter.to_candidate(
                mutation, experiment_id=args.experiment_id, candidate_id=args.candidate_id,
            )
        print(json.dumps(_conversion_payload(conversion), indent=2))
    elif args.command == "report":
        report = build_report(database.parent, benchmark_root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            sys.stdout.write(render_text(report))
    elif args.command == "decisions":
        recorder = DecisionRecorder(database)
        if args.decisions_command == "backfill":
            record = recorder.backfill_ground_truth(
                args.decision_id, ground_truth=args.ground_truth, ground_truth_source=args.source,
            )
            print(json.dumps(record.to_dict(), indent=2))
        elif args.output is None:
            recorder.export_jsonl(sys.stdout, decision_type=args.type, labeled_only=args.labeled_only)
        else:
            with args.output.open("x", encoding="utf-8") as destination:
                count = recorder.export_jsonl(
                    destination, decision_type=args.type, labeled_only=args.labeled_only,
                )
            print(json.dumps({"exported": count, "output": str(args.output)}, indent=2))
    elif args.command == "monitor" and args.monitor_command == "evaluate":
        monitor = ReleaseMonitor(
            database.parent / "monitoring.sqlite3", ExperimentStore(database),
            decisions=DecisionRecorder(database),
        )
        if not args.apply:
            print(json.dumps(monitor.evaluate(args.release).to_dict(), indent=2))
            return 0
        print(json.dumps(_apply_monitoring(args, database, monitor), indent=2))
    elif args.command == "monitor" and args.monitor_command == "collect":
        monitor = ReleaseMonitor(
            database.parent / "monitoring.sqlite3", ExperimentStore(database),
            decisions=DecisionRecorder(database),
        )
        if args.apply:
            _require_role_uid("promoter")
        feed = SessionFeed(
            SessionQueue(database.parent / "sessions.sqlite3"), monitor, switch_margin_seconds=args.switch_margin,
        ).collect(args.release, limit=args.limit)
        if args.apply:
            outcome = {"feed": feed.to_dict(), "result": _apply_monitoring(args, database, monitor)}
        else:
            outcome = {"feed": feed.to_dict(), "assessment": monitor.evaluate(args.release).to_dict()}
        print(json.dumps(outcome, indent=2))
    elif args.command == "monitor" and args.monitor_command == "replay":
        try:
            argv = json.loads(args.runner_command)
        except json.JSONDecodeError as exc:
            raise ValueError("--runner-command must be a JSON array of strings") from exc
        if not isinstance(argv, list):
            raise ValueError("--runner-command must be a JSON array of strings")
        monitor = ReleaseMonitor(database.parent / "monitoring.sqlite3", ExperimentStore(database))
        replayer = ShadowReplayer(
            SessionQueue(database.parent / "sessions.sqlite3"), monitor,
            CommandReplayRunner(argv, timeout=args.timeout),
        )
        if args.source == "shadow-benchmarks":
            report = replayer.replay_benchmarks(
                args.release, BenchmarkStore(benchmark_root).load_public(), limit=args.limit,
            )
        else:
            report = replayer.replay(args.release, limit=args.limit)
        print(json.dumps(report.to_dict(), indent=2))
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
        layout = enforce_isolation("evaluator")
        _require_role_uid("evaluator")
        if layout is not None:
            layout.require_paths({
                "holdout": args.holdout_root,
                "benchmarks_public": benchmark_root,
                "ledger": database.parent,
            })
            if args.repository.resolve() != layout.evaluator_mirror:
                raise PermissionError(
                    f"--repository must be the provisioned evaluator mirror {layout.evaluator_mirror}"
                )
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
        changed_paths = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(args.repository), "diff", "--name-only", "-z",
             stored_candidate["base_commit"], stored_candidate["commit_id"]],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
                 "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.decode("utf-8").split("\0")
        assert_evaluable_candidate_paths(candidate.candidate_type, candidate.target, changed_paths)
        access_store = BenchmarkStore(
            benchmark_root,
            holdout_root=holdout_root,
        )
        benchmarks = access_store.validate()
        # Fail fast, before suite selection is recorded, any task runs or the
        # lineage's sealed holdout is consumed: promotion rejects evaluations
        # under any benchmark version other than the experiment's binding.
        access_store.require_bound_version(
            stored_candidate["experiment_id"],
            store.show_experiment(stored_candidate["experiment_id"])["benchmark_version"],
        )
        suite = args.suite.upper()
        config = config_from_environment()
        impact = impact_scope_for_candidate(
            candidate.candidate_type, candidate.target, config.taxonomy.categories,
        )
        selected, holdout, applied_impact = select_evaluation_suite(benchmarks, suite, impact=impact)
        suite_ids = [item.id for item in selected]
        holdout_ids = [item.id for item in holdout]
        selection_warnings = [
            warning for warning in (smoke_size_warning(suite, len(selected)),) if warning is not None
        ]
        for warning in selection_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        selection_manifest = suite_manifest(
            selected, suite=suite, benchmark_version=access_store.version, impact=applied_impact,
        )
        holdout_manifest = (
            suite_manifest(
                holdout, suite=Suite.SEALED_HOLDOUT, benchmark_version=access_store.version,
                include_task_details=False,
            )
            if holdout else None
        )
        if not args.judge:
            # Refuse before recording anything: an evaluation that never runs must not
            # leave suite_selection/needs_qwen decision rows behind.
            judged = [item.id for item in selected if judge_route(item) == "qwen_judge"]
            if judged:
                raise ValueError(
                    f"benchmark {judged[0]!r} relies on subjective verification and requires a configured "
                    "judge (--judge)"
                )
            if any(judge_route(item) == "qwen_judge" for item in holdout):
                raise ValueError("a sealed holdout task relies on subjective verification and requires a "
                                 "configured judge (--judge)")
        decision_recorder = DecisionRecorder(database)
        decision_recorder.record(**suite_selection_decision_fields(
            suite=suite, candidate_type=candidate.candidate_type, requested_impact=impact,
            applied_impact=applied_impact, selected=selected, holdout=holdout,
            manifest_digest=selection_manifest["digest"],
        ))
        for item in (*selected, *holdout):
            decision_recorder.record(**needs_qwen_decision_fields(item, suite=suite, judge_configured=args.judge))
        seeds = None
        if args.seeds is not None:
            try:
                seeds = [int(value) for value in args.seeds.split(",")]
            except ValueError as exc:
                raise ValueError("--seeds must be comma-separated integers") from exc
            if not seeds or any(value < 0 for value in seeds):
                raise ValueError("--seeds must contain non-negative integers")
        policy = PromotionPolicy.from_config(config.promotion)
        signing_key = Path(private_key_path).read_bytes()
        spend_holdout = None
        if holdout_ids:
            # Spec §16: only finalists (signed, passing SMOKE + CORE for this exact
            # candidate/benchmark/policy identity) are granted the sealed holdout.
            require_holdout_finalist(
                store, args.candidate_id, candidate_hash=candidate.content_hash,
                benchmark_version=access_store.version, policy=policy,
                public_key=evaluator_public_key(signing_key),
            )
            holdout_rows = [item for item in benchmarks if item.split is Split.SEALED_HOLDOUT]
            lineage_id = store.show_experiment(stored_candidate["experiment_id"])["lineage_id"]
            registry = SpentHoldoutRegistry(str(database))
            if holdout_generation_spent(registry, holdout_rows, lineage_id):
                raise AccessDenied("holdout feedback for this candidate/evolution lineage was already consumed")
            # Binding spend happens right before the first holdout task executes.
            spend_holdout = lambda: consume_holdout_generation(registry, holdout_rows, lineage_id)  # noqa: E731
        grant = access_store.access.evaluator_holdout_grant(Role.EVALUATOR) if holdout_ids else None
        artifact_store = ArtifactStore(database.parent / "artifacts")
        args.benchmark_root = public_root
        args.holdout_root = holdout_root
        runner = build_runner(args, artifact_store)
        judge_model = OpenAICompatibleQwen(QwenConfig.from_env()) if args.judge else None
        evaluator = PairedEvaluator(
            access_store,
            HoldoutSpendingRunner(runner, spend_holdout) if spend_holdout is not None else runner,
            policy,
            experiment_store=store,
            evaluator_signing_key=signing_key,
            artifact_store=artifact_store,
            run_policy=RunPolicy.from_config(config.scoring),
            judge=PairwiseJudge(judge_model) if judge_model is not None else None,
        )
        with (
            record_qwen_usage(database.parent, judge_model, command="evaluate-judge")
            if judge_model is not None else nullcontext()
        ):
            report = evaluator.evaluate(
                candidate,
                baseline_commit=stored_candidate["base_commit"],
                benchmark_ids=suite_ids + holdout_ids,
                seeds=seeds,
                suite=suite,
                benchmark_version=access_store.version,
                holdout_grant=grant,
                evaluation_id=f"eval-{suite.lower()}-{args.candidate_id}-{__import__('uuid').uuid4().hex}",
                suite_selection={
                    "suite_manifest_digest": selection_manifest["digest"],
                    "holdout_manifest_digest": holdout_manifest["digest"] if holdout_manifest else None,
                    "impact": applied_impact.to_dict(),
                    "task_ids": suite_ids,
                    "warnings": selection_warnings,
                },
            )
        print(json.dumps(report_payload(report), indent=2))
    elif args.command in {"promote", "rollback"}:
        layout = enforce_isolation("promoter")
        _require_role_uid("promoter")
        if layout is not None:
            layout.require_paths({
                "production_repo": args.repository,
                "releases": args.release_root,
                "deployment_link": args.deployment_link,
                "evaluator_public_key": args.evaluator_public_key,
                "ledger": database.parent,
            })
        manifest_key = os.environ.get("HERMES_IMPROVE_MANIFEST_KEY")
        if manifest_key is None:
            raise RuntimeError("HERMES_IMPROVE_MANIFEST_KEY is required")
        config = config_from_environment()
        if args.command == "promote" and args.candidate_bundle is not None:
            if layout is not None and args.candidate_bundle.resolve().parent != layout.path("handoff"):
                raise PermissionError("--candidate-bundle must be inside the provisioned handoff directory")
            pending = ExperimentStore(database).get_candidate(
                ExperimentStore(database).resolve_promotion_candidate(args.candidate_id)
            )
            import_bundle_ref(
                args.repository, args.candidate_bundle, pending["experiment_id"], pending["commit_id"],
            )
        promoter = GitPromoter(
            ExperimentStore(database),
            args.repository,
            args.release_root,
            args.deployment_link,
            PromotionPolicy.from_config(config.promotion),
            args.evaluator_public_key.read_bytes(),
            operator_public_keys=[
                path.read_bytes() for path in getattr(args, "operator_public_key", [])
            ],
            config_digest=config.digest,
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


# Refusals an operator can act on: wrong input, missing identity or
# configuration, failed integrity or policy checks, blocked rollbacks. They end
# with a one-line error; anything else (a bug) keeps its traceback.
EXPECTED_ERRORS = (
    ImprovementError, ValueError, PermissionError, KeyError, FileNotFoundError, FileExistsError, RuntimeError,
)
# RuntimeError subclasses that signal a bug rather than a refusal.
_UNEXPECTED_RUNTIME_ERRORS = (RecursionError, NotImplementedError)


def main(argv: list[str] | None = None) -> int:
    """Console entry point: :func:`run` with one-line errors for expected refusals.

    Expected refusals print `error: MESSAGE` to stderr and return 1. Set
    `HERMES_IMPROVE_DEBUG=1` to get the full traceback instead.
    """
    try:
        return run(argv)
    except EXPECTED_ERRORS as exc:
        if isinstance(exc, _UNEXPECTED_RUNTIME_ERRORS) or os.environ.get("HERMES_IMPROVE_DEBUG", "") not in {"", "0"}:
            raise
        message = f"not found: {exc}" if type(exc) is KeyError else str(exc) or type(exc).__name__
        print(f"error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
