"""Cross-role candidate handoff without running Git inside another user's repository.

Git executes configuration and hooks from the repository it operates on, so
the evaluator and promoter never run Git in the agent-owned candidate
repository. The agent exports a Git bundle plus a metadata record into the
handoff directory; the evaluator imports it into its own mirror and records
the candidate in the evaluator/promoter ledger after recomputing every hash,
and the promoter fetches the evaluated commit from the bundle into the
production repository. A bundle is plain data parsed by the receiver's Git.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .errors import IntegrityError
from .models import Candidate, CandidateType
from .store import ExperimentStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_BUNDLE_BYTES = 100 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
_CANDIDATE_FIELDS = {
    "id", "experiment_id", "candidate_type", "target", "base_commit",
    "commit_id", "diff_hash", "content_hash", "created_at",
}
_EXPERIMENT_FIELDS = {
    "lineage_id", "hypothesis", "risk", "benchmark_version", "policy_version", "source_sessions",
}


def _git(repository: Path, *args: str) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _branch(experiment_id: str) -> str:
    if not isinstance(experiment_id, str) or not _SAFE_ID.fullmatch(experiment_id):
        raise IntegrityError("experiment ID is not a safe Git branch component")
    return f"refs/heads/candidate/{experiment_id}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_production_baseline(candidate_repository: Path, production_repository: Path) -> str:
    """Agent side: mirror production ``main`` into the agent-owned candidate repository."""
    production = Path(production_repository).resolve()
    # Git serves a local fetch by running upload-pack inside the source
    # repository and strips GIT_CONFIG_* from its environment, so the
    # safe.directory exception for the (more privileged) promoter-owned
    # production repository is scoped to that upload-pack command only.
    upload_pack = f"git -c safe.directory={shlex.quote(str(production))} upload-pack"
    _git(
        Path(candidate_repository), "fetch", "--no-tags", "--no-write-fetch-head",
        f"--upload-pack={upload_pack}", str(production), "+refs/heads/main:refs/heads/main",
    )
    return _git(Path(candidate_repository), "rev-parse", "--verify", "refs/heads/main").decode().strip()


def export_candidate(
    store: ExperimentStore, repository: Path, candidate_id: str, out_dir: Path,
) -> dict[str, Any]:
    """Agent side: write ``<candidate>.bundle`` and ``<candidate>.candidate.json``."""
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError("candidate ID must be a safe identifier")
    candidate = store.get_candidate(candidate_id)
    experiment = store.show_experiment(candidate["experiment_id"])
    branch = _branch(candidate["experiment_id"])
    tip = _git(Path(repository), "rev-parse", "--verify", branch).decode().strip()
    if tip != candidate["commit_id"]:
        raise IntegrityError("candidate branch no longer points to the recorded candidate commit")
    out_dir = Path(out_dir)
    bundle_path = out_dir / f"{candidate_id}.bundle"
    metadata_path = out_dir / f"{candidate_id}.candidate.json"
    if bundle_path.exists() or metadata_path.exists():
        raise ValueError(f"candidate {candidate_id} was already exported")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{candidate_id}-", suffix=".bundle", dir=out_dir)
    os.close(descriptor)
    os.unlink(temporary)
    try:
        _git(Path(repository), "bundle", "create", temporary, branch)
        if os.path.getsize(temporary) > MAX_BUNDLE_BYTES:
            raise IntegrityError("candidate bundle exceeds the 100 MiB limit")
        os.chmod(temporary, 0o444)
        os.link(temporary, bundle_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata = {
        "schema_version": 1,
        "bundle_sha256": _file_sha256(bundle_path),
        "candidate": {name: candidate[name] for name in sorted(_CANDIDATE_FIELDS)},
        "experiment": {name: experiment[name] for name in sorted(_EXPERIMENT_FIELDS)},
    }
    descriptor = os.open(metadata_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(metadata, output, sort_keys=True, indent=2)
        output.write("\n")
    return {"bundle": str(bundle_path), "metadata": str(metadata_path), **metadata}


def import_bundle_ref(
    repository: Path, bundle: Path, experiment_id: str, expected_commit: str,
) -> str:
    """Fetch exactly ``candidate/<experiment>`` at ``expected_commit`` from a bundle."""
    branch = _branch(experiment_id)
    if not _GIT_OBJECT.fullmatch(expected_commit):
        raise IntegrityError("expected candidate commit is not a full Git object ID")
    bundle = Path(bundle)
    if not bundle.is_file() or bundle.is_symlink():
        raise IntegrityError(f"candidate bundle {bundle} is not a regular file")
    if bundle.stat().st_size > MAX_BUNDLE_BYTES:
        raise IntegrityError("candidate bundle exceeds the 100 MiB limit")
    repository = Path(repository)
    _git(repository, "bundle", "verify", "--quiet", str(bundle))
    heads = [line.split() for line in _git(repository, "bundle", "list-heads", str(bundle)).decode().splitlines()]
    if heads != [[expected_commit, branch]]:
        raise IntegrityError("candidate bundle must contain exactly the evaluated candidate branch")
    existing = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "--quiet", branch],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env={"PATH": os.environ.get("PATH", ""), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        check=False,
    )
    current = existing.stdout.decode().strip()
    if existing.returncode == 0 and current != expected_commit:
        raise IntegrityError("receiving repository already has a different commit for this candidate branch")
    if existing.returncode != 0:
        _git(repository, "fetch", "--no-tags", "--no-write-fetch-head", str(bundle), f"{branch}:{branch}")
    if _git(repository, "rev-parse", "--verify", branch).decode().strip() != expected_commit:
        raise IntegrityError("imported candidate branch does not point to the expected commit")
    return branch


def _load_metadata(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as source:
        raw = source.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise IntegrityError("candidate metadata exceeds the 1 MiB limit")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("candidate metadata must be UTF-8 JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "bundle_sha256", "candidate", "experiment"}
        or document["schema_version"] != 1
        or not isinstance(document["candidate"], dict)
        or not isinstance(document["experiment"], dict)
        or set(document["candidate"]) != _CANDIDATE_FIELDS
        or set(document["experiment"]) != _EXPERIMENT_FIELDS
    ):
        raise IntegrityError("candidate metadata has missing or unknown fields")
    candidate = document["candidate"]
    experiment = document["experiment"]
    strings = [*candidate.values(), *(experiment[name] for name in _EXPERIMENT_FIELDS - {"source_sessions"})]
    if not all(isinstance(value, str) and value.strip() and len(value) <= 4096 for value in strings):
        raise IntegrityError("candidate metadata values must be non-empty strings up to 4096 characters")
    sessions = experiment["source_sessions"]
    if not isinstance(sessions, list) or len(sessions) > 100 or not all(
        isinstance(item, str) and 0 < len(item) <= 256 for item in sessions
    ):
        raise IntegrityError("candidate source sessions are invalid")
    if not _SHA256.fullmatch(str(document["bundle_sha256"])):
        raise IntegrityError("candidate bundle digest is invalid")
    return document


def import_candidate(
    ledger: ExperimentStore, handoff_dir: Path, candidate_id: str, mirror: Path,
) -> Candidate:
    """Evaluator side: verify an exported candidate and record it in the ledger."""
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError("candidate ID must be a safe identifier")
    handoff_dir = Path(handoff_dir)
    document = _load_metadata(handoff_dir / f"{candidate_id}.candidate.json")
    meta = document["candidate"]
    experiment = document["experiment"]
    if meta["id"] != candidate_id:
        raise IntegrityError("candidate metadata names a different candidate")
    bundle = handoff_dir / f"{candidate_id}.bundle"
    if _file_sha256(bundle) != document["bundle_sha256"]:
        raise IntegrityError("candidate bundle digest differs from its metadata")
    base, commit = meta["base_commit"], meta["commit_id"]
    if not _GIT_OBJECT.fullmatch(base) or not _GIT_OBJECT.fullmatch(commit):
        raise IntegrityError("candidate commits must be full Git object IDs")
    import_bundle_ref(mirror, bundle, meta["experiment_id"], commit)
    _git(Path(mirror), "merge-base", "--is-ancestor", base, commit)
    diff = _git(Path(mirror), "diff", "--no-ext-diff", "--no-textconv", "--binary", base, commit)
    if hashlib.sha256(diff).hexdigest() != meta["diff_hash"]:
        raise IntegrityError("recomputed candidate diff differs from the exported diff hash")
    try:
        candidate_type = CandidateType(meta["candidate_type"])
    except ValueError as exc:
        raise IntegrityError("candidate metadata has an unknown candidate type") from exc
    candidate = Candidate(
        candidate_id, meta["experiment_id"], candidate_type, meta["target"],
        base, commit, diff.decode("utf-8"), meta["created_at"],
    )
    if candidate.content_hash != meta["content_hash"]:
        raise IntegrityError("recomputed candidate content hash differs from the exported record")
    ledger.create_experiment_with_candidate(
        candidate,
        lineage_id=experiment["lineage_id"],
        hypothesis=experiment["hypothesis"],
        risk=experiment["risk"],
        benchmark_version=experiment["benchmark_version"],
        policy_version=experiment["policy_version"],
        source_sessions=tuple(experiment["source_sessions"]),
    )
    return candidate
