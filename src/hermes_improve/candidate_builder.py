"""Materialize operator-supplied brain changes as immutable Git candidates.

Every candidate lives on its own branch ``candidate/<experiment-id>`` created
from ``main`` in a dedicated Git worktree. The builder records the candidate
commit, the parent baseline, the SHA-256 of the binary diff and the candidate
type together with a new experiment in one SQLite transaction. A branch that
already exists is never reused, so any change to an evaluated candidate needs
a new experiment ID and therefore a new commit, hash, and evaluation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

import yaml

from .config import DEFAULT_CANDIDATE_RISK, RISK_LEVELS, strict_yaml_load
from .ingestion import contains_secret_material
from .models import Candidate, CandidateType
from .skill_paths import validate_skill_target
from .store import ExperimentStore
from .target_paths import (
    TARGET_RULES,
    assert_changed_paths_match_target,
    removed_skill_units,
    target_for_units,
    units_for_paths,
)


_SAFE_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MAX_SKILL_BYTES = 256 * 1024
_MAX_FILE_BYTES = 256 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024
_MAX_FILES = 32
_MAX_SOURCE_SESSIONS = 100
_SKILL_CANDIDATE_TYPES = {
    CandidateType.SKILL_CREATE,
    CandidateType.SKILL_PATCH,
}
_SKILL_TYPES = {
    CandidateType.SKILL_CREATE, CandidateType.SKILL_PATCH,
    CandidateType.SKILL_MERGE, CandidateType.SKILL_SPLIT,
}


def create_skill_candidate(
    repository: Path,
    store: ExperimentStore,
    *,
    experiment_id: str,
    candidate_id: str,
    candidate_type: CandidateType,
    target: str,
    content: str,
    hypothesis: str,
    benchmark_version: str,
    source_sessions: tuple[str, ...] = (),
    policy_version: str = "1",
    worktree_root: Path | None = None,
    parent_experiment_id: str | None = None,
) -> Candidate:
    """Create a ``SKILL_CREATE``/``SKILL_PATCH`` candidate from one SKILL.md text."""
    if not _SAFE_EXPERIMENT.fullmatch(experiment_id):
        raise ValueError("experiment ID must be a safe 1-80 character identifier")
    if candidate_type not in _SKILL_CANDIDATE_TYPES:
        raise ValueError("CLI candidate creation currently supports skill create/patch only")
    target_path = validate_skill_target(target)
    if len(target_path.parts) != 3:
        raise ValueError("skill target must be a flat brain/skills/<slug> directory")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("skill candidate content must be non-empty text")
    if len(content.encode("utf-8")) > _MAX_SKILL_BYTES:
        raise ValueError("skill candidate content exceeds the 256 KiB limit")
    if contains_secret_material({"content": content}):
        raise ValueError("skill candidate content contains credential-like material")
    return create_candidate(
        repository,
        store,
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        files={f"{target}/SKILL.md": content},
        hypothesis=hypothesis,
        benchmark_version=benchmark_version,
        source_sessions=source_sessions,
        policy_version=policy_version,
        worktree_root=worktree_root,
        parent_experiment_id=parent_experiment_id,
    )


def create_candidate(
    repository: Path,
    store: ExperimentStore,
    *,
    experiment_id: str,
    candidate_id: str,
    candidate_type: CandidateType,
    files: Mapping[str, str],
    hypothesis: str,
    benchmark_version: str,
    deletions: Iterable[str] = (),
    source_sessions: tuple[str, ...] = (),
    policy_version: str = "1",
    risk: str | None = None,
    worktree_root: Path | None = None,
    parent_experiment_id: str | None = None,
) -> Candidate:
    """Create an immutable candidate of any type from operator-supplied files.

    ``files`` maps repository paths under ``brain/`` to their full new text;
    ``deletions`` lists tracked paths to remove (merge/split and clean-ups).
    With ``worktree_root`` the worktree ``<root>/<experiment-id>`` is kept for
    evaluation; otherwise a temporary worktree is removed after the commit.
    """
    if not isinstance(experiment_id, str) or not _SAFE_EXPERIMENT.fullmatch(experiment_id):
        raise ValueError("experiment ID must be a safe 1-80 character identifier")
    if not isinstance(candidate_id, str) or not _SAFE_EXPERIMENT.fullmatch(candidate_id):
        raise ValueError("candidate ID must be a safe 1-80 character identifier")
    if not isinstance(candidate_type, CandidateType):
        raise ValueError("candidate type must be a CandidateType")
    if not isinstance(files, Mapping):
        raise ValueError("candidate files must map brain paths to text content")
    written = {path: _validate_content(path, content) for path, content in files.items()}
    removed = tuple(deletions)
    if len(written) + len(removed) > _MAX_FILES:
        raise ValueError(f"candidate may change at most {_MAX_FILES} files")
    if len(set(removed)) != len(removed) or set(removed) & set(written):
        raise ValueError("candidate deletions must be unique and disjoint from written files")
    if sum(len(content.encode("utf-8")) for content in written.values()) > _MAX_TOTAL_BYTES:
        raise ValueError("candidate files exceed the 1 MiB total limit")
    units = units_for_paths(candidate_type, [*written, *removed])
    target = target_for_units(units)
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 4000:
        raise ValueError("candidate hypothesis must be non-empty text up to 4000 characters")
    if contains_secret_material({"hypothesis": hypothesis}):
        raise ValueError("candidate hypothesis contains credential-like material")
    source_sessions = tuple(source_sessions)
    if len(source_sessions) > _MAX_SOURCE_SESSIONS or any(
        not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256
        for session_id in source_sessions
    ):
        raise ValueError("source session IDs must be non-empty strings up to 256 characters (maximum 100)")
    if contains_secret_material({"source_sessions": source_sessions}):
        raise ValueError("source session IDs contain credential-like material")
    risk = DEFAULT_CANDIDATE_RISK[candidate_type] if risk is None else risk
    if risk not in RISK_LEVELS:
        raise ValueError(f"candidate risk must be one of {', '.join(RISK_LEVELS)}")
    lineage_id = experiment_id
    if parent_experiment_id is not None:
        try:
            parent = store.show_experiment(parent_experiment_id)
        except KeyError as exc:
            raise ValueError(f"unknown parent experiment: {parent_experiment_id}") from exc
        lineage_id = str(parent["lineage_id"])

    repo = Path(repository).resolve()
    base_commit = _git(repo, "rev-parse", "--verify", "refs/heads/main").stdout.decode().strip()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", base_commit):
        raise ValueError("main must resolve to a full Git commit ID")
    tree = set(
        item for item in _git(repo, "ls-tree", "-r", "-z", "--name-only", base_commit)
        .stdout.decode("utf-8").split("\0") if item
    )
    _check_type_semantics(candidate_type, units, written, removed, tree)
    _check_case_collisions(written, tree)
    branch = f"candidate/{experiment_id}"
    branch_ref = f"refs/heads/{branch}"
    if _git(repo, "show-ref", "--verify", "--quiet", branch_ref, check=False).returncode == 0:
        raise ValueError(f"candidate branch already exists: {branch}")
    persistent = worktree_root is not None
    if persistent:
        root = Path(worktree_root).resolve()
        if root == repo or repo in root.parents:
            raise ValueError("candidate worktree root must be outside the repository checkout")
        root.mkdir(parents=True, exist_ok=True, mode=0o755)
        if (root / experiment_id).exists():
            raise ValueError(f"candidate worktree already exists: {root / experiment_id}")

    worktree: Path | None = None
    branch_created = False
    succeeded = False
    try:
        if persistent:
            worktree = Path(worktree_root).resolve() / experiment_id
        else:
            worktree = Path(tempfile.mkdtemp(prefix="hermes-candidate-"))
            worktree.rmdir()
        _git(repo, "worktree", "add", "-b", branch, str(worktree), base_commit)
        branch_created = True
        for path, content in written.items():
            destination = worktree.joinpath(*PurePosixPath(path).parts)
            _validate_no_symlink_parents(worktree, destination)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            destination.write_text(content, encoding="utf-8")
        for path in removed:
            _validate_no_symlink_parents(worktree, worktree.joinpath(*PurePosixPath(path).parts))
            _git(worktree, "rm", "-q", "--", path)
        if written:
            _git(worktree, "add", "--", *written)
        if _git(worktree, "diff", "--cached", "--quiet", check=False).returncode == 0:
            raise ValueError("candidate content does not change the existing brain files")
        environment = _git_environment() | {
            "GIT_AUTHOR_NAME": "Hermes Improvement Candidate",
            "GIT_AUTHOR_EMAIL": "hermes-improve@localhost",
            "GIT_COMMITTER_NAME": "Hermes Improvement Candidate",
            "GIT_COMMITTER_EMAIL": "hermes-improve@localhost",
        }
        subject = " ".join(hypothesis.split())[:120]
        _git(
            worktree, "commit", "-m",
            f"Candidate {experiment_id} ({candidate_type.value}): {subject}",
            env=environment,
        )
        commit = _git(worktree, "rev-parse", "HEAD").stdout.decode().strip()
        diff = _git(
            repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", base_commit, commit
        ).stdout.decode()
        if not diff.strip():
            raise ValueError("candidate commit does not contain a diff")
        changed = [
            item for item in _git(repo, "diff", "--name-only", "-z", base_commit, commit)
            .stdout.decode("utf-8").split("\0") if item
        ]
        assert_changed_paths_match_target(candidate_type, target, changed)
        candidate = Candidate(
            candidate_id=candidate_id,
            experiment_id=experiment_id,
            candidate_type=candidate_type,
            target=target,
            base_commit=base_commit,
            commit=commit,
            diff=diff,
        )
        store.create_experiment_with_candidate(
            candidate,
            lineage_id=lineage_id,
            hypothesis=hypothesis,
            risk=risk,
            benchmark_version=benchmark_version,
            policy_version=policy_version,
            source_sessions=source_sessions,
            parent_id=parent_experiment_id,
        )
        succeeded = True
        return candidate
    except Exception:
        if branch_created:
            if worktree is not None and worktree.exists():
                _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
            _git(repo, "branch", "-D", branch, check=False)
        raise
    finally:
        if worktree is not None and worktree.exists() and not (persistent and succeeded):
            _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
            if worktree.exists():
                shutil.rmtree(worktree)


_GIT_OBJECT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_REGULAR_BLOB_MODES = frozenset({"100644", "100755"})


def production_skill_files(repository: Path, skill_unit: str) -> tuple[str, ...]:
    """Every tracked file of ``skill_unit`` (``brain/skills/<name>``) on ``main``.

    The skill must exist with an exact-case ``SKILL.md`` and consist only of
    regular files: a symlink, submodule or anything else makes removal refuse
    rather than delete something whose target lies outside the unit.
    """
    validate_skill_target(skill_unit)
    if len(PurePosixPath(skill_unit).parts) != 3:
        raise ValueError("skill unit must be a flat brain/skills/<name> directory")
    listing = _git(
        Path(repository).resolve(), "ls-tree", "-r", "-z", "--full-tree", "refs/heads/main", "--", "brain/skills",
    ).stdout.decode("utf-8")
    files = []
    for entry in listing.split("\0"):
        if not entry:
            continue
        header, _, path = entry.partition("\t")
        if path != skill_unit and not path.startswith(f"{skill_unit}/"):
            continue
        mode, kind = header.split(" ")[:2]
        if kind != "blob" or mode not in _REGULAR_BLOB_MODES:
            raise ValueError(f"skill contains a non-regular tracked entry and cannot be removed: {path}")
        files.append(path)
    if f"{skill_unit}/SKILL.md" not in files:
        raise ValueError(f"skill does not exist on main: {skill_unit}")
    return tuple(sorted(files))


def removed_skills_in_candidate(repository: Path, base_commit: str, commit: str) -> tuple[str, ...]:
    """Skill units whose ``SKILL.md`` the candidate commit deletes (rename detection off)."""
    if not _GIT_OBJECT.fullmatch(base_commit) or not _GIT_OBJECT.fullmatch(commit):
        raise ValueError("candidate base and commit must be full Git object IDs")
    deleted = _git(
        Path(repository).resolve(), "diff", "--no-renames", "--diff-filter=D", "--name-only", "-z",
        base_commit, commit,
    ).stdout.decode("utf-8").split("\0")
    return removed_skill_units(path for path in deleted if path)


def _validate_content(path: str, content: str) -> str:
    if not isinstance(path, str):
        raise ValueError("candidate file paths must be strings")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"candidate content must be non-empty text: {path}")
    if "\x00" in content:
        raise ValueError(f"candidate content must not contain NUL bytes: {path}")
    if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError(f"candidate content exceeds the 256 KiB limit: {path}")
    if contains_secret_material({"content": content}):
        raise ValueError(f"candidate content contains credential-like material: {path}")
    normalized = content.rstrip() + "\n"
    suffix = PurePosixPath(path).suffix
    if suffix in {".yaml", ".yml"}:
        try:
            document = strict_yaml_load(normalized)
        except yaml.YAMLError as exc:
            raise ValueError(f"candidate file is not valid YAML: {path}: {exc}") from exc
        if not isinstance(document, (dict, list)):
            raise ValueError(f"candidate YAML must contain a mapping or list: {path}")
    elif suffix == ".json":
        try:
            json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate file is not valid JSON: {path}: {exc}") from exc
    elif suffix == ".py":
        try:
            # Syntax check only; the code object is discarded and never executed.
            compile(normalized, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"candidate Python file does not compile: {path}: {exc}") from exc
    return normalized


def _unit_files(unit: str, paths: Iterable[str]) -> set[str]:
    return {path for path in paths if path == unit or path.startswith(f"{unit}/")}


def _check_type_semantics(
    candidate_type: CandidateType,
    units: tuple[str, ...],
    written: Mapping[str, str],
    removed: tuple[str, ...],
    tree: set[str],
) -> None:
    for path in removed:
        if path not in tree:
            raise ValueError(f"cannot delete a path that is not tracked on main: {path}")
    directory = TARGET_RULES[candidate_type].unit_kind == "directory"
    existing = {unit for unit in units if _unit_files(unit, tree)} if directory else set()
    if candidate_type in {CandidateType.SKILL_CREATE, CandidateType.TOOL_CREATE}:
        label = "skill" if candidate_type is CandidateType.SKILL_CREATE else "tool"
        for unit in units:
            if unit in existing:
                raise ValueError(f"{label} already exists: {unit}")
    if candidate_type in {CandidateType.SKILL_PATCH, CandidateType.TOOL_PATCH}:
        label = "skill" if candidate_type is CandidateType.SKILL_PATCH else "tool"
        for unit in units:
            if unit not in existing or (
                candidate_type is CandidateType.SKILL_PATCH and f"{unit}/SKILL.md" not in tree
            ):
                raise ValueError(f"{label} to patch does not exist: {unit}")
    if candidate_type not in _SKILL_TYPES:
        return
    fully_removed = {
        unit for unit in units
        if unit in existing
        and _unit_files(unit, tree) <= set(removed)
        and not _unit_files(unit, written)
    }
    for unit in set(units) - fully_removed:
        entry = f"{unit}/SKILL.md"
        if entry not in written and (entry not in tree or entry in removed):
            raise ValueError(f"skill must keep a SKILL.md entry file: {unit}")
    if candidate_type is CandidateType.SKILL_MERGE:
        if not fully_removed or not set(units) - fully_removed:
            raise ValueError(
                "SKILL_MERGE must remove at least one existing skill and keep or create at least one"
            )
    if candidate_type is CandidateType.SKILL_SPLIT:
        new_units = set(units) - existing
        if not existing or not new_units:
            raise ValueError("SKILL_SPLIT must touch an existing skill and create at least one new skill")


def _check_case_collisions(written: Mapping[str, str], tree: set[str]) -> None:
    known: dict[str, str] = {}
    for path in tree:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            known.setdefault(prefix.casefold(), prefix)
    for path in written:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            existing = known.get(prefix.casefold())
            if existing is not None and existing != prefix:
                raise ValueError(
                    f"candidate path collides case-insensitively with tracked path: {prefix} / {existing}"
                )


def _validate_no_symlink_parents(root: Path, target: Path) -> None:
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("candidate target may not traverse symbolic links")


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env or _git_environment(),
    )
    if check and result.returncode:
        raise RuntimeError(f"Git {arguments[0]} failed: {result.stderr.decode(errors='replace')}")
    return result


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
