from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from .ingestion import contains_secret_material
from .models import Candidate, CandidateType
from .store import ExperimentStore


_SAFE_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MAX_SKILL_BYTES = 256 * 1024
_SKILL_CANDIDATE_TYPES = {
    CandidateType.SKILL_CREATE,
    CandidateType.SKILL_PATCH,
}
_PROTECTED_COMPONENTS = {
    "evaluator", "verifiers", "benchmarks", "holdout", "security",
    "promotion", "ci", ".github", "tests",
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
) -> Candidate:
    if not _SAFE_EXPERIMENT.fullmatch(experiment_id):
        raise ValueError("experiment ID must be a safe 1-80 character identifier")
    if candidate_type not in _SKILL_CANDIDATE_TYPES:
        raise ValueError("CLI candidate creation currently supports skill create/patch only")
    target_path = PurePosixPath(target)
    if (
        target_path.is_absolute()
        or len(target_path.parts) < 3
        or target_path.parts[:2] != ("brain", "skills")
        or any(part in {"", ".", ".."} or part.lower() in _PROTECTED_COMPONENTS for part in target_path.parts)
    ):
        raise ValueError("candidate target must be an unprotected path under brain/skills/")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("skill candidate content must be non-empty text")
    if len(content.encode("utf-8")) > _MAX_SKILL_BYTES:
        raise ValueError("skill candidate content exceeds the 256 KiB limit")
    if contains_secret_material({"content": content}):
        raise ValueError("skill candidate content contains credential-like material")
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 4000:
        raise ValueError("candidate hypothesis must be non-empty text up to 4000 characters")
    if not isinstance(candidate_id, str) or not _SAFE_EXPERIMENT.fullmatch(candidate_id):
        raise ValueError("candidate ID must be a safe 1-80 character identifier")

    repo = Path(repository).resolve()
    base_commit = _git(repo, "rev-parse", "--verify", "refs/heads/main").stdout.decode().strip()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", base_commit):
        raise ValueError("main must resolve to a full Git commit ID")
    branch = f"candidate/{experiment_id}"
    branch_ref = f"refs/heads/{branch}"
    if _git(repo, "show-ref", "--verify", "--quiet", branch_ref, check=False).returncode == 0:
        raise ValueError(f"candidate branch already exists: {branch}")

    worktree: Path | None = None
    branch_created = False
    try:
        worktree = Path(tempfile.mkdtemp(prefix="hermes-candidate-"))
        worktree.rmdir()
        _git(repo, "worktree", "add", "-b", branch, str(worktree), base_commit)
        branch_created = True
        skill_file = worktree.joinpath(*target_path.parts) / "SKILL.md"
        _validate_no_symlink_parents(worktree, skill_file)
        if candidate_type is CandidateType.SKILL_CREATE and skill_file.exists():
            raise ValueError(f"skill already exists: {target}")
        if candidate_type is CandidateType.SKILL_PATCH and not skill_file.is_file():
            raise ValueError(f"skill to patch does not exist: {target}")
        skill_file.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        skill_file.write_text(content.rstrip() + "\n", encoding="utf-8")
        _git(worktree, "add", "--", skill_file.relative_to(worktree).as_posix())
        if _git(worktree, "diff", "--cached", "--quiet", check=False).returncode == 0:
            raise ValueError("candidate content does not change the existing skill")
        environment = _git_environment() | {
            "GIT_AUTHOR_NAME": "Hermes Improvement Candidate",
            "GIT_AUTHOR_EMAIL": "hermes-improve@localhost",
            "GIT_COMMITTER_NAME": "Hermes Improvement Candidate",
            "GIT_COMMITTER_EMAIL": "hermes-improve@localhost",
        }
        _git(worktree, "commit", "-m", f"Candidate {experiment_id}: {hypothesis[:120]}", env=environment)
        commit = _git(worktree, "rev-parse", "HEAD").stdout.decode().strip()
        diff = _git(
            repo, "diff", "--no-ext-diff", "--no-textconv", "--binary", base_commit, commit
        ).stdout.decode()
        if not diff.strip():
            raise ValueError("candidate commit does not contain a diff")
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
            lineage_id=experiment_id,
            hypothesis=hypothesis,
            risk="LOW",
            benchmark_version=benchmark_version,
            policy_version="1",
            source_sessions=source_sessions,
        )
        return candidate
    except Exception:
        if branch_created:
            if worktree is not None and worktree.exists():
                _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
            _git(repo, "branch", "-D", branch, check=False)
        raise
    finally:
        if worktree is not None and worktree.exists():
            _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
            if worktree.exists():
                shutil.rmtree(worktree)


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
