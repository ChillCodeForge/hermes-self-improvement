"""Deliverable files of subjective benchmark criteria (lab finding F24).

The pairwise judge used to see only the agent's final chat text. A task whose
deliverable is a file (the agent writes ``incident.md`` and replies "done")
was therefore unjudgeable. A subjective criterion can declare the files it
grades::

    {"criterion": "Starts with an INCIDENT headline.", "files": ["incident.md"]}

After each run the runner captures exactly those files from the run's result
workspace with :func:`collect_deliverables` and reports them in
``RunObservation.deliverables``; the evaluator hands them to the judge next to
the final answer. Capture is bounded and fail-closed: paths are validated
relative POSIX paths, symlinks and anything resolving outside the workspace
are never read, files above :data:`MAX_DELIVERABLE_BYTES` or not valid UTF-8
are reported by status without content, and content is redacted like the
final output.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

MAX_DELIVERABLE_FILES = 8
MAX_DELIVERABLE_BYTES = 32 * 1024
DELIVERABLE_STATUSES = frozenset({"ok", "missing", "not_a_file", "too_large", "not_utf8"})


def subjective_criterion_files(item: object) -> tuple[str, ...]:
    """Declared deliverable paths of one subjective criterion (``()`` for plain strings)."""
    if isinstance(item, Mapping):
        files = item.get("files", [])
        if isinstance(files, list):
            return tuple(path for path in files if isinstance(path, str))
    return ()


def declared_deliverables(verification: Mapping[str, Any]) -> tuple[str, ...]:
    """Union of the deliverable paths declared by all subjective criteria, in declaration order."""
    criteria = verification.get("subjective", [])
    if not isinstance(criteria, list):
        return ()
    paths: dict[str, None] = {}
    for item in criteria:
        for path in subjective_criterion_files(item):
            paths.setdefault(path, None)
    return tuple(paths)


def validate_subjective_criteria(criteria: object) -> None:
    """Semantic validation of ``verification.subjective`` (strings or criterion objects)."""
    from .sandbox import validate_relative_path

    if not isinstance(criteria, list):
        raise ValueError("benchmark verification.subjective must be an array")
    for item in criteria:
        if isinstance(item, str):
            if not item.strip():
                raise ValueError("benchmark subjective criteria must be non-empty")
            continue
        if not isinstance(item, Mapping) or not isinstance(item.get("criterion"), str) \
                or not item["criterion"].strip():
            raise ValueError("benchmark subjective criteria must be strings or objects with a criterion")
        if "files" not in item:
            continue
        files = item["files"]
        if not isinstance(files, list) or not files or len(files) > MAX_DELIVERABLE_FILES:
            raise ValueError(
                f"subjective criterion files must list 1-{MAX_DELIVERABLE_FILES} workspace-relative paths"
            )
        for path in files:
            normalized = validate_relative_path(path, label="subjective criterion file", allow_root=False)
            if normalized.as_posix() != path:
                raise ValueError("subjective criterion files must be normalized relative POSIX paths")
        if len(set(files)) != len(files):
            raise ValueError("subjective criterion files must be unique")
    if len(declared_deliverables({"subjective": criteria})) > MAX_DELIVERABLE_FILES:
        raise ValueError(f"a benchmark may declare at most {MAX_DELIVERABLE_FILES} deliverable files")


def collect_deliverables(paths: Sequence[str], workspace: Path) -> tuple[dict[str, Any], ...]:
    """Read the declared deliverable files from a run's result workspace (bounded, never outside it)."""
    from .ingestion import redact_text
    from .sandbox import validate_relative_path

    root = Path(workspace).resolve()
    collected = []
    for path in paths:
        relative = validate_relative_path(path, label="deliverable path", allow_root=False)
        entry: dict[str, Any] = {"path": path, "status": "missing", "bytes": 0, "sha256": "", "content": ""}
        target = root.joinpath(*relative.parts)
        if target.is_symlink():
            entry["status"] = "not_a_file"
        elif target.exists():
            resolved = target.resolve()
            if root not in resolved.parents or not resolved.is_file():
                entry["status"] = "not_a_file"
            else:
                with resolved.open("rb") as stream:
                    data = stream.read(MAX_DELIVERABLE_BYTES + 1)
                if len(data) > MAX_DELIVERABLE_BYTES:
                    entry.update(status="too_large", bytes=resolved.stat().st_size)
                else:
                    entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        entry["status"] = "not_utf8"
                    else:
                        entry.update(status="ok", content=redact_text(text))
        collected.append(entry)
    return tuple(collected)


def validate_deliverables(deliverables: object) -> bool:
    """Structural check of ``RunObservation.deliverables`` as reported by a runner."""
    if not isinstance(deliverables, tuple):
        return False
    for item in deliverables:
        if not isinstance(item, Mapping) or set(item) != {"path", "status", "bytes", "sha256", "content"}:
            return False
        if (
            not isinstance(item["path"], str) or item["status"] not in DELIVERABLE_STATUSES
            or isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] < 0
            or not isinstance(item["sha256"], str) or not isinstance(item["content"], str)
            or (item["status"] != "ok" and item["content"])
            or len(item["content"]) > 2 * MAX_DELIVERABLE_BYTES
        ):
            return False
    return len({item["path"] for item in deliverables}) == len(deliverables)
