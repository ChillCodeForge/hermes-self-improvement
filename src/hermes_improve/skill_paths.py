from __future__ import annotations

import re
from pathlib import PurePosixPath


_SAFE_SKILL_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_PROTECTED_COMPONENTS = {
    ".git", "evaluator", "verifiers", "benchmarks", "holdout", "security",
    "promotion", "ci", ".github", "tests",
}


def validate_skill_target(target: str) -> PurePosixPath:
    if not isinstance(target, str) or not target or "\\" in target or "\x00" in target:
        raise ValueError("candidate target must be an unprotected path under brain/skills/")
    path = PurePosixPath(target)
    if (
        path.is_absolute()
        or path.as_posix() != target
        or len(path.parts) < 3
        or path.parts[:2] != ("brain", "skills")
        or path.parts[-1].casefold() == "skill.md"
        or any(not _SAFE_SKILL_COMPONENT.fullmatch(part) for part in path.parts[2:])
        or any(part.casefold() in _PROTECTED_COMPONENTS for part in path.parts[2:])
    ):
        raise ValueError("candidate target must be an unprotected path under brain/skills/")
    return path
