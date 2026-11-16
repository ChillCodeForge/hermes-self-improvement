from __future__ import annotations

from pathlib import PurePosixPath

from .target_paths import PROTECTED_COMPONENTS as _PROTECTED_COMPONENTS
from .target_paths import SAFE_COMPONENT as _SAFE_SKILL_COMPONENT


def validate_skill_target(target: str) -> PurePosixPath:
    """Validate a skill directory target (``brain/skills/<slug>``).

    Kept for the skill-only candidate flow; the general per-type rules live in
    :mod:`hermes_improve.target_paths` and share the same component checks.
    """
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
