"""Role prompts must ship with the package (lab finding F1).

A non-editable install (``scripts/provision.sh --install-from``) contains only
what setuptools packages. Prompts used to live at the repository root and were
resolved relative to the checkout, so every Qwen role failed in an installed
environment. They now live in ``hermes_improve/prompts`` and are package data.
"""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

import hermes_improve
from hermes_improve import judge, prompting

PACKAGE = Path(hermes_improve.__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]


def test_prompt_roots_are_inside_the_package():
    assert prompting.PROMPT_ROOT == PACKAGE / "prompts"
    assert judge.DEFAULT_PROMPT_DIR == PACKAGE / "prompts" / "judge"
    assert not (REPO / "prompts").exists(), "prompts must not be resolved from the checkout root"


def test_every_prompt_file_is_declared_as_package_data():
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["hermes_improve"]
    files = sorted(path for path in (PACKAGE / "prompts").rglob("*") if path.is_file())
    assert files, "no prompt files found"
    for path in files:
        relative = path.relative_to(PACKAGE).as_posix()
        # setuptools package-data globs do not cross directories with "*".
        assert any(
            fnmatch.fnmatch(relative, pattern) and relative.count("/") == pattern.count("/")
            for pattern in patterns
        ), f"{relative} is not shipped as package data"


def test_every_role_prompt_loads_from_the_package():
    for name in ("dreaming/rem.md", "dreaming/deep.md", "dreaming/candidate_content.md",
                 "optimizer/mutate_skill.md", "optimizer/gepa_reflection.md", "optimizer/gepa_system.md"):
        assert prompting.load_prompt(name).strip()
    assert (judge.DEFAULT_PROMPT_DIR / "pairwise_system.md").is_file()
