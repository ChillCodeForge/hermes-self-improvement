"""Shared pytest setup for every test directory.

Tests are split into ``unit/`` (single module, in-process), ``integration/``
(CLI, Git repositories, subprocesses, HTTP stubs, several components),
``security/`` (OS isolation and protected boundaries) and ``e2e/`` (the full
improvement loop). Shared helpers such as ``evolution_doubles`` and
``openai_stub`` live in this directory, which is put on ``sys.path`` here so
every sub-directory can import them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "root_required: OS-isolation test that creates Unix users and switches UIDs; "
        "runs only as root (e.g. inside scripts/docker-test.sh)",
    )
