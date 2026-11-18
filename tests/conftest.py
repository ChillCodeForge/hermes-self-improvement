"""Shared pytest setup for every test directory.

Tests are split into ``unit/`` (single module, in-process), ``integration/``
(CLI, Git repositories, subprocesses, HTTP stubs, several components),
``security/`` (OS isolation and protected boundaries) and ``e2e/`` (the full
improvement loop). Shared helpers such as ``evolution_doubles`` and
``openai_stub`` live in this directory, which is put on ``sys.path`` here so
every sub-directory can import them.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Environment prefixes the package reads host configuration from: the isolation
# layout, role UIDs, signing keys, state directories, model endpoints, metrics.
_HOST_ENV_PREFIXES = ("HERMES_", "QWEN_")


@pytest.fixture(autouse=True)
def hermetic_environment(tmp_path_factory, monkeypatch):
    """Keep every test independent of host provisioning and operator settings.

    Without this, ``hermes-improve`` resolves ``HERMES_IMPROVE_LAYOUT`` to the
    real ``/srv/hermes-improve/config/layout.json`` and state to ``~``: on a
    provisioned host a CLI test would verify the real isolation layout (and fail
    with "process uid 0 is not the promoter user") or touch real state. Every
    ``HERMES_*``/``QWEN_*`` variable is cleared and the layout, the improvement
    home and the Hermes home point at paths below a fresh temporary directory
    that do not exist. Tests that need a value set it themselves with
    ``monkeypatch.setenv``, which overrides this fixture.
    """
    for name in list(os.environ):
        if name.startswith(_HOST_ENV_PREFIXES):
            monkeypatch.delenv(name)
    sandbox = tmp_path_factory.mktemp("hermetic-env")
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(sandbox / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(sandbox / "improve-home"))
    monkeypatch.setenv("HERMES_HOME", str(sandbox / "hermes-home"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "root_required: OS-isolation test that creates Unix users and switches UIDs; "
        "runs only as root (e.g. inside scripts/docker-test.sh)",
    )
