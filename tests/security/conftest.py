from __future__ import annotations

import os

import pytest

_REASON = "requires root to create role users and switch UIDs (run scripts/docker-test.sh)"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "root_required: OS-isolation test that creates Unix users and switches UIDs; "
        "runs only as root (e.g. inside scripts/docker-test.sh)",
    )


def _is_root() -> bool:
    return os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0


def pytest_collection_modifyitems(config, items):
    # Decided at collection time so that module-scoped provisioning fixtures
    # never run for a skipped test.
    if _is_root():
        return
    marked = [item for item in items if item.get_closest_marker("root_required")]
    if marked and os.environ.get("HSI_REQUIRE_ROOT_TESTS") == "1":
        raise pytest.UsageError(f"HSI_REQUIRE_ROOT_TESTS=1 but {len(marked)} tests {_REASON}")
    for item in marked:
        item.add_marker(pytest.mark.skip(reason=_REASON))
