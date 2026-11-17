import os

import pytest


def pytest_collection_modifyitems(config, items):
    # Mirrors tests/security/conftest.py for runs that collect only tests/e2e.
    if os.name == "posix" and os.geteuid() == 0:
        return
    marked = [item for item in items if item.get_closest_marker("root_required") and "e2e" in item.nodeid]
    if marked and os.environ.get("HSI_REQUIRE_ROOT_TESTS") == "1":
        raise pytest.UsageError(f"HSI_REQUIRE_ROOT_TESTS=1 but {len(marked)} e2e tests require root")
    for item in marked:
        item.add_marker(pytest.mark.skip(reason="requires root to create role users (run scripts/docker-test.sh)"))
