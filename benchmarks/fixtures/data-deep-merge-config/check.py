import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

EXPECTED = {
    "service": {"name": "billing", "port": 9090, "features": ["invoices"]},
    "database": {"host": "db.internal", "pool": {"min": 2, "max": 25}},
    "logging": {"level": "debug", "format": "json"},
    "metrics": {"enabled": True},
}


def body(checks):
    merged = json.loads(read("merged.json"))
    checks.check("merged_equal", merged == EXPECTED)
    checks.check("null_removes_key", "legacy_mode" not in merged and "ssl" not in merged.get("database", {}))
    checks.check("base_unchanged", json.loads(read("base.json"))["service"]["port"] == 8080)
    checks.check("override_unchanged", json.loads(read("override.json"))["legacy_mode"] is None)


run(body)
