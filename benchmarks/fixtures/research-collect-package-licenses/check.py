import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

EXPECTED = {
    "attrs": "MIT License",
    "requests": "Apache 2.0",
    "tinycache": "BSD License",
    "yamlish": "MIT",
}


def body(checks):
    data = json.loads(read("licenses.json"))
    checks.check("all_packages", sorted(data) == sorted(EXPECTED))
    checks.check("licenses_exact", data == EXPECTED)


run(body)
