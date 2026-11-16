import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json


def body(checks):
    answer = json.loads(read("answer.json"))
    checks.check("setting", answer.get("setting") == "sync.request_timeout")
    checks.check("default_from_code", answer.get("default_seconds") == 45)
    checks.check("env_var", answer.get("env_var") == "SYNC_REQUEST_TIMEOUT")
    checks.check("no_extra_keys", sorted(answer) == ["default_seconds", "env_var", "setting"])


run(body)
