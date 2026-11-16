import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

EXPECTED_MODELS = {
    "m-large": {"mean_latency_ms": 400.0, "success_rate": 0.667},
    "m-small": {"mean_latency_ms": 125.0, "success_rate": 1.0},
    "m-tiny": {"mean_latency_ms": 55.0, "success_rate": 0.333},
}


def body(checks):
    report = json.loads(read("report.json"))
    models = report.get("models", {})
    checks.check("all_models_present", sorted(models) == sorted(EXPECTED_MODELS))
    for model, expected in EXPECTED_MODELS.items():
        checks.check(f"stats:{model}", models.get(model) == expected)
    checks.check("fastest_reliable_model", report.get("fastest_model") == "m-small")


run(body)
