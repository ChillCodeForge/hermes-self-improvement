import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = "3 E102\n2 E404\n2 E500\n1 E201\n"


def body(checks):
    checks.check("summary_exists", Path("error-summary.txt").is_file())
    checks.check("summary_exact", read("error-summary.txt") == EXPECTED)
    checks.check("log_unchanged", read("app.log").count("\n") == 12)


run(body)
