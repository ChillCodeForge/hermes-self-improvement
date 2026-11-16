import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402


def body(checks):
    checks.check("tests_unchanged", read("test_durations.py") == read(FIXTURE / "pristine" / "test_durations.py"))
    tests = subprocess.run([sys.executable, "-m", "unittest", "-q"], capture_output=True, timeout=60)
    checks.check("unit_tests_pass", tests.returncode == 0)
    sys.path.insert(0, ".")
    from durations import parse_duration
    checks.check("hidden_minutes_seconds", parse_duration("2m10s") == 130)
    try:
        parse_duration("3s2m")
        rejected = False
    except ValueError:
        rejected = True
    checks.check("hidden_rejects_reordered", rejected)


run(body)
