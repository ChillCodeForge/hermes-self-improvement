import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402


def body(checks):
    checks.check("tests_unchanged", read("test_textutil.py") == read(FIXTURE / "pristine" / "test_textutil.py"))
    tests = subprocess.run([sys.executable, "-m", "unittest", "-q"], capture_output=True, timeout=60)
    checks.check("unit_tests_pass", tests.returncode == 0)
    sys.path.insert(0, ".")
    from textutil import slugify
    checks.check("hidden_digits_kept", slugify("Top 10 Tips (2024)") == "top-10-tips-2024")
    checks.check("hidden_default_limit", len(slugify("word " * 40)) <= 60)


run(body)
