import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402


def body(checks):
    checks.check("tests_unchanged", read("test_paginate.py") == read(FIXTURE / "pristine" / "test_paginate.py"))
    tests = subprocess.run([sys.executable, "-m", "unittest", "-q"], capture_output=True, timeout=60)
    checks.check("unit_tests_pass", tests.returncode == 0)
    sys.path.insert(0, ".")
    from paginate import page_count, paginate
    checks.check("hidden_middle_page", paginate(list("abcdefgh"), 2, 3) == ["d", "e", "f"])
    checks.check("hidden_page_count", page_count(7, 7) == 1 and page_count(8, 7) == 2)


run(body)
