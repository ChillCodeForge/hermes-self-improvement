import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = """# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Export reports as CSV.

### Fixed

- Retry uploads when the storage API returns HTTP 503.

## [2.3.0] - 2024-04-02

### Fixed

- Handle empty upload batches.
"""


def body(checks):
    text = read("CHANGELOG.md")
    unreleased = text.split("## [2.3.0]")[0]
    checks.check("entry_in_unreleased", "- Retry uploads when the storage API returns HTTP 503." in unreleased)
    checks.check("released_section_untouched", text.endswith("## [2.3.0] - 2024-04-02\n\n### Fixed\n\n- Handle empty upload batches.\n"))
    checks.check("changelog_exact", text == EXPECTED)


run(body)
