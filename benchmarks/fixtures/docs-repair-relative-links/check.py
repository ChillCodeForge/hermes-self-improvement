import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import re

LINK = re.compile(r"\]\(([^)#]+)(#[^)]*)?\)")
EXPECTED = {
    "docs/index.md": """# Documentation

- [Installation](installation.md) explains prerequisites.
- [Using the CLI](guide/using-the-cli.md) covers every command.
- Back to the [project README](../README.md).
""",
    "docs/installation.md": """# Installation

Install the package, then read [the CLI guide](guide/using-the-cli.md#commands).
""",
    "docs/guide/using-the-cli.md": """# Using the CLI

## Commands

Run `tool sync`. Setup steps live in [installation](../installation.md).
""",
}


def body(checks):
    broken = []
    for path in sorted(Path("docs").rglob("*.md")):
        for target, _anchor in LINK.findall(path.read_text(encoding="utf-8")):
            if not (path.parent / target).exists():
                broken.append(f"{path}:{target}")
    checks.check("no_broken_links", not broken)
    for path, content in EXPECTED.items():
        checks.check(f"exact:{path}", Path(path).is_file() and read(path) == content)
    checks.check("no_files_renamed", sorted(str(p) for p in Path("docs").rglob("*.md")) == sorted(EXPECTED))


run(body)
