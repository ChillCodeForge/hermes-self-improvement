import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = """# hermes-sync

Synchronise skills between machines.

## Options

<!-- options:start -->
| Option | Default | Description |
|---|---|---|
| `--config` | `hermes.toml` | Path to the configuration file. |
| `--dry-run` | - | Print planned changes without writing. |
| `--jobs` | `4` | Number of parallel workers. |
<!-- options:end -->

## License

MIT
"""


def body(checks):
    text = read("README.md")
    checks.check("markers_kept", "<!-- options:start -->" in text and "<!-- options:end -->" in text)
    checks.check("readme_exact", text == EXPECTED)
    checks.check("options_unchanged", '"--jobs"' in read("cli-options.json"))


run(body)
