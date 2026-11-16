import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import tempfile


def body(checks):
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "my documents"
        source.mkdir()
        (source / "annual report.txt").write_text("report\n")
        (source / "q1 notes.md").write_text("notes\n")
        (source / "plain.txt").write_text("plain\n")
        (source / "sub dir").mkdir()
        destination = Path(scratch) / "backup target"
        result = subprocess.run(
            ["sh", str(Path("backup.sh").resolve()), str(source), str(destination)],
            capture_output=True, timeout=30,
        )
        checks.check("script_succeeds", result.returncode == 0)
        names = sorted(item.name for item in destination.iterdir()) if destination.is_dir() else []
        checks.check("all_files_backed_up", names == ["annual report.txt.bak", "plain.txt.bak", "q1 notes.md.bak"])
        checks.check(
            "contents_copied",
            destination.is_dir() and (destination / "annual report.txt.bak").is_file()
            and (destination / "annual report.txt.bak").read_text() == "report\n",
        )
        checks.check("source_untouched", sorted(item.name for item in source.iterdir()) == [
            "annual report.txt", "plain.txt", "q1 notes.md", "sub dir",
        ])


run(body)
