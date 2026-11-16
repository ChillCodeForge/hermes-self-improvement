import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = {
    "photos/2023/12/31/IMG_20231231_235959.jpg": "jpeg-3\n",
    "photos/2024/01/05/IMG_20240105_101500.jpg": "jpeg-1\n",
    "photos/2024/01/05/IMG_20240105_183000.jpg": "jpeg-2\n",
    "photos/2024/02/14/VID_20240214_090000.mp4": "mp4-1\n",
}


def body(checks):
    for path, content in EXPECTED.items():
        checks.check(f"moved:{Path(path).name}", Path(path).is_file() and read(path) == content)
    remaining = sorted(item.name for item in Path("inbox").iterdir())
    checks.check("inbox_keeps_only_notes", remaining == ["notes.txt"])
    checks.check("notes_unchanged", read("inbox/notes.txt") == "call the framer\n")
    files = sorted(str(item) for item in Path("photos").rglob("*") if item.is_file())
    checks.check("no_extra_files", files == sorted(EXPECTED))


run(body)
