import re
from pathlib import Path

pins = dict(line.split() for line in Path("pins.txt").read_text().splitlines() if line.strip())
for dockerfile in sorted(Path("services").rglob("Dockerfile")):
    lines = []
    for line in dockerfile.read_text().splitlines(keepends=True):
        match = re.match(r"(FROM\s+)(\S+)(.*)", line, re.S)
        if match and match.group(2) in pins:
            line = f"{match.group(1)}{match.group(2)}@{pins[match.group(2)]}{match.group(3)}"
        lines.append(line)
    dockerfile.write_text("".join(lines))
