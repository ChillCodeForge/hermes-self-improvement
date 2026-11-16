from pathlib import Path

ADD = [
    "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes",
    "PrivateTmp=yes", "ReadWritePaths=/var/lib/report-worker",
]
lines = Path("report-worker.service").read_text().splitlines()
index = lines.index("Restart=on-failure") + 1
Path("report-worker.service").write_text("\n".join(lines[:index] + ADD + lines[index:]) + "\n")
