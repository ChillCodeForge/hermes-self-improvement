"""Docker CLI stand-in for tests: executes the container entrypoint as a local process.

It understands the subset of ``docker run`` flags the sandbox emits, maps bind
mounts and tmpfs targets to host directories, records the argv, and runs the
entrypoint with container paths rewritten. Kernel isolation is obviously not
reproduced; tests use it to exercise the real container script, snapshot
transport and verifier wiring. Configuration lives in a JSON file next to the
script (``<script>.json``) because the sandbox passes only an allowlisted
environment (PATH, HOME and the daemon-selection variables) to the Docker client.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


VALUE_FLAGS = {"--memory", "--cpus", "--tmpfs", "--mount", "--cidfile", "--env", "--workdir", "--entrypoint"}


def install(directory: Path, *, network: dict | None = None, daemon_unreachable: bool = False) -> Path:
    """Write an executable fake ``docker`` into *directory* and return its path.

    With ``daemon_unreachable`` every command fails like the real CLI does when
    it cannot reach the daemon selected by ``DOCKER_HOST`` (exit 1, no container).
    """
    script = directory / "docker"
    script.write_text(f"#!{sys.executable}\n" + Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    config = {
        "log": str(directory / "docker-calls.jsonl"), "network": network,
        "daemon_unreachable": daemon_unreachable,
    }
    script.with_suffix(".json").write_text(json.dumps(config), encoding="utf-8")
    return script


def calls(directory: Path) -> list[list[str]]:
    log = directory / "docker-calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _config() -> dict:
    return json.loads(Path(sys.argv[0]).with_suffix(".json").read_text(encoding="utf-8"))


def _run(args: list[str], config: dict) -> int:
    mapping: dict[str, str] = {}
    env = {"PATH": os.environ.get("PATH", "")}
    entrypoint = None
    cidfile = None
    index = 0
    scratch = Path(tempfile.mkdtemp(prefix="fake-container-"))
    while index < len(args):
        arg = args[index]
        if arg in VALUE_FLAGS:
            value = args[index + 1]
            index += 2
            if arg == "--tmpfs":
                target = value.split(":", 1)[0]
                host = scratch / target.strip("/").replace("/", "_")
                host.mkdir()
                mapping[target] = str(host)
            elif arg == "--mount":
                fields = dict(item.split("=", 1) for item in value.split(",") if "=" in item)
                mapping[fields["target"]] = fields["source"]
            elif arg == "--env":
                key, _, val = value.partition("=")
                env[key] = val
            elif arg == "--entrypoint":
                entrypoint = value
            elif arg == "--cidfile":
                cidfile = value
            continue
        if arg.startswith("-"):
            index += 1
            continue
        break
    command = args[index + 1:]
    rewritten = [mapping.get(item, item) for item in command]
    process = subprocess.Popen([entrypoint, *rewritten], env=env, start_new_session=True)
    if cidfile:
        Path(cidfile).write_text(f"{process.pid:012x}", encoding="ascii")
    return process.wait()


def main() -> int:
    config = _config()
    args = sys.argv[1:]
    with open(config["log"], "a", encoding="utf-8") as log:
        log.write(json.dumps(args) + "\n")
    if config.get("daemon_unreachable"):
        host = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
        sys.stderr.write(f"Cannot connect to the Docker daemon at {host}. Is the docker daemon running?\n")
        return 1
    if args[:1] == ["run"]:
        return _run(args[1:], config)
    if args[:2] == ["network", "inspect"]:
        if config.get("network") is None:
            sys.stderr.write("Error: No such network\n")
            return 1
        sys.stdout.write(json.dumps([config["network"]]))
        return 0
    if args[:1] == ["kill"]:
        try:
            os.killpg(int(args[1], 16), signal.SIGKILL)
        except ProcessLookupError:
            return 1
        return 0
    sys.stderr.write(f"unsupported fake docker command: {args}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
