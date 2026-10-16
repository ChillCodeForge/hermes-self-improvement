from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import __version__
from .benchmarks import BenchmarkStore
from .models import Role
from .store import ExperimentStore


def _paths() -> tuple[Path, Path]:
    home = Path(os.environ.get("HERMES_IMPROVE_HOME", Path.home() / ".hermes-improve"))
    return home / "experiments.sqlite3", Path(os.environ.get("HERMES_IMPROVE_BENCHMARKS", "benchmarks"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-improve", description="Safety-gated Hermes improvement foundations")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    benchmarks = sub.add_parser("benchmarks")
    benchmark_sub = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    benchmark_sub.add_parser("validate")
    listing = benchmark_sub.add_parser("list")
    listing.add_argument("--role", choices=[role.value for role in Role], default=Role.OPERATOR.value)
    experiments = sub.add_parser("experiments")
    experiment_sub = experiments.add_subparsers(dest="experiment_command", required=True)
    listing = experiment_sub.add_parser("list")
    listing.add_argument("--limit", type=int, default=100)
    show = experiment_sub.add_parser("show")
    show.add_argument("experiment_id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    database, benchmark_root = _paths()
    if args.command == "status":
        print(json.dumps({"version": __version__, "database": str(database), "benchmarks": str(benchmark_root), "phase": "evaluation-foundation"}, indent=2))
    elif args.command == "benchmarks":
        store = BenchmarkStore(benchmark_root)
        if args.benchmark_command == "validate":
            rows = store.validate()
            print(f"valid benchmarks: {len(rows)}")
        else:
            print(json.dumps(store.list(Role(args.role)), indent=2))
    elif args.command == "experiments":
        store = ExperimentStore(database)
        if args.experiment_command == "list":
            print(json.dumps(store.list_experiments(args.limit), indent=2))
        else:
            print(json.dumps(store.show_experiment(args.experiment_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
