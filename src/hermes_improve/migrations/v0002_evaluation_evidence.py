"""Run observations and signed evaluation metrics.

Pre-versioning stores added ``metrics_json`` and ``signature`` with inline
``ALTER TABLE`` statements, so a legacy database may already have either
column. The additions are therefore conditional, which plain SQL cannot
express.
"""

from __future__ import annotations

import sqlite3

from . import add_missing_columns


def upgrade(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS run_observations (
            evaluation_id TEXT NOT NULL REFERENCES evaluation_runs(id),
            task_id TEXT NOT NULL,
            commit_id TEXT NOT NULL,
            seed INTEGER NOT NULL,
            observation_json TEXT NOT NULL,
            PRIMARY KEY (evaluation_id, task_id, commit_id, seed)
        )
    """)
    add_missing_columns(db, "evaluation_runs", (
        ("metrics_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("signature", "TEXT"),
    ))
