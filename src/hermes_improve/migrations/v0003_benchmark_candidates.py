"""Mined benchmark proposals and their operator review columns.

The review columns were added by inline ``ALTER TABLE`` statements before
schema versioning, so a legacy database may already have them.
"""

from __future__ import annotations

import sqlite3

from . import add_missing_columns


def upgrade(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_candidates (
            task_fingerprint TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            eligibility TEXT NOT NULL,
            reproducibility_score REAL NOT NULL,
            ground_truth_type TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            source_sessions_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    add_missing_columns(db, "benchmark_candidates", (
        ("review_status", "TEXT NOT NULL DEFAULT 'PENDING'"),
        ("reviewer", "TEXT"),
        ("review_rationale", "TEXT"),
        ("reviewed_at", "TEXT"),
    ))
