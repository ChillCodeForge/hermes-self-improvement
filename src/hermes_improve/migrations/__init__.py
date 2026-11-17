"""Versioned schema migrations for the experiment ledger (spec sections 3, 26, 37).

Migrations live next to this module as ``vNNNN_<name>.sql`` or
``vNNNN_<name>.py`` (the latter exposes ``upgrade(db)``; it is used only where
SQL cannot express the change, e.g. conditional column additions for
databases created before schema versioning). They are shipped as package
data, so a root-owned installation applies the same files.

``PRAGMA user_version`` is the authoritative schema version. Each migration
runs in its own ``BEGIN IMMEDIATE`` transaction that also appends a row to
``schema_migrations`` and bumps ``user_version``; a failure rolls back the
whole step. Re-opening a database at the current version applies nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import sqlite3
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable

from ..models import utc_now

_FILE = re.compile(r"^v(\d{4})_([a-z0-9_]+)\.(sql|py)$")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class SchemaVersionError(RuntimeError):
    """The database schema cannot be brought to the version this code expects."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    filename: str
    sha256: str
    apply: Callable[[sqlite3.Connection], None]


def _sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    leftover = "\n".join(
        line for line in buffer.splitlines() if line.strip() and not line.strip().startswith("--")
    )
    if leftover:
        raise SchemaVersionError(f"migration script ends with an incomplete statement: {leftover[:80]!r}")
    return statements


def _sql_migration(script: str) -> Callable[[sqlite3.Connection], None]:
    statements = _sql_statements(script)

    def apply(db: sqlite3.Connection) -> None:
        for statement in statements:
            db.execute(statement)

    return apply


def load_migrations() -> tuple[Migration, ...]:
    """All packaged migrations in version order; versions must be 1..N without gaps."""
    found: dict[int, Migration] = {}
    for entry in resources.files(__name__).iterdir():
        match = _FILE.fullmatch(entry.name)
        if match is None:
            continue
        version = int(match.group(1))
        if version in found:
            raise SchemaVersionError(f"duplicate schema migration version {version}")
        content = entry.read_bytes()
        if match.group(3) == "sql":
            apply = _sql_migration(content.decode("utf-8"))
        else:
            module = importlib.import_module(f"{__name__}.{entry.name[:-3]}")
            apply = module.upgrade
        found[version] = Migration(
            version, match.group(2), entry.name, hashlib.sha256(content).hexdigest(), apply,
        )
    ordered = tuple(found[key] for key in sorted(found))
    if [item.version for item in ordered] != list(range(1, len(ordered) + 1)):
        raise SchemaVersionError("schema migrations must be numbered 1..N without gaps")
    return ordered


def add_missing_columns(
    db: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]],
) -> None:
    """Add each column unless a pre-versioning store already added it."""
    if not _IDENTIFIER.fullmatch(table):
        raise ValueError(f"invalid table name {table!r}")
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if not existing:
        raise SchemaVersionError(f"table {table!r} does not exist")
    for column, definition in columns:
        if not _IDENTIFIER.fullmatch(column):
            raise ValueError(f"invalid column name {column!r}")
        if column not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def schema_version(db: sqlite3.Connection) -> int:
    return int(db.execute("PRAGMA user_version").fetchone()[0])


def apply_migrations(path: Path, migrations: Iterable[Migration] | None = None) -> int:
    """Bring the SQLite database at ``path`` to the newest schema; returns its version."""
    steps = tuple(MIGRATIONS if migrations is None else migrations)
    latest = steps[-1].version if steps else 0
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        current = schema_version(db)
        if current > latest:
            raise SchemaVersionError(
                f"database schema version {current} is newer than this code supports ({latest})"
            )
        for migration in steps:
            if migration.version <= current:
                continue
            db.execute("BEGIN IMMEDIATE")
            try:
                # Another process may have migrated while this one waited for the lock.
                current = schema_version(db)
                if migration.version <= current:
                    db.execute("COMMIT")
                    continue
                if migration.version != current + 1:
                    raise SchemaVersionError(
                        f"cannot apply schema migration {migration.version} to version {current}"
                    )
                migration.apply(db)
                db.execute(
                    "INSERT INTO schema_migrations (version,name,sha256,applied_at) VALUES (?,?,?,?)",
                    (migration.version, migration.name, migration.sha256, utc_now()),
                )
                db.execute(f"PRAGMA user_version = {int(migration.version)}")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            current = migration.version
        return current
    finally:
        db.close()


# Loaded last: Python migrations import the helpers defined above.
MIGRATIONS = load_migrations()
SCHEMA_VERSION = MIGRATIONS[-1].version
