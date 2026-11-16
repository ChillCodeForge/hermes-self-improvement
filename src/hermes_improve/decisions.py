"""Decision history for a possible future Laya decision model (spec section 36).

Every automated or rule-based decision that a later learned model could imitate
is recorded with its input features, the options that were available, the
selected option, and the producing source/model version. Ground truth is
backfilled later by an operator or by a downstream outcome; the record then
states whether the original decision was correct.

The history is training data only. Nothing in this package reads it to make a
promotion, rollback, or gate decision, and a learned model built from it must
never replace a deterministic gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

from .ingestion import contains_secret_material
from .models import canonical_json, utc_now


DECISION_TYPES = frozenset({
    "benchmark_eligibility",
    "risk",
    "needs_qwen",
    "regression_severity",
    "suite_selection",
    "experiment_priority",
    "curator_signal",
})
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")
_DECISION_ID = re.compile(r"^[a-f0-9]{32}$")
_MAX_FEATURE_BYTES = 64 * 1024
_MAX_OPTIONS = 64
_MAX_EXPORT_RECORDS = 1_000_000


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    decision_type: str
    input_features: Mapping[str, Any]
    available_options: tuple[str, ...]
    selected_option: str
    confidence: float | None
    source: str
    ground_truth: str | None
    ground_truth_source: str | None
    correct: bool | None
    model_version: str | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision_type": self.decision_type,
            "input_features": dict(self.input_features),
            "available_options": list(self.available_options),
            "selected_option": self.selected_option,
            "confidence": self.confidence,
            "source": self.source,
            "ground_truth": self.ground_truth,
            "ground_truth_source": self.ground_truth_source,
            "correct": self.correct,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
        }


class DecisionRecorder:
    """Append-only decision history stored in the experiment SQLite database.

    Recording is idempotent: the decision ID is derived from the decision type,
    the canonical input features, the options, the selected option, the source,
    and the model version, so repeating the same decision on the same evidence
    does not inflate the dataset.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS decision_history (
                    id TEXT PRIMARY KEY,
                    decision_type TEXT NOT NULL,
                    input_features TEXT NOT NULL,
                    available_options TEXT NOT NULL,
                    selected_option TEXT NOT NULL,
                    confidence REAL,
                    source TEXT NOT NULL,
                    ground_truth TEXT,
                    ground_truth_source TEXT,
                    correct INTEGER,
                    model_version TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            db.execute(
                "CREATE INDEX IF NOT EXISTS decision_history_type ON decision_history(decision_type, timestamp)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def record(
        self,
        *,
        decision_type: str,
        input_features: Mapping[str, Any],
        available_options: Sequence[str],
        selected_option: str,
        source: str,
        confidence: float | None = None,
        model_version: str | None = None,
    ) -> DecisionRecord:
        if not isinstance(decision_type, str) or decision_type not in DECISION_TYPES:
            raise ValueError(f"decision_type must be one of: {', '.join(sorted(DECISION_TYPES))}")
        features = _validate_features(input_features)
        options = _validate_options(available_options)
        _validate_label(selected_option, "selected_option")
        if selected_option not in options:
            raise ValueError("selected_option must be one of available_options")
        _validate_label(source, "source")
        if model_version is not None:
            _validate_label(model_version, "model_version")
        if confidence is not None:
            if (
                isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError("confidence must be a finite number between 0 and 1")
            confidence = float(confidence)
        features_json = canonical_json(features)
        options_json = canonical_json(list(options))
        decision_id = hashlib.sha256(canonical_json([
            decision_type, features_json, options_json, selected_option, source, model_version,
        ]).encode()).hexdigest()[:32]
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM decision_history WHERE id=?", (decision_id,)).fetchone()
            if existing is not None:
                if existing["confidence"] != confidence:
                    raise ValueError("decision was already recorded with a different confidence")
                return _record(existing)
            db.execute(
                """INSERT INTO decision_history
                   (id,decision_type,input_features,available_options,selected_option,confidence,
                    source,ground_truth,ground_truth_source,correct,model_version,timestamp)
                   VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL,?,?)""",
                (decision_id, decision_type, features_json, options_json, selected_option,
                 confidence, source, model_version, utc_now()),
            )
            row = db.execute("SELECT * FROM decision_history WHERE id=?", (decision_id,)).fetchone()
        return _record(row)

    def backfill_ground_truth(
        self, decision_id: str, *, ground_truth: str, ground_truth_source: str,
    ) -> DecisionRecord:
        """Attach the observed correct option; immutable once set."""
        if not isinstance(decision_id, str) or not _DECISION_ID.fullmatch(decision_id):
            raise ValueError("decision ID must be a 32-character lowercase hex digest")
        _validate_label(ground_truth, "ground_truth")
        _validate_label(ground_truth_source, "ground_truth_source")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM decision_history WHERE id=?", (decision_id,)).fetchone()
            if row is None:
                raise KeyError(decision_id)
            options = json.loads(row["available_options"])
            if ground_truth not in options:
                raise ValueError("ground_truth must be one of the decision's available_options")
            if row["ground_truth"] is not None:
                if row["ground_truth"] == ground_truth and row["ground_truth_source"] == ground_truth_source:
                    return _record(row)
                raise ValueError("decision already has an immutable ground truth")
            db.execute(
                """UPDATE decision_history SET ground_truth=?, ground_truth_source=?, correct=?
                   WHERE id=? AND ground_truth IS NULL""",
                (ground_truth, ground_truth_source, int(ground_truth == row["selected_option"]), decision_id),
            )
            updated = db.execute("SELECT * FROM decision_history WHERE id=?", (decision_id,)).fetchone()
        return _record(updated)

    def get(self, decision_id: str) -> DecisionRecord:
        with self._connect() as db:
            row = db.execute("SELECT * FROM decision_history WHERE id=?", (decision_id,)).fetchone()
        if row is None:
            raise KeyError(decision_id)
        return _record(row)

    def iter_records(
        self, *, decision_type: str | None = None, labeled_only: bool = False,
    ) -> Iterator[DecisionRecord]:
        if decision_type is not None and decision_type not in DECISION_TYPES:
            raise ValueError(f"decision_type must be one of: {', '.join(sorted(DECISION_TYPES))}")
        clauses, params = [], []
        if decision_type is not None:
            clauses.append("decision_type=?")
            params.append(decision_type)
        if labeled_only:
            clauses.append("ground_truth IS NOT NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            for row in db.execute(
                f"SELECT * FROM decision_history {where} ORDER BY timestamp, id", params,
            ):
                yield _record(row)

    def counts(self) -> dict[str, dict[str, int]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT decision_type, COUNT(*) AS total,
                          SUM(CASE WHEN ground_truth IS NOT NULL THEN 1 ELSE 0 END) AS labeled,
                          SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct
                   FROM decision_history GROUP BY decision_type ORDER BY decision_type"""
            ).fetchall()
        return {
            row["decision_type"]: {
                "total": row["total"], "labeled": row["labeled"] or 0, "correct": row["correct"] or 0,
            }
            for row in rows
        }

    def export_jsonl(
        self, destination: TextIO, *, decision_type: str | None = None, labeled_only: bool = False,
    ) -> int:
        """Write one JSON object per decision; returns the number of records."""
        count = 0
        for record in self.iter_records(decision_type=decision_type, labeled_only=labeled_only):
            count += 1
            if count > _MAX_EXPORT_RECORDS:
                raise ValueError("decision export exceeds the 1,000,000-record limit")
            destination.write(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        return count


def record_decision(database: Path, **decision: Any) -> DecisionRecord:
    """Record one decision in the experiment database (callable by any package)."""
    return DecisionRecorder(database).record(**decision)


def _record(row: sqlite3.Row) -> DecisionRecord:
    return DecisionRecord(
        id=row["id"],
        decision_type=row["decision_type"],
        input_features=json.loads(row["input_features"]),
        available_options=tuple(json.loads(row["available_options"])),
        selected_option=row["selected_option"],
        confidence=row["confidence"],
        source=row["source"],
        ground_truth=row["ground_truth"],
        ground_truth_source=row["ground_truth_source"],
        correct=None if row["correct"] is None else bool(row["correct"]),
        model_version=row["model_version"],
        timestamp=row["timestamp"],
    )


def _validate_label(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise ValueError(f"{field_name} must be a 1-128 character label of letters, digits, and ._:/@+-")
    if contains_secret_material(value):
        raise ValueError(f"{field_name} must not contain credential-like material")


def _validate_options(options: Sequence[str]) -> tuple[str, ...]:
    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise ValueError("available_options must be a sequence of labels")
    result = tuple(options)
    if not 2 <= len(result) <= _MAX_OPTIONS:
        raise ValueError("available_options must contain between 2 and 64 options")
    for option in result:
        _validate_label(option, "available_options entry")
    if len(set(result)) != len(result):
        raise ValueError("available_options must not contain duplicates")
    return result


def _validate_features(features: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(features, Mapping) or not features:
        raise ValueError("input_features must be a non-empty object")
    try:
        encoded = canonical_json(_json_value(features))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input_features must be finite JSON data: {exc}") from exc
    if len(encoded.encode("utf-8")) > _MAX_FEATURE_BYTES:
        raise ValueError("input_features exceed the 64 KiB limit")
    if contains_secret_material(json.loads(encoded)):
        raise ValueError("input_features must not contain credential-like material")
    return json.loads(encoded)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("feature keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported feature type {type(value).__name__}")
