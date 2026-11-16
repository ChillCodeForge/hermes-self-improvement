"""Read-only connector to the on-disk state of the real Hermes Curator.

The file formats parsed here were taken from the Hermes Agent source recorded
in ``docs/operations.md`` (Curator integration section):

* ``<HERMES_HOME>/skills/.usage.json`` - per-skill usage and lifecycle record
  (``tools/skill_usage.py``: ``_empty_record``/``load_usage``/``save_usage``).
* ``<HERMES_HOME>/skills/.curator_state`` - scheduler state
  (``agent/curator.py``: ``load_state``).
* ``<HERMES_HOME>/logs/curator/<YYYYMMDD-HHMMSS>[-N]/run.json`` - per-run report
  with consolidations (overlap outcome), prunings, archives and state
  transitions (``agent/curator.py``: run report writer).
* ``<HERMES_HOME>/skills/.curator_ledger.jsonl`` - append-only mutation audit
  log (``tools/skill_ledger.py``: ``append_entry``).
* ``<HERMES_HOME>/skills/.curator_backups/<id>/manifest.json`` - snapshot
  backups used by ``hermes curator rollback`` (``agent/curator_backup.py``).
* ``<HERMES_HOME>/pending/skills/<id>.json`` - skill writes staged by the
  ``skills.write_approval`` gate (``tools/write_approval.py``: ``stage_write``).

The connector never writes to the Hermes home. Every file is opened without
following symbolic links, size-bounded, and validated; malformed records are
reported as rejected instead of being silently dropped or trusted.
"""
from __future__ import annotations

import errno
import json
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .curator import validate_curator_skill_name
from .ingestion import contains_secret_material
from .models import ImprovementSignal


_MAX_USAGE_BYTES = 16 * 1024 * 1024
_MAX_STATE_BYTES = 256 * 1024
_MAX_RUN_BYTES = 8 * 1024 * 1024
_MAX_LEDGER_TAIL_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_PENDING_BYTES = 2 * 1024 * 1024
_MAX_PENDING_FILES = 1000
_MAX_BACKUPS = 1000
_MAX_COUNTER = (1 << 63) - 1
_RUN_DIR = re.compile(r"^\d{8}-\d{6}(?:-\d{1,4})?$")
_PENDING_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_LIFECYCLE_STATES = frozenset({"active", "stale", "archived"})
_PENDING_ACTIONS = frozenset({"create", "edit", "patch", "delete", "write_file", "remove_file"})
_PATCH_CHURN_MIN = 3


@dataclass(frozen=True)
class CuratorSkillRecord:
    name: str
    state: str
    pinned: bool
    created_by: str | None
    use_count: int
    view_count: int
    patch_count: int
    patch_generation: int
    last_activity_at: str | None
    archived_at: str | None

    @property
    def activity_count(self) -> int:
        return self.use_count + self.view_count + self.patch_count

    @property
    def curator_managed(self) -> bool:
        return self.created_by == "agent"


@dataclass(frozen=True)
class CuratorRun:
    run_id: str
    started_at: str
    counts: Mapping[str, int]
    consolidated: tuple[Mapping[str, str], ...]
    pruned: tuple[str, ...]
    archived: tuple[str, ...]
    state_transitions: tuple[Mapping[str, str], ...]
    llm_error: str | None


@dataclass(frozen=True)
class PendingSkillWrite:
    pending_id: str
    action: str
    skill: str
    origin: str
    summary: str
    created_at: float
    payload: Mapping[str, Any]


@dataclass
class CuratorSnapshot:
    hermes_home: str
    skills: list[CuratorSkillRecord] = field(default_factory=list)
    scheduler: dict[str, Any] | None = None
    runs: list[CuratorRun] = field(default_factory=list)
    ledger_actions: dict[str, int] = field(default_factory=dict)
    ledger_actors: dict[str, int] = field(default_factory=dict)
    ledger_rollbacks: list[dict[str, str]] = field(default_factory=list)
    ledger_entries_read: int = 0
    ledger_truncated: bool = False
    backups: list[dict[str, Any]] = field(default_factory=list)
    pending_writes: list[PendingSkillWrite] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)


class CuratorConnector:
    """Parses the live Hermes Curator state under a Hermes home, read-only."""

    def __init__(self, hermes_home: Path, *, max_runs: int = 10) -> None:
        if isinstance(max_runs, bool) or not isinstance(max_runs, int) or not 1 <= max_runs <= 100:
            raise ValueError("max_runs must be between 1 and 100")
        home = Path(hermes_home).expanduser()
        if not home.is_absolute():
            raise ValueError("Hermes home must be an absolute path")
        if not home.is_dir():
            raise FileNotFoundError(f"Hermes home does not exist: {home}")
        self.hermes_home = home
        self.max_runs = max_runs

    @property
    def skills_dir(self) -> Path:
        return self.hermes_home / "skills"

    def snapshot(self) -> CuratorSnapshot:
        snapshot = CuratorSnapshot(hermes_home=str(self.hermes_home))
        self._load_usage(snapshot)
        self._load_scheduler(snapshot)
        self._load_runs(snapshot)
        self._load_ledger(snapshot)
        self._load_backups(snapshot)
        self._load_pending(snapshot)
        return snapshot

    def usage_rows(self) -> list[dict[str, Any]]:
        """Rows in the format accepted by :class:`hermes_improve.curator.CuratorAdapter`."""
        return [
            {
                "name": record.name,
                "use_count": record.use_count,
                "patch_count": record.patch_count,
                "activity_count": record.activity_count,
            }
            for record in self.snapshot().skills
        ]

    def pending_write(self, pending_id: str) -> PendingSkillWrite:
        if not isinstance(pending_id, str) or not _PENDING_ID.fullmatch(pending_id):
            raise ValueError("pending write ID must be 1-64 letters, digits, '_' or '-'")
        path = self.hermes_home / "pending" / "skills" / f"{pending_id}.json"
        raw = _read_bounded(path, _MAX_PENDING_BYTES)
        if raw is None:
            raise KeyError(pending_id)
        return _parse_pending(pending_id, _decode_json(raw, f"pending write {pending_id}"))

    def status(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        states: dict[str, int] = {}
        for record in snapshot.skills:
            states[record.state] = states.get(record.state, 0) + 1
        latest = snapshot.runs[0] if snapshot.runs else None
        return {
            "hermes_home": snapshot.hermes_home,
            "skills": {
                "total": len(snapshot.skills),
                "curator_managed": sum(record.curator_managed for record in snapshot.skills),
                "pinned": sum(record.pinned for record in snapshot.skills),
                "by_state": dict(sorted(states.items())),
                "never_used": sum(record.activity_count == 0 for record in snapshot.skills),
            },
            "scheduler": snapshot.scheduler if snapshot.scheduler is not None else "no data",
            "runs": {
                "read": len(snapshot.runs),
                "latest": None if latest is None else {
                    "run_id": latest.run_id,
                    "started_at": latest.started_at,
                    "counts": dict(latest.counts),
                    "llm_error": latest.llm_error,
                },
                "consolidations": sum(len(run.consolidated) for run in snapshot.runs),
                "prunings": sum(len(run.pruned) for run in snapshot.runs),
                "archives": sum(len(run.archived) for run in snapshot.runs),
            },
            "ledger": {
                "entries_read": snapshot.ledger_entries_read,
                "truncated_to_tail": snapshot.ledger_truncated,
                "by_action": dict(sorted(snapshot.ledger_actions.items())),
                "by_actor": dict(sorted(snapshot.ledger_actors.items())),
                "rollbacks": len(snapshot.ledger_rollbacks),
            },
            "backups": {
                "count": len(snapshot.backups),
                "latest": snapshot.backups[0] if snapshot.backups else None,
            },
            "pending_skill_writes": [
                {
                    "pending_id": item.pending_id,
                    "action": item.action,
                    "skill": item.skill,
                    "origin": item.origin,
                    "summary": item.summary,
                }
                for item in snapshot.pending_writes
            ],
            "rejected_records": snapshot.rejected,
        }

    def signals(self, snapshot: CuratorSnapshot | None = None) -> list[ImprovementSignal]:
        snapshot = snapshot or self.snapshot()
        signals: list[ImprovementSignal] = []
        for record in snapshot.skills:
            if record.state == "stale" and not record.pinned:
                signals.append(ImprovementSignal(
                    source="curator",
                    summary=f"Curator marked {record.name} stale",
                    issue=f"Decide whether {record.name} needs an improvement proposal or should stay on the archive path",
                    impact_score=0.3,
                    evidence=_record_evidence(record, "lifecycle_stale"),
                ))
            elif record.state == "archived" and record.use_count > 0:
                signals.append(ImprovementSignal(
                    source="curator",
                    summary=f"Curator archived {record.name} although it was used {record.use_count} times",
                    issue=f"Check whether archiving {record.name} removed a capability that sessions relied on",
                    impact_score=0.5,
                    evidence=_record_evidence(record, "lifecycle_archived_used"),
                ))
            if record.patch_count >= _PATCH_CHURN_MIN and record.patch_count >= record.use_count:
                signals.append(ImprovementSignal(
                    source="curator",
                    summary=f"{record.name} was patched {record.patch_count} times for {record.use_count} uses",
                    issue=f"Repeated in-place patches of {record.name} suggest an unstable skill; evaluate a reviewed candidate instead",
                    impact_score=min(1.0, record.patch_count / (record.patch_count + record.use_count)),
                    evidence=_record_evidence(record, "patch_churn"),
                ))
        for run in snapshot.runs:
            for item in run.consolidated:
                signals.append(ImprovementSignal(
                    source="curator",
                    summary=f"Curator consolidated {item['name']} into {item['into']}",
                    issue=f"Overlap between {item['name']} and {item['into']} must be evaluated as a SKILL_MERGE candidate",
                    impact_score=0.6,
                    evidence={"kind": "overlap_consolidation", "run_id": run.run_id, **item},
                ))
        for rollback in snapshot.ledger_rollbacks:
            signals.append(ImprovementSignal(
                source="curator",
                summary=f"A mutation of {rollback['skill']} was rolled back in the Curator ledger",
                issue=f"A reverted change to {rollback['skill']} indicates a regression the candidate path should have caught",
                impact_score=0.7,
                evidence={"kind": "ledger_rollback", **rollback},
            ))
        for pending in snapshot.pending_writes:
            signals.append(ImprovementSignal(
                source="curator",
                summary=f"Staged {pending.action} of {pending.skill} awaits conversion into a candidate",
                issue=f"Convert pending skill write {pending.pending_id} into an evaluated candidate instead of approving it in place",
                impact_score=0.5,
                evidence={
                    "kind": "pending_write", "pending_id": pending.pending_id,
                    "action": pending.action, "skill": pending.skill, "origin": pending.origin,
                },
            ))
        return signals

    # -- loaders ---------------------------------------------------------------

    def _load_usage(self, snapshot: CuratorSnapshot) -> None:
        raw = _read_bounded(self.skills_dir / ".usage.json", _MAX_USAGE_BYTES)
        if raw is None:
            return
        data = _decode_json(raw, "Curator usage file")
        if not isinstance(data, dict):
            raise ValueError("Curator usage file must contain a JSON object keyed by skill name")
        for name, record in sorted(data.items()):
            try:
                snapshot.skills.append(_parse_usage_record(name, record))
            except ValueError as exc:
                snapshot.rejected.append({"source": ".usage.json", "reason": str(exc)})

    def _load_scheduler(self, snapshot: CuratorSnapshot) -> None:
        raw = _read_bounded(self.skills_dir / ".curator_state", _MAX_STATE_BYTES)
        if raw is None:
            return
        data = _decode_json(raw, "Curator scheduler state")
        if not isinstance(data, dict):
            raise ValueError("Curator scheduler state must be a JSON object")
        result: dict[str, Any] = {}
        for key in ("last_run_at", "last_run_summary"):
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Curator scheduler state {key} must be a string or null")
            result[key] = None if value is None else _clean_text(value, 500)
        if result["last_run_at"] is not None:
            _parse_timestamp(result["last_run_at"], "last_run_at")
        duration = data.get("last_run_duration_seconds")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0):
            raise ValueError("Curator scheduler state last_run_duration_seconds must be non-negative")
        result["last_run_duration_seconds"] = duration
        paused = data.get("paused", False)
        if not isinstance(paused, bool):
            raise ValueError("Curator scheduler state paused must be boolean")
        result["paused"] = paused
        result["run_count"] = _counter(data.get("run_count", 0), "run_count")
        snapshot.scheduler = result

    def _load_runs(self, snapshot: CuratorSnapshot) -> None:
        root = self.hermes_home / "logs" / "curator"
        if not _is_real_directory(root):
            return
        names = sorted((entry.name for entry in os.scandir(root) if _RUN_DIR.fullmatch(entry.name)), reverse=True)
        for name in names[: self.max_runs]:
            run_dir = root / name
            if not _is_real_directory(run_dir):
                snapshot.rejected.append({"source": f"logs/curator/{name}", "reason": "run directory is a symbolic link or not a directory"})
                continue
            raw = _read_bounded(run_dir / "run.json", _MAX_RUN_BYTES)
            if raw is None:
                snapshot.rejected.append({"source": f"logs/curator/{name}", "reason": "run.json is missing"})
                continue
            try:
                snapshot.runs.append(_parse_run(name, _decode_json(raw, f"Curator run {name}")))
            except ValueError as exc:
                snapshot.rejected.append({"source": f"logs/curator/{name}/run.json", "reason": str(exc)})

    def _load_ledger(self, snapshot: CuratorSnapshot) -> None:
        tail = _read_tail(self.skills_dir / ".curator_ledger.jsonl", _MAX_LEDGER_TAIL_BYTES)
        if tail is None:
            return
        raw, truncated = tail
        lines = raw.decode("utf-8-sig", errors="replace").splitlines()
        if truncated and lines:
            lines = lines[1:]
        snapshot.ledger_truncated = truncated
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("ledger entry must be an object")
                action, actor, skill = entry.get("action"), entry.get("actor"), entry.get("skill")
                entry_id, timestamp = entry.get("id"), entry.get("ts")
                if not all(isinstance(value, str) and value for value in (action, actor, skill, entry_id, timestamp)):
                    raise ValueError("ledger entry requires id, ts, actor, action and skill strings")
                _parse_timestamp(timestamp, "ledger ts")
                if actor not in {"curator", "agent", "user"}:
                    raise ValueError(f"ledger actor {actor!r} is not a Hermes ledger actor")
                action = _clean_text(action, 64)
            except (json.JSONDecodeError, ValueError) as exc:
                snapshot.rejected.append({"source": f".curator_ledger.jsonl tail line {number}", "reason": str(exc)})
                continue
            snapshot.ledger_entries_read += 1
            snapshot.ledger_actions[action] = snapshot.ledger_actions.get(action, 0) + 1
            snapshot.ledger_actors[actor] = snapshot.ledger_actors.get(actor, 0) + 1
            if action == "rollback":
                safe_skill = skill if _safe_name(skill) else "[invalid skill name]"
                snapshot.ledger_rollbacks.append({
                    "entry_id": _clean_text(entry_id, 64), "skill": safe_skill, "ts": timestamp, "actor": actor,
                })

    def _load_backups(self, snapshot: CuratorSnapshot) -> None:
        root = self.skills_dir / ".curator_backups"
        if not _is_real_directory(root):
            return
        backups = []
        for entry in sorted(os.scandir(root), key=lambda item: item.name, reverse=True):
            if len(backups) >= _MAX_BACKUPS:
                break
            if entry.name == "blobs" or not _is_real_directory(Path(entry.path)):
                continue
            raw = _read_bounded(Path(entry.path) / "manifest.json", _MAX_MANIFEST_BYTES)
            if raw is None:
                continue
            try:
                manifest = _decode_json(raw, f"backup manifest {entry.name}")
                if not isinstance(manifest, dict) or not isinstance(manifest.get("created_at"), str):
                    raise ValueError("backup manifest requires a created_at string")
                _parse_timestamp(manifest["created_at"], "backup created_at")
                reason = manifest.get("reason", "")
                backups.append({
                    "id": _clean_text(entry.name, 64),
                    "created_at": manifest["created_at"],
                    "reason": _clean_text(reason if isinstance(reason, str) else "", 200),
                    "skill_files": _counter(manifest.get("skill_files", 0), "skill_files"),
                })
            except ValueError as exc:
                snapshot.rejected.append({"source": f".curator_backups/{entry.name}", "reason": str(exc)})
        snapshot.backups = sorted(backups, key=lambda item: item["created_at"], reverse=True)

    def _load_pending(self, snapshot: CuratorSnapshot) -> None:
        root = self.hermes_home / "pending" / "skills"
        if not _is_real_directory(root):
            return
        files = sorted(entry.name for entry in os.scandir(root) if entry.name.endswith(".json"))
        if len(files) > _MAX_PENDING_FILES:
            raise ValueError("pending skill write directory exceeds the 1,000-record limit")
        for name in files:
            pending_id = name[:-5]
            try:
                snapshot.pending_writes.append(self.pending_write(pending_id))
            except (KeyError, ValueError) as exc:
                snapshot.rejected.append({"source": f"pending/skills/{name}", "reason": str(exc)})


def _parse_usage_record(name: Any, record: Any) -> CuratorSkillRecord:
    if not isinstance(name, str) or not _safe_name(name):
        raise ValueError("skill name is not a bounded, clean, credential-free string")
    if not isinstance(record, dict):
        raise ValueError(f"usage record for {name} must be an object")
    state = record.get("state", "active")
    if state not in _LIFECYCLE_STATES:
        raise ValueError(f"usage record for {name} has unknown lifecycle state")
    pinned = record.get("pinned", False)
    if not isinstance(pinned, bool):
        raise ValueError(f"usage record for {name} has a non-boolean pinned flag")
    created_by = record.get("created_by")
    if created_by is not None and (
        not isinstance(created_by, str) or not re.fullmatch(r"[a-z_]{1,32}", created_by)
    ):
        raise ValueError(f"usage record for {name} has an invalid created_by marker")
    stamps = []
    for key in ("last_used_at", "last_viewed_at", "last_patched_at", "archived_at", "created_at"):
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"usage record for {name} has a non-string {key}")
        parsed = _parse_timestamp(value, f"{name} {key}")
        if key.startswith("last_"):
            stamps.append((parsed, value))
    return CuratorSkillRecord(
        name=name,
        state=state,
        pinned=pinned,
        created_by=created_by,
        use_count=_counter(record.get("use_count", 0), f"{name} use_count"),
        view_count=_counter(record.get("view_count", 0), f"{name} view_count"),
        patch_count=_counter(record.get("patch_count", 0), f"{name} patch_count"),
        patch_generation=_counter(record.get("patch_generation", 0), f"{name} patch_generation"),
        last_activity_at=max(stamps, key=lambda item: item[0])[1] if stamps else None,
        archived_at=record.get("archived_at"),
    )


def _parse_run(run_id: str, data: Any) -> CuratorRun:
    if not isinstance(data, dict):
        raise ValueError("run report must be a JSON object")
    started_at = data.get("started_at")
    if not isinstance(started_at, str):
        raise ValueError("run report requires started_at")
    _parse_timestamp(started_at, "run started_at")
    raw_counts = data.get("counts", {})
    if not isinstance(raw_counts, dict):
        raise ValueError("run report counts must be an object")
    counts = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z_]{1,64}", key):
            raise ValueError("run report count keys must be lowercase identifiers")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"run report count {key} must be an integer")
        counts[key] = value
    consolidated = []
    for item in _list(data, "consolidated"):
        if not isinstance(item, dict):
            raise ValueError("consolidated entries must be objects")
        name, into = item.get("name"), item.get("into")
        if not (isinstance(name, str) and _safe_name(name) and isinstance(into, str) and _safe_name(into)):
            raise ValueError("consolidated entries require clean name and into skill names")
        source = item.get("source", "")
        consolidated.append({
            "name": name, "into": into,
            "source": _clean_text(source if isinstance(source, str) else "", 120),
        })
    pruned = []
    for item in _list(data, "pruned"):
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not _safe_name(name):
            raise ValueError("pruned entries require a clean skill name")
        pruned.append(name)
    archived = []
    for name in _list(data, "archived"):
        if not isinstance(name, str) or not _safe_name(name):
            raise ValueError("archived entries must be clean skill names")
        archived.append(name)
    transitions = []
    for item in _list(data, "state_transitions"):
        if not isinstance(item, dict):
            raise ValueError("state transitions must be objects")
        name, before, after = item.get("name"), item.get("from"), item.get("to")
        if not (isinstance(name, str) and _safe_name(name) and before in _LIFECYCLE_STATES and after in _LIFECYCLE_STATES):
            raise ValueError("state transitions require a clean name and known lifecycle states")
        transitions.append({"name": name, "from": before, "to": after})
    llm_error = data.get("llm_error")
    if llm_error is not None and not isinstance(llm_error, str):
        raise ValueError("run report llm_error must be a string or null")
    return CuratorRun(
        run_id=run_id,
        started_at=started_at,
        counts=counts,
        consolidated=tuple(consolidated),
        pruned=tuple(pruned),
        archived=tuple(archived),
        state_transitions=tuple(transitions),
        llm_error=None if llm_error is None else _clean_text(llm_error, 500),
    )


def _parse_pending(pending_id: str, data: Any) -> PendingSkillWrite:
    if not isinstance(data, dict):
        raise ValueError("pending write must be a JSON object")
    if data.get("id") != pending_id:
        raise ValueError("pending write id does not match its file name")
    if data.get("subsystem") != "skills":
        raise ValueError("pending write is not a skills write")
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("pending write payload must be an object")
    action = payload.get("action")
    if action not in _PENDING_ACTIONS or data.get("action") != action:
        raise ValueError("pending write has an unknown or inconsistent skill_manage action")
    skill = payload.get("name")
    if not isinstance(skill, str) or not _safe_name(skill):
        raise ValueError("pending write requires a clean skill name")
    origin = data.get("origin", "foreground")
    if origin not in {"foreground", "background_review"}:
        raise ValueError("pending write origin must be foreground or background_review")
    created_at = data.get("created_at")
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)) or created_at < 0:
        raise ValueError("pending write created_at must be a non-negative epoch timestamp")
    summary = data.get("summary", "")
    for key in ("content", "file_path", "file_content", "old_string", "new_string", "absorbed_into", "category"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"pending write payload {key} must be a string")
    replace_all = payload.get("replace_all", False)
    if not isinstance(replace_all, bool):
        raise ValueError("pending write payload replace_all must be boolean")
    return PendingSkillWrite(
        pending_id=pending_id,
        action=action,
        skill=skill,
        origin=origin,
        summary=_clean_text(summary if isinstance(summary, str) else "", 300),
        created_at=float(created_at),
        payload=dict(payload),
    )


def _record_evidence(record: CuratorSkillRecord, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": record.name,
        "state": record.state,
        "pinned": record.pinned,
        "curator_managed": record.curator_managed,
        "use_count": record.use_count,
        "view_count": record.view_count,
        "patch_count": record.patch_count,
        "activity_count": record.activity_count,
        "last_activity_at": record.last_activity_at,
    }


def _list(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"run report {key} must be a list")
    return value


def _safe_name(name: str) -> bool:
    try:
        validate_curator_skill_name(name)
    except ValueError:
        return False
    return True


def _counter(value: Any, label: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNTER:
        raise ValueError(f"{label} must be a non-negative bounded integer")
    return value


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp") from exc


def _clean_text(value: str, limit: int) -> str:
    cleaned = "".join(character if ord(character) >= 32 else " " for character in value)[:limit]
    return "[REDACTED]" if contains_secret_material(cleaned) else cleaned


def _decode_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _is_real_directory(path: Path) -> bool:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(mode)


def _open_regular(path: Path) -> int | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"refusing to follow symbolic link: {path}") from exc
        raise
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"Curator state path is not a regular file: {path}")
    return descriptor


def _read_bounded(path: Path, limit: int) -> bytes | None:
    descriptor = _open_regular(path)
    if descriptor is None:
        return None
    with os.fdopen(descriptor, "rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{path.name} exceeds the {limit // 1024} KiB read limit")
    return raw


def _read_tail(path: Path, limit: int) -> tuple[bytes, bool] | None:
    descriptor = _open_regular(path)
    if descriptor is None:
        return None
    with os.fdopen(descriptor, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        truncated = size > limit
        if truncated:
            source.seek(size - limit)
        return source.read(limit), truncated
