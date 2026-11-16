"""Hermes home fixture shaped like real Hermes Curator state.

Shapes follow the Hermes Agent source: ``tools/skill_usage.py`` (``.usage.json``
records from ``_empty_record``), ``agent/curator.py`` (``.curator_state`` and
``logs/curator/<stamp>/run.json``), ``tools/skill_ledger.py``
(``.curator_ledger.jsonl`` entries from ``append_entry``),
``agent/curator_backup.py`` (backup ``manifest.json``) and
``tools/write_approval.py`` (``pending/skills/<id>.json`` from ``stage_write``).
"""
from __future__ import annotations

import json
from pathlib import Path


def usage_record(**overrides):
    record = {
        "archived_at": None,
        "created_at": "2026-09-19T09:13:54.124670+00:00",
        "created_by": None,
        "first_seen_at": None,
        "last_patched_at": None,
        "last_reused_patch_generation": 0,
        "last_used_at": None,
        "last_viewed_at": None,
        "patch_count": 0,
        "patch_generation": 0,
        "pinned": False,
        "state": "active",
        "use_count": 0,
        "view_count": 0,
    }
    record.update(overrides)
    return record


def build_hermes_home(root: Path) -> Path:
    home = root / "hermes-home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    (skills / ".usage.json").write_text(json.dumps({
        "airtable": usage_record(),
        "claude-code": usage_record(
            created_by="agent", use_count=2, view_count=4, patch_count=5, patch_generation=5,
            last_used_at="2026-10-01T10:00:00+00:00", last_patched_at="2026-10-02T10:00:00+00:00",
        ),
        "old-notes": usage_record(
            created_by="agent", state="stale", use_count=1, last_used_at="2026-08-01T10:00:00+00:00",
        ),
        "retired-helper": usage_record(
            created_by="agent", state="archived", use_count=3,
            archived_at="2026-10-20T10:00:00+00:00", last_used_at="2026-09-01T10:00:00+00:00",
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    (skills / ".curator_state").write_text(json.dumps({
        "last_report_path": str(home / "logs" / "curator" / "20261109-102239"),
        "last_run_at": "2026-11-09T10:22:39.710128+00:00",
        "last_run_duration_seconds": 0.98,
        "last_run_summary": "auto: 1 archived; llm: 1 consolidated",
        "last_run_summary_shown_at": None,
        "paused": False,
        "run_count": 8,
    }, indent=2), encoding="utf-8")
    run_dir = home / "logs" / "curator" / "20261109-102239"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "started_at": "2026-11-09T10:22:39.710128+00:00",
        "duration_seconds": 0.98,
        "model": "",
        "provider": "",
        "auto_transitions": {"marked_stale": 1, "archived": 1, "reactivated": 0, "checked": 4, "seeded": 0},
        "counts": {
            "before": 4, "after": 3, "delta": -1, "archived_this_run": 1, "added_this_run": 0,
            "consolidated_this_run": 1, "pruned_this_run": 0, "state_transitions": 1,
            "cron_jobs_rewritten": 0, "tool_calls_total": 2,
        },
        "tool_call_counts": {"skill_manage": 2},
        "archived": ["retired-helper"],
        "consolidated": [{
            "name": "retired-helper", "into": "claude-code",
            "source": "absorbed_into (model-declared at delete)", "reason": "same umbrella",
        }],
        "pruned": [],
        "pruned_names": [],
        "added": [],
        "state_transitions": [{"name": "old-notes", "from": "active", "to": "stale"}],
        "cron_rewrites": {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0},
        "llm_final": "",
        "llm_summary": "1 consolidated",
        "llm_error": None,
        "tool_calls": [],
    }, indent=2), encoding="utf-8")
    ledger_entries = [
        {"id": "4b7d257f8fb5", "ts": "2026-09-16T15:17:21.146574+00:00", "actor": "agent", "action": "patch",
         "skill": "claude-code", "evidence": {"session_id": "20260916_170244_f29a78"},
         "before": [{"path": str(skills / "claude-code" / "SKILL.md"), "sha256": "3f" * 32}],
         "after": [{"path": str(skills / "claude-code" / "SKILL.md"), "sha256": "c4" * 32}]},
        {"id": "aa11bb22cc33", "ts": "2026-09-17T15:17:21+00:00", "actor": "user", "action": "pre-rollback",
         "skill": "claude-code", "evidence": {"rollback_target": "4b7d257f8fb5"}, "before": [], "after": []},
        {"id": "dd44ee55ff66", "ts": "2026-09-17T15:17:22+00:00", "actor": "user", "action": "rollback",
         "skill": "claude-code", "evidence": {"rollback_target": "4b7d257f8fb5", "restored": 1, "removed": 0},
         "before": [], "after": []},
    ]
    (skills / ".curator_ledger.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in ledger_entries), encoding="utf-8",
    )
    backup = skills / ".curator_backups" / "2026-11-09T10-22-39Z"
    backup.mkdir(parents=True)
    (backup / "manifest.json").write_text(json.dumps({
        "id": backup.name, "reason": "pre-curator-run", "created_at": "2026-11-09T10:22:39+00:00",
        "archive": "skills.tar.gz", "archive_bytes": 1024, "skill_files": 4,
        "cron_jobs": {"backed_up": False, "jobs_count": 0, "reason": "no cron/jobs.json present"},
    }, indent=2), encoding="utf-8")
    pending = home / "pending" / "skills"
    pending.mkdir(parents=True)
    write_pending(pending, "1a2b3c4d", {
        "action": "edit", "name": "demo",
        "content": "---\nname: demo\ndescription: Improved demo\n---\nUse the improved steps.\n",
    }, summary="rewrite 'demo' - Improved demo (64 chars)", origin="background_review")
    return home


def write_pending(directory: Path, pending_id: str, payload: dict, *, summary: str = "", origin: str = "foreground") -> Path:
    path = directory / f"{pending_id}.json"
    path.write_text(json.dumps({
        "id": pending_id, "subsystem": "skills", "action": payload.get("action", ""),
        "summary": summary, "origin": origin, "created_at": 1762683759.71, "payload": payload,
    }), encoding="utf-8")
    return path
