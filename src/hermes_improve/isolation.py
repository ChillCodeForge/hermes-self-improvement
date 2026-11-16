"""Operating-system isolation layout and fail-closed boundary self-check.

`scripts/provision.sh` creates the Unix users, directories, modes, and POSIX
ACLs described by :data:`LAYOUT_ENTRIES`. This module re-derives every role's
effective access from the real filesystem (owner, group, mode bits, ACL
entries, and directory traversal along the full path) and compares it with
the expected matrix. Evaluate and promote refuse to run when a boundary is
violated. The check detects misconfiguration; the adversary-facing
enforcement is the kernel's permission model that the layout establishes.
"""

from __future__ import annotations

import errno
import grp
import json
import os
import pwd
import re
import stat
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

ROLES = ("agent", "evaluator", "promoter")
DEFAULT_LAYOUT_PATH = Path("/srv/hermes-improve/config/layout.json")
UNPROVISIONED_OPT_OUT = "unprovisioned"
_MAX_LAYOUT_BYTES = 64 * 1024
_MAX_WALK_ENTRIES = 200_000
_MAX_REPORTED_PER_RULE = 5
_SAFE_ROOT = re.compile(r"^/[A-Za-z0-9._/-]*$")
_SAFE_NAME = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
# Membership in these groups grants root-equivalent or credential access.
_PRIVILEGED_GROUPS = frozenset({"root", "sudo", "wheel", "admin", "adm", "disk", "shadow", "docker", "lxd"})
# The evaluator drives the Docker sandbox; docker membership is root-equivalent
# and therefore reported, but it is the documented exception for that role.
_ROLE_ALLOWED_PRIVILEGED = {"agent": frozenset(), "evaluator": frozenset({"docker"}), "promoter": frozenset()}

_PERM_BITS = {"r": 4, "w": 2, "x": 1}

# Repository paths that implement or configure the yardstick (spec §3). They
# must keep a CODEOWNERS rule so that merges require owner review.
PROTECTED_REPOSITORY_PATHS = (
    "/src/hermes_improve/evaluator.py",
    "/src/hermes_improve/verifier.py",
    "/src/hermes_improve/verifiers.py",
    "/src/hermes_improve/policy.py",
    "/src/hermes_improve/promotion.py",
    "/src/hermes_improve/security.py",
    "/src/hermes_improve/isolation.py",
    "/src/hermes_improve/role_handoff.py",
    "/src/hermes_improve/evaluation_signing.py",
    "/src/hermes_improve/benchmarks.py",
    "/src/hermes_improve/docker_runner.py",
    "/src/hermes_improve/statistics.py",
    "/src/hermes_improve/cli.py",
    "/benchmarks/",
    "/config/",
    "/scripts/",
    "/.github/",
    "/tests/security/",
    "/docs/security.md",
    "/pyproject.toml",
)

_ACL_USER_OBJ = 0x01
_ACL_USER = 0x02
_ACL_GROUP_OBJ = 0x04
_ACL_GROUP = 0x08
_ACL_MASK = 0x10
_ACL_OTHER = 0x20


@dataclass(frozen=True)
class LayoutEntry:
    key: str
    relative: str
    kind: str  # "dir" or "file"
    access: Mapping[str, str]
    recursive: bool = True


def _all(value: str) -> dict[str, str]:
    return {role: value for role in ROLES}


def _only(**grants: str) -> dict[str, str]:
    return {role: grants.get(role, "") for role in ROLES}


# Expected access per role. A cell lists exactly the permissions the role must
# hold on the entry itself; every other bit must be denied. For recursive
# entries no descendant may grant a bit outside the cell, and when the cell
# lacks "w" the role must not own any descendant (an owner can chmod).
LAYOUT_ENTRIES: tuple[LayoutEntry, ...] = (
    LayoutEntry("root", "", "dir", _all("rx"), recursive=False),
    LayoutEntry("install", "install", "dir", _all("rx")),
    LayoutEntry("config", "config", "dir", _all("rx")),
    LayoutEntry("layout_file", "config/layout.json", "file", _all("r")),
    LayoutEntry("keys", "keys", "dir", _all("rx")),
    LayoutEntry("evaluator_keys", "keys/evaluator", "dir", _only(evaluator="rx")),
    LayoutEntry("evaluator_private_key", "keys/evaluator/signing.pem", "file", _only(evaluator="r")),
    LayoutEntry("evaluator_public_key", "keys/evaluator.pub", "file", _all("r")),
    LayoutEntry("promoter_keys", "keys/promoter", "dir", _only(promoter="rx")),
    LayoutEntry("manifest_key", "keys/promoter/manifest.key", "file", _only(promoter="r")),
    LayoutEntry("brain", "brain", "dir", _all("rx"), recursive=False),
    LayoutEntry("production_repo", "brain/production.git", "dir", _only(agent="rx", evaluator="rx", promoter="rwx")),
    LayoutEntry("candidate_repo", "brain/candidates.git", "dir", _only(agent="rwx", evaluator="rx")),
    LayoutEntry("workspaces", "workspaces", "dir", _only(agent="rwx", evaluator="rx")),
    LayoutEntry("handoff", "handoff", "dir", _only(agent="rwx", evaluator="rx", promoter="rx")),
    LayoutEntry("releases", "releases", "dir", _only(agent="rx", evaluator="rx", promoter="rwx")),
    LayoutEntry("deploy", "deploy", "dir", _only(agent="rx", evaluator="rx", promoter="rwx")),
    LayoutEntry("benchmarks_public", "benchmarks/public", "dir", _all("rx")),
    LayoutEntry("holdout", "holdout", "dir", _only(evaluator="rx")),
    LayoutEntry("ledger", "ledger", "dir", _only(evaluator="rwx", promoter="rwx")),
    LayoutEntry("agent_state", "agent", "dir", _only(agent="rwx")),
    LayoutEntry("evaluator_state", "evaluator", "dir", _only(evaluator="rwx")),
    LayoutEntry("home_agent", "home/agent", "dir", _only(agent="rwx")),
    LayoutEntry("home_evaluator", "home/evaluator", "dir", _only(evaluator="rwx")),
    LayoutEntry("home_promoter", "home/promoter", "dir", _only(promoter="rwx")),
)
_ENTRY_BY_KEY = {entry.key: entry for entry in LAYOUT_ENTRIES}


@dataclass(frozen=True)
class Identity:
    role: str
    name: str
    uid: int
    gid: int
    groups: frozenset[int]


@dataclass(frozen=True)
class IsolationLayout:
    root: Path
    users: Mapping[str, str]
    ledger_group: str
    cli: Path
    source: Path | None = None

    def path(self, key: str) -> Path:
        entry = _ENTRY_BY_KEY[key]
        return self.root / entry.relative if entry.relative else self.root

    @property
    def deployment_link(self) -> Path:
        return self.path("deploy") / "current"

    @property
    def evaluator_mirror(self) -> Path:
        return self.path("evaluator_state") / "mirror.git"

    def identity(self, role: str) -> Identity:
        name = self.users[role]
        try:
            record = pwd.getpwnam(name)
        except KeyError as exc:
            raise PermissionError(f"isolation user {name!r} for role {role} does not exist") from exc
        return Identity(role, name, record.pw_uid, record.pw_gid,
                        frozenset(os.getgrouplist(name, record.pw_gid)))

    def require_paths(self, expected: Mapping[str, Path | str]) -> None:
        """Refuse CLI arguments that point outside the provisioned layout."""
        for key, value in expected.items():
            if key == "deployment_link":
                wanted = self.deployment_link
                actual = Path(value).parent.resolve() / Path(value).name
            else:
                wanted = self.path(key)
                actual = Path(value).resolve()
            if actual != wanted:
                raise PermissionError(
                    f"{key} must be the provisioned path {wanted}, not {actual}"
                )


@dataclass
class IsolationReport:
    role: str | None
    layout: str
    violations: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_entries: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "role": self.role,
            "layout": self.layout,
            "checked_entries": self.checked_entries,
            "violations": self.violations,
            "unverified": self.unverified,
            "warnings": self.warnings,
        }


def load_layout(path: Path | str, *, required_owner: int = 0) -> IsolationLayout:
    """Load the provisioned layout; it must be a root-owned, non-shared-writable file."""
    path = Path(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise PermissionError(f"isolation layout {path} does not exist") from exc
    if not stat.S_ISREG(info.st_mode):
        raise PermissionError(f"isolation layout {path} must be a regular file")
    if info.st_uid != required_owner or info.st_mode & 0o022:
        raise PermissionError(
            f"isolation layout {path} must be owned by uid {required_owner} and not group/other writable"
        )
    with path.open("rb") as source:
        raw = source.read(_MAX_LAYOUT_BYTES + 1)
    if len(raw) > _MAX_LAYOUT_BYTES:
        raise ValueError("isolation layout exceeds 64 KiB")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("isolation layout must be UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "root", "users", "ledger_group", "cli",
    }:
        raise ValueError("isolation layout has missing or unknown fields")
    if document["schema_version"] != 1:
        raise ValueError("unsupported isolation layout schema version")
    root = document["root"]
    if not isinstance(root, str) or not _SAFE_ROOT.fullmatch(root) or root == "/" or ".." in root.split("/"):
        raise ValueError("isolation layout root must be a safe absolute path")
    if os.path.realpath(root) != os.path.normpath(root):
        raise PermissionError("isolation layout root must not traverse symbolic links")
    users = document["users"]
    if not isinstance(users, dict) or set(users) != set(ROLES):
        raise ValueError("isolation layout must name exactly the agent, evaluator, and promoter users")
    names = [users[role] for role in ROLES]
    group = document["ledger_group"]
    if not all(isinstance(name, str) and _SAFE_NAME.fullmatch(name) for name in [*names, group]):
        raise ValueError("isolation layout user and group names must be safe Unix names")
    if len(set(names)) != len(names):
        raise ValueError("agent, evaluator, and promoter must be distinct users")
    cli = document["cli"]
    if not isinstance(cli, str) or not cli.startswith("/"):
        raise ValueError("isolation layout cli must be an absolute path")
    return IsolationLayout(Path(root), dict(users), group, Path(cli), path)


def parse_posix_acl(blob: bytes) -> list[tuple[int, int, int]]:
    """Decode a Linux ``system.posix_acl_access`` xattr into (tag, perm, id) tuples."""
    if len(blob) < 4 or (len(blob) - 4) % 8:
        raise ValueError("malformed POSIX ACL xattr")
    (version,) = struct.unpack_from("<I", blob, 0)
    if version != 2:
        raise ValueError(f"unsupported POSIX ACL xattr version {version}")
    return [struct.unpack_from("<HHI", blob, offset) for offset in range(4, len(blob), 8)]


def _read_acl(path: str) -> list[tuple[int, int, int]] | None:
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        raise RuntimeError("the isolation check requires Linux extended attribute support")
    try:
        return parse_posix_acl(getxattr(path, "system.posix_acl_access", follow_symlinks=False))
    except OSError as exc:
        if exc.errno in {errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP}:
            return None
        raise


def permission_bits(
    st_mode: int, st_uid: int, st_gid: int,
    acl: list[tuple[int, int, int]] | None, uid: int, groups: Iterable[int],
) -> int:
    """POSIX.1e access evaluation for one identity; returns r/w/x as 4/2/1 bits."""
    groups = frozenset(groups)
    if uid == 0:
        return 7
    mode = st_mode & 0o777
    if acl is None:
        if uid == st_uid:
            return (mode >> 6) & 7
        if st_gid in groups:
            return (mode >> 3) & 7
        return mode & 7
    entries = {tag: [] for tag in (_ACL_USER_OBJ, _ACL_USER, _ACL_GROUP_OBJ, _ACL_GROUP, _ACL_MASK, _ACL_OTHER)}
    for tag, perm, ident in acl:
        if tag not in entries:
            raise ValueError(f"unknown POSIX ACL tag {tag:#x}")
        entries[tag].append((perm & 7, ident))
    mask = entries[_ACL_MASK][0][0] if entries[_ACL_MASK] else 7
    if uid == st_uid:
        return entries[_ACL_USER_OBJ][0][0] if entries[_ACL_USER_OBJ] else (mode >> 6) & 7
    for perm, ident in entries[_ACL_USER]:
        if ident == uid:
            return perm & mask
    matched = False
    bits = 0
    if st_gid in groups and entries[_ACL_GROUP_OBJ]:
        matched = True
        bits |= entries[_ACL_GROUP_OBJ][0][0] & mask
    for perm, ident in entries[_ACL_GROUP]:
        if ident in groups:
            matched = True
            bits |= perm & mask
    if matched:
        return bits
    return entries[_ACL_OTHER][0][0] if entries[_ACL_OTHER] else mode & 7


def _bits_text(bits: int, kind: str) -> str:
    letters = "rw" if kind == "file" else "rwx"
    text = "".join(letter for letter in letters if bits & _PERM_BITS[letter])
    return text or "-"


class _Inspector:
    """Caches stat/ACL lookups and computes effective access including traversal."""

    def __init__(self, identities: Mapping[str, Identity]) -> None:
        self.identities = identities
        self._meta: dict[str, tuple[os.stat_result, list | None] | OSError] = {}

    def meta(self, path: str) -> tuple[os.stat_result, list | None]:
        cached = self._meta.get(path)
        if cached is None:
            try:
                info = os.lstat(path)
                cached = (info, None if stat.S_ISLNK(info.st_mode) else _read_acl(path))
            except OSError as exc:
                cached = exc
            self._meta[path] = cached
        if isinstance(cached, OSError):
            raise cached
        return cached

    def own_bits(self, path: str, identity: Identity) -> int:
        info, acl = self.meta(path)
        return permission_bits(info.st_mode, info.st_uid, info.st_gid, acl, identity.uid, identity.groups)

    def can_traverse_to(self, path: str, identity: Identity) -> bool | None:
        """Whether every ancestor grants search; None if an ancestor is not inspectable."""
        parent = os.path.dirname(path)
        chain = []
        while True:
            chain.append(parent)
            if parent == "/":
                break
            parent = os.path.dirname(parent)
        for ancestor in reversed(chain):
            try:
                if not self.own_bits(ancestor, identity) & 1:
                    return False
            except PermissionError:
                return None
        return True


def _privileged_group_names(gids: Iterable[int]) -> set[str]:
    names = set()
    for gid in gids:
        try:
            names.add(grp.getgrgid(gid).gr_name)
        except KeyError:
            continue
    return names & _PRIVILEGED_GROUPS


def _check_identities(layout: IsolationLayout, report: IsolationReport) -> dict[str, Identity]:
    identities: dict[str, Identity] = {}
    for role in ROLES:
        try:
            identities[role] = layout.identity(role)
        except PermissionError as exc:
            report.violations.append(str(exc))
    uids = [identity.uid for identity in identities.values()]
    if len(set(uids)) != len(uids):
        report.violations.append("role users must have distinct UIDs")
    for role, identity in identities.items():
        if identity.uid == 0 or identity.gid == 0:
            report.violations.append(f"{role} user {identity.name} must not be root or in the root group")
        privileged = _privileged_group_names(identity.groups)
        forbidden = privileged - _ROLE_ALLOWED_PRIVILEGED[role]
        if forbidden:
            report.violations.append(
                f"{role} user {identity.name} is in privileged groups: {', '.join(sorted(forbidden))}"
            )
        for allowed in sorted(privileged & _ROLE_ALLOWED_PRIVILEGED[role]):
            report.warnings.append(f"{role} user {identity.name} is in root-equivalent group {allowed}")
        other_primary = {
            identities[other].gid for other in identities if other != role
        } & identity.groups
        if other_primary:
            report.violations.append(f"{role} user {identity.name} is a member of another role's primary group")
    try:
        ledger_gid = grp.getgrnam(layout.ledger_group).gr_gid
    except KeyError:
        report.violations.append(f"ledger group {layout.ledger_group!r} does not exist")
    else:
        if "agent" in identities and ledger_gid in identities["agent"].groups:
            report.violations.append("agent user must not be a member of the ledger group")
    return identities


def _check_root_ancestors(layout: IsolationLayout, inspector: _Inspector, report: IsolationReport) -> None:
    """No role may be able to rename or replace the layout root through a parent directory."""
    path = str(layout.root)
    while path != "/":
        parent = os.path.dirname(path)
        try:
            info, _ = inspector.meta(parent)
        except OSError as exc:
            report.unverified.append(f"cannot inspect layout ancestor {parent}: {exc.strerror}")
            path = parent
            continue
        sticky = bool(info.st_mode & stat.S_ISVTX)
        for role, identity in inspector.identities.items():
            if inspector.own_bits(parent, identity) & 2 and (not sticky or info.st_uid == identity.uid):
                report.violations.append(f"{role} can replace the layout root through writable ancestor {parent}")
        path = parent


def _check_entry(
    entry: LayoutEntry, path: str, inspector: _Inspector, report: IsolationReport, budget: list[int],
) -> None:
    try:
        info, _ = inspector.meta(path)
    except FileNotFoundError:
        report.violations.append(f"{entry.key} is missing: {path}")
        return
    except PermissionError:
        for role, identity in inspector.identities.items():
            if inspector.can_traverse_to(path, identity) is not False and entry.access[role]:
                report.unverified.append(f"{entry.key} is not inspectable by this identity")
                break
        return
    expected_type = stat.S_ISDIR if entry.kind == "dir" else stat.S_ISREG
    if not expected_type(info.st_mode):
        report.violations.append(f"{entry.key} must be a real {entry.kind} (not a symlink or other type): {path}")
        return
    letters = "rw" if entry.kind == "file" else "rwx"
    for role, identity in inspector.identities.items():
        traversable = inspector.can_traverse_to(path, identity)
        if traversable is None:
            report.unverified.append(f"{entry.key}: ancestors not inspectable for {role}")
            continue
        bits = inspector.own_bits(path, identity) if traversable else 0
        wanted = sum(_PERM_BITS[letter] for letter in entry.access[role])
        actual = bits & sum(_PERM_BITS[letter] for letter in letters)
        if actual != wanted:
            report.violations.append(
                f"{role} ({identity.name}) has {_bits_text(actual, entry.kind)} on {entry.key} "
                f"{path}; expected {entry.access[role] or '-'}"
            )
        if "w" not in entry.access[role] and info.st_uid == identity.uid:
            report.violations.append(f"{role} ({identity.name}) owns protected {entry.key} {path}")
    report.checked_entries += 1
    if entry.kind == "dir" and entry.recursive:
        _walk_subtree(entry, path, inspector, report, budget)


def _walk_subtree(
    entry: LayoutEntry, top: str, inspector: _Inspector, report: IsolationReport, budget: list[int],
) -> None:
    limited = {
        role: identity for role, identity in inspector.identities.items()
        if entry.access[role] != "rwx"
    }
    if not limited:
        return
    reported: dict[str, int] = {}
    stack = [top]
    while stack:
        directory = stack.pop()
        try:
            children = list(os.scandir(directory))
        except PermissionError:
            for role, identity in limited.items():
                if entry.access[role] and inspector.own_bits(directory, identity) & 5:
                    report.unverified.append(
                        f"{entry.key}: {directory} is not listable by this identity; {role} access below it unverified"
                    )
            continue
        for child in children:
            budget[0] -= 1
            if budget[0] < 0:
                raise PermissionError(
                    f"isolation check exceeded {_MAX_WALK_ENTRIES} entries; refusing to report an unverified layout"
                )
            try:
                info, _ = inspector.meta(child.path)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode):
                continue
            is_dir = stat.S_ISDIR(info.st_mode)
            for role, identity in limited.items():
                allowed = sum(_PERM_BITS[letter] for letter in entry.access[role])
                if not entry.access[role]:
                    # The role cannot reach the subtree at all (checked on the
                    # top-level entry); descendants are unreachable to it.
                    continue
                bits = inspector.own_bits(child.path, identity)
                if not is_dir:
                    bits &= 6
                excess = bits & ~allowed
                owned = "w" not in entry.access[role] and info.st_uid == identity.uid
                if (excess or owned) and reported.get(role, 0) < _MAX_REPORTED_PER_RULE:
                    reported[role] = reported.get(role, 0) + 1
                    problem = "owns" if owned else f"has {_bits_text(excess, 'dir' if is_dir else 'file')} on"
                    report.violations.append(f"{role} ({identity.name}) {problem} {child.path} inside {entry.key}")
            report.checked_entries += 1
            if is_dir:
                stack.append(child.path)


def _check_deployment(layout: IsolationLayout, inspector: _Inspector, report: IsolationReport) -> None:
    link = layout.deployment_link
    if not os.path.lexists(link):
        report.warnings.append("no active production release (deployment link absent)")
        return
    if not os.path.islink(link):
        report.violations.append(f"deployment pointer {link} must be a symlink to a release")
        return
    target = Path(os.path.realpath(link))
    if target.parent != layout.path("releases") or not target.is_dir():
        report.violations.append(f"deployment pointer resolves outside the release root: {target}")


def check_production_checkout(
    path: Path | str, layout: IsolationLayout | None, report: IsolationReport, *, role: str | None,
) -> None:
    """A Hermes deployment checkout must be read-only for everyone but the promoter."""
    top = os.path.realpath(path)
    if not os.path.isdir(top):
        report.violations.append(f"production checkout {top} is not a directory")
        return
    identities: dict[str, Identity] = {}
    if layout is not None:
        for name in ("agent", "evaluator"):
            try:
                identities[name] = layout.identity(name)
            except PermissionError as exc:
                report.violations.append(str(exc))
    inspector = _Inspector(identities)
    check_live = role != "promoter" and os.geteuid() != 0
    count = 0
    for directory, subdirs, files in os.walk(top):
        for name in [".", *subdirs, *files]:
            entry_path = directory if name == "." else os.path.join(directory, name)
            count += 1
            if count > _MAX_WALK_ENTRIES:
                report.violations.append("production checkout too large to verify")
                return
            if os.path.islink(entry_path):
                report.violations.append(f"production checkout contains a symlink: {entry_path}")
                continue
            if check_live and os.access(entry_path, os.W_OK, effective_ids=True):
                report.violations.append(f"production checkout entry is writable by this process: {entry_path}")
            for other, identity in identities.items():
                info, _ = inspector.meta(entry_path)
                if inspector.own_bits(entry_path, identity) & 2 or info.st_uid == identity.uid:
                    report.violations.append(f"production checkout entry is modifiable by {other}: {entry_path}")
    report.checked_entries += count


def _check_current_process(
    layout: IsolationLayout, role: str, identities: Mapping[str, Identity],
    inspector: _Inspector, report: IsolationReport,
) -> None:
    identity = identities.get(role)
    if identity is None:
        return
    if os.geteuid() != identity.uid or os.getuid() != identity.uid:
        report.violations.append(
            f"process uid {os.getuid()}/euid {os.geteuid()} is not the {role} user {identity.name} ({identity.uid})"
        )
        return
    if os.getegid() != identity.gid:
        report.violations.append(f"process egid {os.getegid()} is not the {role} primary group {identity.gid}")
    extra = set(os.getgroups()) - identity.groups - {identity.gid}
    if extra:
        report.violations.append(f"process holds supplementary groups not assigned to {identity.name}: {sorted(extra)}")
    # Cross-check the static model against the kernel's own decision.
    for entry in LAYOUT_ENTRIES:
        path = str(layout.path(entry.key))
        try:
            inspector.meta(path)
        except FileNotFoundError:
            continue  # already reported as missing
        except PermissionError:
            pass  # not inspectable here, so the kernel must deny every bit below
        for letter in ("r", "w", "x") if entry.kind == "dir" else ("r", "w"):
            mode = {"r": os.R_OK, "w": os.W_OK, "x": os.X_OK}[letter]
            granted = os.access(path, mode, effective_ids=True)
            if granted != (letter in entry.access[role]):
                report.violations.append(
                    f"kernel {'grants' if granted else 'denies'} {letter} on {entry.key} to {role}; "
                    f"expected {entry.access[role] or '-'}"
                )


def check_isolation(
    layout: IsolationLayout, role: str | None = None, *, production: Iterable[Path | str] = (),
) -> IsolationReport:
    if role is not None and role not in ROLES:
        raise ValueError(f"unknown isolation role {role!r}")
    report = IsolationReport(role, str(layout.source or layout.root))
    identities = _check_identities(layout, report)
    inspector = _Inspector(identities)
    _check_root_ancestors(layout, inspector, report)
    budget = [_MAX_WALK_ENTRIES]
    for entry in LAYOUT_ENTRIES:
        _check_entry(entry, str(layout.path(entry.key)), inspector, report, budget)
    _check_deployment(layout, inspector, report)
    if os.path.islink(layout.deployment_link) and not report.violations:
        check_production_checkout(layout.deployment_link, layout, report, role=role)
    for path in production:
        check_production_checkout(path, layout, report, role=role)
    if role is not None:
        _check_current_process(layout, role, identities, inspector, report)
    return report


def resolve_layout_path(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get("HERMES_IMPROVE_LAYOUT") or DEFAULT_LAYOUT_PATH)


def enforce_isolation(role: str, environ: Mapping[str, str] | None = None) -> IsolationLayout | None:
    """Fail closed unless the provisioned layout verifies for ``role``.

    Without a layout file the command is refused, except when the operator
    explicitly sets ``HERMES_IMPROVE_ISOLATION=unprovisioned`` for development
    or single-user testing; that mode prints a warning and gives no OS-level
    separation guarantees.
    """
    environ = os.environ if environ is None else environ
    path = resolve_layout_path(environ)
    if not path.exists():
        if environ.get("HERMES_IMPROVE_ISOLATION") == UNPROVISIONED_OPT_OUT:
            print(
                f"warning: running {role} command without a provisioned isolation layout ({path}); "
                "no operating-system role separation is verified",
                file=sys.stderr,
            )
            return None
        raise PermissionError(
            f"no provisioned isolation layout at {path}; run scripts/provision.sh or set "
            "HERMES_IMPROVE_LAYOUT (HERMES_IMPROVE_ISOLATION=unprovisioned only for development)"
        )
    layout = load_layout(path)
    report = check_isolation(layout, role)
    if not report.ok:
        raise PermissionError("isolation boundary violated: " + "; ".join(report.violations))
    return layout
