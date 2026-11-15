#!/usr/bin/env bash
# Provision the Generator / Evaluator / Promoter operating-system isolation layout.
#
# Idempotent; Linux only; must run as root. Creates the role users and groups,
# the directory layout with owners, modes, and POSIX ACLs that match
# hermes_improve.isolation.LAYOUT_ENTRIES, signing keys, per-role launcher
# environments, and finally runs `hermes-improve security check` as root.
#
# Usage:
#   scripts/provision.sh [--root DIR] (--cli PATH | --install-from SRC) [--python PY]
#
#   --root DIR          layout root (default /srv/hermes-improve)
#   --cli PATH          existing root-owned hermes-improve executable to launch for every role
#   --install-from SRC  create DIR/install/venv and pip-install SRC into it (root-owned)
#   --python PY         interpreter with `cryptography` for evaluator key generation
#                       (default: the install venv's python, else python3)
set -euo pipefail
umask 022

ROOT=/srv/hermes-improve
CLI=""
INSTALL_FROM=""
PYTHON=""
AGENT=hermes-agent
EVALUATOR=hermes-evaluator
PROMOTER=hermes-promoter
LEDGER_GROUP=hermes-ledger
NOLOGIN=/usr/sbin/nologin

die() { echo "provision: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
    --cli) CLI="${2:?--cli needs a value}"; shift 2 ;;
    --install-from) INSTALL_FROM="${2:?--install-from needs a value}"; shift 2 ;;
    --python) PYTHON="${2:?--python needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == Linux ]] || die "Linux is required"
[[ "$(id -u)" -eq 0 ]] || die "must run as root"
for tool in useradd groupadd usermod getent setfacl getfacl git setpriv stat base64 head; do
  command -v "$tool" >/dev/null || die "required tool missing: $tool (install acl, util-linux, passwd, git)"
done
[[ "$ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$ROOT" != / && "/$ROOT/" != */../* ]] || die "unsafe --root: $ROOT"
ROOT="${ROOT%/}"
[[ -z "$CLI" || -z "$INSTALL_FROM" ]] || die "use either --cli or --install-from"
[[ -x /usr/sbin/nologin ]] || NOLOGIN=/bin/false

# --- groups and users -------------------------------------------------------
ensure_group() {
  getent group "$1" >/dev/null || groupadd --system "$1"
}
ensure_user() {
  local name="$1"
  ensure_group "$name"
  if ! getent passwd "$name" >/dev/null; then
    useradd --system --gid "$name" --no-create-home --home-dir /nonexistent --shell "$NOLOGIN" "$name"
  fi
  [[ "$(id -u "$name")" -ne 0 ]] || die "$name must not have UID 0"
  [[ "$(id -gn "$name")" == "$name" ]] || die "$name must have primary group $name"
}
for user in "$AGENT" "$EVALUATOR" "$PROMOTER"; do ensure_user "$user"; done
ensure_group "$LEDGER_GROUP"
usermod -a -G "$LEDGER_GROUP" "$EVALUATOR"
usermod -a -G "$LEDGER_GROUP" "$PROMOTER"

# No role may hold sudo or other root-equivalent groups; evaluator may hold docker
# (documented as root-equivalent, required by the Docker sandbox runner).
privileged='^(root|sudo|wheel|admin|adm|disk|shadow|docker|lxd)$'
for user in "$AGENT" "$EVALUATOR" "$PROMOTER"; do
  for group in $(id -nG "$user"); do
    if [[ "$group" =~ $privileged ]] && ! [[ "$user" == "$EVALUATOR" && "$group" == docker ]]; then
      die "$user is a member of privileged group $group; remove it before provisioning"
    fi
  done
  for other in "$AGENT" "$EVALUATOR" "$PROMOTER"; do
    if [[ "$other" != "$user" ]] && id -nG "$user" | tr ' ' '\n' | grep -qx "$other"; then
      die "$user must not be a member of the $other group"
    fi
  done
done
if id -nG "$AGENT" | tr ' ' '\n' | grep -qx "$LEDGER_GROUP"; then
  die "$AGENT must not be a member of $LEDGER_GROUP"
fi
for sudoers in /etc/sudoers /etc/sudoers.d/*; do
  [[ -f "$sudoers" ]] || continue
  if grep -Eq "(^|[^A-Za-z0-9_-])($AGENT|$EVALUATOR|$PROMOTER)([^A-Za-z0-9_-]|$)" "$sudoers"; then
    die "$sudoers grants rights to a Hermes role user; remove that entry"
  fi
done
if command -v sudo >/dev/null; then
  for user in "$AGENT" "$EVALUATOR" "$PROMOTER"; do
    if sudo -n -l -U "$user" 2>/dev/null | grep -q "may run"; then
      die "$user has sudo rights"
    fi
  done
fi

# --- helpers ----------------------------------------------------------------
ensure_dir() {  # path owner group mode
  local path="$1"
  [[ ! -L "$path" ]] || die "$path must not be a symlink"
  mkdir -p "$path"
  chown "$2:$3" "$path"
  chmod "$4" "$path"
}
reset_acl() {  # path: remove extended and default ACL entries recursively
  setfacl -R -b "$1"
  find "$1" -type d -exec setfacl -k {} +
}
grant_readers() {  # path user...: recursive rX plus inherited default ACL
  local path="$1"; shift
  local spec="" default="d:u::rwx,d:g::r-x,d:o::---"
  for reader in "$@"; do
    spec+="${spec:+,}u:$reader:rX"
    default+=",d:u:$reader:r-x"
  done
  setfacl -R -m "$spec" "$path"
  find "$path" -type d -exec setfacl -m "$default" {} +
}
init_bare_repo() {  # path owner
  if [[ ! -e "$1/HEAD" ]]; then
    (cd / && setpriv --reuid="$2" --regid="$2" --init-groups --reset-env -- \
      env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git init --quiet --bare -b main "$1")
  fi
}
tree_owner_mode() {  # path owner group: owner-only writes inside a tree
  chown -R "$2:$3" "$1"
  chmod -R u+rwX,go-w "$1"
}

# --- layout -----------------------------------------------------------------
parent="$(dirname "$ROOT")"
mkdir -p "$parent"
ensure_dir "$ROOT" root root 0755
ensure_dir "$ROOT/install" root root 0755
ensure_dir "$ROOT/config" root root 0755
ensure_dir "$ROOT/config/roles" root root 0755
ensure_dir "$ROOT/keys" root root 0755
ensure_dir "$ROOT/keys/evaluator" root "$EVALUATOR" 0750
ensure_dir "$ROOT/keys/promoter" root "$PROMOTER" 0750
ensure_dir "$ROOT/brain" root root 0755
ensure_dir "$ROOT/brain/production.git" "$PROMOTER" "$PROMOTER" 0750
ensure_dir "$ROOT/brain/candidates.git" "$AGENT" "$AGENT" 0750
ensure_dir "$ROOT/workspaces" "$AGENT" "$AGENT" 0750
ensure_dir "$ROOT/handoff" "$AGENT" "$AGENT" 0750
ensure_dir "$ROOT/releases" "$PROMOTER" "$PROMOTER" 0755
ensure_dir "$ROOT/deploy" "$PROMOTER" "$PROMOTER" 0755
ensure_dir "$ROOT/benchmarks" root root 0755
ensure_dir "$ROOT/benchmarks/public" root root 0755
ensure_dir "$ROOT/holdout" root "$EVALUATOR" 0750
ensure_dir "$ROOT/ledger" root "$LEDGER_GROUP" 3770
ensure_dir "$ROOT/ledger/artifacts" root "$LEDGER_GROUP" 3770
ensure_dir "$ROOT/agent" "$AGENT" "$AGENT" 0700
ensure_dir "$ROOT/evaluator" "$EVALUATOR" "$EVALUATOR" 0700
ensure_dir "$ROOT/home" root root 0755
ensure_dir "$ROOT/home/agent" "$AGENT" "$AGENT" 0700
ensure_dir "$ROOT/home/evaluator" "$EVALUATOR" "$EVALUATOR" 0700
ensure_dir "$ROOT/home/promoter" "$PROMOTER" "$PROMOTER" 0700

# Protected components: code, configuration, public ground truth -> root only.
tree_owner_mode "$ROOT/install" root root
tree_owner_mode "$ROOT/config" root root
tree_owner_mode "$ROOT/benchmarks/public" root root
# Sealed holdout: evaluator read-only, nobody else.
chown -R "root:$EVALUATOR" "$ROOT/holdout"
find "$ROOT/holdout" -type d -exec chmod 0750 {} +
find "$ROOT/holdout" -type f -exec chmod 0640 {} +
# Production: promoter-owned, never writable by group/other.
tree_owner_mode "$ROOT/brain/production.git" "$PROMOTER" "$PROMOTER"
chmod -R go-w "$ROOT/releases" "$ROOT/deploy"
find "$ROOT/releases" "$ROOT/deploy" -mindepth 1 ! -type l ! -user "$PROMOTER" -exec chown -h "$PROMOTER" {} +
# Agent-owned areas.
for dir in brain/candidates.git workspaces handoff agent home/agent; do
  tree_owner_mode "$ROOT/$dir" "$AGENT" "$AGENT"
done
tree_owner_mode "$ROOT/evaluator" "$EVALUATOR" "$EVALUATOR"
tree_owner_mode "$ROOT/home/evaluator" "$EVALUATOR" "$EVALUATOR"
tree_owner_mode "$ROOT/home/promoter" "$PROMOTER" "$PROMOTER"

for dir in install config keys brain/production.git brain/candidates.git workspaces handoff \
           releases deploy benchmarks/public holdout ledger agent evaluator home; do
  reset_acl "$ROOT/$dir"
done
grant_readers "$ROOT/brain/production.git" "$AGENT" "$EVALUATOR"
grant_readers "$ROOT/brain/candidates.git" "$EVALUATOR"
grant_readers "$ROOT/workspaces" "$EVALUATOR"
grant_readers "$ROOT/handoff" "$EVALUATOR" "$PROMOTER"
# Restore modes that the ACL mask calculation may have widened or narrowed.
chmod 0750 "$ROOT/brain/production.git" "$ROOT/brain/candidates.git" "$ROOT/workspaces" "$ROOT/handoff"

# Ledger: shared by evaluator and promoter, root-owned database files, sticky +
# setgid directory so neither role can delete or replace the other's files.
for db in experiments.sqlite3 monitoring.sqlite3; do
  [[ -e "$ROOT/ledger/$db" ]] || : > "$ROOT/ledger/$db"
  chown "root:$LEDGER_GROUP" "$ROOT/ledger/$db"
  chmod 0660 "$ROOT/ledger/$db"
done
find "$ROOT/ledger" -type d -exec setfacl -m "d:u::rwx,d:g::rwx,d:o::---" {} +
chmod -R o-rwx "$ROOT/ledger"

init_bare_repo "$ROOT/brain/production.git" "$PROMOTER"
init_bare_repo "$ROOT/brain/candidates.git" "$AGENT"
init_bare_repo "$ROOT/evaluator/mirror.git" "$EVALUATOR"

# --- protected install ------------------------------------------------------
if [[ -n "$INSTALL_FROM" ]]; then
  [[ -d "$INSTALL_FROM" ]] || die "--install-from must be a source directory"
  if [[ ! -x "$ROOT/install/venv/bin/hermes-improve" ]]; then
    python3 -m venv "$ROOT/install/venv"
    "$ROOT/install/venv/bin/pip" install --quiet --disable-pip-version-check "$INSTALL_FROM"
  fi
  tree_owner_mode "$ROOT/install" root root
  CLI="$ROOT/install/venv/bin/hermes-improve"
fi
if [[ -z "$CLI" && -x "$ROOT/install/venv/bin/hermes-improve" ]]; then
  CLI="$ROOT/install/venv/bin/hermes-improve"
fi
[[ -n "$CLI" ]] || die "pass --cli PATH or --install-from SRC"
[[ "$CLI" = /* && -x "$CLI" ]] || die "--cli must be an absolute executable path"
cli_dir="$(dirname "$(readlink -f "$CLI")")"
while [[ "$cli_dir" != / ]]; do
  read -r owner perms < <(stat -c '%u %a' "$cli_dir")
  [[ "$owner" -eq 0 && $(( 8#$perms & 8#022 )) -eq 0 ]] || \
    die "CLI directory $cli_dir must be root-owned and not group/other writable"
  cli_dir="$(dirname "$cli_dir")"
done
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/install/venv/bin/python" ]]; then PYTHON="$ROOT/install/venv/bin/python"; else PYTHON=python3; fi
fi

# --- keys -------------------------------------------------------------------
if [[ ! -s "$ROOT/keys/promoter/manifest.key" ]]; then
  ( umask 077; head -c 48 /dev/urandom | base64 -w0 > "$ROOT/keys/promoter/manifest.key" )
fi
chown "root:$PROMOTER" "$ROOT/keys/promoter/manifest.key"
chmod 0440 "$ROOT/keys/promoter/manifest.key"
if [[ ! -s "$ROOT/keys/evaluator/signing.pem" ]]; then
  ( umask 077; "$PYTHON" - "$ROOT/keys/evaluator/signing.pem" "$ROOT/keys/evaluator.pub" <<'PY'
import sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
key = Ed25519PrivateKey.generate()
with open(sys.argv[1], "xb") as private:
    private.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
with open(sys.argv[2], "wb") as public:
    public.write(key.public_key().public_bytes(serialization.Encoding.PEM,
                                               serialization.PublicFormat.SubjectPublicKeyInfo))
PY
  ) || die "evaluator key generation failed; pass --python with the cryptography package"
fi
[[ -s "$ROOT/keys/evaluator.pub" ]] || die "evaluator public key missing next to existing private key"
chown "root:$EVALUATOR" "$ROOT/keys/evaluator/signing.pem"
chmod 0440 "$ROOT/keys/evaluator/signing.pem"
chown root:root "$ROOT/keys/evaluator.pub"
chmod 0644 "$ROOT/keys/evaluator.pub"

# --- layout description and launcher environments --------------------------
write_root_file() {  # path; content on stdin; atomically, root:root 0644
  local temporary
  temporary="$(mktemp "$1.XXXXXX")"
  cat > "$temporary"
  chown root:root "$temporary"
  chmod 0644 "$temporary"
  mv -f "$temporary" "$1"
}
write_root_file "$ROOT/config/layout.json" <<EOF
{
  "schema_version": 1,
  "root": "$ROOT",
  "users": {"agent": "$AGENT", "evaluator": "$EVALUATOR", "promoter": "$PROMOTER"},
  "ledger_group": "$LEDGER_GROUP",
  "cli": "$CLI"
}
EOF
EVALUATOR_UID="$(id -u "$EVALUATOR")"
PROMOTER_UID="$(id -u "$PROMOTER")"
common_env() {
  cat <<EOF
HERMES_IMPROVE_CLI=$CLI
HERMES_IMPROVE_LAYOUT=$ROOT/config/layout.json
HERMES_IMPROVE_BENCHMARKS=$ROOT/benchmarks/public
HERMES_EVALUATOR_UID=$EVALUATOR_UID
HERMES_PROMOTER_UID=$PROMOTER_UID
EOF
}
{ common_env; cat <<EOF
HERMES_IMPROVE_USER=$AGENT
HERMES_IMPROVE_HOME=$ROOT/agent
HOME=$ROOT/home/agent
TMPDIR=$ROOT/workspaces
EOF
} | write_root_file "$ROOT/config/roles/agent.env"
{ common_env; cat <<EOF
HERMES_IMPROVE_USER=$EVALUATOR
HERMES_IMPROVE_HOME=$ROOT/ledger
HERMES_EVALUATOR_PRIVATE_KEY=$ROOT/keys/evaluator/signing.pem
HOME=$ROOT/home/evaluator
TMPDIR=$ROOT/home/evaluator
EOF
} | write_root_file "$ROOT/config/roles/evaluator.env"
{ common_env; cat <<EOF
HERMES_IMPROVE_USER=$PROMOTER
HERMES_IMPROVE_HOME=$ROOT/ledger
HERMES_IMPROVE_MANIFEST_KEY_FILE=$ROOT/keys/promoter/manifest.key
HOME=$ROOT/home/promoter
TMPDIR=$ROOT/home/promoter
EOF
} | write_root_file "$ROOT/config/roles/promoter.env"

# --- verification -----------------------------------------------------------
"$CLI" security check --layout "$ROOT/config/layout.json" >/dev/null || {
  "$CLI" security check --layout "$ROOT/config/layout.json" >&2 || true
  die "isolation self-check failed after provisioning"
}
echo "provisioned $ROOT (agent=$AGENT evaluator=$EVALUATOR promoter=$PROMOTER)"
