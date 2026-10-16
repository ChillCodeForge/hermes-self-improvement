from __future__ import annotations

import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import IntegrityError, PolicyRejected
from .evaluation_signing import verify_report_signature
from .models import CandidateType, canonical_json, utc_now
from .policy import PromotionPolicy
from .store import ExperimentStore


_SAFE_NAME = re.compile(r"^brain-v[1-9][0-9]*$")
_SAFE_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SKILL_TARGET = re.compile(r"^brain/skills/[A-Za-z0-9._/-]+$")


class GitPromoter:
    """Promotes an evaluated skills-only Git commit using an atomic symlink.

    Candidate contents are read via `git archive` and are never executed by the
    promoter. This still must run under the dedicated least-privilege promoter
    Unix identity described in the deployment design.
    """

    def __init__(
        self,
        store: ExperimentStore,
        repository: Path,
        release_root: Path,
        deployment_link: Path,
        policy: PromotionPolicy,
        evaluator_public_key: bytes,
    ) -> None:
        self.store = store
        self.repository = Path(repository).resolve()
        self.release_root = Path(release_root).resolve()
        self.deployment_link = Path(deployment_link).absolute()
        self.policy = policy
        self.evaluator_public_key = evaluator_public_key
        self.release_root.mkdir(parents=True, exist_ok=True)
        self.deployment_link.parent.mkdir(parents=True, exist_ok=True)
        if self.deployment_link.exists() and not self.deployment_link.is_symlink():
            raise ValueError("deployment path must be absent or an existing release symlink")

    def promote(
        self,
        candidate_id: str,
        *,
        release_name: str,
        manifest_key: bytes,
    ) -> dict[str, Any]:
        if not _SAFE_NAME.fullmatch(release_name):
            raise ValueError("release name must match brain-vN")
        if len(manifest_key) < 32:
            raise ValueError("manifest signing key must contain at least 32 bytes")
        lock_path = self.release_root / ".promotion.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._promote_locked(candidate_id, release_name, manifest_key)

    def _promote_locked(self, candidate_id: str, release_name: str, manifest_key: bytes) -> dict[str, Any]:
        evidence = self.store.promotion_evidence(candidate_id)
        candidate = evidence["candidate"]
        evaluation = evidence["evaluation"]
        results = evidence["task_results"]
        if candidate["experiment_decision"] != "passed" or not evaluation["passed"]:
            raise PolicyRejected("candidate does not have a passing completed evaluation")
        if evaluation["candidate_hash"] != candidate["content_hash"]:
            raise IntegrityError("evaluated candidate hash differs from immutable candidate record")
        if not evaluation["signature"]:
            raise IntegrityError("evaluation has no trusted evaluator signature")
        signed_evaluation = {
            "candidate_id": candidate_id,
            "candidate_hash": evaluation["candidate_hash"],
            "benchmark_version": evaluation["benchmark_version"],
            "policy_version": evaluation["policy_version"],
            "outcomes": [item["result"] for item in sorted(results, key=lambda item: item["task_id"])],
            "passed": bool(evaluation["passed"]),
            "reasons": json.loads(evaluation["reasons"]),
            "created_at": evaluation["created_at"],
            "metrics": json.loads(evaluation["metrics_json"]),
            "run_observations": [
                {**item, "artifact_digests": item.get("artifact_digests", [])}
                for item in sorted(
                    evidence["run_observations"],
                    key=lambda row: (row["task_id"], row["commit"], row["seed"]),
                )
            ],
        }
        verify_report_signature(
            signed_evaluation, evaluation["signature"], self.evaluator_public_key
        )
        if evaluation["policy_version"] != self.policy.version:
            raise PolicyRejected("evaluation policy version differs from current promotion policy")
        if evaluation["benchmark_version"] != candidate["experiment_benchmark_version"]:
            raise IntegrityError("evaluation benchmark version differs from experiment manifest")
        self.policy.authorize_type(CandidateType(candidate["candidate_type"]))
        suites = {
            str(item["result"].get("suite", "")).upper()
            for item in results
        }
        splits = {str(item["split"]) for item in results}
        if "FULL" not in suites or "SEALED_HOLDOUT" not in suites or "SEALED_HOLDOUT" not in splits:
            raise PolicyRejected("Full and sealed holdout task evidence are both required")
        evaluation_metrics = json.loads(evaluation["metrics_json"])

        baseline = str(candidate["baseline_commit"])
        commit = str(candidate["commit_id"])
        experiment_id = str(candidate["experiment_id"])
        if not _SAFE_EXPERIMENT.fullmatch(experiment_id):
            raise IntegrityError("experiment ID is not a safe Git branch component")
        if not _GIT_OBJECT.fullmatch(baseline) or not _GIT_OBJECT.fullmatch(commit):
            raise IntegrityError("stored baseline/candidate commit is not a full Git object ID")
        branch = f"refs/heads/candidate/{experiment_id}"
        branch_commit = self._git("rev-parse", "--verify", branch).decode().strip()
        if branch_commit != commit:
            raise IntegrityError("candidate branch no longer points to the evaluated commit")
        production_ref = "refs/heads/main"
        production_commit = self._git("rev-parse", "--verify", production_ref).decode().strip()
        if production_commit != baseline:
            raise IntegrityError("production Git ref has changed since evaluation")
        self._git("merge-base", "--is-ancestor", baseline, commit)
        actual_diff = self._git("diff", "--no-ext-diff", "--no-textconv", "--binary", baseline, commit)
        if hashlib.sha256(actual_diff).hexdigest() != candidate["diff_hash"]:
            raise IntegrityError("candidate diff differs from the evaluated diff")
        changed_paths = self._git("diff", "--name-only", "-z", baseline, commit).decode().split("\0")
        changed_paths = [path for path in changed_paths if path]
        if not changed_paths or any(not path.startswith("brain/skills/") for path in changed_paths):
            raise PolicyRejected("automatic promotion is restricted to brain/skills/ paths")
        target = str(candidate["target"])
        if (
            not _SKILL_TARGET.fullmatch(target)
            or ".." in Path(target).parts
            or any(path != target and not path.startswith(f"{target.rstrip('/')}/") for path in changed_paths)
        ):
            raise PolicyRejected("changed paths do not match a safe skills-only candidate target")
        if self._ref_exists(f"refs/tags/{release_name}"):
            raise ValueError(f"release tag already exists: {release_name}")
        destination = self.release_root / release_name
        if destination.exists():
            raise ValueError(f"release directory already exists: {destination}")

        staging = Path(tempfile.mkdtemp(prefix=f".{release_name}-", dir=self.release_root))
        manifest_path = self.release_root / f"{release_name}.manifest.json"
        old_target = os.readlink(self.deployment_link) if self.deployment_link.is_symlink() else None
        branch_updated = False
        tag_created = False
        deployed = False
        try:
            self._extract_archive(commit, staging)
            os.chmod(staging, 0o555)
            os.replace(staging, destination)
            diff_hash = hashlib.sha256(actual_diff).hexdigest()
            manifest_body = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "candidate_commit": commit,
                "candidate_hash": candidate["content_hash"],
                "baseline_commit": baseline,
                "benchmark_version": evaluation["benchmark_version"],
                "policy_version": evaluation["policy_version"],
                "full_suite_passed": True,
                "holdout_passed": True,
                "regression_summary": json.loads(evaluation["reasons"]),
                "security_passed": True,
                "baseline_metrics": {
                    "success_rate": evaluation_metrics["baseline_success_rate"],
                    "tool_error_rate": evaluation_metrics["baseline_tool_error_rate"],
                    "tokens_per_success": evaluation_metrics["baseline_tokens_per_success"],
                    "latency_per_success_ms": evaluation_metrics["baseline_latency_per_success_ms"],
                    "correction_rate": evaluation_metrics["baseline_correction_rate"],
                },
                "artifact_type": candidate["candidate_type"],
                "changed_paths": changed_paths,
                "diff_sha256": diff_hash,
                "release_target": release_name,
                "created_at": utc_now(),
            }
            signature = hmac.new(manifest_key, canonical_json(manifest_body).encode(), hashlib.sha256).hexdigest()
            manifest = {**manifest_body, "manifest_sha256": hashlib.sha256(canonical_json(manifest_body).encode()).hexdigest(), "signature_hmac_sha256": signature}
            self._write_manifest(manifest_path, manifest)

            self._git("update-ref", production_ref, commit, baseline)
            branch_updated = True
            object_format = self._git("rev-parse", "--show-object-format").decode().strip()
            zero_oid_length = {"sha1": 40, "sha256": 64}.get(object_format)
            if zero_oid_length is None:
                raise IntegrityError(f"unsupported Git object format: {object_format!r}")
            self._git("update-ref", f"refs/tags/{release_name}", commit, "0" * zero_oid_length)
            tag_created = True
            self._switch_link(destination)
            deployed = True
            self.store.record_release(
                release_name, experiment_id, candidate["content_hash"], "PROVISIONAL",
                {**manifest, "manifest_path": str(manifest_path), "deployment_path": str(destination)},
            )
            return manifest
        except BaseException:
            if deployed:
                self._restore_link(old_target)
            if tag_created:
                self._git("update-ref", "-d", f"refs/tags/{release_name}", commit)
            if branch_updated:
                self._git("update-ref", production_ref, baseline, commit)
            if destination.exists():
                self._remove_tree(destination)
            manifest_path.unlink(missing_ok=True)
            raise
        finally:
            if staging.exists():
                os.chmod(staging, 0o755)
                shutil.rmtree(staging)

    def rollback(self, release_name: str, *, manifest_key: bytes) -> str:
        if not _SAFE_NAME.fullmatch(release_name):
            raise ValueError("release name must match brain-vN")
        if len(manifest_key) < 32:
            raise ValueError("manifest signing key must contain at least 32 bytes")
        with (self.release_root / ".promotion.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            target_release = self.store.get_release(release_name)
            if target_release["status"] == "ROLLED_BACK":
                raise ValueError("release is already marked rolled back")
            target_commit = str(target_release["manifest"]["candidate_commit"])
            target_path = (self.release_root / release_name).resolve()
            if self.release_root not in target_path.parents or not target_path.is_dir():
                raise IntegrityError("release checkout is missing or outside the configured release root")
            manifest_path = self.release_root / f"{release_name}.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            signature = manifest.pop("signature_hmac_sha256", None)
            digest = manifest.pop("manifest_sha256", None)
            expected_digest = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
            expected_signature = hmac.new(manifest_key, canonical_json(manifest).encode(), hashlib.sha256).hexdigest()
            if digest != expected_digest or not isinstance(signature, str) or not hmac.compare_digest(signature, expected_signature):
                raise IntegrityError("release manifest signature or digest verification failed")
            manifest["manifest_sha256"] = digest
            manifest["signature_hmac_sha256"] = signature
            if target_release["manifest"].get("candidate_commit") != target_commit:
                raise IntegrityError("release database and signed manifest disagree")
            current = self._git("rev-parse", "--verify", "refs/heads/main").decode().strip()
            if current == target_commit:
                raise ValueError("requested release is already active")
            self._git("merge-base", "--is-ancestor", target_commit, current)
            old_link = os.readlink(self.deployment_link) if self.deployment_link.is_symlink() else None
            self._git("update-ref", "refs/heads/main", target_commit, current)
            try:
                self._switch_link(target_path)
                self.store.transition_rollback(current, release_name)
            except BaseException:
                self._restore_link(old_link)
                self._git("update-ref", "refs/heads/main", current, target_commit)
                raise
            return current

    def _git(self, *args: str) -> bytes:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.repository), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
        return result.stdout

    def _ref_exists(self, ref: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(self.repository), "show-ref", "--verify", "--quiet", ref],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _extract_archive(self, commit: str, destination: Path) -> None:
        tree = self._git("ls-tree", "-r", "-l", commit).decode("utf-8")
        total_size = 0
        for entry in tree.splitlines():
            metadata, separator, _path = entry.partition("\t")
            fields = metadata.split()
            if not separator or len(fields) < 3:
                raise IntegrityError("Git returned an invalid tree entry")
            if fields[0] == "120000":
                raise IntegrityError("release archives may not contain symbolic links")
            if len(fields) >= 4 and fields[3].isdigit():
                total_size += int(fields[3])
        if total_size > 100 * 1024 * 1024:
            raise IntegrityError("candidate brain exceeds the 100 MiB release limit")
        archive = self._git("archive", "--format=tar", commit)
        root = destination.resolve()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            for member in bundle:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise IntegrityError(f"unsafe path in candidate archive: {member.name!r}")
                target = root.joinpath(*relative.parts)
                if root not in target.resolve().parents and target.resolve() != root:
                    raise IntegrityError(f"archive path escapes release: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    os.chmod(target, 0o755)
                    continue
                if not member.isfile():
                    raise IntegrityError(f"unsupported file type in candidate archive: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise IntegrityError(f"archive file has no data: {member.name!r}")
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(fd, "wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(target, 0o444)
        for directory, subdirs, _ in os.walk(root, topdown=False):
            os.chmod(directory, 0o555)

    def _write_manifest(self, path: Path, manifest: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        for directory, subdirs, files in os.walk(path, topdown=False):
            for name in files:
                os.chmod(Path(directory) / name, 0o600)
            for name in subdirs:
                os.chmod(Path(directory) / name, 0o700)
            os.chmod(directory, 0o700)
        shutil.rmtree(path)

    def _switch_link(self, target: Path) -> None:
        temporary = self.deployment_link.with_name(f".{self.deployment_link.name}.next")
        temporary.unlink(missing_ok=True)
        try:
            temporary.symlink_to(target, target_is_directory=True)
            os.replace(temporary, self.deployment_link)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_link(self, target: str | None) -> None:
        if self.deployment_link.is_symlink():
            self.deployment_link.unlink()
        elif self.deployment_link.exists():
            raise IntegrityError("deployment pointer was replaced by a non-symlink during rollback")
        if target is not None:
            temporary = self.deployment_link.with_name(f".{self.deployment_link.name}.restore")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(target, target_is_directory=True)
            os.replace(temporary, self.deployment_link)
