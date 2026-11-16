"""Signed operator approvals for manually gated candidate types (spec section 10).

A candidate type whose gate is ``manual`` in the protected promotion config can
only be promoted with an Ed25519 approval signed by an operator key that the
promoter was configured to trust. The signed payload binds the experiment ID,
candidate ID, immutable candidate hash, candidate type and policy version, so
an approval cannot be replayed for a changed candidate or a different policy.
Approvals are stored append-only.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .errors import PolicyRejected
from .ingestion import contains_secret_material
from .models import CandidateType, canonical_json, utc_now
from .policy import PromotionPolicy
from .store import ExperimentStore


APPROVAL_PURPOSE = "hermes-improve-promotion-approval"
_SAFE_OPERATOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")


def public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def load_public_key(public_key_pem: bytes, *, role: str) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{role} public key must be Ed25519")
    return key


def approve_promotion(
    store: ExperimentStore,
    experiment_id: str,
    *,
    operator: str,
    rationale: str,
    private_key_pem: bytes,
    policy: PromotionPolicy,
) -> dict[str, Any]:
    """Sign and append an operator approval for a passed, manually gated experiment."""
    if not isinstance(operator, str) or not _SAFE_OPERATOR.fullmatch(operator):
        raise ValueError("operator must be a safe 1-128 character identifier")
    if (
        not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2048
        or "\x00" in rationale
    ):
        raise ValueError("approval rationale must be non-empty text up to 2048 characters")
    if contains_secret_material({"operator": operator, "rationale": rationale}):
        raise ValueError("approval metadata must not contain credential-like material")
    experiment = store.show_experiment(experiment_id)
    if len(experiment["candidates"]) != 1:
        raise ValueError("approvals require an experiment with exactly one candidate")
    candidate_id = experiment["candidates"][0]["id"]
    evidence = store.promotion_evidence(candidate_id)
    candidate = evidence["candidate"]
    if candidate["experiment_decision"] != "passed" or not evidence["evaluation"]["passed"]:
        raise PolicyRejected("only experiments with a passing completed evaluation can be approved")
    candidate_type = CandidateType(candidate["candidate_type"])
    if not policy.requires_manual_approval(candidate_type):
        raise ValueError(
            f"{candidate_type.value} is auto-promotable under policy {policy.version}; no approval is needed"
        )
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("operator approval key must be Ed25519")
    payload = {
        "schema_version": 1,
        "purpose": APPROVAL_PURPOSE,
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "candidate_hash": candidate["content_hash"],
        "candidate_type": candidate_type.value,
        "policy_version": policy.version,
        "operator": operator,
        "rationale": rationale.strip(),
        "operator_key_fingerprint": public_key_fingerprint(key.public_key()),
        "approved_at": utc_now(),
    }
    signature = key.sign(canonical_json(payload).encode("utf-8")).hex()
    approval_id = store.record_promotion_approval(payload, signature)
    return {"approval_id": approval_id, **payload, "signature": signature}


def verify_promotion_approval(
    records: Iterable[Mapping[str, Any]],
    *,
    experiment_id: str,
    candidate_id: str,
    candidate_hash: str,
    candidate_type: CandidateType,
    policy_version: str,
    operator_public_keys: Sequence[Ed25519PublicKey],
) -> dict[str, Any]:
    """Return the newest approval that is validly signed by a trusted operator
    key and bound to exactly this candidate and policy, else reject."""
    trusted = {public_key_fingerprint(key): key for key in operator_public_keys}
    expected = {
        "schema_version": 1,
        "purpose": APPROVAL_PURPOSE,
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "candidate_type": candidate_type.value,
        "policy_version": policy_version,
    }
    for record in reversed(list(records)):
        payload = record.get("payload")
        signature = record.get("signature")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            continue
        if any(payload.get(key) != value for key, value in expected.items()):
            continue
        key = trusted.get(payload.get("operator_key_fingerprint"))
        if key is None:
            continue
        try:
            key.verify(bytes.fromhex(signature), canonical_json(payload).encode("utf-8"))
        except (InvalidSignature, ValueError):
            continue
        return {**payload, "signature": signature}
    raise PolicyRejected(
        f"{candidate_type.value} requires a signed operator approval from a trusted key bound to "
        "this experiment, candidate hash and policy version"
    )
