from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .errors import IntegrityError
from .models import EvaluationReport, canonical_json


def sign_report(report: EvaluationReport, private_key_pem: bytes) -> str:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("evaluator signing key must be Ed25519")
    return key.sign(canonical_json(report_payload(report)).encode("utf-8")).hex()


def verify_report_signature(
    payload: dict, signature_hex: str, public_key_pem: bytes
) -> None:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("evaluator verification key must be Ed25519")
    try:
        key.verify(bytes.fromhex(signature_hex), canonical_json(payload).encode("utf-8"))
    except (InvalidSignature, ValueError) as exc:
        raise IntegrityError("evaluator report signature verification failed") from exc


def report_payload(report: EvaluationReport) -> dict:
    from .models import evaluation_payload

    return evaluation_payload(report)
