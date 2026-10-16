from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .errors import IntegrityError


class ArtifactStore:
    """Content-addressed artifact store; payloads are immutable by digest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes) -> tuple[str, Path]:
        digest = hashlib.sha256(content).hexdigest()
        target = self.root / digest[:2] / digest[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            self.verify(digest)
            return digest, target
        fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                self.verify(digest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return digest, target

    def get(self, digest: str) -> bytes:
        path = self.path_for(digest)
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != digest:
            raise IntegrityError(f"artifact digest mismatch: expected {digest}, got {actual}")
        return content

    def verify(self, digest: str) -> Path:
        self.get(digest)
        return self.path_for(digest)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("artifact digest must be a lowercase SHA-256 hex string")
        return self.root / digest[:2] / digest[2:]
