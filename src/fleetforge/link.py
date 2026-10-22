"""A model server reached over an unreliable link. It serves one current bundle, in parts and in byte ranges."""

from __future__ import annotations

import random

from .artifact import Bundle

PARTS = ("manifest", "signature", "model", "model_card", "sbom")


class LinkError(Exception):
    """The connection dropped before any bytes of this request arrived."""


class Server:
    def __init__(self, bundle: Bundle, rng: random.Random | None = None, p_drop: float = 0.0, p_short: float = 0.0, p_corrupt: float = 0.0) -> None:
        self.bundle = bundle
        self.rng = rng or random.Random(0)
        self.p_drop, self.p_short, self.p_corrupt = p_drop, p_short, p_corrupt
        self.bytes_sent = 0
        self.requests = 0

    def size(self, part: str) -> int:
        return len(getattr(self.bundle, part))

    def read(self, part: str, offset: int, n: int) -> bytes:
        """Up to n bytes of `part` from `offset`. May raise LinkError, return fewer bytes than asked, or flip a byte."""
        self.requests += 1
        data = getattr(self.bundle, part)
        if self.rng.random() < self.p_drop:
            raise LinkError("dropped")
        chunk = data[offset : offset + n]
        if len(chunk) > 1 and self.rng.random() < self.p_short:
            chunk = chunk[: self.rng.randint(1, len(chunk) - 1)]
        if chunk and self.rng.random() < self.p_corrupt:
            i = self.rng.randrange(len(chunk))
            chunk = chunk[:i] + bytes([chunk[i] ^ 0x01]) + chunk[i + 1 :]
        self.bytes_sent += len(chunk)
        return chunk
