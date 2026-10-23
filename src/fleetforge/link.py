"""A model server reached over an unreliable link. It serves one current bundle, in parts and in byte ranges."""

from __future__ import annotations

import lzma
import random

from .artifact import Bundle, chunk_index, parse_manifest

PARTS = ("manifest", "signature", "model", "model_card", "sbom", "index")


class LinkError(Exception):
    """The connection dropped before any bytes of this request arrived."""


class Server:
    def __init__(self, bundle: Bundle, rng: random.Random | None = None, p_drop: float = 0.0, p_short: float = 0.0, p_corrupt: float = 0.0) -> None:
        self.bundle = bundle
        self.rng = rng or random.Random(0)
        self.p_drop, self.p_short, self.p_corrupt = p_drop, p_short, p_corrupt
        self.bytes_sent = 0
        self.requests = 0
        self._index: bytes | None = None
        self._packed: dict[int, bytes] = {}

    def _part(self, part: str) -> bytes:
        if part == "index":  # the chunk-hash list the signed manifest pins
            if self._index is None:
                self._index = chunk_index(self.bundle.model, parse_manifest(self.bundle.manifest).chunk_size)
            return self._index
        return getattr(self.bundle, part)

    def size(self, part: str) -> int:
        return len(self._part(part))

    def _damage(self, data: bytes) -> bytes:
        """Apply the link's faults to one response: it may not arrive, arrive short, or arrive with a flipped byte."""
        self.requests += 1
        if self.rng.random() < self.p_drop:
            raise LinkError("dropped")
        if len(data) > 1 and self.rng.random() < self.p_short:
            data = data[: self.rng.randint(1, len(data) - 1)]
        if data and self.rng.random() < self.p_corrupt:
            i = self.rng.randrange(len(data))
            data = data[:i] + bytes([data[i] ^ 0x01]) + data[i + 1 :]
        self.bytes_sent += len(data)
        return data

    def read(self, part: str, offset: int, n: int) -> bytes:
        """Up to n bytes of `part` from `offset`. May raise LinkError, return fewer bytes than asked, or flip a byte."""
        return self._damage(self._part(part)[offset : offset + n])

    def read_chunk(self, i: int, compress: bool = True) -> tuple[bytes, bool]:
        """Chunk i of the model as (payload, is_lzma). The payload is compressed when that makes it smaller, and it
        is the unit of transfer: it arrives whole (possibly damaged) or not at all."""
        size = parse_manifest(self.bundle.manifest).chunk_size
        raw = self.bundle.model[i * size : (i + 1) * size]
        if compress:
            packed = self._packed.setdefault(i, lzma.compress(raw, preset=6))
            if len(packed) < len(raw):
                return self._damage(packed), True
        return self._damage(raw), False
