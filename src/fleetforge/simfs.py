"""A small simulated file store with power-cut injection.

Model (close to a journaling file system on flash, deliberately simple):

  write      goes to a volatile buffer; it is durable only after `sync` of that file
  sync       makes the file's buffered writes durable
  rename     atomic and durable immediately, but moves whatever the file currently holds: if the source was not
             synced first, its unsynced bytes can still be lost (the classic rename-before-fsync bug)
  delete, truncate   atomic and durable immediately

Every call is one operation. A cut at operation k raises PowerCut before that operation takes effect, except a
`write`, which is torn: it enters the volatile buffer first, so it may or may not survive. After a cut, `crash`
builds the post-reboot store under one of four patterns for what happens to buffered (unsynced) writes:

  lose   none survive          keep   all survive
  half   the first half of the buffered writes survive, in order
  alt    every other buffered write survives (out-of-order loss)
"""

from __future__ import annotations


class PowerCut(Exception):
    pass


PATTERNS = ("lose", "keep", "half", "alt")


class SimFS:
    def __init__(self) -> None:
        self.dur: dict[str, bytearray] = {}
        self.buf: dict[str, list[tuple[int, bytes]]] = {}
        self.ops = 0
        self.cut_at: int | None = None
        self.history: list[str] = []

    # ------------------------------------------------------------------ plumbing
    def _tick(self, what: str) -> None:
        self.ops += 1
        self.history.append(what)
        if self.cut_at is not None and self.ops == self.cut_at:
            raise PowerCut(what)

    @staticmethod
    def _apply(data: bytearray, off: int, chunk: bytes) -> None:
        if off > len(data):
            data.extend(b"\x00" * (off - len(data)))
        data[off : off + len(chunk)] = chunk

    def _view(self, path: str) -> bytearray | None:
        if path not in self.dur and path not in self.buf:
            return None
        v = bytearray(self.dur.get(path, b""))
        for off, chunk in self.buf.get(path, []):
            self._apply(v, off, chunk)
        return v

    # ------------------------------------------------------------------ reads (free)
    def exists(self, path: str) -> bool:
        return path in self.dur or path in self.buf

    def read(self, path: str) -> bytes | None:
        v = self._view(path)
        return None if v is None else bytes(v)

    def size(self, path: str) -> int:
        v = self._view(path)
        return 0 if v is None else len(v)

    def listdir(self, prefix: str) -> list[str]:
        return sorted(p for p in {*self.dur, *self.buf} if p.startswith(prefix))

    # ------------------------------------------------------------------ operations
    def write(self, path: str, offset: int, data: bytes) -> None:
        self.buf.setdefault(path, []).append((offset, bytes(data)))
        self.dur.setdefault(path, bytearray())
        self._tick(f"write {path}@{offset}+{len(data)}")  # torn: already buffered when the cut fires

    def sync(self, path: str) -> None:
        self._tick(f"sync {path}")
        for off, chunk in self.buf.pop(path, []):
            self._apply(self.dur.setdefault(path, bytearray()), off, chunk)

    def rename(self, src: str, dst: str) -> None:
        self._tick(f"rename {src}->{dst}")
        if src not in self.dur and src not in self.buf:
            raise FileNotFoundError(src)
        self.dur[dst] = self.dur.pop(src, bytearray())
        self.buf.pop(dst, None)
        if src in self.buf:
            self.buf[dst] = self.buf.pop(src)

    def delete(self, path: str) -> None:
        self._tick(f"delete {path}")
        self.dur.pop(path, None)
        self.buf.pop(path, None)

    def truncate(self, path: str, n: int) -> None:
        self._tick(f"truncate {path} {n}")
        v = self._view(path)
        if v is None:
            return
        self.dur[path] = v[:n]
        self.buf.pop(path, None)

    # ------------------------------------------------------------------ crash
    def crash(self, pattern: str) -> "SimFS":
        """The store after power returns: durable data plus whichever buffered writes the pattern keeps."""
        new = SimFS()
        paths = set(self.dur) | set(self.buf)
        for p in paths:
            d = bytearray(self.dur.get(p, b""))
            pend = self.buf.get(p, [])
            if pattern == "keep":
                kept = pend
            elif pattern == "half":
                kept = pend[: (len(pend) + 1) // 2]
            elif pattern == "alt":
                kept = pend[::2]
            elif pattern == "lose":
                kept = []
            else:
                raise ValueError(pattern)
            for off, chunk in kept:
                self._apply(d, off, chunk)
            new.dur[p] = d
        return new
