"""Exhaustive power-cut sweep for the update agent, with a naive in-place updater as the control.

For every operation count k in a full update run, and every post-cut pattern, the run is cut at k, the store
crashes into its post-reboot state, invariants are checked on that state, and the device is then driven to the end
with no further faults (and also with a second cut at every operation of the recovery run).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from . import agent as AG
from .agent import Agent, Bricked, provision
from .artifact import Bundle, make_bundle, make_keyset, parse_manifest, public_bytes
from .link import Server
from .simfs import PATTERNS, PowerCut, SimFS

NOW = 1_800_000_000
TARGET = "jetson-orin-nano-trt10"
NAME = "detector"


def _key(seed: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed.encode()).digest())


def _blob(seed: int, n: int) -> bytes:
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


@dataclass
class World:
    root_pub: bytes
    keyset: tuple
    b1: Bundle
    b2: Bundle
    seq1: int = 1
    seq2: int = 2
    h1: str = ""
    h2: str = ""
    base: SimFS = field(default_factory=SimFS)

    def server(self) -> Server:
        return Server(self.b2)


def make_world(model_size: int = 6 * AG.CHUNK + 100, reuse: float = 0.0) -> World:
    """`reuse` is the fraction of the new model's chunks that are identical to the factory model's (from the start)."""
    root, signer = _key("root"), _key("signer")
    keyset = make_keyset(root, 1, [("k1", public_bytes(signer), NOW + 10**6)])
    old, new = _blob(1, model_size), _blob(2, model_size)
    keep = int(reuse * model_size) // AG.CHUNK * AG.CHUNK
    new = old[:keep] + new[keep:]
    mk = lambda seq, model: make_bundle(signer, "k1", NAME, seq, f"1.{seq}", TARGET, model, b"card", b"{}", NOW - 100, NOW + 10**5)  # noqa: E731
    b1, b2 = mk(1, old), mk(2, new)
    w = World(public_bytes(root), keyset, b1, b2)
    w.h1, w.h2 = parse_manifest(b1.manifest).model_sha256, parse_manifest(b2.manifest).model_sha256
    fs = SimFS()
    provision(fs, w.root_pub, TARGET, b1, keyset)
    fs.ops, fs.history = 0, []
    w.base = fs
    return w


def clone(fs: SimFS) -> SimFS:
    c = SimFS()
    c.dur = {p: bytearray(v) for p, v in fs.dur.items()}
    c.buf = {p: list(v) for p, v in fs.buf.items()}
    return c


def drive(fs: SimFS, w: World, mode: str, server: Server | None = None, max_lives: int = 12) -> int:
    """Run the device until the update has run its course. A PowerCut propagates to the caller.

    One life is one boot: if a trial is pending the boot runs it and the supervisor confirms (or, in 'crashloop'
    mode, the model keeps crashing so the device reboots until the trial is abandoned); otherwise the agent checks
    for an update and, if it staged one, the device reboots into the trial.
    """
    server = server or w.server()
    for life in range(max_lives):
        a = Agent(fs, w.root_pub, TARGET)
        running = a.boot()
        st = a._load()
        if not (st["pending"] and running == st["pending"]["slot"]):
            if a.update(server, NOW) == "pending":
                a = Agent(fs, w.root_pub, TARGET)
                running = a.boot()  # reboot into the trial
                st = a._load()
        if st["pending"] and running == st["pending"]["slot"]:
            if mode == "crashloop":
                for _ in range(AG.MAX_TRIES):
                    Agent(fs, w.root_pub, TARGET).boot()  # the model crashes before it can be confirmed
            else:
                a.confirm(healthy=(mode == "good"))
        if _finished(fs, w, mode):
            return life
    raise AssertionError("did not finish")


def _finished(fs: SimFS, w: World, mode: str) -> bool:
    st = Agent(fs, w.root_pub, TARGET)._load()
    if st["pending"]:
        return False
    if mode == "good":
        return st["installed"].get(NAME) == w.h2
    return w.h2 in st["bad"]


def invariants(fs: SimFS, w: World) -> list[str]:
    """Properties that must hold on the store right after any crash. Returns violated ones (empty if all hold)."""
    bad = []
    try:
        a = Agent(fs, w.root_pub, TARGET).audit()
    except Bricked as e:
        return [f"state unreadable: {e}"]
    st = a["state"]
    if not a["active_ok"]:
        bad.append("active slot does not verify")
    seq = st["floor"].get(NAME, 0)
    if seq not in (w.seq1, w.seq2):
        bad.append(f"floor {seq}")
    want = w.h1 if seq == w.seq1 else w.h2
    if st["installed"].get(NAME) != want:
        bad.append("installed hash does not match the floor")
    b = Agent(fs, w.root_pub, TARGET)._bundle(st["active"])
    if b is not None and parse_manifest(b.manifest).seq != seq:
        bad.append("active slot holds a different seq than the floor")
    if st["pending"] and st["pending"]["slot"] == st["active"]:
        bad.append("pending points at the active slot")
    if st["pending"] and not a["pending_ok"]:
        pass  # allowed: boot drops a damaged pending slot
    return bad


def final_check(fs: SimFS, w: World, mode: str) -> list[str]:
    st = Agent(fs, w.root_pub, TARGET)._load()
    a = Agent(fs, w.root_pub, TARGET).audit()
    bad = []
    if not a["active_ok"]:
        bad.append("final active slot does not verify")
    if mode == "good" and st["floor"].get(NAME) != w.seq2:
        bad.append("update did not complete")
    if mode != "good" and (st["floor"].get(NAME) != w.seq1 or w.h2 not in st["bad"]):
        bad.append("bad model not rolled back and remembered")
    return bad


@dataclass
class SweepResult:
    runs: int = 0
    failures: list = field(default_factory=list)
    ops: int = 0
    recovery_ops_max: int = 0


def count_ops(w: World, mode: str) -> int:
    fs = clone(w.base)
    drive(fs, w, mode)
    return fs.ops


def run_cut(w: World, mode: str, k: int, pattern: str, j: int | None = None, pattern2: str | None = None) -> list[str]:
    fs = clone(w.base)
    fs.cut_at = k
    try:
        drive(fs, w, mode)
        return []  # cut point beyond the run: nothing happened
    except PowerCut:
        pass
    except Bricked as e:
        return [f"bricked before the cut: {e}"]
    fs = fs.crash(pattern)
    v = invariants(fs, w)
    if v:
        return [f"after cut {k}/{pattern}: {x}" for x in v]
    fs.cut_at = j
    try:
        drive(fs, w, mode)
    except PowerCut:
        fs = fs.crash(pattern2 or pattern)
        v = invariants(fs, w)
        if v:
            return [f"after second cut {j}: {x}" for x in v]
        fs.cut_at = None
        try:
            drive(fs, w, mode)
        except (Bricked, AssertionError) as e:
            return [f"after cuts {k},{j}: {type(e).__name__} {e}"]
    except (Bricked, AssertionError) as e:
        return [f"after cut {k}/{pattern}: {type(e).__name__} {e}"]
    fs.cut_at = None
    f = final_check(fs, w, mode)
    return [f"cuts {k},{j}/{pattern}: {x}" for x in f]


def single_cut_sweep(w: World, mode: str) -> SweepResult:
    n = count_ops(w, mode)
    r = SweepResult(ops=n)
    for k in range(1, n + 1):
        for p in PATTERNS:
            r.runs += 1
            f = run_cut(w, mode, k, p)
            if f:
                r.failures.append((k, p, f[0]))
    return r


def double_cut_sweep(w: World, mode: str, patterns=("lose", "half")) -> SweepResult:
    n = count_ops(w, mode)
    r = SweepResult(ops=n)
    for k in range(1, n + 1):
        for p in patterns:
            # how many operations does the recovery run take after a cut at k?
            fs = clone(w.base)
            fs.cut_at = k
            try:
                drive(fs, w, mode)
                continue
            except PowerCut:
                pass
            fs = fs.crash(p)
            fs.cut_at = None
            drive(fs, w, mode)
            m = fs.ops
            r.recovery_ops_max = max(r.recovery_ops_max, m)
            for j in range(1, m + 1):
                r.runs += 1
                f = run_cut(w, mode, k, p, j)
                if f:
                    r.failures.append((k, j, p, f[0]))
    return r


# ---------------------------------------------------------------------------------- the control
class NaiveUpdater:
    """Overwrites the active model in place, syncing only at the end, with no verification before use."""

    def __init__(self, fs: SimFS, w: World) -> None:
        self.fs, self.w = fs, w

    def update(self) -> None:
        b = self.w.b2
        path = AG.slot_path("A", "model")
        for off in range(0, len(b.model), AG.CHUNK):
            self.fs.write(path, off, b.model[off : off + AG.CHUNK])
        self.fs.sync(path)
        for name, data in (("manifest", b.manifest), ("signature", b.signature)):
            p = AG.slot_path("A", name)
            self.fs.write(p, 0, data)
            self.fs.sync(p)


def naive_sweep(w: World) -> SweepResult:
    fs = clone(w.base)
    NaiveUpdater(fs, w).update()
    n = fs.ops
    r = SweepResult(ops=n)
    for k in range(1, n + 1):
        for p in PATTERNS:
            r.runs += 1
            fs = clone(w.base)
            fs.cut_at = k
            try:
                NaiveUpdater(fs, w).update()
                continue
            except PowerCut:
                pass
            fs = fs.crash(p)
            try:
                ok = Agent(fs, w.root_pub, TARGET).audit()["active_ok"]
            except Bricked:
                ok = False
            if not ok:
                r.failures.append((k, p, "no valid model on the device"))
    return r
