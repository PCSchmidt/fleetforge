"""The device agent: pull-based, resumable model updates into A/B slots on a power-cut-prone store.

Layout on the store:

  state.json                       the only authority: active slot, optional pending trial, floors, installed
                                   hashes, rejected hashes, the signed keyset. Replaced only by write-temp, sync,
                                   rename, so it is always either the old or the new complete record.
  slots/A/*, slots/B/*             model, manifest, signature, model_card, sbom (+ model.part and journal while
                                   a download is in progress)

Rules that make interruption safe:

  * The running model's slot is never written. An update goes to the other slot.
  * A downloaded slot is verified in full (signature, floor, hashes) before state.json points at it.
  * state.json points at it only as `pending`; the floor rises and it becomes `active` only on `confirm(healthy)`.
    A pending model that is unhealthy, or fails to be confirmed within MAX_TRIES boots, is dropped and its hash
    remembered, and the device keeps running the previous active model.
  * Download progress is a journal of (offset, length, chunk hash) lines, each written only after the chunk it
    describes was synced; resuming trusts only the journalled prefix whose hashes still match the file.

The agent never executes the model. Nothing here is a network stack, a filesystem, or a bootloader.
"""

from __future__ import annotations

import hashlib
import json

from .artifact import Bundle, Device, ParseError, parse_manifest, sha256
from .link import LinkError, Server
from .simfs import SimFS

CHUNK = 4096
RETRIES = 6
MAX_TRIES = 3
SLOTS = ("A", "B")
FILES = ("model", "manifest", "signature", "model_card", "sbom")


class Bricked(Exception):
    """The device has no valid model to run. Every property test is aimed at never raising this."""


def other(slot: str) -> str:
    return "B" if slot == "A" else "A"


def slot_path(slot: str, name: str) -> str:
    return f"slots/{slot}/{name}"


class Agent:
    def __init__(self, fs: SimFS, root_public: bytes, target: str, agent_version: int = 1) -> None:
        self.fs = fs
        self.root_public = root_public
        self.target = target
        self.agent_version = agent_version
        self.resume = True  # tests turn this off to measure what resuming saves

    # ------------------------------------------------------------------ persistent state
    def _load(self) -> dict:
        raw = self.fs.read("state.json")
        if raw is None:
            raise Bricked("no state")
        try:
            st = json.loads(raw)
            check = st.pop("check")
            if check != hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest():
                raise ValueError("checksum")
            return st
        except (ValueError, KeyError, TypeError) as e:
            raise Bricked(f"state unreadable: {e}") from e

    def _save(self, st: dict) -> None:
        body = dict(st)
        body["check"] = hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()
        self.fs.write("state.tmp", 0, json.dumps(body, sort_keys=True).encode())
        self.fs.sync("state.tmp")
        self.fs.rename("state.tmp", "state.json")

    def _device(self, st: dict) -> Device:
        d = Device(self.root_public, self.target, self.agent_version)
        r = d.apply_keyset(bytes.fromhex(st["keyset"]["raw"]), bytes.fromhex(st["keyset"]["sig"]))
        assert r.ok, r
        d.floor = dict(st["floor"])
        d.installed = dict(st["installed"])
        return d

    def _bundle(self, slot: str) -> Bundle | None:
        parts = {f: self.fs.read(slot_path(slot, f)) for f in FILES}
        if any(p is None for p in parts.values()):
            return None
        return Bundle(parts["manifest"], parts["signature"], parts["model"], parts["model_card"], parts["sbom"])

    def _slot_ok(self, d: Device, slot: str) -> bool:
        b = self._bundle(slot)
        return b is not None and d.verify_installed(b).ok

    def _wipe(self, slot: str) -> None:
        # the manifest goes first so a half-wiped slot can never look valid
        for f in ("manifest", "signature", "model", "model_card", "sbom", "model.part", "journal"):
            if self.fs.exists(slot_path(slot, f)):
                self.fs.delete(slot_path(slot, f))

    # ------------------------------------------------------------------ boot
    def audit(self) -> dict:
        """Read-only view for the tests: which slots verify, and what the state says."""
        st = self._load()
        d = self._device(st)
        return {"state": st, "active_ok": self._slot_ok(d, st["active"]), "pending_ok": bool(st["pending"]) and self._slot_ok(d, st["pending"]["slot"])}

    def boot(self) -> str:
        """Decide which slot to run. Returns the slot letter. Raises Bricked if there is nothing valid to run."""
        st = self._load()
        d = self._device(st)
        running = st["active"]
        p = st["pending"]
        if p:
            tries = p["tries"] + 1
            if tries > MAX_TRIES:
                st["bad"].append(p["hash"])
                st["pending"] = None
                self._save(st)
            elif not self._slot_ok(d, p["slot"]):
                st["pending"] = None  # the slot is damaged, not necessarily the model: do not blacklist it
                self._save(st)
            else:
                st["pending"] = {**p, "tries": tries}
                self._save(st)
                running = p["slot"]
        if running == st["active"] and not self._slot_ok(d, running):
            raise Bricked(f"active slot {running} does not verify")
        return running

    def confirm(self, healthy: bool) -> str:
        """Called by the supervisor once the pending model has had its chance to prove itself."""
        st = self._load()
        p = st["pending"]
        if not p:
            return "nothing"
        if healthy:
            st["previous"] = st["active"]
            st["active"] = p["slot"]
            st["floor"][p["name"]] = max(st["floor"].get(p["name"], 0), p["seq"])
            st["installed"][p["name"]] = p["hash"]
            st["pending"] = None
            self._save(st)
            return "promoted"
        st["bad"].append(p["hash"])
        st["pending"] = None
        self._save(st)
        return "rolled_back"

    # ------------------------------------------------------------------ download
    def _fetch_small(self, server: Server, part: str) -> bytes | None:
        out = b""
        total = server.size(part)
        fails = 0
        while len(out) < total:
            try:
                chunk = server.read(part, len(out), CHUNK)
            except LinkError:
                chunk = b""
            if not chunk:
                fails += 1
                if fails > RETRIES:
                    return None
                continue
            out += chunk
        return out

    def _put_file(self, path: str, data: bytes) -> None:
        self.fs.write(path + ".tmp", 0, data)
        self.fs.sync(path + ".tmp")
        self.fs.rename(path + ".tmp", path)

    def _resume_point(self, slot: str, head: bytes) -> int:
        """Reset or repair slot's download state against the manifest hash; return how many model bytes are good."""
        jpath, mpath = slot_path(slot, "journal"), slot_path(slot, "model.part")
        j = self.fs.read(jpath)
        if j is None or not j.startswith(head) or not self.resume:
            self._wipe(slot)
            self.fs.write(jpath, 0, head)
            self.fs.sync(jpath)
            return 0
        # after the final rename the finished model sits at `model`, not `model.part`
        part = self.fs.read(mpath) or self.fs.read(slot_path(slot, "model")) or b""
        good_end, good_len = 0, len(head)
        pos = len(head)
        while True:
            nl = j.find(b"\n", pos)
            if nl < 0:
                break
            try:
                off, ln, h = j[pos:nl].decode().split()
                off, ln = int(off), int(ln)
            except ValueError:
                break
            if off != good_end or len(part[off : off + ln]) != ln or sha256(part[off : off + ln]) != h:
                break
            good_end += ln
            pos = nl + 1
            good_len = pos
        if len(j) > good_len:
            self.fs.truncate(jpath, good_len)
        if len(part) > good_end and self.fs.exists(mpath):
            self.fs.truncate(mpath, good_end)
        return good_end

    def update(self, server: Server, now: int) -> str:
        """Fetch and stage the server's current bundle if it is acceptable. Idempotent and resumable."""
        st = self._load()
        if st["pending"]:
            return "pending"
        d = self._device(st)
        manifest, sig = self._fetch_small(server, "manifest"), self._fetch_small(server, "signature")
        if manifest is None or sig is None:
            return "link_down"
        r = d.verify_header(manifest, sig, now)
        if not r.ok:
            return "rejected:" + r.code
        m = parse_manifest(manifest)
        if m.model_sha256 in st["bad"]:
            return "rejected:bad_model"
        if st["installed"].get(m.name) == m.model_sha256:
            return "up_to_date"
        slot = other(st["active"])
        head = b"M " + sha256(manifest).encode() + b"\n"
        off = self._resume_point(slot, head)
        mpath, jpath = slot_path(slot, "model.part"), slot_path(slot, "journal")
        fails = 0
        while off < m.model_size:
            want = min(CHUNK, m.model_size - off)
            try:
                chunk = server.read("model", off, want)
            except LinkError:
                chunk = b""
            if not chunk:
                fails += 1
                if fails > RETRIES:
                    return "link_down"
                continue
            chunk = chunk[:want]
            self.fs.write(mpath, off, chunk)
            self.fs.sync(mpath)
            self.fs.write(jpath, self.fs.size(jpath), f"{off} {len(chunk)} {sha256(chunk)}\n".encode())
            self.fs.sync(jpath)
            off += len(chunk)
        card, sbom = self._fetch_small(server, "model_card"), self._fetch_small(server, "sbom")
        if card is None or sbom is None:
            return "link_down"
        self._put_file(slot_path(slot, "model_card"), card)
        self._put_file(slot_path(slot, "sbom"), sbom)
        self._put_file(slot_path(slot, "manifest"), manifest)
        self._put_file(slot_path(slot, "signature"), sig)
        if self.fs.exists(mpath):
            self.fs.rename(mpath, slot_path(slot, "model"))
        b = self._bundle(slot)
        res = d.verify(b, now) if b else None
        if res is None or not res.ok:
            self._wipe(slot)  # corrupted in transit: start over next time
            return "verify_failed:" + (res.code if res else "incomplete")
        st["pending"] = {"slot": slot, "tries": 0, "name": m.name, "seq": m.seq, "hash": m.model_sha256}
        self._save(st)
        return "pending"


def provision(fs: SimFS, root_public: bytes, target: str, bundle: Bundle, keyset: tuple[bytes, bytes]) -> None:
    """Factory step: put a first model in slot A and write the initial state. Not part of the power-cut sweep."""
    m = parse_manifest(bundle.manifest)
    for name, data in zip(FILES, (bundle.model, bundle.manifest, bundle.signature, bundle.model_card, bundle.sbom)):
        fs.write(slot_path("A", name), 0, data)
        fs.sync(slot_path("A", name))
    st = {
        "active": "A",
        "previous": None,
        "pending": None,
        "floor": {m.name: m.seq},
        "installed": {m.name: m.model_sha256},
        "bad": [],
        "keyset": {"raw": keyset[0].hex(), "sig": keyset[1].hex()},
    }
    Agent(fs, root_public, target)._save(st)
