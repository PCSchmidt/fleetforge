"""Signed model artifacts and the device-side check that decides whether to accept one.

An artifact bundle is: manifest bytes (JSON), an Ed25519 signature over those exact bytes, the model file, and two
attachments (model card, SBOM) whose hashes the manifest pins. A device holds a pinned root public key, a keyset
(signed by the root, versioned, replaced wholesale so omission revokes), and a per-model rollback floor.

This is TUF-inspired but much smaller than TUF: one signing role under one root, no threshold signatures, no
timestamp/snapshot roles. It is offline-capable (a local key, no transparency log, no certificate chain), which is
a deliberate first step toward an air-gapped story; Sigstore/cosign-style keyless signing is not implemented.

Verification order (cheap and bounded first, trust decisions on the exact signed bytes):
  size cap -> strict parse -> key lookup -> signature -> target -> rollback floor -> validity window ->
  agent version -> model size and hash -> attachment hashes.
Nothing is parsed from the model file; the device never executes anything here. `verify` changes no state;
`commit` raises the floor and must be called only after the new model is installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MANIFEST_FORMAT = "fleetforge-manifest/2"
KEYSET_FORMAT = "fleetforge-keyset/1"
MANIFEST_DOMAIN = b"fleetforge/manifest/v1\x00"
KEYSET_DOMAIN = b"fleetforge/keyset/v1\x00"
MAX_MANIFEST = 16 * 1024
MAX_KEYSET = 16 * 1024
CLOCK_SKEW = 300  # seconds a manifest's creation time may be ahead of the device clock
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
KEYID_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_INT = 2**53
DEFAULT_CHUNK = 4096
MIN_CHUNK, MAX_CHUNK = 512, 16 * 1024 * 1024


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Result:
    ok: bool
    code: str
    detail: str = ""


OK = Result(True, "ok")


@dataclass(frozen=True)
class Bundle:
    manifest: bytes
    signature: bytes
    model: bytes
    model_card: bytes
    sbom: bytes


# ------------------------------------------------------------------------------------------------ strict JSON
def _no_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ParseError(f"duplicate key {k!r}")
        d[k] = v
    return d


def _loads(raw: bytes):
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_no_dupes,
            parse_constant=lambda c: (_ for _ in ()).throw(ParseError(f"non-finite constant {c}")),
        )
    except ParseError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise ParseError(f"not valid JSON: {e}") from e


def _int(d, key, lo, hi):
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ParseError(f"'{key}' must be an integer in [{lo}, {hi}]")
    return v


def _str(d, key, rx):
    v = d.get(key)
    if not isinstance(v, str) or not rx.match(v):
        raise ParseError(f"'{key}' is malformed")
    return v


def _closed(d, allowed, where):
    if not isinstance(d, dict):
        raise ParseError(f"{where} must be an object")
    extra = set(d) - set(allowed)
    missing = set(allowed) - set(d)
    if extra or missing:
        raise ParseError(f"{where}: unexpected {sorted(extra)} missing {sorted(missing)}")


@dataclass(frozen=True)
class Manifest:
    name: str
    seq: int
    version: str
    target: str
    model_sha256: str
    model_size: int
    created: int
    not_after: int
    min_agent: int
    key_id: str
    metrics: dict = field(default_factory=dict)
    model_card_sha256: str = ""
    sbom_sha256: str = ""
    chunk_size: int = DEFAULT_CHUNK
    index_sha256: str = ""


def parse_manifest(raw: bytes) -> Manifest:
    if len(raw) > MAX_MANIFEST:
        raise ParseError("manifest too large")
    d = _loads(raw)
    keys = ("format", "name", "seq", "version", "target", "model", "chunks", "attachments", "created", "not_after", "min_agent", "key_id", "metrics")
    _closed(d, keys, "manifest")
    if d["format"] != MANIFEST_FORMAT:
        raise ParseError("unknown manifest format")
    _closed(d["model"], ("sha256", "size"), "model")
    _closed(d["attachments"], ("model_card_sha256", "sbom_sha256"), "attachments")
    _closed(d["chunks"], ("size", "index_sha256"), "chunks")
    m = d["metrics"]
    if not isinstance(m, dict) or len(m) > 16:
        raise ParseError("metrics must be an object of at most 16 entries")
    for k, v in m.items():
        if not isinstance(k, str) or not 0 < len(k) <= 32 or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ParseError("metrics must map short names to finite numbers")
    ver = d["version"]
    if not isinstance(ver, str) or not 0 < len(ver) <= 32 or not ver.isprintable():
        raise ParseError("'version' is malformed")
    created, not_after = _int(d, "created", 0, MAX_INT), _int(d, "not_after", 0, MAX_INT)
    if not_after <= created:
        raise ParseError("not_after must be after created")
    return Manifest(
        name=_str(d, "name", NAME_RE),
        seq=_int(d, "seq", 1, MAX_INT),
        version=ver,
        target=_str(d, "target", NAME_RE),
        model_sha256=_str(d["model"], "sha256", HEX64),
        model_size=_int(d["model"], "size", 0, 2**40),
        created=created,
        not_after=not_after,
        min_agent=_int(d, "min_agent", 0, MAX_INT),
        key_id=_str(d, "key_id", KEYID_RE),
        metrics=dict(m),
        model_card_sha256=_str(d["attachments"], "model_card_sha256", HEX64),
        sbom_sha256=_str(d["attachments"], "sbom_sha256", HEX64),
        chunk_size=_int(d["chunks"], "size", MIN_CHUNK, MAX_CHUNK),
        index_sha256=_str(d["chunks"], "index_sha256", HEX64),
    )


# ------------------------------------------------------------------------------------------------ signing side
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def chunk_index(model: bytes, chunk_size: int) -> bytes:
    """The model's chunk hashes, concatenated: SHA-256 of each `chunk_size` piece (the last may be shorter)."""
    return b"".join(hashlib.sha256(model[i : i + chunk_size]).digest() for i in range(0, len(model), chunk_size))


def make_bundle(
    key: Ed25519PrivateKey,
    key_id: str,
    name: str,
    seq: int,
    version: str,
    target: str,
    model: bytes,
    model_card: bytes,
    sbom: bytes,
    created: int,
    not_after: int,
    min_agent: int = 0,
    metrics: dict | None = None,
    chunk_size: int = DEFAULT_CHUNK,
) -> Bundle:
    doc = {
        "format": MANIFEST_FORMAT,
        "name": name,
        "seq": seq,
        "version": version,
        "target": target,
        "model": {"sha256": sha256(model), "size": len(model)},
        "chunks": {"size": chunk_size, "index_sha256": sha256(chunk_index(model, chunk_size))},
        "attachments": {"model_card_sha256": sha256(model_card), "sbom_sha256": sha256(sbom)},
        "created": created,
        "not_after": not_after,
        "min_agent": min_agent,
        "key_id": key_id,
        "metrics": metrics or {},
    }
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return Bundle(raw, key.sign(MANIFEST_DOMAIN + raw), model, model_card, sbom)


def make_keyset(root: Ed25519PrivateKey, version: int, keys: list[tuple[str, bytes, int]]) -> tuple[bytes, bytes]:
    """keys: (key_id, raw public key bytes, not_after). Returns (document bytes, signature)."""
    doc = {
        "format": KEYSET_FORMAT,
        "version": version,
        "keys": [{"id": k, "public": pub.hex(), "not_after": na} for k, pub, na in keys],
    }
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return raw, root.sign(KEYSET_DOMAIN + raw)


def public_bytes(key: Ed25519PrivateKey) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


# ------------------------------------------------------------------------------------------------ device side
class Device:
    """What a device remembers: a pinned root key, the current keyset, and the highest accepted `seq` per model."""

    def __init__(self, root_public: bytes, target: str, agent_version: int = 1) -> None:
        self.root = Ed25519PublicKey.from_public_bytes(root_public)
        self.target = target
        self.agent_version = agent_version
        self.keyset_version = 0
        self.keys: dict[str, tuple[Ed25519PublicKey, int]] = {}
        self.floor: dict[str, int] = {}
        self.installed: dict[str, str] = {}  # name -> model sha256 of the accepted seq

    # --- keyset ---
    def apply_keyset(self, raw: bytes, signature: bytes) -> Result:
        if len(raw) > MAX_KEYSET:
            return Result(False, "keyset_malformed", "too large")
        try:
            self.root.verify(signature, KEYSET_DOMAIN + raw)
        except InvalidSignature:
            return Result(False, "keyset_bad_signature")
        try:
            d = _loads(raw)
            _closed(d, ("format", "version", "keys"), "keyset")
            if d["format"] != KEYSET_FORMAT:
                raise ParseError("unknown keyset format")
            version = _int(d, "version", 1, MAX_INT)
            if not isinstance(d["keys"], list) or not 1 <= len(d["keys"]) <= 16:
                raise ParseError("keys must be a list of 1 to 16 entries")
            parsed: dict[str, tuple[Ed25519PublicKey, int]] = {}
            for k in d["keys"]:
                _closed(k, ("id", "public", "not_after"), "key")
                kid = _str(k, "id", KEYID_RE)
                pub = _str(k, "public", HEX64)
                if kid in parsed:
                    raise ParseError("duplicate key id")
                parsed[kid] = (Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)), _int(k, "not_after", 0, MAX_INT))
        except (ParseError, ValueError) as e:
            return Result(False, "keyset_malformed", str(e))
        if version <= self.keyset_version:
            return Result(False, "keyset_rollback", f"version {version} <= {self.keyset_version}")
        self.keyset_version, self.keys = version, parsed
        return OK

    # --- artifacts ---
    def verify_header(self, manifest: bytes, signature: bytes, now: int) -> Result:
        """Everything that can be decided from the manifest and signature alone, so a device can refuse an update
        before downloading the model."""
        try:
            m = parse_manifest(manifest)
        except ParseError as e:
            return Result(False, "manifest_malformed", str(e))
        key = self.keys.get(m.key_id)
        if key is None:
            return Result(False, "unknown_key", m.key_id)
        pub, key_not_after = key
        if now > key_not_after:
            return Result(False, "key_expired", m.key_id)
        try:
            pub.verify(signature, MANIFEST_DOMAIN + manifest)
        except InvalidSignature:
            return Result(False, "bad_signature")
        if m.target != self.target:
            return Result(False, "wrong_target", f"{m.target} != {self.target}")
        floor = self.floor.get(m.name, 0)
        if m.seq < floor:
            return Result(False, "rollback", f"seq {m.seq} < floor {floor}")
        if m.seq == floor and self.installed.get(m.name) != m.model_sha256:
            return Result(False, "rollback", f"seq {m.seq} equals the floor with a different model")
        if now > m.not_after:
            return Result(False, "expired")
        if m.created > now + CLOCK_SKEW:
            return Result(False, "not_yet_valid")
        if m.min_agent > self.agent_version:
            return Result(False, "agent_too_old", f"needs {m.min_agent}, have {self.agent_version}")
        return OK

    def verify_body(self, b: Bundle) -> Result:
        """Model size and hash and attachment hashes against an already-verified manifest."""
        m = parse_manifest(b.manifest)
        if len(b.model) != m.model_size:
            return Result(False, "model_size", f"{len(b.model)} != {m.model_size}")
        if sha256(b.model) != m.model_sha256:
            return Result(False, "model_hash")
        if sha256(b.model_card) != m.model_card_sha256:
            return Result(False, "attachment_hash", "model card")
        if sha256(b.sbom) != m.sbom_sha256:
            return Result(False, "attachment_hash", "sbom")
        if sha256(chunk_index(b.model, m.chunk_size)) != m.index_sha256:
            return Result(False, "chunk_index")
        return OK

    def verify(self, b: Bundle, now: int) -> Result:
        r = self.verify_header(b.manifest, b.signature, now)
        return r if not r.ok else self.verify_body(b)

    def verify_installed(self, b: Bundle) -> Result:
        """Integrity of a bundle already on the device: key, signature and hashes, but not the validity window, the
        floor or the target, which only matter when deciding to accept something new."""
        try:
            m = parse_manifest(b.manifest)
        except ParseError as e:
            return Result(False, "manifest_malformed", str(e))
        key = self.keys.get(m.key_id)
        if key is None:
            return Result(False, "unknown_key", m.key_id)
        try:
            key[0].verify(b.signature, MANIFEST_DOMAIN + b.manifest)
        except InvalidSignature:
            return Result(False, "bad_signature")
        return self.verify_body(b)

    def commit(self, b: Bundle) -> None:
        """Record an installed artifact. Call only after `verify` accepted it and the install succeeded."""
        m = parse_manifest(b.manifest)
        self.floor[m.name] = max(self.floor.get(m.name, 0), m.seq)
        self.installed[m.name] = m.model_sha256
