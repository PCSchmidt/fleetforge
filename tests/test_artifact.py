import hashlib
import json
import math

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fleetforge import artifact as A

NOW = 1_800_000_000
TARGET = "jetson-orin-nano-trt10"


def key(seed: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(seed.encode()).digest())


ROOT, SIGNER, OTHER = key("root"), key("signer"), key("other")
MODEL = bytes(range(256)) * 8  # 2 KiB
CARD, SBOM = b"model card v1", b'{"components":[]}'


def bundle(seq=1, signer=SIGNER, key_id="k1", model=MODEL, target=TARGET, created=NOW - 100, not_after=NOW + 10_000, min_agent=0, name="detector", **kw):
    return A.make_bundle(signer, key_id, name, seq, f"1.{seq}", target, model, CARD, SBOM, created, not_after, min_agent, {"map50": 0.61}, **kw)


def device(keys=None, version=1, agent=1) -> A.Device:
    d = A.Device(A.public_bytes(ROOT), TARGET, agent)
    keys = keys if keys is not None else [("k1", A.public_bytes(SIGNER), NOW + 100_000)]
    raw, sig = A.make_keyset(ROOT, version, keys)
    assert d.apply_keyset(raw, sig).ok
    return d


def code(r):
    return r.code


def test_a_valid_bundle_is_accepted_and_verify_changes_nothing():
    d = device()
    b = bundle()
    assert d.verify(b, NOW).ok
    assert d.floor == {} and d.installed == {}
    d.commit(b)
    assert d.floor == {"detector": 1}


def test_rollback_replay_and_idempotent_reinstall():
    d = device()
    b1, b2 = bundle(1), bundle(2, model=MODEL + b"x")
    d.commit(b1)
    assert d.verify(b2, NOW).ok
    d.commit(b2)
    assert code(d.verify(b1, NOW)) == "rollback"  # a valid old artifact replayed
    assert d.verify(b2, NOW).ok  # reinstalling the same one is allowed
    assert code(d.verify(bundle(2, model=MODEL + b"y"), NOW)) == "rollback"  # same seq, different model


def test_rolling_back_means_re_signing_the_old_model_with_a_higher_seq():
    d = device()
    d.commit(bundle(1))
    d.commit(bundle(2, model=MODEL + b"bad"))
    again = bundle(3)  # the original model, new seq
    assert d.verify(again, NOW).ok


def test_the_floor_is_per_model_name():
    d = device()
    d.commit(bundle(5, name="detector"))
    assert d.verify(bundle(1, name="tracker"), NOW).ok


@pytest.mark.parametrize("part", ["manifest", "signature", "model", "model_card", "sbom"])
def test_every_single_bit_flip_in_every_part_is_rejected(part):
    d = device()
    b = bundle()
    raw = bytearray(getattr(b, part))
    for i in range(len(raw)):
        for bit in range(8):
            raw[i] ^= 1 << bit
            tampered = A.Bundle(**{**b.__dict__, part: bytes(raw)}) if hasattr(b, "__dict__") else None
            assert not d.verify(tampered, NOW).ok, (part, i, bit)
            raw[i] ^= 1 << bit
    assert d.verify(b, NOW).ok


def test_truncated_extended_and_malformed_signatures():
    d = device()
    b = bundle()
    for sig in (b"", b.signature[:63], b.signature + b"\x00", b"\x00" * 64):
        assert not d.verify(A.Bundle(b.manifest, sig, b.model, b.model_card, b.sbom), NOW).ok
    assert code(d.verify(A.Bundle(b.manifest, b.signature, b.model[:-1], b.model_card, b.sbom), NOW)) == "model_size"
    assert code(d.verify(A.Bundle(b.manifest, b.signature, b.model + b"\x00", b.model_card, b.sbom), NOW)) == "model_size"
    assert code(d.verify(A.Bundle(b.manifest + b" ", b.signature, b.model, b.model_card, b.sbom), NOW)) == "bad_signature"


def test_wrong_unknown_expired_and_revoked_keys():
    d = device()
    assert code(d.verify(bundle(signer=OTHER), NOW)) == "bad_signature"  # claims k1, signed by someone else
    assert code(d.verify(bundle(signer=OTHER, key_id="k9"), NOW)) == "unknown_key"
    short = device([("k1", A.public_bytes(SIGNER), NOW - 1)])
    assert code(short.verify(bundle(), NOW)) == "key_expired"
    # revocation: a newer keyset without k1 makes previously valid artifacts fail
    raw, sig = A.make_keyset(ROOT, 2, [("k2", A.public_bytes(OTHER), NOW + 10_000)])
    assert d.apply_keyset(raw, sig).ok
    assert code(d.verify(bundle(), NOW)) == "unknown_key"
    assert d.verify(bundle(signer=OTHER, key_id="k2"), NOW).ok


def test_keyset_attacks():
    d = device()
    raw, sig = A.make_keyset(ROOT, 2, [("k1", A.public_bytes(SIGNER), NOW + 1)])
    assert d.apply_keyset(raw, sig).ok
    assert code(d.apply_keyset(raw, sig)) == "keyset_rollback"  # same version
    old, osig = A.make_keyset(ROOT, 1, [("k1", A.public_bytes(SIGNER), NOW + 10**6)])
    assert code(d.apply_keyset(old, osig)) == "keyset_rollback"
    forged, fsig = A.make_keyset(OTHER, 9, [("evil", A.public_bytes(OTHER), NOW + 10**6)])
    assert code(d.apply_keyset(forged, fsig)) == "keyset_bad_signature"
    assert code(d.apply_keyset(raw + b" ", sig)) == "keyset_bad_signature"
    dup, dsig = A.make_keyset(ROOT, 9, [("a", A.public_bytes(SIGNER), 1), ("a", A.public_bytes(OTHER), 1)])
    assert code(d.apply_keyset(dup, dsig)) == "keyset_malformed"
    assert d.keyset_version == 2


def test_a_keyset_signature_is_not_a_manifest_signature_and_domains_are_separate():
    d = device()
    b = bundle()
    no_domain = SIGNER.sign(b.manifest)
    assert code(d.verify(A.Bundle(b.manifest, no_domain, b.model, b.model_card, b.sbom), NOW)) == "bad_signature"
    wrong_domain = SIGNER.sign(A.KEYSET_DOMAIN + b.manifest)
    assert code(d.verify(A.Bundle(b.manifest, wrong_domain, b.model, b.model_card, b.sbom), NOW)) == "bad_signature"


def test_target_agent_and_validity_window():
    d = device(agent=3)
    assert code(d.verify(bundle(target="other-board"), NOW)) == "wrong_target"
    assert code(d.verify(bundle(min_agent=4), NOW)) == "agent_too_old"
    assert d.verify(bundle(min_agent=3), NOW).ok
    assert d.verify(bundle(not_after=NOW), NOW).ok  # valid through not_after
    assert code(d.verify(bundle(not_after=NOW), NOW + 1)) == "expired"
    assert d.verify(bundle(created=NOW + A.CLOCK_SKEW), NOW).ok
    assert code(d.verify(bundle(created=NOW + A.CLOCK_SKEW + 1), NOW)) == "not_yet_valid"


def test_mix_and_match_is_rejected():
    d = device()
    a, b = bundle(1), bundle(2, model=MODEL + b"b")
    assert code(d.verify(A.Bundle(a.manifest, a.signature, b.model, a.model_card, a.sbom), NOW)) in ("model_size", "model_hash")
    assert code(d.verify(A.Bundle(b.manifest, a.signature, b.model, b.model_card, b.sbom), NOW)) == "bad_signature"
    swapped_card = A.Bundle(a.manifest, a.signature, a.model, b"another card", a.sbom)
    assert code(d.verify(swapped_card, NOW)) == "attachment_hash"
    swapped_sbom = A.Bundle(a.manifest, a.signature, a.model, a.model_card, b"{}")
    assert code(d.verify(swapped_sbom, NOW)) == "attachment_hash"
    same_size = bytes([MODEL[0] ^ 1]) + MODEL[1:]
    assert code(d.verify(A.Bundle(a.manifest, a.signature, same_size, a.model_card, a.sbom), NOW)) == "model_hash"


def signed(doc_bytes: bytes) -> A.Bundle:
    """A manifest the real key genuinely signed, so only parsing can reject it."""
    return A.Bundle(doc_bytes, SIGNER.sign(A.MANIFEST_DOMAIN + doc_bytes), MODEL, CARD, SBOM)


def good_doc(**over):
    doc = json.loads(bundle().manifest)
    doc.update(over)
    return doc


def dumps(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("seq"),
        lambda d: d.update(format="fleetforge-manifest/1"),  # the retired format
        lambda d: d["chunks"].update(size=100),
        lambda d: d["chunks"].update(size=2**40),
        lambda d: d["chunks"].update(size=4096.0),
        lambda d: d["chunks"].update(index_sha256="xyz"),
        lambda d: d["chunks"].update(extra=1),
        lambda d: d.pop("chunks"),
        lambda d: d.update(seq=True),
        lambda d: d.update(seq=1.5),
        lambda d: d.update(seq=0),
        lambda d: d.update(seq=-3),
        lambda d: d.update(seq=2**60),
        lambda d: d.update(name="Detector"),
        lambda d: d.update(name="../etc/passwd"),
        lambda d: d.update(name=""),
        lambda d: d.update(target="a b"),
        lambda d: d.update(not_after=d["created"]),
        lambda d: d.update(metrics={"m": float("inf")}),
        lambda d: d.update(metrics={"m": "high"}),
        lambda d: d.update(metrics={str(i): 1 for i in range(17)}),
        lambda d: d["model"].update(sha256="ZZ" * 32),
        lambda d: d["model"].update(extra=1),
        lambda d: d["attachments"].pop("sbom_sha256"),
        lambda d: d.update(key_id="a b"),
        lambda d: d.update(version="x" * 33),
    ],
)
def test_validly_signed_but_malformed_manifests_are_rejected_by_the_parser(mutate):
    doc = good_doc()
    mutate(doc)
    try:
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    except ValueError:
        pytest.skip("not serialisable")
    r = device().verify(signed(raw), NOW)
    assert not r.ok and r.code == "manifest_malformed", r


def test_duplicate_keys_nan_non_utf8_and_oversize_are_rejected():
    d = device()
    good = bundle().manifest
    dup = good[:-1] + b',"seq":999}'
    assert code(d.verify(signed(dup), NOW)) == "manifest_malformed"
    assert code(d.verify(signed(good.replace(b'"map50":0.61', b'"map50":NaN')), NOW)) == "manifest_malformed"
    assert code(d.verify(signed(b"\xff\xfe" + good), NOW)) == "manifest_malformed"
    assert code(d.verify(signed(good + b" " * A.MAX_MANIFEST), NOW)) == "manifest_malformed"  # valid JSON, signed, but over the size cap
    assert code(d.verify(signed(b"[" * 100000), NOW)) == "manifest_malformed"  # deep nesting does not crash
    assert code(d.verify(signed(b"{}"), NOW)) == "manifest_malformed"


def test_commit_never_lowers_the_floor():
    d = device()
    d.commit(bundle(5))
    d.commit(bundle(2))
    assert d.floor["detector"] == 5


def test_signing_is_deterministic_and_keys_are_reproducible():
    assert bundle().manifest == bundle().manifest and bundle().signature == bundle().signature  # Ed25519 is deterministic


def test_a_signed_manifest_that_pins_the_wrong_chunk_index_is_refused_at_the_body_check():
    b = bundle()
    doc = json.loads(b.manifest)
    doc["chunks"]["index_sha256"] = A.sha256(b"not the index")
    r = device().verify(signed(dumps(doc)), NOW)
    assert not r.ok and r.code == "chunk_index"


def test_chunk_index_is_one_hash_per_chunk_with_a_short_last_chunk():
    idx = A.chunk_index(MODEL, 1000)
    assert len(idx) == 32 * math.ceil(len(MODEL) / 1000)
    assert idx[-32:] == hashlib.sha256(MODEL[2000:]).digest()
    assert A.chunk_index(b"", 1000) == b""
