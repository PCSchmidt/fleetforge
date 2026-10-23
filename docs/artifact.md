# Signed model artifacts (milestone 0.2)

What a device checks before it will accept a model, and the attacks the check has been tested against. Python on a laptop; no device, no installer and no network are involved, and the check does not install anything (that is the device agent, milestone 0.3).

## Format

A bundle is five byte strings:

| Part | What it is |
| --- | --- |
| manifest | JSON: format, model name, `seq`, human version, target, model SHA-256 and size, a chunk size and the SHA-256 of the model's chunk-hash index (format version 2; see [delta.md](delta.md)), hashes of the two attachments, `created`, `not_after`, minimum agent version, signing key id, a few reported metrics |
| signature | Ed25519 over `b"fleetforge/manifest/v1\0" + manifest bytes`, exactly as shipped |
| model | the model file |
| model card, SBOM | attachments whose SHA-256 the manifest pins (the SBOM here is whatever bytes the build supplies; no SBOM generation is built) |

A device holds a **pinned root public key**, a **keyset** (signed by the root, versioned, replaced wholesale so leaving a key out revokes it, each key with its own expiry), the **target** it is (for example `jetson-orin-nano-trt10`), its **agent version**, and for each model name a **rollback floor** (the highest `seq` it has accepted) with the hash installed at that `seq`.

This is TUF-inspired and much smaller than TUF: one signing role under one root, no threshold signatures, no separate timestamp or snapshot roles, so it does not defend against a compromised signing key being used to freeze a device on stale metadata beyond the `not_after` window. Signing is with a local key; Sigstore/cosign-style keyless signing, certificates and transparency logs are not implemented, and the source plan's cosign step remains open.

## What the device does, in order

Size cap, strict parse (closed field set, exact types, duplicate JSON keys and NaN rejected), key lookup, key expiry, signature over the exact received bytes (nothing is re-serialised), target, rollback floor, validity window (with 300 s of clock skew allowed for `created`), minimum agent version, model size and SHA-256, attachment hashes. The first failure is reported as a code: `manifest_malformed`, `unknown_key`, `key_expired`, `bad_signature`, `wrong_target`, `rollback`, `expired`, `not_yet_valid`, `agent_too_old`, `model_size`, `model_hash`, `attachment_hash`, `chunk_index`; keyset updates add `keyset_bad_signature`, `keyset_malformed`, `keyset_rollback`.

`verify` changes no state. `commit` raises the floor (never lowers it) and is to be called only after a successful install. A second artifact with the same `seq` is accepted only if it is the same model (a harmless reinstall). To roll back a bad model you do not install the old file: you re-sign the old model with a **higher** `seq`, so the floor never goes down.

## What has been tested

All in `tests/test_artifact.py` and `tests/test_artifact_props.py` (50 tests, a few seconds):

- **Every bit.** Flipping each single bit of the manifest, the signature, the model, the model card and the SBOM is rejected: all of them, not a sample (3,912 manifest bits, 16,384 model bits, 512 signature bits, 104 and 136 for the attachments).
- **Signature and key attacks.** Truncated, extended, empty and all-zero signatures; an artifact signed by the wrong key under a trusted id; an unknown key id; an expired key; revocation through a newer keyset; a keyset signed by someone else, tampered, repeated, rolled back, or with duplicate ids; a signature made without the domain prefix or with the keyset's prefix.
- **Replay and mix-and-match.** An old valid artifact replayed after a newer one; the same `seq` with a different model; a model, manifest, model card or SBOM swapped in from another valid bundle; a same-size model with one byte changed.
- **Parser attacks on genuinely signed manifests** (so only the parser stands in the way): extra and missing fields, wrong format string, boolean, float, zero, negative and huge `seq`, path-traversal and upper-case names, `not_after` before `created`, infinite and string metrics, too many metrics, bad hashes, duplicate keys, NaN, invalid UTF-8, valid JSON padded past the size cap, 100,000 nested brackets (no crash).
- **Properties** (Hypothesis, 300 examples each): over random sequences of installs the floor never falls and nothing below it is ever accepted; arbitrary bytes in every part are never accepted and never crash; any well-formed artifact signed by a trusted key is accepted.
- **Mutation testing** (`scripts/mutation_artifact.py`, `results/mutation_artifact.txt`): 26 injected faults (signature check removed, domain separation dropped, each rollback and equality rule loosened, each hash and size check removed, expiry and skew off by one, keyset rollback allowed, duplicate keys and unknown fields allowed, the floor lowered, booleans accepted as integers, the size cap removed, the name pattern loosened); all 26 are caught. The first run missed one, the size cap, because my test used whitespace that is not valid JSON, so it failed for another reason; the test now uses a validly signed, oversize manifest.

Speed on this laptop, for scale only (not a device measurement): about 90 microseconds to verify a 2 KB bundle and about 0.02 s for a 50 MB model, dominated by hashing.

## What this does not establish

- **No real device, key storage or clock.** The root key would live in OTP or a secure element on a device; here it is a byte string. A device without a trusted clock cannot enforce `not_after`, key expiry or the `created` check, and an attacker who can set its clock back can replay expired artifacts.
- **Root and signing-key compromise are out of scope.** A stolen signing key can sign anything until the root revokes it, which requires the device to receive the new keyset; a device that never connects keeps trusting the old one.
- **Freeze and withholding attacks** (an attacker that blocks updates so a vulnerable model stays) are limited only by `not_after`; there is no timestamp role.
- **The metrics are signed but not enforced.** The manifest carries reported accuracy numbers; deciding whether a model is good enough is the build farm's regression gate and the shadow evaluation (milestone 0.1), not this check.
- **Atomicity.** Whether a power cut or truncated download between `verify` and `commit` can leave a device without a valid model is milestone 0.3 and IronBoot's territory; nothing here protects against it.
- **Implementation risk.** Ed25519 and SHA-256 come from the `cryptography` package and the standard library; the code is my own and has had no external review.

## Reproduce

```
pip install numpy cryptography hypothesis pytest
pytest
python scripts/mutation_artifact.py     # about a minute
```
