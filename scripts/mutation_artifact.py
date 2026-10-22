"""Mutation test for the artifact verifier: break it in specific ways and check the test suite notices.

Run from the repo root: python scripts/mutation_artifact.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fleetforge" / "artifact.py"

MUTANTS = [
    ("signature check skipped", "            pub.verify(b.signature, MANIFEST_DOMAIN + b.manifest)\n", "            pass\n"),
    ("manifest domain separation dropped", "pub.verify(b.signature, MANIFEST_DOMAIN + b.manifest)", "pub.verify(b.signature, b.manifest)"),
    ("keyset signature check skipped", "            self.root.verify(signature, KEYSET_DOMAIN + raw)\n", "            pass\n"),
    ("keyset domain separation dropped", "self.root.verify(signature, KEYSET_DOMAIN + raw)", "self.root.verify(signature, raw)"),
    ("rollback below floor allowed", "if m.seq < floor:", "if m.seq < floor - 1000000:"),
    ("same seq with a different model allowed", "if m.seq == floor and self.installed.get(m.name) != m.model_sha256:", "if False:"),
    ("same seq reinstall refused", "if m.seq == floor and self.installed.get(m.name) != m.model_sha256:", "if m.seq == floor:"),
    ("target not checked", "if m.target != self.target:", "if False:"),
    ("model hash not checked", "if sha256(b.model) != m.model_sha256:", "if False:"),
    ("model size not checked", "if len(b.model) != m.model_size:", "if False:"),
    ("model card hash not checked", "if sha256(b.model_card) != m.model_card_sha256:", "if False:"),
    ("sbom hash not checked", "if sha256(b.sbom) != m.sbom_sha256:", "if False:"),
    ("expiry off by one", "if now > m.not_after:", "if now >= m.not_after:"),
    ("expiry not checked", "if now > m.not_after:", "if False:"),
    ("future manifests allowed", "if m.created > now + CLOCK_SKEW:", "if False:"),
    ("skew boundary off by one", "if m.created > now + CLOCK_SKEW:", "if m.created >= now + CLOCK_SKEW:"),
    ("key expiry not checked", "if now > key_not_after:", "if False:"),
    ("agent version not checked", "if m.min_agent > self.agent_version:", "if False:"),
    ("keyset rollback allowed", "if version <= self.keyset_version:", "if version < self.keyset_version:"),
    ("duplicate JSON keys allowed", "        if k in d:\n            raise ParseError", "        if False:\n            raise ParseError"),
    ("unknown fields allowed", "    if extra or missing:", "    if missing:"),
    ("floor can be lowered", "self.floor[m.name] = max(self.floor.get(m.name, 0), m.seq)", "self.floor[m.name] = m.seq"),
    ("booleans accepted as integers", "if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:", "if not isinstance(v, int) or not lo <= v <= hi:"),
    ("manifest size cap removed", "    if len(raw) > MAX_MANIFEST:\n        raise ParseError(\"manifest too large\")\n", ""),
    ("not_after before created allowed", "    if not_after <= created:\n        raise ParseError(\"not_after must be after created\")\n", ""),
    ("name pattern loosened", 'NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")', 'NAME_RE = re.compile(r"^.{1,64}$")'),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests/test_artifact.py", "tests/test_artifact_props.py"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    return r.returncode == 0


def main() -> None:
    assert run_tests(), "the unmutated suite must pass first"
    original = SRC.read_text(encoding="utf-8")
    killed = 0
    for label, old, new in MUTANTS:
        assert original.count(old) == 1, f"{label}: pattern occurs {original.count(old)} times"
        SRC.write_text(original.replace(old, new), encoding="utf-8")
        try:
            survived = run_tests()
        finally:
            SRC.write_text(original, encoding="utf-8")
        killed += not survived
        print(f"{'SURVIVED' if survived else 'killed':9} {label}", flush=True)
    print(f"\n{killed}/{len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    main()
