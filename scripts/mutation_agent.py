"""Mutation test for the update agent: break it in specific ways and check the test suite (which includes the
exhaustive power-cut sweeps) notices.

Run from the repo root: python scripts/mutation_agent.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fleetforge" / "agent.py"

MUTANTS = [
    ("state replaced without syncing the temp file", '        self.fs.sync("state.tmp")\n', ""),
    ("state overwritten in place instead of renamed", '        self.fs.rename("state.tmp", "state.json")', '        self.fs.write("state.json", 0, self.fs.read("state.tmp"))\n        self.fs.sync("state.json")'),
    ("update written into the running slot", 'slot = other(st["active"])', 'slot = st["active"]'),
    ("full verification before staging skipped", "if res is None or not res.ok:", "if res is None:"),
    ("header verification skipped", '        if not r.ok:\n            return "rejected:" + r.code', "        if False:\n            pass"),
    ("boot does not count tries", 'tries = p["tries"] + 1', 'tries = p["tries"]'),
    ("rolled-back model not remembered", '        st["bad"].append(p["hash"])\n        st["pending"] = None\n        self._save(st)\n        return "rolled_back"', '        st["pending"] = None\n        self._save(st)\n        return "rolled_back"'),
    ("rejected hashes not consulted", 'if m.model_sha256 in st["bad"]:', "if False:"),
    ("unhealthy model promoted", "        if healthy:", "        if True:"),
    ("chunk data not synced before the journal line", "            self.fs.write(mpath, off, chunk)\n            self.fs.sync(mpath)\n", "            self.fs.write(mpath, off, chunk)\n"),
    ("journal chunk hashes not checked on resume", " or sha256(part[off : off + ln]) != h:", ":"),
    ("journal not tied to the manifest", "if j is None or not j.startswith(head) or not self.resume:", "if j is None or not self.resume:"),
    ("pending slot not verified at boot", 'elif not self._slot_ok(d, p["slot"]):', "elif False:"),
    ("update runs while a trial is pending", '        if st["pending"]:\n            return "pending"\n        d = self._device(st)', "        d = self._device(st)"),
    ("already-installed model re-downloaded", '        if st["installed"].get(m.name) == m.model_sha256:\n            return "up_to_date"\n', ""),
    ("active slot not verified at boot", "if running == st[\"active\"] and not self._slot_ok(d, running):", "if False:"),
    ("floor never raised on promotion", '            st["floor"][p["name"]] = max(st["floor"].get(p["name"], 0), p["seq"])\n', ""),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
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
