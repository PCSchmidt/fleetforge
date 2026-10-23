"""Mutation test for the control plane (decision logic and e-processes). Run from the repo root."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fleetforge" / "fleet.py"

MUTANTS = [
    ("harm bet has the wrong sign", "y = -max(-self.c, min(self.c, m)) / self.c  # positive when the candidate is worse", "y = max(-self.c, min(self.c, m)) / self.c"),
    ("harm alarm threshold halved", "if self.bet.update(report) >= math.log(1.0 / p.beta):", "if self.bet.update(report) >= math.log(1.0 / p.beta) / 2:"),
    ("harm alarm never raised", "if self.bet.update(report) >= math.log(1.0 / p.beta):", "if False:"),
    ("crash alarm never raised", "if crashed and not self.fixed and self.crash_rate.update(exposure_min) >= math.log(1.0 / p.crash_beta):", "if False:"),
    ("crash test ignores exposure time", "return self.n * math.log(self.lam1 / self.lam0) - (self.lam1 - self.lam0) * exposure_min", "return self.n * math.log(self.lam1 / self.lam0)"),
    ("stages open without waiting for reports", "if self.next_stage < len(self.members) and self.resolved[self.next_stage - 1] >= self._need(self.next_stage - 1):", "if self.next_stage < len(self.members):"),
    ("promote after the first stage", "if st == len(self.members) - 1 and self.resolved[st] >= self._need(st):", "if True:"),
    ("canary threshold sign flipped", "if canary_vals and float(np.mean(self.canary_vals)) < p.threshold:".replace("canary_vals", "self.canary_vals").replace("self.self.", "self."), "if self.canary_vals and float(np.mean(self.canary_vals)) > p.threshold:"),
    ("fixed canary rolls out before the canary reports", "if self.fixed:\n            if st == 0 and self.resolved[0] >= self._need(0):", "if self.fixed:\n            if st == 0 and self.resolved[0] >= 1:"),
    ("everyone in one stage for the staged policy", "stages = policy.stages", "stages = (cfg.n,)"),
    ("devices not drawn at random", "self.members.append([int(i) for i in order[pos : pos + take]])", "self.members.append([int(i) for i in sorted(order)[pos : pos + take]])"),
    ("alarm leaves devices serving (no rollback time)", "back = {int(i): t_alarm + ((dev.phase[i] - t_alarm) % cfg.poll_min)", "back = {int(i): t_alarm"),
]


def run_tests() -> bool:
    """True if the suite passes (the mutant survived). A run that hangs or is far slower than normal counts as caught:
    a mutant that stops a rollout from ever finishing is noticed by the tests that wait for it."""
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests/test_fleet.py", "-k", "not real"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def main() -> None:
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
