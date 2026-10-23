"""Monte Carlo evaluation of rollout policies on a simulated fleet of 300 devices.

    python scripts/run_fleet_eval.py [replications]     # default 400; writes results/fleet_eval.txt
"""

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetforge import fleet as F  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 400
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def med(v):
    return f"{np.median(v):6.0f}" if len(v) else "   n/a"


def evaluate(cfg, cand, policies, seed=0):
    res = {p.name: [] for p in policies}
    for r in range(R):
        dev = F.make_devices(cfg, cand, np.random.default_rng(seed * 100000 + r))
        for p in policies:
            res[p.name].append(F.run_rollout(p, cfg, dev))
    return res


def row(name, os_, model_mb):
    rb = [o for o in os_ if o.rolled_back]
    pr = [o for o in os_ if not o.rolled_back]
    p, lo, hi = wilson(len(rb), len(os_))
    served = np.mean([o.served for o in os_])
    return (
        f"    {name:<26} rolled back {100 * p:5.1f}% [{100 * lo:4.1f},{100 * hi:5.1f}]   "
        f"minutes to rollback decision {med([o.t_alarm for o in rb])}, all devices back {med([o.t_rolled_back for o in rb])}   "
        f"minutes to 100% when promoted {med([o.t_promoted for o in pr])}   "
        f"devices that served it {served:6.1f}   device-hours on it {np.mean([o.exposure_dev_hours for o in os_]):8.1f}   "
        f"model data sent {served * model_mb / 1024:5.1f} GB"
    )


t0 = time.time()
cfg = F.FleetConfig()
policies = [F.EProcessStaged(), F.AllAtOnce(), F.FixedCanary()]
log(f"Rollout of a candidate model to {cfg.n} simulated devices, {R} fleets per row; same fleets for every policy.")
log(f"Devices poll every {cfg.poll_min:.0f} min; a trial lasts {cfg.trial_min:.0f} min; device-to-device spread of the true gain {cfg.sigma_dev}; "
    f"link drop rates {cfg.drop_lo}-{cfg.drop_hi}; healthy-model crash rate {cfg.crash_base}.")
log("Staged e-process: cohorts 5, 25, 100, rest; harm level 5%. Fixed canary: 5 devices, roll back if their mean report < -0.01 or 2 crash.")
log("'device-hours on it' for a promoted candidate counts to a 24 h horizon. Gain is in accuracy points / 100 (0.02 = +2 points).\n")
cands = [
    F.Candidate("better (+2 points)", 0.02),
    F.Candidate("equal (0)", 0.0),
    F.Candidate("slightly worse (-1 point)", -0.01),
    F.Candidate("worse (-3 points)", -0.03),
    F.Candidate("crashes on 40% of devices", 0.0, 0.4),
]
for cand in cands:
    log(f"  {cand.name}")
    res = evaluate(cfg, cand, policies)
    for p in policies:
        log(row(p.name, res[p.name], cfg.model_mb))
    log()

log("Sensitivity of the staged e-process to the device-to-device spread of the true gain and to the clipping bound c (equal and worse candidates).")
for sigma in (0.0, 0.02, 0.04):
    for c in (0.05, 0.1, 0.2):
        cfg2 = F.FleetConfig(sigma_dev=sigma)
        pol = [F.EProcessStaged(c=c)]
        eq = evaluate(cfg2, F.Candidate("equal", 0.0), pol, seed=7)[pol[0].name]
        bad = evaluate(cfg2, F.Candidate("worse", -0.03), pol, seed=8)[pol[0].name]
        sl = evaluate(cfg2, F.Candidate("slightly worse", -0.01), pol, seed=9)[pol[0].name]
        pe, le, he = wilson(sum(o.rolled_back for o in eq), R)
        log(
            f"  sigma_dev {sigma:.2f}, c {c:.2f}: equal candidate rolled back {100 * pe:4.1f}% [{100 * le:3.1f},{100 * he:4.1f}] (target <= 5%); "
            f"-1 point caught {100 * np.mean([o.rolled_back for o in sl]):5.1f}%; -3 points caught {100 * np.mean([o.rolled_back for o in bad]):5.1f}% "
            f"(median devices exposed {np.median([o.served for o in bad if o.rolled_back]) if any(o.rolled_back for o in bad) else float('nan'):.0f})"
        )
log(f"\nTotal time {time.time() - t0:.0f} s")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "fleet_eval.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
