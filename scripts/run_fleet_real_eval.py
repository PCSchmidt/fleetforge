"""Run the staged rollout with a real update agent on each of 300 simulated devices, next to the abstract model.

    python scripts/run_fleet_real_eval.py [fleets_per_candidate]    # default 10; writes results/fleet_real_eval.txt
"""

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetforge import agent as AG  # noqa: E402
from fleetforge import fleet as F  # noqa: E402
from fleetforge import fleet_real as FR  # noqa: E402
from fleetforge import sweep as S  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = F.FleetConfig()
world = S.make_world(32 * AG.CHUNK)  # 128 KiB, the model size the link calibration was taken with
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


def med(v):
    return f"{np.median(v):6.0f}" if len(v) else "   n/a"


t0 = time.time()
log(f"Staged e-process rollout to {cfg.n} devices, {N} fleets per candidate. 'real' = a real update agent, store and lossy link on every device;")
log("'abstract' = the Monte Carlo device model, on the same simulated measurements (same seeds). Minutes of simulated time, medians.\n")
total_viol = total_dev = 0
for cand in (F.Candidate("better (+2 points)", 0.02), F.Candidate("worse (-3 points)", -0.03), F.Candidate("crashes on 40% of devices", 0.0, 0.4)):
    real, abst = [], []
    for seed in range(N):
        real.append(FR.run_real(F.EProcessStaged(), cfg, cand, seed, world))
        abst.append(F.run_rollout(F.EProcessStaged(), cfg, F.make_devices(cfg, cand, np.random.default_rng(seed))))
    total_viol += sum(len(r.violations) for r in real)
    total_dev += N * cfg.n
    log(f"  {cand.name}")
    log(f"    rolled back: real {sum(r.rolled_back for r in real)}/{N}, abstract {sum(o.rolled_back for o in abst)}/{N}")
    log(f"    minutes to rollback decision: real {med([r.t_alarm for r in real if r.rolled_back])}, abstract {med([o.t_alarm for o in abst if o.rolled_back])}")
    log(f"    minutes until every device is back on production: real {med([r.t_rolled_back for r in real if r.rolled_back])}, abstract {med([o.t_rolled_back for o in abst if o.rolled_back])}")
    log(f"    minutes to 100% when promoted: real {med([r.t_promoted for r in real if not r.rolled_back])}, abstract {med([o.t_promoted for o in abst if not o.rolled_back])}")
    log(f"    devices that served the candidate: real {med([r.served for r in real])}, abstract {med([o.served for o in abst])}")
    log(f"    update calls per device: {np.mean([r.update_calls for r in real]) / cfg.n:.2f}; model data sent per fleet: {np.mean([r.bytes_sent for r in real]) / 1e6:.1f} MB")
log(f"\nPer-device checks at the end of every run (active model verifies, nobody left in a trial, floor matches what the device ended on, crashes remembered): "
    f"{total_viol} violations over {total_dev} device-runs.")
log(f"Total time {time.time() - t0:.0f} s")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "fleet_real_eval.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
