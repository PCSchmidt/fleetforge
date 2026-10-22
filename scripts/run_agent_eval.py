"""Evaluation for the update agent: the exhaustive power-cut sweeps and a lossy-link bandwidth table.

    python scripts/run_agent_eval.py        # about a minute; writes results/agent_eval.txt
"""

import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetforge import agent as AG  # noqa: E402
from fleetforge import sweep as S  # noqa: E402
from fleetforge.agent import Agent  # noqa: E402
from fleetforge.link import Server  # noqa: E402
from fleetforge.simfs import PATTERNS  # noqa: E402

out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


t0 = time.time()
w = S.make_world()
log(f"Model {len(w.b2.model)} bytes in chunks of {AG.CHUNK}; trial tries allowed {AG.MAX_TRIES}.\n")
log("Power-cut sweeps. A cut at every storage operation of a full update, under 4 post-cut patterns (unsynced writes: all lost, all kept,")
log("first half kept, every other one kept); invariants are checked on the store right after each cut, then the device is driven to the end.")
log("Double cuts add a second cut at every operation of the recovery run.\n")
total_runs = total_fail = 0
for mode, label in (("good", "good model, confirmed healthy"), ("bad", "bad model, found unhealthy"), ("crashloop", "model that crashes before it can be confirmed")):
    r1 = S.single_cut_sweep(w, mode)
    r2 = S.double_cut_sweep(w, mode, patterns=PATTERNS)
    total_runs += r1.runs + r2.runs
    total_fail += len(r1.failures) + len(r2.failures)
    log(f"  {label}: {r1.ops} operations; single cuts {r1.runs} runs, {len(r1.failures)} failures; double cuts {r2.runs} runs, {len(r2.failures)} failures")
log(f"\n  total {total_runs} cut runs, {total_fail} failures")
n = S.naive_sweep(w)
log(f"\nControl: an updater that overwrites the running model in place, {n.ops} operations x 4 patterns = {n.runs} runs: "
    f"{len(n.failures)} leave the device with no valid model ({100 * len(n.failures) / n.runs:.0f}%).")

log("\nLossy link. 128 KiB model, 200 devices per row; a request is dropped with the given probability, short reads 20%, one corrupted byte per 100 chunks.")
log("An update call gives up after 6 failed requests and the next call resumes (or, in the right column, starts over). Up to 60 calls.\n")
log(f"  {'drop':>5} | {'with resume':^44} | {'without resume':^44}")
log(f"  {'':>5} | {'done':>6} {'calls':>6} {'bytes sent / size':>18} {'verify fails':>12} | {'done':>6} {'calls':>6} {'bytes sent / size':>18} {'verify fails':>12}")
big = S.make_world(model_size=32 * AG.CHUNK)
for p_drop in (0.0, 0.1, 0.3, 0.5):
    cells = []
    for resume in (True, False):
        done = calls = vf = 0
        sent = 0
        for seed in range(200):
            fs = S.clone(big.base)
            server = Server(big.b2, random.Random(seed), p_drop=p_drop, p_short=0.2, p_corrupt=0.01)
            for c in range(1, 61):
                a = Agent(fs, big.root_pub, S.TARGET)
                a.resume = resume
                res = a.update(server, S.NOW)
                vf += res.startswith("verify_failed")
                if res == "pending":
                    done += 1
                    calls += c
                    break
            sent += server.bytes_sent
        cells.append(f"{100 * done / 200:5.0f}% {calls / max(done, 1):6.1f} {sent / 200 / len(big.b2.model):18.2f} {vf / 200:12.2f}")
    log(f"  {p_drop:5.1f} | {cells[0]} | {cells[1]}")
log(f"\nTotal time {time.time() - t0:.0f} s")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "agent_eval.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
