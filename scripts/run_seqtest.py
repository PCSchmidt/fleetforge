"""Experiments for the sequential shadow-evaluation test. Writes results/seqtest.txt and results/seqtest.json.

Run from the repo root:  python scripts/run_seqtest.py   (a few minutes)
"""

import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetforge import methods as M  # noqa: E402
from fleetforge import streams as S  # noqa: E402

N = 40000  # frames per stream
REPS = 2000
CHUNK = 500
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def run(params, rules, reps=REPS, seed=0):
    """Return {rule: (outcome, stop)} over `reps` independent streams, simulated in chunks."""
    rng = np.random.default_rng(seed)
    res = {k: ([], []) for k in rules}
    for _ in range(reps // CHUNK):
        d = S.simulate(params, CHUNK, N, rng)
        for k, f in rules.items():
            o, s = f(d)
            res[k][0].append(o)
            res[k][1].append(s)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in res.items()}


def pct(o, which):
    p, lo, hi = wilson(int((o == which).sum()), len(o))
    return f"{100 * p:5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]"


def main():
    t0 = time.time()
    log(f"Sequential shadow evaluation: {REPS} simulated streams of {N} frames per cell; one-sided level 2.5% per direction.")
    log("Frame model: disagreement rate q = 0.10; the candidate's gain varies with scene (AR(1), sd sigma_z = 0.03, correlation time tau).")
    log("Cells show the percentage of streams ending in the stated decision, with a 95% Wilson interval.\n")

    block_sizes = [250, 1000, 4000]
    rules = {
        "fixed z-test, frames independent (1 look)": lambda d: M.fixed_naive(d),
        "peek every 100 frames, z-test": lambda d: M.naive_peek(d, 100),
        "iid mixture sequence": lambda d: M.iid_mixture(d, 100),
    }
    for b in block_sizes:
        rules[f"fixed z-test on blocks B={b} (1 look)"] = lambda d, b=b: M.fixed_block(d, b)
        rules[f"block betting B={b}"] = lambda d, b=b: M.block_betting(d, b)
        rules[f"block mixture B={b}"] = lambda d, b=b: M.block_mixture(d, b)

    # ---------------------------------------------------------------- E1 validity under no difference
    log("E1. Candidate truly equal to production (delta = 0). Wrongly promoting is a type I error; the target is <= 2.5%.")
    results = {}
    for tau, sigma in [(1, 0.0), (100, 0.03), (500, 0.03), (2000, 0.03)]:
        label = "independent frames" if sigma == 0 else f"scene variation sd 0.03, correlation time {tau} frames"
        res = run(S.StreamParams(0.0, 0.10, sigma, tau), rules, seed=1)
        results[f"E1 {label}"] = {k: {"promote": int((v[0] == 1).sum()), "reject": int((v[0] == -1).sum())} for k, v in res.items()}
        log(f"\n  {label}")
        for k, (o, s) in res.items():
            log(f"    {k:<44} promote {pct(o, 1)}   reject {pct(o, -1)}")

    # ---------------------------------------------------------------- E2 candidate slightly worse
    log("\nE2. Candidate slightly worse (delta = -0.01), scene variation tau = 500. Any promotion is an error.")
    res = run(S.StreamParams(-0.01, 0.10, 0.03, 500), rules, seed=2)
    results["E2"] = {k: {"promote": int((v[0] == 1).sum()), "reject": int((v[0] == -1).sum())} for k, v in res.items()}
    for k, (o, s) in res.items():
        log(f"    {k:<44} promote {pct(o, 1)}   reject {pct(o, -1)}")

    # ---------------------------------------------------------------- E3 power and speed
    log("\nE3. Candidate truly better, scene variation tau = 100 frames. Decisions in frames (median and 90th percentile among promotions).")
    keep = {k: v for k, v in rules.items() if "B=1000" in k or k.startswith("peek") or k.startswith("iid") or k.startswith("fixed z-test, ")}
    keep = dict(keep)
    for n_pre in (10000, 20000):
        keep[f"fixed z-test on blocks B=1000, stream cut at {n_pre}"] = lambda d, n_pre=n_pre: M.fixed_block(d[:, :n_pre], 1000)
    for delta in [0.005, 0.01, 0.02, 0.04]:
        res = run(S.StreamParams(delta, 0.10, 0.03, 100), keep, seed=3)
        results[f"E3 delta={delta}"] = {}
        log(f"\n  true gain {100 * delta:.1f} accuracy points")
        for k, (o, s) in res.items():
            sel = s[o == 1]
            med = f"{int(np.median(sel)):>6}" if len(sel) else "   n/a"
            p90 = f"{int(np.percentile(sel, 90)):>6}" if len(sel) else "   n/a"
            log(f"    {k:<44} promote {pct(o, 1)}   median {med}  p90 {p90} frames")
            results[f"E3 delta={delta}"][k] = {"promote": int((o == 1).sum()), "median": float(np.median(sel)) if len(sel) else None}

    # ---------------------------------------------------------------- E4 injected bad model
    log("\nE4. Injected bad model (delta = -0.03): how fast is it rejected? tau = 100.")
    res = run(S.StreamParams(-0.03, 0.10, 0.03, 100), keep, seed=4)
    for k, (o, s) in res.items():
        sel = s[o == -1]
        med = f"{int(np.median(sel)):>6}" if len(sel) else "   n/a"
        p90 = f"{int(np.percentile(sel, 90)):>6}" if len(sel) else "   n/a"
        log(f"    {k:<44} reject {pct(o, -1)}   median {med}  p90 {p90} frames   (wrongly promoted {pct(o, 1)})")
        results.setdefault("E4", {})[k] = {"reject": int((o == -1).sum()), "median": float(np.median(sel)) if len(sel) else None}

    # ---------------------------------------------------------------- E5 sensitivity to the clipping bound c
    log("\nE5. Betting rule: sensitivity to the design bound c on a block's gain (B = 1000, tau = 100).")
    cs = {f"block betting B=1000 c={c}": (lambda d, c=c: M.block_betting(d, 1000, c=c)) for c in (0.03, 0.05, 0.1, 0.2)}
    for delta in (0.0, -0.01, 0.01, 0.02):
        res = run(S.StreamParams(delta, 0.10, 0.03, 100), cs, seed=5)
        log(f"\n  true gain {100 * delta:+.1f} points")
        for k, (o, st) in res.items():
            sel = st[o == 1]
            med = f"{int(np.median(sel)):>6}" if len(sel) else "   n/a"
            log(f"    {k:<44} promote {pct(o, 1)}   reject {pct(o, -1)}   median promote {med} frames")

    # ---------------------------------------------------------------- E6 held-out settings for the chosen default c = 0.05
    log("\nE6. Held-out settings (not used when c = 0.05 was chosen): different disagreement rate, scene variation and correlation time.")
    for q, sigma, tau, b, gain in [(0.20, 0.06, 300, 1500, 0.02), (0.05, 0.015, 100, 1000, 0.01), (0.10, 0.05, 150, 1000, 0.02)]:
        rules6 = {
            f"block betting B={b} c=0.05": lambda d, b=b: M.block_betting(d, b, c=0.05),
            f"block mixture B={b}": lambda d, b=b: M.block_mixture(d, b, q_guess=0.1),
            f"fixed z-test on blocks B={b} (1 look)": lambda d, b=b: M.fixed_block(d, b),
        }
        log(f"\n  q = {q}, scene variation sd {sigma}, tau {tau}, block {b} frames")
        for delta, tag in [(0.0, "no difference"), (-gain / 2, "slightly worse"), (gain, "better")]:
            res = run(S.StreamParams(delta, q, sigma, tau), rules6, seed=6)
            log(f"   {tag} (true gain {100 * delta:+.1f} points)")
            for k, (o, st) in res.items():
                sel = st[o == 1]
                med = f"{int(np.median(sel)):>6}" if len(sel) else "   n/a"
                log(f"    {k:<44} promote {pct(o, 1)}   reject {pct(o, -1)}   median promote {med} frames")

    log(f"\nTotal time {time.time() - t0:.0f} s")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "seqtest.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    (ROOT / "results" / "seqtest.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
