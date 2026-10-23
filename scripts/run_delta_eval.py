"""Bytes on the wire for an update with the real agent, on real model pairs.

    python scripts/run_delta_eval.py <export_dir>      # sentinel-node/experiments/detector/out/export

The factory model is the old file; the server offers the new one. The agent fetches only chunks it does not already
hold, compressed when that helps, and verifies every chunk against the signed index. 'lzma whole' is the new file
compressed in one piece, for scale. Lossless link; the last rows repeat one pair on a lossy one.
"""

import lzma
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetforge import sweep as S  # noqa: E402
from fleetforge.agent import Agent, provision  # noqa: E402
from fleetforge.artifact import make_bundle, make_keyset, public_bytes  # noqa: E402
from fleetforge.link import Server  # noqa: E402
from fleetforge.simfs import SimFS  # noqa: E402

export = pathlib.Path(sys.argv[1])
PAIRS = [
    ("trained checkpoint, seed 1 -> seed 2 (retrained)", "fp16_s1/best.pt", "fp16_s2/best.pt"),
    ("ONNX, min-max -> entropy calibration (same weights)", "study_minmax_c0/best.onnx", "study_entropy_c0/best.onnx"),
    ("ONNX, FP16 build -> INT8 build (same weights)", "fp16/best.onnx", "int8/best.onnx"),
    ("TensorRT INT8 engine, min-max -> entropy", "study_minmax_c0/best.engine", "study_entropy_c0/best.engine"),
    ("TensorRT engine, FP16 -> INT8", "fp16/best.engine", "int8/best.engine"),
]
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


root_key, signer = S._key("root"), S._key("signer")
keyset = make_keyset(root_key, 1, [("k1", public_bytes(signer), S.NOW + 10**6)])


def update(old: bytes, new: bytes, chunk: int, p_drop=0.0, p_corrupt=0.0, seed=0):
    mk = lambda seq, model: make_bundle(signer, "k1", S.NAME, seq, f"v{seq}", S.TARGET, model, b"card", b"{}", S.NOW - 100, S.NOW + 10**5, chunk_size=chunk)  # noqa: E731
    b1, b2 = mk(1, old), mk(2, new)
    fs = SimFS()
    provision(fs, public_bytes(root_key), S.TARGET, b1, keyset)
    agent = Agent(fs, public_bytes(root_key), S.TARGET)
    server = Server(b2, random.Random(seed), p_drop=p_drop, p_corrupt=p_corrupt)
    calls = 0
    while True:
        calls += 1
        res = agent.update(server, S.NOW)
        if res == "pending" or calls > 60:
            break
    return res, server.bytes_sent, agent.stats, calls


log(f"{'pair':<56} {'new size':>9} {'lzma whole':>10} | {'chunks 4 KiB':^24} | {'chunks 64 KiB':^24}")
log(f"{'':<56} {'':>9} {'':>10} | {'wire bytes':>10} {'reused':>12} | {'wire bytes':>10} {'reused':>12}")
for label, a, b in PAIRS:
    old, new = (export / a).read_bytes(), (export / b).read_bytes()
    whole = len(lzma.compress(new, preset=6))
    cells = []
    for chunk in (4096, 65536):
        res, sent, st, _ = update(old, new, chunk)
        assert res == "pending", res
        n = st["fetched"] + st["reused"]
        cells.append(f"{100 * sent / len(new):>9.1f}% {st['reused']:>4}/{n:<5}")
    log(f"{label:<56} {len(new):>9} {100 * whole / len(new):>9.0f}% | {cells[0]:>24} | {cells[1]:>24}")
log("\n(wire bytes include the manifest, signature, chunk index and attachments; percentages are of the new file's size)")

old, new = (export / PAIRS[3][1]).read_bytes(), (export / PAIRS[3][2]).read_bytes()
log("\nOne pair (TensorRT INT8 engine, min-max -> entropy, 4 KiB chunks) on a lossy link, 10 devices each:")
log(f"  {'drop rate':>9} {'corrupted chunks':>17} {'done':>6} {'calls':>6} {'wire bytes / size':>18}")
for p_drop, p_corrupt in ((0.0, 0.0), (0.0, 0.02), (0.3, 0.02), (0.5, 0.02)):
    done = calls_sum = 0
    sent = 0.0
    for seed in range(10):
        res, s, st, calls = update(old, new, 4096, p_drop, p_corrupt, seed)
        done += res == "pending"
        calls_sum += calls
        sent += s / len(new)
    log(f"  {p_drop:>9.1f} {p_corrupt:>17.2f} {done:>5}/10 {calls_sum / 10:>6.1f} {sent / 10:>18.2f}")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "delta_eval.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
