"""How much of an update could a delta save, measured on real model pairs?

    python scripts/measure_delta.py <export_dir>      # sentinel-node/experiments/detector/out/export

Pairs (old -> new) are real artifacts from the sentinel-node detector study: independently trained checkpoints,
and ONNX / TensorRT files built from the same weights with different INT8 calibration. For each pair:

  full            the new file as it is, and compressed with lzma (what shipping it whole costs)
  fixed chunks    bytes in 4 KiB / 64 KiB aligned chunks of the new file whose SHA-256 does not occur in the old file
  content chunks  the same with content-defined chunk boundaries (a gear-hash chunker, ~4 KiB average), which
                  survives insertions and shifts that break fixed alignment
  xor + lzma      new XOR old (aligned, old padded with zeros), lzma-compressed: a simple byte-wise patch

Savings are fractions of the full (uncompressed) size so the columns compare directly.
"""

import hashlib
import lzma
import pathlib
import sys
import zlib

import numpy as np

root = pathlib.Path(sys.argv[1])


def fixed_new_bytes(old: bytes, new: bytes, c: int) -> int:
    have = {hashlib.sha256(old[i : i + c]).digest() for i in range(0, len(old), c)}
    return sum(len(new[i : i + c]) for i in range(0, len(new), c) if hashlib.sha256(new[i : i + c]).digest() not in have)


_GEAR = np.random.default_rng(7).integers(0, 2**63, size=256, dtype=np.uint64)


def cdc_chunks(b: bytes, avg_bits: int = 12, lo: int = 1024, hi: int = 16384):
    """Gear-hash content-defined chunking: cut where the hash's low `avg_bits` bits are zero."""
    mask = np.uint64((1 << avg_bits) - 1)
    arr = np.frombuffer(b, dtype=np.uint8)
    h = np.uint64(0)
    start = 0
    out = []
    g = _GEAR
    for i in range(len(arr)):
        h = np.uint64((int(h) << 1 & 0xFFFFFFFFFFFFFFFF) + int(g[arr[i]]) & 0xFFFFFFFFFFFFFFFF)
        n = i - start + 1
        if (n >= lo and (int(h) & int(mask)) == 0) or n >= hi:
            out.append((start, i + 1))
            start, h = i + 1, np.uint64(0)
    if start < len(arr):
        out.append((start, len(arr)))
    return out


def cdc_new_bytes(old: bytes, new: bytes) -> int:
    have = {hashlib.sha256(old[a:b]).digest() for a, b in cdc_chunks(old)}
    return sum(b - a for a, b in cdc_chunks(new) if hashlib.sha256(new[a:b]).digest() not in have)


def xor_lzma(old: bytes, new: bytes) -> int:
    a = np.frombuffer(new, dtype=np.uint8)
    o = np.zeros(len(a), dtype=np.uint8)
    n = min(len(old), len(a))
    o[:n] = np.frombuffer(old, dtype=np.uint8)[:n]
    return len(lzma.compress((a ^ o).tobytes(), preset=6))


PAIRS = [
    ("trained checkpoint, seed 0 -> seed 1 (retrained)", "fp16/best.pt", "fp16_s1/best.pt"),
    ("trained checkpoint, seed 1 -> seed 2 (retrained)", "fp16_s1/best.pt", "fp16_s2/best.pt"),
    ("ONNX INT8, min-max calibration -> entropy calibration (same weights)", "study_minmax_c0/best.onnx", "study_entropy_c0/best.onnx"),
    ("ONNX INT8, calibration set 0 -> set 1 (same weights, same method)", "study_entropy_c0/best.onnx", "study_entropy_c1/best.onnx"),
    ("ONNX FP16 build -> INT8 build (same weights)", "fp16/best.onnx", "int8/best.onnx"),
    ("TensorRT INT8 engine, min-max -> entropy (same weights)", "study_minmax_c0/best.engine", "study_entropy_c0/best.engine"),
    ("TensorRT engine, FP16 -> INT8 (same weights)", "fp16/best.engine", "int8/best.engine"),
    ("TensorRT INT8 engine, seed 1 -> seed 2 (retrained)", "int8_s1/best.engine", "int8_s2/best.engine"),
]

print(f"{'pair':<70} {'size':>9} {'lzma full':>10} {'fixed 4K':>9} {'fixed 64K':>10} {'content ~4K':>12} {'xor+lzma':>9}")
rows = []
for label, a, b in PAIRS:
    old, new = (root / a).read_bytes(), (root / b).read_bytes()
    full = len(new)
    comp = len(lzma.compress(new, preset=6))
    f4, f64 = fixed_new_bytes(old, new, 4096), fixed_new_bytes(old, new, 65536)
    cd = cdc_new_bytes(old, new)
    xl = xor_lzma(old, new)
    rows.append((label, full, comp, f4, f64, cd, xl))
    print(f"{label:<70} {full:>9} {100 * comp / full:>9.0f}% {100 * f4 / full:>8.0f}% {100 * f64 / full:>9.0f}% {100 * cd / full:>11.0f}% {100 * xl / full:>8.0f}%", flush=True)
print("\ncolumns after 'size' are bytes that would still have to be sent, as a percentage of the new file's size")
