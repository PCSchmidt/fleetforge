# Chunk hashes, reuse and compression: what updates can save (milestone 0.6, partly)

Three changes to the update path, measured on real model files from the sentinel-node detector study: every chunk of a model is verified against a signed index as it arrives (so one damaged byte costs one chunk), chunks the device already holds are reused, and chunks are compressed on the wire. The short answer on bandwidth: **a delta helps a lot when the new file is a deterministic function of the weights the device already has (ONNX files built with different calibration: 99% saved), not at all when the weights were retrained (97% to 100% of the chunks differ), and compressing a compiled TensorRT engine saves more than any delta of it (about 45%).** All of this is a laptop simulation on these files; no device, no real network.

## What was built

- **Signed chunk index.** The manifest (format version 2) carries a chunk size and the SHA-256 of the *index*: the model's chunk hashes concatenated, 32 bytes per chunk. A device fetches the index, checks it against the signed pin, then checks every chunk against its entry as it arrives. A flat list rather than a Merkle tree, because the index is small (at 256 KiB chunks, 100 MB of model needs a 12.8 KB list) and a flat list needs no proofs; a tree would only pay for models of tens of gigabytes. The device refuses an update whose served index does not match the pin before it fetches any of the model.
- **Per-chunk retry.** A chunk that arrives damaged, short or undecodable is asked for again; the rest of the download is kept. The download journal records each verified chunk's hash, so a resume after a power cut also re-checks the prefix it trusts.
- **Reuse.** A chunk of the new model whose hash matches a chunk of the running model is copied locally and never crosses the link.
- **Compression.** The server sends each chunk lzma-compressed when that is smaller. The device decompresses with an output limit of one chunk, so a hostile payload cannot expand without bound (tested with a 50 MB bomb that fits in under 100 KB). Integrity is always checked on the raw chunk.
- **Failure budget.** An update call now gives up after six *consecutive* failed requests, not six in total. With thousands of chunk requests the old total budget could not finish any large download (found when I first ran real model sizes); the 0.3 lossy-link table was regenerated.

## How much could a delta save? (upper bounds, `scripts/measure_delta.py`)

Percentage of the new file that would still have to be sent. "Fixed chunks" reuses identical aligned chunks; "content chunks" uses content-defined boundaries (so insertions do not break alignment); "xor + lzma" compresses the byte-wise difference; "lzma whole" is just compressing the new file. Full table: [results/delta_measure.txt](../results/delta_measure.txt).

| Pair (old to new) | Size | lzma whole | Fixed 4 KiB | Fixed 64 KiB | Content ~4 KiB | xor + lzma |
| --- | --- | --- | --- | --- | --- | --- |
| trained checkpoint, seed 0 to seed 1 (retrained) | 5.5 MB | 88% | 97% | 100% | 98% | 86% |
| ONNX, min-max to entropy INT8 calibration (same weights) | 10.7 MB | 84% | **0%** | 1% | **0%** | 0% |
| ONNX, calibration set 0 to set 1 | 10.7 MB | 84% | 0% | 1% | 0% | 0% |
| ONNX, FP16 build to INT8 build (same weights) | 10.7 MB | 84% | 0% | 1% | 0% | 0% |
| TensorRT INT8 engine, min-max to entropy | 5.1 MB | 55% | 100% | 100% | 66% | 80% |
| TensorRT engine, FP16 to INT8 | 5.1 MB | 53% | 100% | 100% | 100% | 83% |
| TensorRT INT8 engine, seed 1 to seed 2 | 5.1 MB | 54% | 100% | 100% | 90% | 82% |

- The ONNX files are the weights plus a graph; calibration and precision are applied when TensorRT builds the engine, so the ONNX files of different calibrations are nearly byte-identical and a delta of them is a few bytes.
- Retrained checkpoints differ in almost every weight: no chunk is shared, and the byte-wise difference compresses no better than the file itself (86% against 88%).
- A compiled engine shares nothing with another build of the "same" model (the engine embeds timing-chosen kernels and layout), and its xor difference is *worse* than compressing the new engine alone. A delta format for engines is not worth building; compression is.

## With the real agent (`scripts/run_delta_eval.py`)

The factory model is the old file, the server holds the new one, and the agent does a real chunked, verified, resumable update on a lossless link. Wire bytes include the manifest, signature, index and attachments; [results/delta_eval.txt](../results/delta_eval.txt).

| Pair | New size | lzma whole | 4 KiB chunks: wire / reused | 64 KiB chunks: wire / reused |
| --- | --- | --- | --- | --- |
| trained checkpoint, seed 1 to seed 2 | 5.5 MB | 88% | 92.2% / 37 of 1340 | 88.2% / 0 of 84 |
| ONNX, min-max to entropy | 10.7 MB | 84% | **0.8%** / 2605 of 2606 | **0.1%** / 162 of 163 |
| ONNX, FP16 build to INT8 build | 10.7 MB | 84% | 0.8% / 2605 of 2606 | 0.1% / 162 of 163 |
| TensorRT INT8 engine, min-max to entropy | 5.1 MB | 55% | 65.3% / 3 of 1250 | 58.9% / 0 of 79 |
| TensorRT engine, FP16 to INT8 | 5.1 MB | 53% | 63.6% / 0 of 1258 | 57.3% / 0 of 79 |

- **Chunk size is a trade.** For the ONNX pairs the wire cost is almost entirely the index (32 bytes per chunk: 83 KB at 4 KiB chunks, 5 KB at 64 KiB), so larger chunks win; for engines larger chunks also compress better (one lzma stream per chunk has a smaller window than one for the whole file, so 4 KiB chunks cost 10 to 11 points over compressing the file whole, 64 KiB chunks about 4). Smaller chunks only pay when changes are scattered and small, which none of these pairs showed. A fixed 64 KiB to 256 KiB chunk size is the sensible default for models of this size; 4 KiB was used in the power-cut sweeps because it gives many operations per update.
- **Damage costs one chunk.** One pair on a lossy link (TensorRT engine, 4 KiB chunks, 10 devices per row):

| Request drop rate | Chunks corrupted | Finished | Update calls | Wire bytes / new size |
| --- | --- | --- | --- | --- |
| 0.0 | 0 | 10/10 | 1.0 | 0.65 |
| 0.0 | 2% | 10/10 | 1.5 | 0.74 |
| 0.3 | 2% | 10/10 | 1.6 | 0.67 |
| 0.5 | 2% | 10/10 | 18.5 | 0.79 |

  (Before this change a single flipped byte forced a full re-download; here 2% corrupted chunks cost about 14% more bytes than the clean run.) Part of the call count at a 0.5 drop rate is that each call gives up after six consecutive failed requests.

## What this does not establish

- **Files from one project.** A 5 MB detector in several builds. Large language models, fine-tuned from a base (where a low-rank or sparse delta might share a lot), are not in this data; I have no such pair on disk and did not test one.
- **No delta of retrained weights was tried beyond a byte-wise xor.** Compression of weight differences in a numeric format (for example quantised differences of float tensors) could do better than xor-lzma; I did not try it.
- **"Ship the ONNX and build the engine on the device"** would turn the engine's 55% into the ONNX's 0.1% to 1%, but it needs a TensorRT build on every device (time, memory, nondeterminism between builds, and a larger trusted surface). I did not measure engine build time or reproducibility.
- **Simulated link and store**, as in the other milestones; the compression and hashing cost on a small CPU is not measured.
- **The index is not itself compressed or delta-encoded** (it is 32 bytes per chunk of mostly unique hashes, which would not compress); for very small chunks it dominates, as shown.

## Reproduce

```
pip install numpy cryptography hypothesis pytest
python scripts/measure_delta.py <sentinel-node>/experiments/detector/out/export    # about 90 s
python scripts/run_delta_eval.py <sentinel-node>/experiments/detector/out/export   # about 4 minutes
```
