# The update agent (milestone 0.3)

A small device-side agent that pulls a signed model over a bad link and swaps it in without ever leaving the device without a valid model, tested by cutting power at every storage operation. Everything runs against a simulated file store and a simulated server in Python; there is no real flash, file system, network or device, and the model is never executed.

## Design

- **A/B slots.** The running model's slot is never written. An update goes to the other slot, is verified in full there (the 0.2 checks: signature, target, floor, validity window, size and hashes), and only then does `state.json` mention it, as a *pending trial*.
- **One authority.** `state.json` holds the active slot, the pending trial, the floors, installed hashes, rejected hashes and the signed keyset. It is replaced only by write-temp, sync, rename, and carries its own checksum.
- **Trial, then promote.** A pending model runs; the supervisor confirms it healthy (the floor rises and the slot becomes active, the old one `previous`) or reports it unhealthy (dropped, its hash remembered, never retried). A pending model that is not confirmed within 3 boots is dropped the same way, so a model that crashes on start cannot hold the device.
- **Chunk-verified, resumable download.** The signed manifest pins a chunk-hash index (milestone 0.6, [delta.md](delta.md)); the device fetches the index, checks it against the pin, and verifies every chunk against it on arrival, so a damaged chunk is asked for again and nothing else is re-sent. Chunks already in the running model are reused, and chunks cross the link lzma-compressed when that is smaller. Progress is a journal of (offset, length, chunk hash) lines, each written only after its chunk was synced; on restart only the journalled prefix whose hashes still match the file is trusted, and a manifest that changed since the journal was written resets the download.
- **Cheap refusal.** The manifest and signature (a few hundred bytes) are fetched and checked first; a wrong-target, rolled-back, expired or badly signed update is refused before any of the model is downloaded (tested: under 2 KB transferred).

## The store model

`SimFS` counts every operation. Writes are volatile until their file is synced; rename and delete are atomic; a rename moves whatever the file holds, so renaming an unsynced file can lose its contents. A cut at operation k stops the run before that operation, except a write, which is torn. After a cut the store is rebuilt under one of four patterns for what happens to unsynced writes: all lost, all kept, the first half kept, every other one kept.

## Results

`scripts/run_agent_eval.py` (output in [results/agent_eval.txt](../results/agent_eval.txt)); the sweeps are also in the test suite.

**Power cuts.** A full update is 52 storage operations (58 for the crash-looping case). A cut at each operation under each of 4 patterns, then every further cut at each operation of the recovery run, for a good model, a bad model and a crash-looping model: **20,238 runs, 0 failures**. After every cut the invariants hold: the state reads, the active slot verifies, the floor matches the installed model and the active slot's contents, and a pending trial never points at the active slot; and after recovery the device ends where it should (new model promoted, or old model kept with the bad one remembered).

**Control.** An updater that overwrites the running model in place leaves the device with no valid model in **34 of 48** cut runs (71%).

**Mutation testing** (`scripts/mutation_agent.py`, `results/mutation_agent.txt`): 22 injected faults (state not synced or overwritten in place, update written into the running slot, verification skipped, tries not counted, unhealthy model promoted, bad hashes not remembered or not consulted, chunk data unsynced before the journal line, journal checks removed, boot skipping verification, floor never raised, received chunks not checked against the signed index, the index not checked against the manifest, a cumulative instead of consecutive failure budget, held chunks never reused, and others); 21 are caught. The survivor is the output limit on decompression: removing it leaves every result the same, because the length check still rejects an oversized chunk afterwards, but memory is no longer bounded; that is a resource property no output-based test can see, and it has a direct test of the decoder instead. Earlier rounds missed three (an update starting while a trial is pending, the journal's chunk-hash check, and the failure budget); each now has a direct test.

**Lossy link** (128 KiB model, 4 KiB chunks, 200 devices per row, short reads 20%, one corrupted byte per 100 chunk transfers; a call gives up after 6 *consecutive* failed requests; this table was regenerated after the chunk-verification and failure-budget changes, and its figures differ from the first version of this page):

| Request drop rate | With resume: done / calls / bytes sent ÷ size | Without resume: done / calls / bytes sent ÷ size |
| --- | --- | --- |
| 0.0 | 100% / 1.1 / 1.18 | 100% / 1.1 / 1.18 |
| 0.1 | 100% / 1.1 / 1.17 | 100% / 1.1 / 1.17 |
| 0.3 | 100% / 1.2 / 1.18 | 100% / 1.2 / 1.25 |
| 0.5 | 100% / 2.3 / 1.20 | 100% / 3.2 / 2.12 |

(The 1.18 at zero drops is the short reads: a truncated chunk is thrown away and asked for again, and the discarded bytes count as sent.)

Resuming matters least where the link is good and most where it is bad: at a 50% drop rate a device that resumes finishes after 2.3 calls with 1.2 times the model's size sent, while one that starts over each call sends 2.1 times. (With the earlier total-failure budget, a 30% drop rate left 45% of the starting-over devices unfinished; the consecutive-failure budget removes that, which is why this table is more forgiving than the first version.)

## What this does not establish

- **Corruption in transit** was a full re-download in the first version of this agent (integrity was checked only against the whole-model hash); it now costs one chunk, because the signed manifest pins a chunk index (see [delta.md](delta.md)). What is still not protected: a *signed* model that is itself bad (that is the health check and the sequential test's job).
- **Repeated power loss drops a good model.** Every boot of a pending model counts as a try, so three reboots before the supervisor confirms (several quick power cuts after an update) drop the model and remember its hash. The device stays safe on the old model and a corrected model with a new `seq` is accepted, but the same file will not be retried. The count cannot tell a crash from a power cut.
- **The simulation is mine.** The file model (volatile until synced, atomic rename) follows POSIX-like behaviour but is not a measurement of any flash or file system; real devices add wear levelling, write reordering across files, directory syncs and hardware faults. The four loss patterns are a sample, not every subset of lost writes.
- **Corruption of the running model after the fact** (bit rot, a failing flash) is reported as `Bricked` at boot, not repaired; there is no recovery slot or fallback to the previous model.
- **Not built:** keyset updates through the agent, the supervisor/health check itself (it is a flag), a real transport, progress reporting, bandwidth throttling, and any coordination across a fleet (that is the control plane, milestone 0.4).

## Reproduce

```
pip install numpy cryptography hypothesis pytest
pytest                                   # 168 tests, about 5 minutes
python scripts/run_agent_eval.py         # about a minute
python scripts/mutation_agent.py         # about 4 minutes
```
