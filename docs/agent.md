# The update agent (milestone 0.3)

A small device-side agent that pulls a signed model over a bad link and swaps it in without ever leaving the device without a valid model, tested by cutting power at every storage operation. Everything runs against a simulated file store and a simulated server in Python; there is no real flash, file system, network or device, and the model is never executed.

## Design

- **A/B slots.** The running model's slot is never written. An update goes to the other slot, is verified in full there (the 0.2 checks: signature, target, floor, validity window, size and hashes), and only then does `state.json` mention it, as a *pending trial*.
- **One authority.** `state.json` holds the active slot, the pending trial, the floors, installed hashes, rejected hashes and the signed keyset. It is replaced only by write-temp, sync, rename, and carries its own checksum.
- **Trial, then promote.** A pending model runs; the supervisor confirms it healthy (the floor rises and the slot becomes active, the old one `previous`) or reports it unhealthy (dropped, its hash remembered, never retried). A pending model that is not confirmed within 3 boots is dropped the same way, so a model that crashes on start cannot hold the device.
- **Resumable download.** Progress is a journal of (offset, length, chunk hash) lines, each written only after its chunk was synced. On restart only the journalled prefix whose hashes still match the file is trusted; anything after is truncated. A manifest that changed since the journal was written resets the download.
- **Cheap refusal.** The manifest and signature (a few hundred bytes) are fetched and checked first; a wrong-target, rolled-back, expired or badly signed update is refused before any of the model is downloaded (tested: under 2 KB transferred).

## The store model

`SimFS` counts every operation. Writes are volatile until their file is synced; rename and delete are atomic; a rename moves whatever the file holds, so renaming an unsynced file can lose its contents. A cut at operation k stops the run before that operation, except a write, which is torn. After a cut the store is rebuilt under one of four patterns for what happens to unsynced writes: all lost, all kept, the first half kept, every other one kept.

## Results

`scripts/run_agent_eval.py` (output in [results/agent_eval.txt](../results/agent_eval.txt)); the sweeps are also in the test suite.

**Power cuts.** A full update is 52 storage operations (58 for the crash-looping case). A cut at each operation under each of 4 patterns, then every further cut at each operation of the recovery run, for a good model, a bad model and a crash-looping model: **20,238 runs, 0 failures**. After every cut the invariants hold: the state reads, the active slot verifies, the floor matches the installed model and the active slot's contents, and a pending trial never points at the active slot; and after recovery the device ends where it should (new model promoted, or old model kept with the bad one remembered).

**Control.** An updater that overwrites the running model in place leaves the device with no valid model in **34 of 48** cut runs (71%).

**Mutation testing** (`scripts/mutation_agent.py`, `results/mutation_agent.txt`): 17 injected faults (state not synced or overwritten in place, update written into the running slot, verification skipped, tries not counted, unhealthy model promoted, bad hashes not remembered or not consulted, chunk data unsynced before the journal line, journal checks removed, boot skipping verification, floor never raised, and others); all 17 caught. The first run missed two: an update starting while a trial is pending, and the journal's chunk-hash check; both now have direct tests (the second uses injected bit rot in the partial file).

**Lossy link** (128 KiB model, 200 devices per row, short reads 20%, one corrupted byte per 100 chunks; a call gives up after 6 failed requests):

| Request drop rate | With resume: done / calls / bytes sent ÷ size | Without resume: done / calls / bytes sent ÷ size |
| --- | --- | --- |
| 0.0 | 100% / 1.6 / 1.53 | 100% / 1.6 / 1.53 |
| 0.1 | 100% / 1.8 / 1.62 | 100% / 1.7 / 1.71 |
| 0.3 | 100% / 4.0 / 1.44 | 45% / 26 / 20.0 |
| 0.5 | 100% / 8.5 / 1.42 | 0% / n/a / 11.5 (in 60 calls) |

Resuming is what makes a bad link workable: at a 30% drop rate every device finishes with 1.4 times the model's size transferred, against 45% finishing (after 20 times the size) without it.

## What this does not establish

- **A single flipped byte costs a full re-download.** Integrity is checked against the model hash only at the end (the 1.5 times at zero drops comes from the corruption rate, not from drops). The chunk hashes in the journal come from the agent itself, so they catch local damage but not corruption in transit. Putting chunk hashes (a Merkle tree) in the signed manifest would fix this; it needs a new manifest format and is not built. This is also the prerequisite for delta updates (milestone 0.6).
- **Repeated power loss drops a good model.** Every boot of a pending model counts as a try, so three reboots before the supervisor confirms (several quick power cuts after an update) drop the model and remember its hash. The device stays safe on the old model and a corrected model with a new `seq` is accepted, but the same file will not be retried. The count cannot tell a crash from a power cut.
- **The simulation is mine.** The file model (volatile until synced, atomic rename) follows POSIX-like behaviour but is not a measurement of any flash or file system; real devices add wear levelling, write reordering across files, directory syncs and hardware faults. The four loss patterns are a sample, not every subset of lost writes.
- **Corruption of the running model after the fact** (bit rot, a failing flash) is reported as `Bricked` at boot, not repaired; there is no recovery slot or fallback to the previous model.
- **Not built:** keyset updates through the agent, the supervisor/health check itself (it is a flag), a real transport, progress reporting, bandwidth throttling, and any coordination across a fleet (that is the control plane, milestone 0.4).

## Reproduce

```
pip install numpy cryptography hypothesis pytest
pytest                                   # 125 tests, about 12 s
python scripts/run_agent_eval.py         # about a minute
python scripts/mutation_agent.py         # about 4 minutes
```
