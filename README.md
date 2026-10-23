# FleetForge

Edge MLOps: ship a model to many devices as a signed artifact, roll it out by canary, and decide on the device whether to promote or roll back.

**Status:** milestones 0.1 to 0.4 and part of 0.6, no real devices. The update path now verifies every chunk against a signed index, reuses chunks the device already has and compresses on the wire; measured on real model files (an ONNX rebuilt with different calibration costs 0.1 to 0.8% of the file, retrained weights and compiled engines get no help from deltas, only from compression) in [docs/delta.md](docs/delta.md). 0.4 is a staged-rollout control plane with sequential harm tests, compared on 300 simulated devices: a fixed canary rule rolled back a good model 13% of the time and caught a 1-point regression 46.5% of the time; the staged e-process rolled back 0.8% and caught all of them, exposing about a tenth of the fleet to a 3-point regression, at the price of a 3.8 h rollout against 1.1 h ([docs/fleet.md](docs/fleet.md)). 0.3 is a device update agent (A/B slots, resumable download, trial-then-promote) that kept a valid model through 20,238 simulated power-cut runs, where an in-place updater was bricked in 71% of its runs ([docs/agent.md](docs/agent.md)). 0.2 is a signed model artifact format with a device-side verifier (Ed25519, pinned root key and keyset, rollback floor, target, validity window), tested against every single-bit tamper and a list of replay, mix-and-match and parser attacks, with all 26 injected faults caught ([docs/artifact.md](docs/artifact.md)). 0.1: a sequential shadow-evaluation test (when is it justified to promote a candidate model that runs beside production, looking at the data as it arrives, when neighbouring frames are correlated?). No artifact signing, no device agent, no control plane, no real devices or models yet. See [docs/scope.md](docs/scope.md) for the plan and [docs/sequential-test.md](docs/sequential-test.md) for the results and their limits.

## Headline (simulated streams, not real models)

- Checking a dashboard and promoting when the difference looks significant ("peeking") wrongly promoted an equal candidate on **25%** of streams (nominal 2.5%), and about **50%** when scenes vary.
- Rules that treat frames as independent also fail when the candidate's advantage varies by scene (28% to 40% false promotions), even if they are valid for peeking.
- Testing on **blocks** of frames at least about four correlation times long restores the error rate; shorter blocks do not.
- Validity costs frames: to match the power of a fixed test at 1 point of true gain, the sequential rules needed about 26,000 frames against 20,000. They stop earlier on larger gains and caught a 3-point regression in a median of 10,000 to 11,000 frames.

## Run

```
pip install numpy cryptography hypothesis pytest
pytest
python scripts/run_seqtest.py
```
