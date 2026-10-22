# FleetForge

Edge MLOps: ship a model to many devices as a signed artifact, roll it out by canary, and decide on the device whether to promote or roll back.

**Status:** milestone 0.1 only, in simulation: a sequential shadow-evaluation test (when is it justified to promote a candidate model that runs beside production, looking at the data as it arrives, when neighbouring frames are correlated?). No artifact signing, no device agent, no control plane, no real devices or models yet. See [docs/scope.md](docs/scope.md) for the plan and [docs/sequential-test.md](docs/sequential-test.md) for the results and their limits.

## Headline (simulated streams, not real models)

- Checking a dashboard and promoting when the difference looks significant ("peeking") wrongly promoted an equal candidate on **25%** of streams (nominal 2.5%), and about **50%** when scenes vary.
- Rules that treat frames as independent also fail when the candidate's advantage varies by scene (28% to 40% false promotions), even if they are valid for peeking.
- Testing on **blocks** of frames at least about four correlation times long restores the error rate; shorter blocks do not.
- Validity costs frames: to match the power of a fixed test at 1 point of true gain, the sequential rules needed about 26,000 frames against 20,000. They stop earlier on larger gains and caught a 3-point regression in a median of 10,000 to 11,000 frames.

## Run

```
pip install numpy pytest
pytest
python scripts/run_seqtest.py
```
