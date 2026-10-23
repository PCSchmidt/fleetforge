# The control plane and a simulated fleet (milestone 0.4)

A control plane that rolls a candidate model out to a fleet in stages, watches what the devices report, and rolls everything back when the evidence says the candidate is worse or crashing. It is compared with two common alternatives on 300 simulated devices. The headline: **a fixed canary rule rolled back a perfectly good model 13% of the time and missed a 1-point regression half the time; the staged sequential rule did neither, and exposed about a tenth of the fleet to a 3-point regression instead of all of it. The price is time: a good model reached 100% of the fleet in about 3.8 hours, against 1.1 hours for all-at-once.** Everything is simulated; the devices' measurements come from the stream model of milestone 0.1, not from real models.

## Setup

- **Devices.** Each polls the control plane every 5 minutes (at its own phase), fetches the candidate over its own lossy link (drop rate 5 to 40%), and serves it as a trial for 30 minutes, with production running beside it on the same frames (milestone 0.1). It then reports one number: the mean paired difference over the trial. A device may instead crash during the trial.
- **True gain.** Each device has its own (fleet mean `delta`, device-to-device standard deviation 0.02 = 2 accuracy points), so a small canary can differ from the fleet by chance. Scene-to-scene variation within a device is as in 0.1.
- **Units of independence.** The control plane treats each *device* as one observation, because a device's gain persists through its trial and a random canary is a random sample of devices. Treating blocks or frames as independent would false-alarm in a small cohort; I did not test that variant.
- **Policies.**
  - *Staged e-process*: cohorts of 5, 25, 100 and the rest; a stage opens when 80% of the previous one has reported; a betting e-process (milestone 0.1) over device reports tests "is the candidate worse?" at the 5% level, and a Poisson likelihood-ratio e-process over exposure time tests the crash rate (baseline 2% of devices per trial against 15%, level 2.5%). Any alarm rolls everything back; finishing the last stage promotes.
  - *All at once*: the same two e-processes, but every device gets the candidate immediately.
  - *Fixed canary*: 5 devices; roll back if their mean report is below -0.01 (one accuracy point) or two of them crashed; otherwise ship to everyone with no further watching. This is the dashboard rule.
- The control plane's decisions are one class (`ControlPlane`) used by both the Monte Carlo and the real-agent fleet below.

## Results (400 simulated fleets per row, same fleets for every policy)

Full table: [results/fleet_eval.txt](../results/fleet_eval.txt). Times are minutes of simulated time (medians); "served" is the mean number of devices that ever ran the candidate; the 95% intervals are Wilson.

| True gain of the candidate | Policy | Rolled back | Minutes to the rollback decision | Devices that served it | Minutes to 100% when promoted |
| --- | --- | --- | --- | --- | --- |
| +2 points | staged e-process | 0.0% [0.0, 1.0] | | 300 | 230 |
| | all at once | 0.0% [0.0, 1.0] | | 300 | 63 |
| | fixed canary | 0.0% [0.0, 1.0] | | 300 | 111 |
| 0 (equal) | staged e-process | 0.8% [0.3, 2.2] | 214 | 300 | 231 |
| | all at once | 1.8% [0.9, 3.6] | 48 | 300 | 63 |
| | fixed canary | **13.0% [10.1, 16.7]** | 51 | 262 | 111 |
| -1 point | staged e-process | 100% [99.0, 100] | 150 | 119 | n/a |
| | all at once | 100% [99.0, 100] | 40 | 291 | n/a |
| | fixed canary | **46.5% [41.7, 51.4]** | 53 | 163 | 112 |
| -3 points | staged e-process | 100% [99.0, 100] | 99 | **30** | n/a |
| | all at once | 100% [99.0, 100] | 37 | 285 | n/a |
| | fixed canary | 98.0% [96.1, 99.0] | 53 | 11 | 115 |
| crashes on 40% of devices | staged e-process | 100% [99.0, 100] | 57 | 13 | n/a |
| | all at once | 100% [99.0, 100] | 13 | 112 | n/a |
| | fixed canary | **66.5% [61.7, 70.9]** | 38 | 104 | 110 |

Reading it:

- **The fixed canary is the wrong shape.** With 5 devices and a 2-point spread between devices, the canary's mean has a standard deviation of about 1 point, which is the size of the threshold. So it rolls back a good model often (13% of equal ones) and lets a 1-point regression through half the time. For a regression of 3 points it is fast and cheap (11 devices exposed on average). It also stops looking after the canary: a model that crashes on 40% of devices got through 33% of the time.
- **Staged e-process error rates are within their levels.** Rolling back an equal or better candidate happened in 0.0 to 0.8% of fleets (the two alarms together are allowed up to 7.5%); it never missed a 1-point or larger regression or a crashing model in 400 fleets.
- **It is not free.** A 1-point regression is caught after 150 minutes with 119 devices exposed, because 5 or 25 devices cannot show it (an estimate from the effect size and spread, not a measured number: the e-process needs several dozen devices at this spread). All-at-once catches it in 40 minutes, with 291 devices exposed. A good model takes 230 minutes to reach everyone staged, against 63 all at once.
- **Mean device-hours on a bad model** (not in the table) are dominated by the rare runs that ship it: the fixed canary's 140 device-hours for the 3-point candidate come from its 2% of fleets that promoted it (about 6,800 device-hours each), not from the rollbacks.

### Sensitivity

`sigma_dev` (device-to-device spread) and the e-process clipping bound `c` (set to 0.1 before running anything and not tuned), staged e-process, 400 fleets each:

| Spread | c | Equal candidate rolled back | -1 point caught | -3 points caught |
| --- | --- | --- | --- | --- |
| 0.00 | 0.05 / 0.10 / 0.20 | 0.2% / 0.0% / 0.0% | 100% | 100% |
| 0.02 | 0.05 / 0.10 / 0.20 | 2.2% / 1.5% / 0.8% | 100% | 100% |
| 0.04 | 0.05 / 0.10 / 0.20 | 3.8% / 2.5% / 2.0% | 80.5% / 88.5% / 92.2% | 100% |

Error rates stay under the 5% target throughout. With more heterogeneity between devices (spread 4 points) a 1-point regression is harder to see (80 to 92% caught within the simulated fleet), and a larger `c` caught more of them.

## With real agents on every device

`scripts/run_fleet_real_eval.py` ([results/fleet_real_eval.txt](../results/fleet_real_eval.txt)): the staged policy with a real update agent (milestone 0.3), store and lossy link on each of 300 devices, 10 fleets per candidate, on the same simulated measurements. The control plane's decisions are the same and agree (every worse or crashing candidate rolled back, every better one promoted). Timing is somewhat shorter with real agents (a promoted rollout took 202 minutes against the Monte Carlo's 227; the rollback of a 3-point candidate 92 against 97; the crashing candidate 56 against 77) because the Monte Carlo's staging-time model, fitted to the link table in `docs/agent.md`, is conservative. At the end of every run, on every device, the active model verifies, no device is left mid-trial, the floor matches the model the device ended on, and local crashes were remembered: 0 violations over 9,000 device-runs. A promoted rollout moved about 43 MB of model data in total at 128 KiB per model (2.4 update calls per device); a bad model stopped after 30 devices moved 4 MB.

## Two mistakes I made on the way

Both were found by checking healthy candidates against the stated error levels, and the second changed the table above.

1. My first crash rule was "roll back on two crashes", copied from the canary rule. With a 0.5% baseline crash rate and 300 devices, two crashes among healthy devices is more likely than not (the staged policy rolled back 42% of equal candidates).
2. The count-based fix (a likelihood-ratio test over crashed and not-crashed devices in the order their notices arrive) was also wrong: a crash notice arrives early in a trial while a healthy device resolves only at its end, so crashes are over-represented early and the all-at-once policy rolled back 17 to 19% of good models. The rule now is a Poisson likelihood-ratio process on the crash *hazard*, with the fleet's exposure time in it, and it passes a direct validity test (an alarm rate under 2.5% when the hazard is at the baseline, with devices starting at staggered times).

## What this does not establish

- **The devices' measurements are simulated** from the milestone-0.1 stream model, with device gains drawn from a normal distribution. Real gains may have heavy tails, structure by hardware class or location, and effects that appear only on rare scenes (a candidate better on average and much worse on one site). One hardware class is simulated.
- **Devices are assumed independent given the fleet mean.** Correlated failures (a network outage, one site's cameras, a bad batch of hardware) would break the e-processes' guarantees and the canary's sampling argument.
- **Only harm is tested.** The staged policy promotes if no alarm is raised; it does not demand evidence that the candidate is better, and a candidate slightly worse than a threshold the test cannot resolve with 300 devices will be promoted. The time to catch a regression of size d grows roughly as 1/d squared.
- **Design numbers are mine:** stage sizes, the 80% advance rule, the 30-minute trial, the 5% and 2.5% levels, the crash baseline of 2% and the alternative of 15%. They were fixed before the runs and not tuned, except as shown in the sensitivity table.
- **The abstract device model's staging times are fitted, not measured on real devices;** the real-agent run is not a real network either. No power cuts are injected in the fleet runs (that is milestone 0.3).
- **Not built:** a real control-plane service (this is a library driven by a simulation), authentication of devices and operators, persistence of the control plane's own state, devices that are offline for days, hardware classes, multiple candidates at once, and the bad-model *detection by on-device metrics other than the paired accuracy difference and crashes* (latency, confidence drift).

## Reproduce

```
pip install numpy cryptography hypothesis pytest
pytest                                         # 152 tests, about 5 minutes
python scripts/run_fleet_eval.py 400           # about an hour
python scripts/run_fleet_real_eval.py 10       # about 20 seconds
```
