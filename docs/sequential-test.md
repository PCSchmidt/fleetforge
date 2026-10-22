# Sequential shadow evaluation (milestone 0.1)

Question: a candidate model runs beside production on a device and both are scored on the same frames. When is it justified to promote the candidate, or to roll it back, and how many frames does that take? The usual practice, checking a dashboard and promoting when the difference looks significant, is shown below to fail badly. This page compares it with tests that stay valid when you look repeatedly and when neighbouring frames are correlated.

Everything here is **simulated**. The frame model is my construction (below), not data from real models or devices. The results show how the statistical rules behave under that model; they are not a measurement of any fleet.

## Setup

For each frame the paired difference is `d = +1` (candidate right, production wrong), `-1` (the reverse), or `0` (they agree). A stream has disagreement rate `q` (0.10 in most runs). The candidate's advantage is not constant: it varies with the scene, as an AR(1) process with standard deviation `sigma_z` (0.03) and correlation time `tau` frames. The mean over time is the true gain `delta`. Per-frame correlation is small (lag-1 about 0.009) but it accumulates: with tau = 500, block means vary several times more than independent frames would.

Rules compared, all at one-sided level 2.5% per direction, on streams of 40,000 frames, 2,000 simulated streams per cell (intervals are 95% Wilson):

| Rule | What it assumes |
| --- | --- |
| fixed z-test, one look, frames independent | independent frames, one look at a pre-chosen length |
| peek every 100 frames, z-test | same test repeated; stop at the first crossing |
| iid mixture sequence | independent frames; valid at every look |
| fixed z-test on blocks | blocks independent; one look |
| block mixture | blocks independent; valid at every look; variance is a plug-in estimate (approximate) |
| block betting | blocks independent and a design bound `c` on a block's gain (default 0.05, block means clipped to +-c); valid at every look, no distributional assumption |

A block is a fixed run of consecutive frames (for instance a few minutes of video); its mean difference is one observation. The unit of independence is the block, not the frame. The betting rule is an e-process: a bet on the sign of each block's mean, sized from earlier blocks only, whose wealth reaching 1/alpha is, by Ville's inequality, an error of probability at most alpha under the null.

## Results

Full tables: [results/seqtest.txt](../results/seqtest.txt).

**1. Looking repeatedly breaks the ordinary test.** With the candidate exactly equal to production, "peek every 100 frames" wrongly promotes **24.9%** of streams with independent frames (nominal 2.5%) and about **50%** once scenes vary. It also promoted 2.9% of streams for a candidate that was 3 points worse (a clearly bad model), because early noise can cross the line before the evidence settles.

**2. Treating correlated frames as independent breaks every frame-level rule, sequential or not.** With scene variation and correlation time 500 frames, the one-look test on frames wrongly promotes **27.8%**, and the iid mixture sequence **40.4%** (nominal 2.5%). A sequential rule that is valid for independent data is not valid here.

**3. Block-level rules restore the error rate if blocks are long enough compared with the correlation time.** Block betting wrongly promotes (percent, at no true difference):

| Block length vs correlation time | Block betting | Block mixture |
| --- | --- | --- |
| 10 times (B=1000, tau=100) | 0.3 | 0.1 |
| 2.5 times (B=250, tau=100) | 3.4 | 1.8 |
| 8 times (B=4000, tau=500) | 0.0 | 0.1 |
| 2 times (B=1000, tau=500) | **3.1** | **3.1** |
| half (B=1000, tau=2000) | **19.1** | **19.4** |
| 2 times (B=4000, tau=2000) | 0.1 | 1.7 |

So blocks of about four correlation times keep the guarantee; two is mildly over the target and shorter is badly over. The cost is that fewer, longer blocks leave fewer decisions: with tau = 2000 and a 40,000-frame stream there are only about five blocks of the needed length, and the rules can say very little. This is a property of the data, not of the rules.

**4. Held-out check.** I set the betting bound `c` to 0.05 after seeing the main simulation (power rose from 22% to 80% at a 1-point gain going from c = 0.1 to 0.05; larger bounds were far less powerful and 0.03 had a little more power and 0.9% / 1.4% error). To see whether that was just tuning, I ran three settings not used for the choice (disagreement rates 0.2 and 0.05, other scene variation and correlation times, blocks of 1000 to 1500 frames, always at least 4 times the correlation time): at no true difference block betting wrongly promoted 1.2%, 0.1% and 1.1% and wrongly rejected 1.4%, 0.0% and 1.1%; block mixture 2.1%, 0.1%, 0.9% and 1.7%, 0.1%, 1.1%. All are within the 2.5% target.

**5. Cost of validity: sequential rules need more frames than a one-look test at the same frame cap.** Candidate 1 point better (tau = 100, blocks of 1000):

| Rule | Promoted | Median frames when promoted |
| --- | --- | --- |
| fixed z-test on blocks, 40,000 frames | 97.6% | 40,000 |
| fixed z-test on blocks, cut at 20,000 | 79.1% | 20,000 |
| block mixture | 75.5% | 25,000 |
| block betting | 79.9% | 26,000 |

So to reach about 80% power at a 1-point gain, the sequential rules ran about 26,000 frames where a fixed test run for exactly 20,000 matched them. What sequential rules buy is the ability to stop early on larger effects and to look at any time: for a 2-point gain the block mixture promoted every stream by a median of 12,000 frames and betting by 15,000, against 10,000 frames for a fixed test that was sized for it; for a 4-point gain, 10,000 and 9,000 frames. The 10,000-frame floor on the mixture rule is its minimum of 10 blocks before it may decide. Block mixture is somewhat faster than betting; betting has the guarantee that does not rest on a plug-in variance.

**6. Catching a bad model.** A candidate 3 points worse was rejected in 100% of streams by every block rule: median 11,000 frames (betting) and 10,000 (mixture, at its 10-block floor). None wrongly promoted it. "Peek every 100 frames" rejected it in a median of 300 frames, faster, but also wrongly promoted it 2.9% of the time.

**7. A slightly worse candidate (1 point worse, tau = 500).** Peeking wrongly promoted it 29.2% of the time and the iid mixture 7.1%; block betting 0.1%, block mixture 0.1%, the fixed block test 0.0%.

## What this does not establish

- **Real fleets are not this model.** Real error patterns are heavy-tailed, non-stationary, and tied to scenes in ways an AR(1) gain does not capture; blocks would be aligned to real sessions or time windows, which may be unequal in length (unequal blocks need weights, not built).
- **The correlation time must be known or estimated.** The rules' validity depends on block length relative to it, and per-frame autocorrelation of `d` is almost invisible (lag-1 about 0.009), so estimating tau from the frame series is a weak method. Comparing the variance of block means across block sizes may work; I have not tested it.
- **The betting bound `c` is a design assumption** and was tuned on this simulation (with the held-out check above). A block with a gain beyond +-c is clipped, which tests the clipped mean; that matters only if such blocks are common.
- **One metric and one candidate.** No multiple comparisons, no latency or confidence-drift criteria, no guard against a candidate that is better on average but much worse on a rare scene, no non-inferiority margin.
- **Only the null and a few alternatives.** The error rates are measured at no difference and at slightly worse; no sweep over disagreement rate and scene variation beyond the held-out three.
- **Block mixture is approximate** (its error rate was 3.1% in two cells when blocks were only twice the correlation time).

## Reproduce

```
pip install numpy pytest
pytest                       # 18 tests, about 3 s
python scripts/run_seqtest.py   # about 80 s; writes results/seqtest.txt and seqtest.json
```
