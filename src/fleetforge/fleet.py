"""A control plane that rolls a candidate model out to a simulated fleet in stages and decides when to roll back.

Model, in minutes of simulated time:

  * A device polls the control plane every `poll_min` minutes (at its own phase). When assigned the candidate it
    fetches it over its own lossy link (several polls if the link is bad), then serves it as a trial.
  * Its trial lasts `trial_blocks` blocks of `block_min` minutes. At the end it reports one number: the mean paired
    difference d (candidate right and production wrong, minus the reverse) over the trial's frames. The candidate
    runs beside production on the same frames (milestone 0.1), so this is the shadow-evaluation measurement.
  * A device may instead crash during the trial and report that.
  * Each device has its own true gain (fleet mean `delta`, device-to-device spread `sigma_dev`), so a small canary
    cohort can differ from the fleet by chance.

The control plane treats each device's report as one observation. Devices are the units of independence, not
frames or blocks, because a device's gain persists across its whole trial and a random canary cohort is a random
sample of devices. Policies:

  EProcessStaged   stages of increasing size; a betting e-process over device reports (harm test: is the candidate
                   worse?) runs across all reports, and any alarm rolls everything back. A stage opens when 80% of
                   the previous stage's devices have reported. After the last stage, promote.
  AllAtOnce        the same e-process, but every device gets the candidate at once.
  FixedCanary      the usual dashboard rule: a canary of 5 devices; roll back if the canary's mean report is below
                   a threshold (or two canary devices crashed); otherwise roll out to everyone with no further guard.
                   The e-process policies also run a likelihood-ratio e-process on the crash rate (CrashBet).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .streams import StreamParams, simulate


@dataclass(frozen=True)
class FleetConfig:
    n: int = 300
    poll_min: float = 5.0
    block_min: float = 3.0
    trial_blocks: int = 10
    block_frames: int = 1000
    download_min: float = 3.0
    q: float = 0.10
    sigma_dev: float = 0.02
    sigma_z: float = 0.03
    tau: float = 100.0
    drop_lo: float = 0.05
    drop_hi: float = 0.40
    crash_base: float = 0.005  # chance a healthy candidate crashes on a device during its trial
    model_mb: float = 25.0
    horizon_min: float = 24 * 60.0  # exposure to a promoted bad model is counted up to this time

    @property
    def trial_min(self) -> float:
        return self.trial_blocks * self.block_min


@dataclass(frozen=True)
class Candidate:
    name: str
    delta: float  # true fleet-mean gain over production
    p_crash: float = 0.0  # extra per-device chance of crashing in the trial


def mean_calls(drop: float) -> float:
    """Mean number of update calls until a download completes, fitted to the lossy-link table in docs/agent.md."""
    return 1.6 + 2.8 * drop + 20.0 * drop * drop


@dataclass
class Devices:
    """Everything about each device that does not depend on the policy."""

    n: int
    phase: np.ndarray  # minutes into the poll interval at which it polls
    calls: np.ndarray  # update calls needed to fetch the candidate
    report: np.ndarray  # mean paired difference over the trial (nan if it crashes)
    crash_at: np.ndarray  # minutes into the trial when it crashes (inf if it does not)
    delta_dev: np.ndarray
    order: np.ndarray  # random order in which devices are drawn into cohorts
    drop: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per-request drop probability of each device's link


def make_devices(cfg: FleetConfig, cand: Candidate, rng: np.random.Generator) -> Devices:
    n = cfg.n
    delta_dev = np.clip(rng.normal(cand.delta, cfg.sigma_dev, size=n), -cfg.q, cfg.q)
    frames = cfg.trial_blocks * cfg.block_frames
    d = simulate(StreamParams(0.0, cfg.q, cfg.sigma_z, cfg.tau), n, frames, rng, delta_rep=delta_dev)
    report = d.mean(axis=1, dtype=np.float64)
    drop = rng.uniform(cfg.drop_lo, cfg.drop_hi, size=n)
    calls = 1 + rng.poisson(np.maximum(mean_calls(drop) - 1.0, 0.0))
    crashes = rng.random(n) < (cfg.crash_base + cand.p_crash)
    crash_at = np.where(crashes, rng.uniform(1.0, cfg.trial_min, size=n), np.inf)
    report = np.where(crashes, np.nan, report)
    return Devices(n, rng.uniform(0, cfg.poll_min, size=n), calls, report, crash_at, delta_dev, rng.permutation(n), drop)


class OnlineBet:
    """Betting e-process against 'the candidate is not worse' (harm). Same recurrence as methods._bet_wealth."""

    def __init__(self, c: float = 0.1, lam_max: float = 0.9, prior_var: float = 0.04) -> None:
        self.c, self.lam_max, self.prior_var = c, lam_max, prior_var
        self.i = 0
        self.s = 0.0
        self.ss = 0.0
        self.logw = 0.0

    def update(self, m: float) -> float:
        y = -max(-self.c, min(self.c, m)) / self.c  # positive when the candidate is worse
        mu = self.s / (self.i + 1)
        var = (self.prior_var + self.ss) / (self.i + 1)
        lam = max(0.0, min(self.lam_max, mu / (var + mu * mu)))
        self.logw += math.log1p(lam * y)
        self.s += y
        self.ss += (y - self.s / (self.i + 1)) ** 2
        self.i += 1
        return self.logw


class CrashRate:
    """Poisson likelihood-ratio e-process on the crash hazard: H0 'devices crash at most at rate lam0' against lam1.

    logW = N log(lam1/lam0) - (lam1 - lam0) * E, with N the crashes seen so far and E the total device-minutes of
    exposure (time spent serving the candidate within the trial window). It is a supermartingale when the true
    hazard is at most lam0, whatever the order in which crashes and non-crashes are observed, which is why exposure
    time has to be part of it. (Counting crashes against a fixed number cannot be right across fleet sizes, and
    counting crashes as they arrive against non-crashes that arrive only at the end of the trial inflates the
    evidence: both were tried first and both false-alarmed on healthy models.)
    """

    def __init__(self, p0: float, p1: float, trial_min: float) -> None:
        self.lam0, self.lam1 = -math.log(1 - p0) / trial_min, -math.log(1 - p1) / trial_min
        self.n = 0

    def update(self, exposure_min: float) -> float:
        """Call at each crash notice, with the exposure accumulated up to that time."""
        self.n += 1
        return self.n * math.log(self.lam1 / self.lam0) - (self.lam1 - self.lam0) * exposure_min


@dataclass
class Outcome:
    rolled_back: bool
    t_alarm: float | None  # when the control plane decided to roll back
    t_rolled_back: float | None  # when the last device that served the candidate was back on production
    t_promoted: float | None  # when the last device was on the candidate for good
    served: int  # devices that ever served the candidate
    exposure_dev_hours: float  # device-hours spent serving the candidate before rollback (or up to the horizon if promoted)
    cause: str = ""
    reports: int = 0


@dataclass(frozen=True)
class EProcessStaged:
    stages: tuple = (5, 25, 100, 10**9)
    beta: float = 0.05
    c: float = 0.1
    crash_beta: float = 0.025
    crash_p0: float = 0.02
    crash_p1: float = 0.15
    advance_at: float = 0.8
    name: str = "staged e-process"


@dataclass(frozen=True)
class AllAtOnce:
    beta: float = 0.05
    c: float = 0.1
    crash_beta: float = 0.025
    crash_p0: float = 0.02
    crash_p1: float = 0.15
    advance_at: float = 0.8
    name: str = "all at once, e-process"


@dataclass(frozen=True)
class FixedCanary:
    canary: int = 5
    threshold: float = -0.01
    crash_alarm: int = 2
    advance_at: float = 0.8
    name: str = "fixed canary rule"


def _stage_times(cfg: FleetConfig, dev: Devices, idx: int, t_assign: float) -> tuple[float, float]:
    """(time the device starts serving the candidate, time its trial report or crash notice reaches the control plane)."""
    first = t_assign + ((dev.phase[idx] - t_assign) % cfg.poll_min)  # first poll at or after the assignment
    start = first + (dev.calls[idx] - 1) * cfg.poll_min + cfg.download_min  # one poll per further update call
    t_event = start + (dev.crash_at[idx] if math.isfinite(dev.crash_at[idx]) else cfg.trial_min)
    return start, t_event + ((dev.phase[idx] - t_event) % cfg.poll_min)


class ControlPlane:
    """The rollout decision logic, with no notion of time or devices beyond ids: feed it each device's resolution
    (report or crash) in the order they reach the control plane and it says what to do next."""

    def __init__(self, policy, cfg: FleetConfig, order) -> None:
        self.policy = policy
        self.fixed = isinstance(policy, FixedCanary)
        if isinstance(policy, EProcessStaged):
            stages = policy.stages
        elif isinstance(policy, AllAtOnce):
            stages = (cfg.n,)
        else:
            stages = (policy.canary, 10**9)
        self.members: list[list[int]] = []
        pos = 0
        for size in stages:
            take = min(size, len(order) - pos)
            if take <= 0:
                break
            self.members.append([int(i) for i in order[pos : pos + take]])
            pos += take
        self.stage_of = {i: k for k, m in enumerate(self.members) for i in m}
        self.resolved = [0] * len(self.members)
        self.bet = None if self.fixed else OnlineBet(policy.c)
        self.crash_rate = None if self.fixed else CrashRate(policy.crash_p0, policy.crash_p1, cfg.trial_min)
        self.crashes = 0
        self.canary_vals: list[float] = []
        self.next_stage = 1
        self.reports = 0
        self.cause = ""

    def initial(self) -> list[int]:
        return list(self.members[0])

    def _need(self, stage: int) -> int:
        return math.ceil(self.policy.advance_at * len(self.members[stage]))

    def resolve(self, i: int, crashed: bool, report: float, exposure_min: float = 0.0) -> tuple[str, list[int]]:
        """Returns ('continue' | 'alarm' | 'promote', devices newly assigned the candidate).

        `exposure_min` is the device-minutes the fleet has spent serving the candidate (within trials) up to now."""
        p = self.policy
        st = self.stage_of[i]
        self.resolved[st] += 1
        if crashed and not self.fixed and self.crash_rate.update(exposure_min) >= math.log(1.0 / p.crash_beta):
            self.cause = "crashes"
            return "alarm", []
        if crashed:
            if self.fixed and st == 0:
                self.crashes += 1
                if self.crashes >= p.crash_alarm:
                    self.cause = "crashes"
                    return "alarm", []
        else:
            self.reports += 1
            if self.bet is not None:
                if self.bet.update(report) >= math.log(1.0 / p.beta):
                    self.cause = "harm"
                    return "alarm", []
            elif st == 0:
                self.canary_vals.append(report)
        if self.fixed:
            if st == 0 and self.resolved[0] >= self._need(0):
                if self.canary_vals and float(np.mean(self.canary_vals)) < p.threshold:
                    self.cause = "canary mean"
                    return "alarm", []
                rest = self.members[1] if len(self.members) > 1 else []  # roll out to everyone; no further guard
                return "promote", list(rest)
            return "continue", []
        new: list[int] = []
        if self.next_stage < len(self.members) and self.resolved[self.next_stage - 1] >= self._need(self.next_stage - 1):
            new = list(self.members[self.next_stage])
            self.next_stage += 1
        if st == len(self.members) - 1 and self.resolved[st] >= self._need(st):
            return "promote", new
        return "continue", new


def run_rollout(policy, cfg: FleetConfig, dev: Devices) -> Outcome:
    n = dev.n
    cp = ControlPlane(policy, cfg, dev.order)
    start = np.full(n, np.inf)
    heap: list[tuple[float, int]] = []

    def assign(ids: list[int], t: float) -> None:
        for i in ids:
            start[i], done = _stage_times(cfg, dev, i, t)
            heapq.heappush(heap, (done, i))

    life = np.where(np.isfinite(dev.crash_at), dev.crash_at, cfg.trial_min)

    def exposure(t: float) -> float:
        ok = np.isfinite(start)
        return float(np.clip(t - start[ok], 0.0, life[ok]).sum())

    assign(cp.initial(), 0.0)
    t_alarm = t_ok = None
    while heap:
        t, i = heapq.heappop(heap)
        crashed_i = bool(math.isnan(dev.report[i]))
        verdict, new = cp.resolve(i, crashed_i, float(dev.report[i]), exposure(t) if crashed_i else 0.0)
        assign(new, t)
        if verdict == "alarm":
            t_alarm = t
            break
        if verdict == "promote":
            t_ok = t
            break
    if t_alarm is not None:
        # devices serving at the alarm go back to production at their next poll; ones not yet serving are cancelled
        back = {int(i): t_alarm + ((dev.phase[i] - t_alarm) % cfg.poll_min) for i in np.nonzero(np.isfinite(start))[0] if start[i] <= t_alarm}
        expo = sum(back[i] - start[i] for i in back) / 60.0
        return Outcome(True, t_alarm, max(back.values()) if back else t_alarm, None, len(back), expo, cp.cause, cp.reports)
    assert t_ok is not None, "rollout neither alarmed nor finished"
    finish = {}
    for i in range(n):
        if math.isfinite(start[i]):
            ready = max(t_ok, start[i])
            finish[i] = ready + ((dev.phase[i] - ready) % cfg.poll_min)
    expo = sum(cfg.horizon_min - start[i] for i in finish) / 60.0
    return Outcome(False, None, None, max(finish.values()), len(finish), expo, "", cp.reports)
