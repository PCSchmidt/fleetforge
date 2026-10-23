"""The same rollout, but with a real update agent (milestone 0.3) on every device.

Each device has its own simulated store holding the factory model, its own lossy link to a server holding the
candidate, and the real agent: `update` fetches, verifies (signature, floor, hashes) and stages the candidate,
`boot` starts the trial, `confirm(False)` rolls back and `confirm(True)` promotes. Only the model's behaviour during
the trial (its report or crash) comes from the simulated measurements; the control plane is the same
`ControlPlane` the Monte Carlo uses. Time advances one simulated minute at a time and a device acts only on its
own polls.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import sweep as S
from .agent import Agent
from .fleet import Candidate, ControlPlane, Devices, FleetConfig, make_devices
from .link import Server


@dataclass
class RealOutcome:
    rolled_back: bool
    t_alarm: float | None = None
    t_rolled_back: float | None = None
    t_promoted: float | None = None
    served: int = 0
    update_calls: int = 0
    bytes_sent: int = 0
    violations: list = field(default_factory=list)  # per-device invariant failures at the end of the run


def run_real(policy, cfg: FleetConfig, cand: Candidate, seed: int, world: S.World | None = None) -> RealOutcome:
    world = world or S.make_world()
    rng = np.random.default_rng(seed)
    dev: Devices = make_devices(cfg, cand, rng)
    n = cfg.n
    fss = [S.clone(world.base) for _ in range(n)]
    servers = [Server(world.b2, random.Random(seed * 1000 + i), p_drop=float(dev.drop[i]), p_short=0.2, p_corrupt=0.002) for i in range(n)]
    agents = [Agent(fss[i], world.root_pub, S.TARGET) for i in range(n)]
    phase = np.floor(dev.phase).astype(int) % int(cfg.poll_min)
    cp = ControlPlane(policy, cfg, dev.order)

    life = np.where(np.isfinite(dev.crash_at), dev.crash_at, cfg.trial_min)

    def exposure(now: float) -> float:
        return float(sum(min(max(now - begun, 0.0), life[j]) for j, begun in begun_at.items()))

    begun_at: dict[int, float] = {}
    fetching: set[int] = set()
    serving: dict[int, float] = {}  # device -> time its trial started
    resolved: set[int] = set()
    crashed_locally: set[int] = set()
    ever_served: set[int] = set()
    finished: dict[int, str] = {}  # device -> 'rolled_back' | 'promoted'
    finish_t: dict[int, float] = {}
    out = RealOutcome(False)
    state = "running"  # running | alarm | promoted
    for i in cp.initial():
        fetching.add(i)
    t = 0
    horizon = int(cfg.horizon_min)
    while t < horizon:
        for i in range(n):
            if t % int(cfg.poll_min) != phase[i] or i in finished:
                continue
            a = agents[i]
            if state == "alarm":
                if i in serving or i in fetching:
                    if i in serving:
                        a.confirm(False)
                    fetching.discard(i)
                    serving.pop(i, None)
                    finished[i] = "rolled_back"
                    finish_t[i] = t
                continue
            if i in fetching:
                out.update_calls += 1
                res = a.update(servers[i], S.NOW)
                if res == "pending":
                    a.boot()  # reboot into the trial
                    fetching.discard(i)
                    serving[i] = t + cfg.download_min
                    begun_at[i] = serving[i]
                    ever_served.add(i)
                    if state == "promoted":
                        a.confirm(True)
                        serving.pop(i)
                        finished[i] = "promoted"
                        finish_t[i] = t
                continue
            if i in serving:
                if state == "promoted":
                    a.confirm(True)
                    serving.pop(i)
                    finished[i] = "promoted"
                    finish_t[i] = t
                    continue
                begun = serving[i]
                crash_time = begun + dev.crash_at[i]
                if math.isfinite(dev.crash_at[i]) and t >= crash_time and i not in resolved:
                    a.confirm(False)  # the model crashed on the device: it falls back by itself and tells the control plane
                    crashed_locally.add(i)
                    serving.pop(i)
                    resolved.add(i)
                    finished[i] = "rolled_back"
                    finish_t[i] = t
                    verdict, new = cp.resolve(i, True, float("nan"), exposure(float(t)))
                elif not math.isfinite(dev.crash_at[i]) and t >= begun + cfg.trial_min and i not in resolved:
                    resolved.add(i)
                    verdict, new = cp.resolve(i, False, float(dev.report[i]))
                else:
                    continue
                fetching.update(new)
                if verdict == "alarm":
                    state, out.t_alarm = "alarm", float(t)
                elif verdict == "promote":
                    state = "promoted"
                    fetching.update(new)
        if state != "running" and not fetching and not serving:
            break
        t += 1
    # end-of-run accounting and per-device checks
    out.served = len(ever_served)
    out.bytes_sent = sum(s.bytes_sent for s in servers)
    out.rolled_back = state == "alarm"
    if out.rolled_back:
        out.t_rolled_back = float(max((finish_t[i] for i, v in finished.items() if v == "rolled_back" and i not in crashed_locally), default=out.t_alarm or 0))
    else:
        out.t_promoted = float(max((finish_t[i] for i, v in finished.items() if v == "promoted"), default=0))
    for i in range(n):
        st = agents[i]._load()
        aud = agents[i].audit()
        if not aud["active_ok"]:
            out.violations.append((i, "active slot does not verify"))
        if st["pending"] is not None:
            out.violations.append((i, "left in a trial"))
        want = world.seq2 if finished.get(i) == "promoted" else world.seq1
        if finished.get(i) == "promoted" or finished.get(i) == "rolled_back" or i not in ever_served:
            if st["floor"].get(S.NAME) != want:
                out.violations.append((i, f"floor {st['floor'].get(S.NAME)} expected {want}"))
        if i in crashed_locally and world.h2 not in st["bad"]:
            out.violations.append((i, "crash not remembered"))
    return out
