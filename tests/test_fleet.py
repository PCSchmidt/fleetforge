import math

import numpy as np
import pytest

from fleetforge import fleet as F
from fleetforge import methods as M

CFG = F.FleetConfig()


def devices(delta, seed=0, p_crash=0.0, cfg=CFG):
    return F.make_devices(cfg, F.Candidate("c", delta, p_crash), np.random.default_rng(seed))


def test_online_bet_matches_the_batch_e_process_on_the_negated_stream():
    rng = np.random.default_rng(0)
    m = rng.normal(-0.01, 0.03, size=40)
    bet = F.OnlineBet(c=0.1)
    online = [bet.update(float(v)) for v in m]
    batch = M._bet_wealth(np.clip(-m[None, :], -0.1, 0.1) / 0.1)[0]  # harm = betting that the mean is negative
    assert np.allclose(online, batch, atol=1e-9)


def test_crash_rate_process_hand_case_and_validity():
    cr = F.CrashRate(0.02, 0.15, 30.0)
    lam0, lam1 = -math.log(0.98) / 30, -math.log(0.85) / 30
    assert cr.update(100.0) == pytest.approx(math.log(lam1 / lam0) - (lam1 - lam0) * 100.0)
    # simulate a fleet whose crash hazard equals the baseline p0, devices starting at staggered times
    rng = np.random.default_rng(0)
    alarms = 0
    reps = 400
    for _ in range(reps):
        starts = np.sort(rng.uniform(0, 200, size=300))
        crash_t = starts - np.log(1 - rng.random(300)) / lam0  # exponential time to crash from each start
        crash_t = np.where(crash_t < starts + 30.0, crash_t, np.inf)  # only crashes within the 30-minute trial count
        cr, bad = F.CrashRate(0.02, 0.15, 30.0), False
        for tc in np.sort(crash_t[np.isfinite(crash_t)]):
            expo = float(np.clip(tc - starts, 0, 30.0).sum())
            if cr.update(expo) >= math.log(1 / 0.025):
                bad = True
                break
        alarms += bad
    assert alarms / reps <= 0.025 + 2.5 * math.sqrt(0.025 * 0.975 / reps)


def test_device_generation_shapes_and_crashes():
    d = devices(0.0, p_crash=1.0)
    assert d.n == 300 and np.isnan(d.report).all() and np.isfinite(d.crash_at).all()
    d = devices(0.0)
    assert np.isnan(d.report).sum() < 15 and (d.calls >= 1).all()
    assert abs(np.nanmean(d.report)) < 0.01 and np.all(np.abs(d.delta_dev) <= CFG.q)


def test_event_times_land_on_each_devices_poll_grid_and_are_ordered():
    d = devices(0.0)
    for i in range(0, 300, 7):
        start, ev = F._stage_times(CFG, d, i, 17.0)
        assert start >= 17.0 + CFG.download_min and ev >= start
        assert (ev - d.phase[i]) % CFG.poll_min == pytest.approx(0.0, abs=1e-6) or (ev - d.phase[i]) % CFG.poll_min == pytest.approx(CFG.poll_min, abs=1e-6)
        if math.isinf(d.crash_at[i]):
            assert ev >= start + CFG.trial_min


@pytest.mark.parametrize("policy", [F.EProcessStaged(), F.AllAtOnce(), F.FixedCanary()])
@pytest.mark.parametrize("delta,p_crash", [(0.02, 0.0), (-0.03, 0.0), (0.0, 0.5)])
def test_outcome_invariants(policy, delta, p_crash):
    for seed in range(5):
        o = F.run_rollout(policy, CFG, devices(delta, seed, p_crash))
        assert 0 < o.served <= CFG.n
        if o.rolled_back:
            assert o.t_alarm is not None and o.t_rolled_back >= o.t_alarm and o.t_promoted is None
            assert o.exposure_dev_hours > 0
        else:
            assert o.t_promoted is not None and o.t_promoted > 0 and o.t_alarm is None


def test_runs_are_deterministic():
    a = F.run_rollout(F.EProcessStaged(), CFG, devices(-0.02, 3))
    b = F.run_rollout(F.EProcessStaged(), CFG, devices(-0.02, 3))
    assert a == b


def test_staging_limits_how_many_devices_a_bad_model_reaches():
    staged, atonce = [], []
    for seed in range(40):
        d = devices(-0.03, seed)
        staged.append(F.run_rollout(F.EProcessStaged(), CFG, d).served)
        atonce.append(F.run_rollout(F.AllAtOnce(), CFG, d).served)
    assert np.median(staged) < 0.2 * np.median(atonce)


def test_a_clearly_bad_model_is_caught_and_a_good_one_promoted():
    caught = np.mean([F.run_rollout(F.EProcessStaged(), CFG, devices(-0.03, s)).rolled_back for s in range(60)])
    kept = np.mean([not F.run_rollout(F.EProcessStaged(), CFG, devices(0.02, s)).rolled_back for s in range(60)])
    assert caught > 0.95 and kept > 0.95


def test_the_harm_alarm_error_rate_for_an_equal_candidate_is_within_its_level_and_the_fixed_rule_is_not():
    n = 300
    ep = np.mean([F.run_rollout(F.EProcessStaged(), CFG, devices(0.0, 1000 + s)).rolled_back for s in range(n)])
    fx = np.mean([F.run_rollout(F.FixedCanary(), CFG, devices(0.0, 1000 + s)).rolled_back for s in range(n)])
    assert ep <= 0.05 + 2.5 * math.sqrt(0.05 * 0.95 / n)  # nominal 5% plus Monte Carlo slack
    assert fx > 0.10  # the dashboard rule rolls back a perfectly good model often


def test_crashes_trigger_rollback_and_the_canary_alone_can_miss_them():
    staged = [F.run_rollout(F.EProcessStaged(), CFG, devices(0.0, s, 0.4)).rolled_back for s in range(40)]
    fixed = [F.run_rollout(F.FixedCanary(), CFG, devices(0.0, s, 0.4)).rolled_back for s in range(40)]
    assert np.mean(staged) > 0.95 and np.mean(fixed) < np.mean(staged)


def test_the_fixed_canary_rule_stops_looking_after_the_canary():
    d = devices(0.0, 5)
    canary = set(int(i) for i in d.order[:5])
    d.report[[i for i in range(300) if i not in canary]] = -0.2  # everyone but the canary is badly hurt
    for i in canary:
        d.report[i] = 0.02
    d.crash_at[:] = np.inf
    o = F.run_rollout(F.FixedCanary(), CFG, d)
    assert not o.rolled_back and o.served == 300  # shipped to all despite what the rest would have shown
    assert F.run_rollout(F.EProcessStaged(), CFG, d).rolled_back


def test_staged_rollout_takes_longer_for_a_good_model_than_all_at_once():
    d = devices(0.02, 2)
    staged, atonce = F.run_rollout(F.EProcessStaged(), CFG, d), F.run_rollout(F.AllAtOnce(), CFG, d)
    assert staged.t_promoted > 2 * atonce.t_promoted


# ---------------------------------------------------------------------------------------------- real agents
@pytest.fixture(scope="module")
def world():
    from fleetforge import agent as AG
    from fleetforge import sweep as S

    return S.make_world(8 * AG.CHUNK)


@pytest.mark.parametrize("delta,p_crash,expect_rollback", [(0.02, 0.0, False), (-0.03, 0.0, True), (0.0, 0.4, True)])
def test_real_agents_end_every_device_in_a_consistent_state(world, delta, p_crash, expect_rollback):
    from fleetforge import fleet_real as FR

    o = FR.run_real(F.EProcessStaged(), CFG, F.Candidate("c", delta, p_crash), seed=3, world=world)
    assert o.rolled_back == expect_rollback
    assert not o.violations, o.violations[:3]
    if expect_rollback:
        assert o.t_rolled_back >= o.t_alarm and o.served < CFG.n
    else:
        assert o.t_promoted > 0 and o.served == CFG.n


def test_real_agents_reproduce_the_control_plane_decisions_of_the_abstract_model(world):
    from fleetforge import fleet_real as FR

    for seed in range(4):
        cand = F.Candidate("c", -0.03)
        real = FR.run_real(F.EProcessStaged(), CFG, cand, seed, world)
        abstract = F.run_rollout(F.EProcessStaged(), CFG, F.make_devices(CFG, cand, np.random.default_rng(seed)))
        assert real.rolled_back == abstract.rolled_back
        assert real.served <= 2 * abstract.served + 10  # same order of exposure, not identical timing


@pytest.mark.parametrize("policy", [F.AllAtOnce(), F.EProcessStaged()])
def test_healthy_candidates_are_not_rolled_back_more_often_than_the_levels_allow(policy):
    n = 150
    for delta in (0.02, 0.0):
        rate = np.mean([F.run_rollout(policy, CFG, devices(delta, 5000 + s)).rolled_back for s in range(n)])
        # two independent alarms (harm 5%, crash 2.5%), so up to about 7.5% plus Monte Carlo slack
        assert rate <= 0.075 + 2.5 * math.sqrt(0.075 * 0.925 / n), (policy.name, delta, rate)


def test_the_fixed_canary_waits_for_most_of_its_canary_before_deciding():
    d = devices(0.0, 8)
    d.crash_at[:] = np.inf
    canary = [int(i) for i in d.order[:5]]
    d.calls[:] = 1
    d.phase[:] = 0.0
    for k, i in enumerate(canary):
        d.phase[i] = 0.5 * k  # canary devices report in this order, half a minute apart
    d.report[:] = 0.0
    d.report[canary] = [0.02, -0.1, -0.1, -0.1, -0.1]  # the first to report looks fine; the mean of the first four does not
    o = F.run_rollout(F.FixedCanary(), CFG, d)
    assert o.rolled_back and o.cause == "canary mean"


def test_devices_return_to_production_at_their_next_poll_not_instantly():
    for seed in range(5):
        o = F.run_rollout(F.EProcessStaged(), CFG, devices(-0.03, seed))
        assert o.rolled_back
        assert o.t_alarm < o.t_rolled_back <= o.t_alarm + CFG.poll_min + 1e-9
