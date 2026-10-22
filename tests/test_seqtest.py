import math

import numpy as np
import pytest

from fleetforge import methods as M
from fleetforge import streams as S


def rng(seed=0):
    return np.random.default_rng(seed)


def lag1(d):
    x = d.astype(float) - d.mean()
    return float((x[:, :-1] * x[:, 1:]).mean() / x.var())


def test_stream_mean_matches_delta_and_values_are_ternary():
    d = S.simulate(S.StreamParams(delta=0.02), 200, 5000, rng())
    assert set(np.unique(d)) <= {-1, 0, 1}
    assert abs(d.mean() - 0.02) < 0.002
    assert abs((d != 0).mean() - 0.10) < 0.005


def test_scene_variation_creates_correlation_and_none_means_independent():
    iid = S.simulate(S.StreamParams(), 200, 5000, rng(1))
    cor = S.simulate(S.StreamParams(sigma_z=0.03, tau=200), 200, 5000, rng(1))
    assert abs(lag1(iid)) < 0.01
    assert 0.004 < lag1(cor) < 0.02  # about sigma_z^2 / var(d) = 0.0009 / 0.1: small per frame, but it accumulates
    # which shows in block means: they vary several times more than independent frames would allow
    iid_var = M.block_means(iid, 1000).var()
    cor_var = M.block_means(cor, 1000).var()
    assert cor_var > 2.5 * iid_var
    assert abs(cor.mean()) < 0.003  # the variation averages out: the overall gain is still delta


def test_bad_stream_parameters_are_rejected():
    for kw in ({"q": 0.0}, {"delta": 0.5}, {"sigma_z": -1.0}, {"tau": 0.0}):
        with pytest.raises(ValueError):
            S.StreamParams(**kw)


@pytest.mark.parametrize(
    "rule",
    [M.fixed_naive, M.naive_peek, M.iid_mixture, lambda d: M.fixed_block(d, 100), lambda d: M.block_betting(d, 100), lambda d: M.block_mixture(d, 100)],
)
def test_no_disagreement_means_no_decision(rule):
    d = np.zeros((3, 4000), dtype=np.int8)
    o, s = rule(d)
    assert (o == 0).all() and (s == 4000).all()


def test_combine_picks_the_earliest_crossing():
    up = np.array([[False, True, False], [False, False, False], [True, False, False]])
    dn = np.array([[False, False, True], [False, False, True], [False, True, False]])
    o, s = M._combine(up, dn, np.array([100, 200, 300]), 300)
    assert o.tolist() == [1, -1, 1] and s.tolist() == [200, 300, 100]


def test_betting_wealth_matches_a_hand_computation():
    logw = M._bet_wealth(np.array([[1.0, 1.0]]))
    assert logw[0, 0] == 0.0  # no bet before any data
    # second bet: mean 0.5, variance (0.04 + 0) / 2 = 0.02, lambda = 0.5 / (0.02 + 0.25) clipped to 0.9
    assert math.isclose(logw[0, 1], math.log(1 + 0.9), rel_tol=1e-12)


def test_betting_never_bets_against_the_direction_it_has_seen():
    # a stream that has only gone down gives lambda = 0 for the 'up' test: wealth stays at 1
    assert np.allclose(M._bet_wealth(-np.ones((1, 20))), 0.0)


def test_extreme_blocks_are_clipped_and_stay_finite():
    d = np.ones((2, 4000), dtype=np.int8)
    o, s = M.block_betting(d, 100)
    assert (o == 1).all() and (s <= 4000).all()
    logw = M._bet_wealth(np.clip(M.block_means(d, 100), -0.1, 0.1) / 0.1)
    assert np.isfinite(logw).all()


def test_peeking_inflates_the_error_rate_and_the_sequential_rules_do_not():
    d = S.simulate(S.StreamParams(), 1000, 20000, rng(2))
    peek = (M.naive_peek(d)[0] == 1).mean()
    assert peek > 0.10  # nominal 2.5%
    for rule in (lambda x: M.iid_mixture(x), lambda x: M.block_betting(x, 500), lambda x: M.block_mixture(x, 500)):
        assert (rule(d)[0] == 1).mean() <= 0.04


def test_block_betting_is_valid_for_a_slightly_worse_candidate():
    d = S.simulate(S.StreamParams(delta=-0.01, sigma_z=0.03, tau=100), 1000, 20000, rng(3))
    assert (M.block_betting(d, 1000)[0] == 1).mean() <= 0.04


def test_blocks_shorter_than_the_correlation_time_break_the_guarantee():
    d = S.simulate(S.StreamParams(sigma_z=0.03, tau=2000), 1000, 20000, rng(4))
    short = (M.block_betting(d, 100)[0] == 1).mean()
    long = (M.block_betting(d, 4000)[0] == 1).mean()
    assert short > 0.04 > long - 0.0  # short blocks inflate; long blocks stay within the target


def test_a_clear_improvement_is_promoted_and_a_clear_regression_rejected():
    up = S.simulate(S.StreamParams(delta=0.04), 300, 20000, rng(5))
    dn = S.simulate(S.StreamParams(delta=-0.04), 300, 20000, rng(6))
    assert (M.block_betting(up, 500)[0] == 1).mean() > 0.95
    assert (M.block_betting(dn, 500)[0] == -1).mean() > 0.95


def test_results_are_reproducible():
    a = M.block_betting(S.simulate(S.StreamParams(delta=0.01), 50, 8000, rng(7)), 500)
    b = M.block_betting(S.simulate(S.StreamParams(delta=0.01), 50, 8000, rng(7)), 500)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
