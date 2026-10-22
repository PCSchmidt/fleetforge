"""Decision rules for promoting a candidate model from a stream of paired differences d_t in {-1, 0, +1}.

Every rule returns (outcome, stop) arrays over replications: outcome is +1 (promote: the candidate is better),
-1 (reject: it is worse) or 0 (undecided by the end), and stop is the number of frames used when it decided
(the full stream length if undecided). All are one-sided at level `alpha` per direction.

Rules, in order of how much they assume:

  fixed_naive     z-test on the final mean, frames treated as independent. Valid only for one look and independent frames.
  naive_peek      the same z-test repeated every `every` frames, stopping at the first crossing. Repeated looks inflate
                  the error rate; this is what you get from "check the dashboard and promote when it looks good".
  iid_mixture     a normal-mixture confidence sequence on the frame mean (valid at every look) with plug-in variance.
                  Handles peeking, still assumes independent frames.
  fixed_block     z-test on block means at the end of the stream. Handles correlated frames, one look.
  block_betting   a betting e-process on block means (valid at every look, no variance or distribution assumption
                  beyond block means lying in [-1, 1]). Handles peeking, and correlation if blocks are long enough
                  to be roughly independent.
"""

from __future__ import annotations

import math

import numpy as np

Z = 1.959964  # one-sided 2.5%


def _first(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a boolean (reps, looks) array: whether any look is true, and the first such look index."""
    any_ = mask.any(axis=1)
    return any_, mask.argmax(axis=1)


def _combine(up: np.ndarray, down: np.ndarray, frames_at: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Combine 'promote' and 'reject' crossing masks (reps, looks) into outcome and stop frame, earliest wins."""
    up_any, up_i = _first(up)
    dn_any, dn_i = _first(down)
    outcome = np.zeros(up.shape[0], dtype=np.int8)
    stop = np.full(up.shape[0], n_frames, dtype=np.int64)
    both = up_any & dn_any
    take_up = up_any & (~dn_any | (up_i <= dn_i))
    take_dn = dn_any & (~up_any | (dn_i < up_i))
    assert not (take_up & take_dn).any() and (take_up | take_dn).sum() == (up_any | dn_any).sum()
    del both
    outcome[take_up] = 1
    outcome[take_dn] = -1
    stop[take_up] = frames_at[up_i[take_up]]
    stop[take_dn] = frames_at[dn_i[take_dn]]
    return outcome, stop


def _checkpoints(d: np.ndarray, every: int):
    n = d.shape[1]
    idx = np.arange(every - 1, n, every)
    cs = np.cumsum(d, axis=1, dtype=np.int64)[:, idx].astype(np.float64)
    cs2 = np.cumsum(np.abs(d), axis=1, dtype=np.int64)[:, idx].astype(np.float64)  # d^2 = |d|
    frames = idx + 1
    mean = cs / frames
    var = np.maximum(cs2 / frames - mean**2, 0.0)
    return frames, mean, var


def fixed_naive(d: np.ndarray, z: float = Z) -> tuple[np.ndarray, np.ndarray]:
    n = d.shape[1]
    frames, mean, var = _checkpoints(d, n)
    se = np.sqrt(var[:, -1] / n)
    zs = np.divide(mean[:, -1], se, out=np.zeros(d.shape[0]), where=se > 0)
    outcome = np.where(zs > z, 1, np.where(zs < -z, -1, 0)).astype(np.int8)
    return outcome, np.full(d.shape[0], n, dtype=np.int64)


def naive_peek(d: np.ndarray, every: int = 100, z: float = Z) -> tuple[np.ndarray, np.ndarray]:
    frames, mean, var = _checkpoints(d, every)
    se = np.sqrt(var / frames)
    zs = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
    return _combine(zs > z, zs < -z, frames, d.shape[1])


def iid_mixture(d: np.ndarray, every: int = 100, alpha: float = 0.025, n_opt: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Normal-mixture confidence sequence on the frame mean: stop when the mixture martingale exceeds 1/alpha.

    With S_n the running sum and V_n = n * (running variance), M_n = sqrt(rho/(V+rho)) exp(S^2 / (2 (V+rho))) is a
    martingale for independent sub-Gaussian increments; P(any M_n >= 1/alpha) <= alpha, so each direction's error is
    at most alpha. The variance is a plug-in estimate, so this is approximate even for independent frames.
    """
    frames, mean, var = _checkpoints(d, every)
    v = frames * np.maximum(var, 1e-12)
    rho = n_opt * 0.1
    s = mean * frames
    log_m = 0.5 * np.log(rho / (v + rho)) + s**2 / (2.0 * (v + rho))
    cross = log_m >= math.log(1.0 / alpha)
    return _combine(cross & (s > 0), cross & (s < 0), frames, d.shape[1])


def block_means(d: np.ndarray, block: int) -> np.ndarray:
    k = d.shape[1] // block
    return d[:, : k * block].reshape(d.shape[0], k, block).mean(axis=2, dtype=np.float64)


def fixed_block(d: np.ndarray, block: int, z: float = Z) -> tuple[np.ndarray, np.ndarray]:
    m = block_means(d, block)
    k = m.shape[1]
    sd = m.std(axis=1, ddof=1)
    se = sd / math.sqrt(k)
    zs = np.divide(m.mean(axis=1), se, out=np.zeros(m.shape[0]), where=se > 0)
    outcome = np.where(zs > z, 1, np.where(zs < -z, -1, 0)).astype(np.int8)
    return outcome, np.full(d.shape[0], d.shape[1], dtype=np.int64)


def _bet_wealth(x: np.ndarray, lam_max: float = 0.9, prior_var: float = 0.04) -> np.ndarray:
    """log-wealth path (reps, K) of a betting e-process against H0: E[x_k | past] <= 0, for x_k in [-1, 1].

    The bet size lambda_k is computed from blocks before k only (an aGRAPA-style plug-in: mean over variance plus
    mean squared, clipped to [0, lam_max] so wealth stays positive), so the wealth is a non-negative
    supermartingale under H0 and, by Ville's inequality, P(wealth ever reaches 1/alpha) <= alpha.
    """
    reps, k = x.shape
    logw = np.zeros((reps, k))
    s = np.zeros(reps)
    ss = np.zeros(reps)
    cur = np.zeros(reps)
    for i in range(k):
        mu = s / (i + 1)  # one pseudo-observation of 0 shrinks the early estimates
        var = (prior_var + ss) / (i + 1)
        lam = np.clip(mu / (var + mu**2), 0.0, lam_max)
        cur = cur + np.log1p(lam * x[:, i])
        logw[:, i] = cur
        s = s + x[:, i]
        mean_now = s / (i + 1)
        ss = ss + (x[:, i] - mean_now) ** 2
    return logw


def block_betting(d: np.ndarray, block: int, c: float = 0.05, alpha: float = 0.025, beta: float = 0.025) -> tuple[np.ndarray, np.ndarray]:
    """Betting e-process on block means, clipped to [-c, c] (default c = 0.05, chosen after looking at results on the main simulation settings; see docs) and scaled to [-1, 1].

    `c` is a design assumption: a block's accuracy gain (candidate minus production) is not expected to exceed c
    in magnitude. Clipping makes the test valid for the clipped block mean whatever happens; it matches the test of
    the real mean unless blocks beyond +-c are common.
    """
    x = np.clip(block_means(d, block), -c, c) / c
    up = _bet_wealth(x) >= math.log(1.0 / alpha)
    down = _bet_wealth(-x) >= math.log(1.0 / beta)
    frames = (np.arange(x.shape[1]) + 1) * block
    return _combine(up, down, frames, d.shape[1])


def block_mixture(d: np.ndarray, block: int, alpha: float = 0.025, k_opt: int = 40, k_min: int = 10, q_guess: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Normal-mixture confidence sequence on block means with plug-in variance: no bound c needed, but approximate
    (valid for independent blocks only as the variance estimate converges), and it makes no decision before
    `k_min` blocks."""
    m = block_means(d, block)
    reps, k = m.shape
    n = np.arange(1, k + 1)
    cs = np.cumsum(m, axis=1)
    mean = cs / n
    var = np.maximum(np.cumsum(m**2, axis=1) / n - mean**2, 1e-12)
    v = n * var
    rho = k_opt * q_guess / block
    log_m = 0.5 * np.log(rho / (v + rho)) + cs**2 / (2.0 * (v + rho))
    cross = (log_m >= math.log(1.0 / alpha)) & (n >= k_min)
    frames = n * block
    return _combine(cross & (cs > 0), cross & (cs < 0), frames, d.shape[1])
