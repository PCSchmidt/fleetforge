"""Simulated paired shadow-evaluation streams.

On each frame, production and candidate models are scored. Only the difference matters:

    d_t = +1  candidate right, production wrong
    d_t = -1  candidate wrong, production right
    d_t =  0  they agree (both right or both wrong)

`q` is the disagreement rate: the fraction of frames on which the two models differ. It is the knob that sets how
noisy the comparison is (two nearly identical models disagree rarely, so a small difference is easy to see).

The candidate's advantage is not constant. It varies over time, because the scene changes (night, rain, a new
location) and the candidate is better on some scenes and worse on others:

    delta_t = delta + z_t,    z_t an AR(1) process with standard deviation `sigma_z` and correlation time `tau`

Given delta_t, a frame is +1 with probability (q + delta_t) / 2 and -1 with probability (q - delta_t) / 2, so the
mean of d_t is delta_t and the overall mean is `delta` (the true accuracy gain, in accuracy points / 100).
Consecutive frames are therefore positively correlated through z_t; with sigma_z = 0 they are independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamParams:
    delta: float = 0.0  # true accuracy gain of the candidate over production
    q: float = 0.10  # disagreement rate
    sigma_z: float = 0.0  # standard deviation of the scene-to-scene variation of the gain
    tau: float = 200.0  # correlation time of that variation, in frames

    def __post_init__(self) -> None:
        if not 0 < self.q <= 1:
            raise ValueError("q must be in (0, 1]")
        if self.sigma_z < 0 or self.tau <= 0:
            raise ValueError("sigma_z must be >= 0 and tau > 0")
        if abs(self.delta) > self.q:
            raise ValueError("|delta| cannot exceed the disagreement rate q")


def simulate(params: StreamParams, reps: int, frames: int, rng: np.random.Generator) -> np.ndarray:
    """Return an int8 array (reps, frames) of d_t in {-1, 0, +1}."""
    q = params.q
    delta_t = np.full((reps, frames), params.delta, dtype=np.float32)
    if params.sigma_z > 0:
        rho = np.exp(-1.0 / params.tau)
        innov_sd = params.sigma_z * np.sqrt(1.0 - rho * rho)
        z = rng.normal(0.0, params.sigma_z, size=reps)  # stationary start
        eps = rng.normal(0.0, innov_sd, size=(frames, reps)).astype(np.float32)
        for t in range(frames):
            z = rho * z + eps[t]
            delta_t[:, t] += z
    np.clip(delta_t, -q, q, out=delta_t)
    u = rng.random(size=(reps, frames), dtype=np.float32)
    p_plus = 0.5 * (q + delta_t)
    p_minus = 0.5 * (q - delta_t)
    d = np.zeros((reps, frames), dtype=np.int8)
    d[u < p_plus] = 1
    d[(u >= p_plus) & (u < p_plus + p_minus)] = -1
    return d
