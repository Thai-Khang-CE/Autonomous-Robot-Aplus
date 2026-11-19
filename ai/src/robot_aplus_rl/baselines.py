"""Non-learning reference policies. Both use ONLY the 11-D deployable observation."""

from __future__ import annotations

import math

import numpy as np


class RandomPolicy:
    """Uniformly random actions in [-1, 1]^2 (seeded for reproducibility)."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32), None


class GoalSeekingHeuristic:
    """Reactive 'turn toward goal, slow down and steer away near obstacles' controller.

    Rays are indexed counter-clockwise from the heading: 0 = front, 1 = front-left (45 deg),
    7 = front-right (-45 deg), 2 = left, 6 = right. Distances are normalised by max_range.
    """

    name = "heuristic"

    def __init__(self, num_rays: int = 8, max_range: float = 3.5, safe_distance: float = 0.55,
                 heading_gain: float = 2.0) -> None:
        if num_rays != 8:
            raise ValueError("heuristic assumes 8 rays")
        self.max_range = max_range
        self.safe = safe_distance / max_range
        self.k = heading_gain

    def predict(self, obs, deterministic: bool = True):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        rays = obs[:8]
        alpha = math.atan2(obs[-2], obs[-1])
        u_w = float(np.clip(self.k * alpha / math.pi * 2.0, -1.0, 1.0))
        front = min(rays[0], 0.9 * rays[1], 0.9 * rays[7])
        # forward speed: slow when the goal is behind or an obstacle is close in front
        speed = max(0.0, math.cos(alpha)) * min(1.0, front / (2 * self.safe))
        if front < self.safe:
            left = rays[1] + rays[2]
            right = rays[7] + rays[6]
            u_w = 1.0 if left >= right else -1.0
            speed = 0.0 if rays[0] < 0.7 * self.safe else 0.15
        u_v = 2.0 * speed - 1.0
        return np.array([u_v, u_w], dtype=np.float32), None
