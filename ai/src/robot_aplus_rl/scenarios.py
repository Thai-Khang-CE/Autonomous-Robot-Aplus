"""Seeded scenario generation (obstacles, start pose, goal) and fixed scenario sets.

Seed ranges are disjoint so that held-out evaluation scenarios are never used for training:

* training resets draw scenario seeds in [0, 1_000_000) from ``env.np_random``;
* validation (checkpoint monitoring during training) uses seeds 1_000_000 + i;
* the final held-out TEST set uses seeds 2_000_000 + i.

Every generated scenario is checked for collision-free start/goal and a navigable route
(grid BFS in the robot's configuration space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robot_aplus_rl.config import EnvConfig
from robot_aplus_rl.geometry import clearance
from robot_aplus_rl.planning import is_reachable

CATEGORIES = ("open", "indoor", "narrow")
_CATEGORY_CODE = {"open": 11, "indoor": 22, "narrow": 33, "custom": 99}

TRAIN_SEED_LIMIT = 1_000_000
VALIDATION_SEED_OFFSET = 1_000_000
TEST_SEED_OFFSET = 2_000_000


@dataclass(frozen=True, eq=False)
class Scenario:
    scenario_id: str
    category: str
    seed: int
    obstacles: np.ndarray  # (N, 3): cx, cy, radius
    start: tuple[float, float, float]  # x, y, theta
    goal: tuple[float, float]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "seed": self.seed,
            "obstacles": np.asarray(self.obstacles).round(4).tolist(),
            "start": [round(v, 4) for v in self.start],
            "goal": [round(v, 4) for v in self.goal],
        }


def custom_scenario(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    obstacles: list[tuple[float, float, float]] | np.ndarray = (),
    scenario_id: str = "custom",
) -> Scenario:
    obs = np.asarray(obstacles, dtype=np.float64).reshape(-1, 3)
    return Scenario(scenario_id, "custom", -1, obs, tuple(map(float, start)), tuple(map(float, goal)))


def _random_circles(rng, n, r_lo, r_hi, cfg: EnvConfig) -> list[tuple[float, float, float]]:
    out = []
    for _ in range(n):
        r = rng.uniform(r_lo, r_hi)
        cx, cy = rng.uniform(r, cfg.world_size - r, size=2)
        out.append((cx, cy, r))
    return out


def _narrow_wall(rng, cfg: EnvConfig) -> tuple[list[tuple[float, float, float]], float, bool]:
    """A straight partition across the room made of touching circles, with one doorway."""
    W = cfg.world_size
    wall_pos = rng.uniform(0.4 * W, 0.6 * W)
    gap_centre = rng.uniform(0.2 * W, 0.8 * W)
    gap_width = rng.uniform(0.60, 0.80)  # minimum surface-to-surface doorway width [m]
    r_post = 0.15
    circles = []
    for c in np.arange(0.0, W + 1e-9, 0.2):
        if c + r_post <= gap_centre - gap_width / 2 or c - r_post >= gap_centre + gap_width / 2:
            circles.append((wall_pos, float(c), r_post))
    horizontal = bool(rng.random() < 0.5)
    if horizontal:
        circles = [(y, x, r) for x, y, r in circles]
    return circles, wall_pos, horizontal


def _sample_free_point(rng, obstacles, cfg: EnvConfig, extra: float, region=None, tries=400):
    lo = cfg.robot_radius + extra
    hi = cfg.world_size - lo
    for _ in range(tries):
        if region is None:
            x, y = rng.uniform(lo, hi, size=2)
        else:
            (xl, xh), (yl, yh) = region
            x, y = rng.uniform(max(lo, xl), min(hi, xh)), rng.uniform(max(lo, yl), min(hi, yh))
        if clearance(x, y, obstacles, cfg.world_size) >= cfg.robot_radius + extra:
            return float(x), float(y)
    return None


def generate_scenario(category: str, seed: int, cfg: EnvConfig, max_attempts: int = 100) -> Scenario:
    """Deterministically generate a valid scenario of the given category from ``seed``."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    rng = np.random.default_rng([_CATEGORY_CODE[category], int(seed)])
    W = cfg.world_size
    for attempt in range(max_attempts):
        start_region = goal_region = None
        meta: dict = {"attempt": attempt}
        if category == "open":
            circles = _random_circles(rng, int(rng.integers(1, 4)), 0.20, 0.40, cfg)
        elif category == "indoor":
            circles = _random_circles(rng, int(rng.integers(5, 9)), 0.20, 0.50, cfg)
        else:
            circles, wall_pos, horizontal = _narrow_wall(rng, cfg)
            extra = _random_circles(rng, int(rng.integers(1, 4)), 0.20, 0.35, cfg)
            # keep the extra clutter away from the doorway line
            axis = 1 if horizontal else 0
            circles += [c for c in extra if abs(c[axis] - wall_pos) > c[2] + 0.6]
            near, far = (0.0, wall_pos - 0.6), (wall_pos + 0.6, W)
            full = (0.0, W)
            a, b = (near, far) if rng.random() < 0.5 else (far, near)
            start_region = (full, a) if horizontal else (a, full)
            goal_region = (full, b) if horizontal else (b, full)
            meta.update(wall_pos=float(wall_pos), horizontal=horizontal)
        obstacles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
        start_xy = _sample_free_point(rng, obstacles, cfg, cfg.start_clearance, start_region)
        if start_xy is None:
            continue
        goal_xy = None
        for _ in range(50):
            cand = _sample_free_point(rng, obstacles, cfg, cfg.goal_clearance, goal_region)
            if cand is not None and math.dist(cand, start_xy) >= cfg.min_start_goal_distance:
                goal_xy = cand
                break
        if goal_xy is None:
            continue
        if not is_reachable(
            start_xy, goal_xy, obstacles, W, cfg.robot_radius, cfg.planning_margin,
            cfg.planning_resolution,
        ):
            continue
        theta = float(rng.uniform(-math.pi, math.pi))
        return Scenario(
            f"{category}-{seed}", category, int(seed), obstacles,
            (start_xy[0], start_xy[1], theta), goal_xy, meta,
        )
    raise RuntimeError(f"could not generate a valid {category} scenario for seed {seed}")


def scenario_set(name: str, per_category: int, cfg: EnvConfig) -> list[Scenario]:
    """Fixed, ordered scenario list: ``validation`` or ``test`` (held-out)."""
    offsets = {"validation": VALIDATION_SEED_OFFSET, "test": TEST_SEED_OFFSET}
    if name not in offsets:
        raise ValueError(f"unknown scenario set {name!r}")
    return [
        generate_scenario(cat, offsets[name] + i, cfg)
        for cat in CATEGORIES
        for i in range(per_category)
    ]
