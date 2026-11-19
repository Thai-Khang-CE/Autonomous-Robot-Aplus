"""Navigation2DEnv: goal-directed navigation of a differential-drive disc robot in a 2D room.

Observation (11, float32):  [l_1..l_8, d_goal_norm, sin(alpha), cos(alpha)]
    l_k         range reading of ray k divided by max_range, in [0, 1]
    d_goal_norm goal distance divided by the world diagonal, clipped to [0, 1]
    alpha       goal bearing relative to the robot heading, wrapped to [-pi, pi)
Action (2, float32): (u_v, u_w) in [-1, 1]^2
    v = (u_v + 1) / 2 * v_max  (forward only, no reverse)
    w = u_w * w_max
Episode outcomes (info["outcome"]): "success" | "collision" | "timeout"
    success / collision -> terminated=True, timeout -> truncated=True

The policy never sees obstacle coordinates; those appear only in ``info`` diagnostics and plots.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robot_aplus_rl.config import EnvConfig, load_env_config
from robot_aplus_rl.geometry import first_contact_fraction, goal_distance_and_bearing, unicycle_step
from robot_aplus_rl.rewards import compute_reward
from robot_aplus_rl.scenarios import TRAIN_SEED_LIMIT, Scenario, generate_scenario
from robot_aplus_rl.sensors import cast_rays, ray_angles

OUTCOME_SUCCESS = "success"
OUTCOME_COLLISION = "collision"
OUTCOME_TIMEOUT = "timeout"


class Navigation2DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        config: EnvConfig | None = None,
        scenario_pool: list[Scenario] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else load_env_config()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.scenario_pool = list(scenario_pool) if scenario_pool else None
        self._pool_index = 0

        n = self.cfg.num_rays
        low = np.concatenate([np.zeros(n + 1), [-1.0, -1.0]]).astype(np.float32)
        high = np.ones(n + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._ray_offsets = ray_angles(n)
        cats = list(self.cfg.train_category_weights)
        w = np.asarray([self.cfg.train_category_weights[c] for c in cats], dtype=np.float64)
        self._train_categories, self._train_probs = cats, w / w.sum()

        self.scenario: Scenario | None = None
        self._done = True

    # ------------------------------------------------------------------ helpers
    @property
    def pose(self) -> tuple[float, float, float]:
        return self._x, self._y, self._theta

    def _goal_terms(self) -> tuple[float, float]:
        gx, gy = self.scenario.goal
        return goal_distance_and_bearing(self._x, self._y, self._theta, gx, gy)

    def _ranges(self) -> np.ndarray:
        return cast_rays(
            self._x, self._y, self._theta, self._ray_offsets, self.scenario.obstacles,
            self.cfg.world_size, self.cfg.max_range,
        )

    def _observation(self) -> np.ndarray:
        dist, bearing = self._goal_terms()
        obs = np.empty(self.cfg.num_rays + 3, dtype=np.float32)
        obs[: self.cfg.num_rays] = self._ranges() / self.cfg.max_range
        obs[self.cfg.num_rays] = min(dist / self.cfg.world_diagonal, 1.0)
        obs[-2] = math.sin(bearing)
        obs[-1] = math.cos(bearing)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def action_to_velocity(self, action: np.ndarray) -> tuple[float, float]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {np.shape(action)}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"action contains non-finite values: {a}")
        a = np.clip(a, -1.0, 1.0)
        v = (a[0] + 1.0) / 2.0 * self.cfg.max_linear_velocity
        w = a[1] * self.cfg.max_angular_velocity
        return float(v), float(w)

    def _select_scenario(self, options: dict[str, Any] | None) -> Scenario:
        options = options or {}
        if "scenario" in options:
            return options["scenario"]
        if "category" in options or "scenario_seed" in options:
            return generate_scenario(options["category"], int(options["scenario_seed"]), self.cfg)
        if self.scenario_pool:
            sc = self.scenario_pool[self._pool_index % len(self.scenario_pool)]
            self._pool_index += 1
            return sc
        cat = self._train_categories[
            int(self.np_random.choice(len(self._train_categories), p=self._train_probs))
        ]
        return generate_scenario(cat, int(self.np_random.integers(0, TRAIN_SEED_LIMIT)), self.cfg)

    def _info(self) -> dict[str, Any]:
        dist, bearing = self._goal_terms()
        return {
            "scenario_id": self.scenario.scenario_id,
            "category": self.scenario.category,
            "distance_to_goal": dist,
            "goal_bearing": bearing,
            "step": self._steps,
            "pose": (self._x, self._y, self._theta),
        }

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.scenario = self._select_scenario(options)
        self._x, self._y, self._theta = self.scenario.start
        self._steps = 0
        self._done = False
        self._episode_return = 0.0
        self._path_length = 0.0
        self._initial_distance = self._goal_terms()[0]
        self._trajectory = [(self._x, self._y, self._theta)]
        return self._observation(), self._info()

    def step(self, action):
        if self.scenario is None or self._done:
            raise RuntimeError("step() called before reset() or after the episode ended")
        v, w = self.action_to_velocity(action)
        prev_dist = self._goal_terms()[0]
        x0, y0 = self._x, self._y
        x1, y1, theta1 = unicycle_step(x0, y0, self._theta, v, w, self.cfg.dt)

        contact = first_contact_fraction(
            (x0, y0), (x1, y1), self.cfg.robot_radius, self.scenario.obstacles, self.cfg.world_size
        )
        collided = contact is not None
        if collided:  # stop the robot at the first contact point (no penetration)
            x1, y1 = x0 + contact * (x1 - x0), y0 + contact * (y1 - y0)
        self._path_length += math.hypot(x1 - x0, y1 - y0)
        self._x, self._y, self._theta = x1, y1, theta1
        self._steps += 1
        self._trajectory.append((x1, y1, theta1))

        new_dist = self._goal_terms()[0]
        reached = (not collided) and new_dist <= self.cfg.goal_tolerance
        reward, components = compute_reward(prev_dist, new_dist, reached, collided, self.cfg.reward)
        terminated = reached or collided
        truncated = (not terminated) and self._steps >= self.cfg.max_episode_steps
        self._episode_return += reward

        info = self._info()
        info.update(reward_components=components, linear_velocity=v, angular_velocity=w)
        if terminated or truncated:
            self._done = True
            outcome = (
                OUTCOME_SUCCESS if reached else OUTCOME_COLLISION if collided else OUTCOME_TIMEOUT
            )
            info.update(
                outcome=outcome,
                is_success=reached,
                is_collision=collided,
                is_timeout=outcome == OUTCOME_TIMEOUT,
                episode_summary=self.episode_summary(outcome),
            )
        obs = self._observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def episode_summary(self, outcome: str | None = None) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "category": self.scenario.category,
            "scenario_seed": self.scenario.seed,
            "outcome": outcome,
            "episode_return": self._episode_return,
            "episode_length": self._steps,
            "path_length": self._path_length,
            "initial_distance": self._initial_distance,
            "final_distance": self._goal_terms()[0],
            "time_to_goal": self._steps * self.cfg.dt if outcome == OUTCOME_SUCCESS else None,
            "trajectory": [tuple(map(float, p)) for p in self._trajectory],
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        from robot_aplus_rl.visualization import render_rgb

        return render_rgb(self.scenario, self._trajectory, self.cfg)
