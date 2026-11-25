"""Episode-level evaluation on fixed scenario lists and metric aggregation.

Metrics (computed only from recorded episodes):
    success_rate   SR = N_success / N
    collision_rate CR = N_collision / N
    timeout_rate   N_timeout / N
    spl_approx     mean_i S_i * L_i / max(P_i, L_i), with L_i the APPROXIMATE shortest
                   collision-free path length from robot_aplus_rl.planning.reference_path
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from robot_aplus_rl.config import EnvConfig
from robot_aplus_rl.navigation_env import Navigation2DEnv
from robot_aplus_rl.planning import path_length, reference_path
from robot_aplus_rl.scenarios import Scenario


def reference_lengths(scenarios: list[Scenario], cfg: EnvConfig) -> dict[str, float]:
    out = {}
    for sc in scenarios:
        ref = reference_path(sc.start[:2], sc.goal, sc.obstacles, cfg.world_size, cfg.robot_radius,
                             cfg.planning_margin, cfg.planning_resolution)
        out[sc.scenario_id] = path_length(ref) if ref is not None else float("nan")
    return out


def run_episode(policy, env: Navigation2DEnv, scenario: Scenario, deterministic: bool) -> dict:
    obs, _ = env.reset(seed=0, options={"scenario": scenario})
    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info["episode_summary"]


def evaluate_on_scenarios(
    policy,
    scenarios: list[Scenario],
    cfg: EnvConfig,
    policy_label: str,
    algorithm: str,
    train_seed: int | None,
    deterministic: bool,
    ref_lengths: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    env = Navigation2DEnv(cfg)
    rows, trajectories = [], {}
    for sc in scenarios:
        s = run_episode(policy, env, sc, deterministic)
        L = (ref_lengths or {}).get(sc.scenario_id, float("nan"))
        success = s["outcome"] == "success"
        if not math.isfinite(L):
            spl = float("nan")  # no valid reference path -> SPL not computed
        else:
            spl = L / max(s["path_length"], L) if success else 0.0
        rows.append({
            "policy": policy_label,
            "algorithm": algorithm,
            "train_seed": train_seed,
            "deterministic": deterministic,
            "scenario_id": sc.scenario_id,
            "category": sc.category,
            "scenario_seed": sc.seed,
            "outcome": s["outcome"],
            "success": int(success),
            "collision": int(s["outcome"] == "collision"),
            "timeout": int(s["outcome"] == "timeout"),
            "episode_return": s["episode_return"],
            "episode_length": s["episode_length"],
            "path_length": s["path_length"],
            "time_to_goal": s["time_to_goal"],
            "initial_distance": s["initial_distance"],
            "final_distance": s["final_distance"],
            "reference_path_length": L,
            "spl_term_approx": spl,
        })
        trajectories[sc.scenario_id] = s["trajectory"]
    env.close()
    return pd.DataFrame(rows), trajectories


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(df: pd.DataFrame) -> dict:
    n = len(df)
    succ = df[df["success"] == 1]
    lo, hi = wilson_interval(int(df["success"].sum()), n)
    return {
        "episodes": n,
        "success_rate": df["success"].mean(),
        "success_ci95_low": lo,
        "success_ci95_high": hi,
        "collision_rate": df["collision"].mean(),
        "timeout_rate": df["timeout"].mean(),
        "mean_return": df["episode_return"].mean(),
        "std_return": df["episode_return"].std(ddof=0),
        "mean_length": df["episode_length"].mean(),
        "mean_path_length": df["path_length"].mean(),
        "mean_time_to_goal_success": succ["time_to_goal"].mean() if len(succ) else float("nan"),
        "mean_final_distance": df["final_distance"].mean(),
        "spl_approx": df["spl_term_approx"].mean(),
    }


def save_trajectories(path: Path, trajectories: dict[str, list]) -> None:
    compact = {k: np.round(np.asarray(v), 4).tolist() for k, v in trajectories.items()}
    path.write_text(json.dumps(compact), encoding="utf-8")
