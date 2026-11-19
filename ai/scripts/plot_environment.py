"""Environment visualisation generated directly from the simulator code (not a screenshot).

1. environment_overview: one scenario per category with the robot footprint, the 8 range rays
   (lengths = actual cast_rays output at the start pose), obstacles, goal and reference path.
2. observation_example: the 11-D observation vector returned by env.reset() for that scenario.
Outputs to results/figures/ and reports/week09/figures/.
"""

from __future__ import annotations

import json
import math
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robot_aplus_rl import Navigation2DEnv, load_env_config  # noqa: E402
from robot_aplus_rl.config import PROJECT_ROOT  # noqa: E402
from robot_aplus_rl.planning import path_length, reference_path  # noqa: E402
from robot_aplus_rl.scenarios import generate_scenario  # noqa: E402
from robot_aplus_rl.sensors import cast_rays, ray_angles  # noqa: E402
from robot_aplus_rl.visualization import draw_scenario  # noqa: E402

FIG = PROJECT_ROOT / "results" / "figures"
REP = PROJECT_ROOT / "reports" / "week09" / "figures"
RAY = "#2a78d6"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        shutil.copy(FIG / f"{name}.{ext}", REP / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def main():
    cfg = load_env_config()
    env = Navigation2DEnv(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    records = {}
    for ax, cat in zip(axes, ("open", "indoor", "narrow")):
        sc = generate_scenario(cat, 2_000_000, cfg)  # first held-out test scenario of each type
        obs, info = env.reset(options={"scenario": sc})
        x, y, th = sc.start
        d = cast_rays(x, y, th, ray_angles(cfg.num_rays), sc.obstacles, cfg.world_size, cfg.max_range)
        ref = reference_path(sc.start[:2], sc.goal, sc.obstacles, cfg.world_size, cfg.robot_radius,
                             cfg.planning_margin, cfg.planning_resolution)
        draw_scenario(ax, sc, None, cfg, reference=ref,
                      title=f"{sc.scenario_id}  ({len(sc.obstacles)} circles)\n"
                            f"start-goal {info['distance_to_goal']:.2f} m, L* = {path_length(ref):.2f} m")
        for k, (a, r) in enumerate(zip(ray_angles(cfg.num_rays), d)):
            ex, ey = x + r * math.cos(th + a), y + r * math.sin(th + a)
            ax.plot([x, ex], [y, ey], color=RAY, lw=0.9, alpha=0.8,
                    label="range rays" if k == 0 else None)
            ax.plot(ex, ey, "o", ms=2.5, color=RAY)
        ax.grid(False)
        records[sc.scenario_id] = {"observation": np.round(obs, 4).tolist(),
                                   "ray_distances_m": np.round(d, 3).tolist()}
    axes[0].legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    fig.suptitle("Navigation2DEnv: first held-out test scenario of each category "
                 "(generated from the simulator; rays = actual sensor output at the start pose)",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "environment_overview")
    (FIG / "environment_overview_observations.json").write_text(json.dumps(records, indent=1))

    # observation vector bar chart for the indoor scenario
    sid = "indoor-2000000"
    obs = np.asarray(records[sid]["observation"])
    labels = [f"$\\ell_{k + 1}$\n{45 * k}°" for k in range(8)] + ["$\\tilde d_g$", "sin α", "cos α"]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    colours = [RAY] * 8 + ["#1baf7a"] + ["#eb6834"] * 2
    ax.bar(range(11), obs, color=colours, width=0.7)
    for i, v in enumerate(obs):
        ax.text(i, v + (0.04 if v >= 0 else -0.12), f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xticks(range(11), labels, fontsize=7)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("value")
    ax.set_title(f"Observation returned by env.reset() for {sid} (float32, 11-D)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "observation_example")


if __name__ == "__main__":
    main()
