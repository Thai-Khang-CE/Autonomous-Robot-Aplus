"""Static plots of scenarios and trajectories (matplotlib, non-interactive backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from robot_aplus_rl.config import EnvConfig  # noqa: E402
from robot_aplus_rl.scenarios import Scenario  # noqa: E402

OUTCOME_COLOURS = {"success": "#1b7837", "collision": "#b2182b", "timeout": "#d95f02", None: "#444"}


def draw_scenario(
    ax,
    scenario: Scenario,
    trajectory,
    cfg: EnvConfig,
    title: str | None = None,
    outcome: str | None = None,
    reference: list | None = None,
):
    W = cfg.world_size
    ax.add_patch(Rectangle((0, 0), W, W, fill=False, lw=2.0, ec="black"))
    for cx, cy, r in np.asarray(scenario.obstacles):
        ax.add_patch(Circle((cx, cy), r, fc="#7f7f7f", ec="#3f3f3f", lw=0.6))
    gx, gy = scenario.goal
    ax.add_patch(Circle((gx, gy), cfg.goal_tolerance, fc="#4dac26", alpha=0.25, ec="#1b7837"))
    ax.plot(gx, gy, marker="*", ms=14, color="#1b7837", label="goal", ls="none")
    sx, sy, st = scenario.start
    ax.add_patch(Circle((sx, sy), cfg.robot_radius, fill=False, ec="#2166ac", lw=1.2))
    ax.arrow(sx, sy, 0.35 * np.cos(st), 0.35 * np.sin(st), width=0.02, color="#2166ac")
    ax.plot(sx, sy, "o", color="#2166ac", ms=4, label="start")
    if reference is not None:
        ref = np.asarray(reference)
        ax.plot(ref[:, 0], ref[:, 1], ls=":", color="#555", lw=1.2, label="reference path")
    if trajectory is not None and len(trajectory) > 0:
        tr = np.asarray(trajectory)
        colour = OUTCOME_COLOURS.get(outcome, "#444")
        ax.plot(tr[:, 0], tr[:, 1], color=colour, lw=1.6, label="trajectory")
        ex, ey = tr[-1, 0], tr[-1, 1]
        ax.add_patch(Circle((ex, ey), cfg.robot_radius, fill=False, ec=colour, lw=1.2, ls="--"))
    ax.set_xlim(-0.1, W + 0.1)
    ax.set_ylim(-0.1, W + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def render_rgb(scenario: Scenario, trajectory, cfg: EnvConfig) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    draw_scenario(ax, scenario, trajectory, cfg, title=scenario.scenario_id)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_trajectory_figure(path_stem, scenario, trajectory, cfg, title, outcome, reference=None):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    draw_scenario(ax, scenario, trajectory, cfg, title=title, outcome=outcome, reference=reference)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=160)
    plt.close(fig)
