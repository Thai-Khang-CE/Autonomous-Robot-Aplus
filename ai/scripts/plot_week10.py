"""Generate Week 10 scientific figures from saved trial, episode and trajectory data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robot_aplus_rl.config import load_env_config
from robot_aplus_rl.visualization import draw_scenario
from evaluate_week10 import valid_reference
from week10 import checked_config, load_scenarios, read_json

ALGORITHMS = ("ppo", "sac")
GROUPS = ("baseline", "tuned", "historical")
COLORS = {"baseline": "#64809b", "tuned": "#087f73", "historical": "#9a622b",
          "success": "#087f73", "collision": "#b24d3d", "timeout": "#b58234"}
OUTCOMES = ("success", "collision", "timeout")


def save(fig, output: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def optimization(root: Path, output: Path):
    fig, axs = plt.subplots(1, 2, figsize=(9.1, 3.05), layout="constrained", sharey=True)
    for algo, ax in zip(ALGORITHMS, axs):
        trials = pd.read_csv(root / "tuning" / f"{algo}_trials.csv")
        trials = trials[trials.state.eq("COMPLETE")].sort_values("number")
        selection = read_json(root / "tuning" / f"{algo}_selection.json")
        ax.plot(trials.number, trials.value.cummax(), color="#748696", lw=1.2, ls="--", label="Best to date")
        ax.scatter(trials.number, trials.value, s=35, color=COLORS["baseline"], zorder=4, label="Trial")
        ax.scatter(selection["trial"], selection["validation_score"], s=115, marker="*",
                   color=COLORS["tuned"], zorder=5, label="Selected alternative")
        ax.set_xticks(trials.number)
        ax.set_title(algo.upper(), loc="left", weight="bold")
        ax.set_xlabel("Optuna trial number")
        ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylabel("Validation score")
    axs[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    save(fig, output, "optimization_history")


def outcomes(frame, output: Path, algo: str):
    frame = frame[frame.algorithm.eq(algo)]
    cats = ["all", "open", "indoor", "narrow"]
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3), layout="constrained", sharey=True)
    for ax, outcome in zip(axes, OUTCOMES):
        x = np.arange(len(cats))
        for index, group in enumerate(GROUPS):
            item = frame[frame.group.eq(group)]
            center = x + (index - 1) * 0.24
            metric = f"{outcome}_rate"
            populations = [100 * item[item.category.eq(cat)][metric].to_numpy() for cat in cats]
            ax.bar(center, [np.mean(v) for v in populations], width=0.22,
                   color=COLORS[group], alpha=0.75, label=group.capitalize())
            for cx, data in zip(center, populations):
                ax.scatter(cx + np.linspace(-0.045, 0.045, len(data)), data, color="#27323b",
                           s=9, zorder=4, edgecolors="white", linewidths=0.25)
        ax.set_title(outcome.capitalize(), weight="bold")
        ax.set_xticks(x, [cat.capitalize() for cat in cats], rotation=22, ha="right")
        ax.set_ylim(0, 106)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Episodes (%)")
    axes[2].legend(loc="upper right", fontsize=8, framealpha=.90)
    save(fig, output, f"{algo}_comparison")


def learning_curves(root: Path, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.1), layout="constrained", sharey=True)
    for algo, ax in zip(ALGORITHMS, axes):
        runs = read_json(root / "manifests" / f"{algo}_confirmation.json")["runs"]
        for group in GROUPS[:2]:
            runs_in_group = [run for tag, run in runs.items() if tag.startswith(group)]
            grid = np.arange(2048, runs_in_group[0]["executed_steps"] + 1, 1024)
            curves = []
            for run in runs_in_group:
                rows = pd.read_csv(root / "logs/confirmation" / algo / run["run_id"] / "training_episodes.csv")
                curves.append([
                    rows.loc[rows.timestep.between(step - 2047, step), "episode_return"].mean()
                    for step in grid
                ])
            mean = np.nanmean(curves, axis=0)
            sd = np.nanstd(curves, axis=0, ddof=1)
            ax.plot(grid, mean, color=COLORS[group], label=group.capitalize())
            ax.fill_between(grid, mean - sd, mean + sd, color=COLORS[group], alpha=0.13)
        ax.axhline(0, lw=.7, color="#c0c6c9")
        ax.set_title(algo.upper(), loc="left", weight="bold")
        ax.set_xlabel("Training transitions")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean training return")
    axes[1].legend(fontsize=8, loc="lower right")
    save(fig, output, "training_curves")


def narrow(root: Path, output: Path):
    cfg = load_env_config(checked_config(root, "environment.yaml"))
    sc = next(sc for sc in load_scenarios(root, "test") if sc.category == "narrow")
    reference = valid_reference(sc, cfg)
    test_dir = root / "evaluation/test"
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.9), layout="constrained", sharey=True)
    for ax, (algo, group) in zip(axes, [(a, g) for a in ALGORITHMS for g in GROUPS[:2]]):
        label = f"{algo}-{group}-seed201"
        rows = pd.read_csv(test_dir / "episodes" / f"{label}.csv").set_index("scenario_id")
        tr = read_json(test_dir / "trajectories" / f"{label}.json")[sc.scenario_id]
        result = rows.loc[sc.scenario_id, "outcome"]
        draw_scenario(ax, sc, tr, cfg, f"{algo.upper()} {group}\n{result.capitalize()}", result, reference)
        ax.set_xticks([0, 3, 6])
        ax.set_yticks([0, 3, 6])
        if ax is not axes[0]:
            ax.set_ylabel("")
    save(fig, output, "narrow_trajectories")


def footprint(frame, output: Path):
    frame = frame[frame.category.eq("all")]
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 2.9), sharey=True, layout="constrained")
    for algo, ax in zip((*ALGORITHMS, "heuristic"), axes):
        item = frame[frame.algorithm.eq(algo)]
        x = np.arange(2)
        for index, status in enumerate(OUTCOMES):
            points = [
                100 * item.loc[item.radius_m.eq(r), f"{status}_rate"].to_numpy()
                for r in (0.18, 0.25)
            ]
            center = x + (index - 1) * .24
            ax.bar(center, [np.mean(v) for v in points], width=.22,
                   color=COLORS[status], label=status.capitalize())
            for cx, data in zip(center, points):
                ax.scatter(cx + np.linspace(-.04, .04, len(data)), data, s=9, color="#27323b",
                           edgecolors="white", linewidths=.25)
        ax.set_xticks(x, ["0.18 m", "0.25 m"])
        ax.set_title(algo.upper() if algo != "heuristic" else "Heuristic", weight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Paired episodes (%)")
    axes[2].legend(fontsize=7, loc="upper right")
    save(fig, output, "footprint_robustness")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--screen-only", action="store_true")
    args = cli.parse_args()
    root = args.output
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#303a42", "text.color": "#1f2a30"})
    optimization(root, output)
    if not args.screen_only:
        frame = pd.read_csv(root / "comparisons/final_metrics.csv")
        for algo in ALGORITHMS:
            outcomes(frame, output, algo)
        learning_curves(root, output)
        narrow(root, output)
        footprint(pd.read_csv(root / "comparisons/footprint_metrics.csv"), output)
    print(f"Saved figures in {output}")


if __name__ == "__main__":
    main()
