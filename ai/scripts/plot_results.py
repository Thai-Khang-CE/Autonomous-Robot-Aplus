"""Produce report figures from REAL logs only (training CSVs, evaluations.npz, test episodes).

Figures are written to results/figures/ (PNG + PDF) and copied to reports/week09/figures/.
A figure whose input data is missing is skipped with a printed notice; nothing is synthesised.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from robot_aplus_rl.config import PROJECT_ROOT, load_env_config  # noqa: E402
from robot_aplus_rl.scenarios import generate_scenario  # noqa: E402
from robot_aplus_rl.visualization import draw_scenario  # noqa: E402

RES = PROJECT_ROOT / "results"
FIG = RES / "figures"
REPORT_FIG = PROJECT_ROOT / "reports" / "week09" / "figures"
EVAL = RES / "evaluation"
# Validated reference categorical palette (slot order), random baseline in neutral grey.
COL = {"ppo": "#2a78d6", "sac": "#eb6834", "heuristic": "#1baf7a", "random": "#8c8b86"}
NAME = {"ppo": "PPO", "sac": "SAC", "heuristic": "Heuristic", "random": "Random"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 110, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
        shutil.copy(FIG / f"{name}.{ext}", REPORT_FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def run_dirs(algo: str, label: str) -> list[Path]:
    """Completed runs, plus runs still in progress (their logs are real but partial)."""
    return sorted(p for p in (RES / algo).glob(f"{label}_seed*") if (p / "run_info.json").exists()
                  and json.loads((p / "run_info.json").read_text())["status"] in ("completed", "running")
                  and (p / "training_episodes.csv").exists())


def max_step(runs) -> int:
    return int(min(pd.read_csv(r / "training_episodes.csv")["timestep"].max() for r in runs))


def rolling_curve(run: Path, column: str, window: int = 100, grid=None):
    df = pd.read_csv(run / "training_episodes.csv")
    y = df[column].rolling(window, min_periods=max(10, window // 5)).mean()
    if grid is None:
        return df["timestep"].to_numpy(), y.to_numpy()
    return grid, np.interp(grid, df["timestep"], y.bfill(), left=np.nan)


def plot_training(label: str, column: str, ylabel: str, fname: str, total: int) -> None:
    grid = np.linspace(2000, total, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    have = False
    for ax, algo in zip(axes, ("ppo", "sac")):
        runs = run_dirs(algo, label)
        curves = []
        for r in runs:
            _, y = rolling_curve(r, column, grid=grid)
            curves.append(y)
            ax.plot(grid, y, color=COL[algo], lw=0.8, alpha=0.35)
        if curves:
            have = True
            m = np.nanmean(curves, axis=0)
            ax.plot(grid, m, color=COL[algo], lw=2, label=f"{NAME[algo]} mean of {len(runs)} seed(s)")
            ax.legend(loc="lower right", frameon=False)
        ax.set_title(f"{NAME[algo]} ({label}, {len(runs)} seed{'s' if len(runs) != 1 else ''})")
        ax.set_xlabel("environment steps")
    axes[0].set_ylabel(ylabel)
    if have:
        save(fig, fname)
    else:
        plt.close(fig)
        print("skip", fname, "(no completed runs)")


def plot_single_algo(algo: str, label: str, total: int) -> None:
    """Return and training success (rolling 100 episodes) for one algorithm, every seed shown."""
    runs = run_dirs(algo, label)
    if not runs:
        print("skip", f"{algo}_reward_curve_{label}", "(no completed runs)")
        return
    total = min(total, max_step(runs))
    grid = np.linspace(2000, total, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    shades = [COL[algo], "#5f5e5a", "#a3a29c"]
    for r, c in zip(runs, shades):
        for ax, col in zip(axes, ("episode_return", "is_success")):
            _, y = rolling_curve(r, col, grid=grid)
            ax.plot(grid, y, color=c, lw=1.4, label=r.name.replace("_", " "))
    axes[0].set_ylabel("episode return (rolling 100)")
    axes[1].set_ylabel("training success (rolling 100)")
    axes[1].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xlabel("environment steps")
    axes[1].legend(frameon=False, loc="lower right")
    running = any(json.loads((r / "run_info.json").read_text())["status"] == "running" for r in runs)
    fig.suptitle(f"{NAME[algo]} training curves ({label}, {len(runs)} seed(s)"
                 + (f"; IN PROGRESS, logged up to {total:,} steps" if running else "")
                 + "; stochastic training policy)", fontsize=9)
    save(fig, f"{algo}_reward_curve_{label}")


def plot_training_comparison(label: str, total: int) -> None:
    if not (run_dirs("ppo", label) or run_dirs("sac", label)):
        print("skip", f"training_comparison_{label}", "(no completed runs)")
        return
    runs_all = run_dirs("ppo", label) + run_dirs("sac", label)
    total = min(total, max_step(runs_all))
    grid = np.linspace(2000, total, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for algo in ("ppo", "sac"):
        runs = run_dirs(algo, label)
        if not runs:
            continue
        for ax, col, yl in zip(axes, ("episode_return", "is_success"),
                               ("episode return (rolling 100)", "training success (rolling 100)")):
            ys = np.array([rolling_curve(r, col, grid=grid)[1] for r in runs])
            m = np.nanmean(ys, axis=0)
            ax.plot(grid, m, color=COL[algo], lw=2, label=f"{NAME[algo]} (n={len(runs)})")
            if len(runs) > 1:
                ax.fill_between(grid, np.nanmin(ys, 0), np.nanmax(ys, 0), color=COL[algo], alpha=0.15,
                                lw=0)
            ax.set_ylabel(yl)
            ax.set_xlabel("environment steps")
    axes[0].legend(frameon=False)
    axes[1].set_ylim(-0.02, 1.02)
    fig.suptitle(f"Training comparison, {label} runs, first {total:,} steps (line = seed mean, band = min-max)",
                 fontsize=10)
    save(fig, f"training_comparison_{label}")


def plot_validation(label: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    have = False
    for algo in ("ppo", "sac"):
        runs = run_dirs(algo, label)
        series = []
        for r in runs:
            f = r / "evaluations.npz"
            if f.exists():
                d = np.load(f)
                series.append((d["timesteps"], d["successes"].mean(axis=1)))
        if not series:
            continue
        have = True
        n = min(len(t) for t, _ in series)
        t = series[0][0][:n]
        ys = np.array([s[:n] for _, s in series])
        for y in ys:
            ax.plot(t, y, color=COL[algo], lw=0.8, alpha=0.35)
        ax.plot(t, ys.mean(0), color=COL[algo], lw=2, marker="o", ms=3,
                label=f"{NAME[algo]} (n={len(series)})")
    if not have:
        plt.close(fig)
        print("skip", f"validation_success_{label}", "(no completed runs)")
        return
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("validation success rate")
    ax.set_title("Deterministic validation (30 fixed scenarios)")
    ax.legend(frameon=False)
    save(fig, f"validation_success_{label}")


def plot_eval_bars(groups: dict[str, str]) -> None:
    f = EVAL / "summary_by_label.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    s["group"] = s["label"].str.replace(r"_seed\d+", "", regex=True)
    cats = ["open", "indoor", "narrow", "all"]
    for metric, title in (("success_rate", "Success rate"), ("collision_rate", "Collision rate")):
        fig, ax = plt.subplots(figsize=(6.4, 2.9))
        keys = [k for k in groups if k in set(s.group)]
        width = 0.8 / max(len(keys), 1)
        for i, key in enumerate(keys):
            algo = groups[key]
            d = s[s.group == key]
            means = [d[d.category == c][metric].mean() for c in cats]
            lo = [m - d[d.category == c][metric].min() for m, c in zip(means, cats)]
            hi = [d[d.category == c][metric].max() - m for m, c in zip(means, cats)]
            x = np.arange(len(cats)) + (i - (len(keys) - 1) / 2) * width
            n = d.train_seed.nunique() if d.train_seed.notna().any() else 1
            ax.bar(x, means, width * 0.92, color=COL[algo], label=f"{NAME[algo]}" + (" 100k ckpt" if "interim" in key else "") +
                   (f" (mean of {n} seeds, bar = min-max)" if n > 1 else ""),
                   yerr=[lo, hi] if n > 1 else None, error_kw={"lw": 0.8, "ecolor": INK})
        ax.set_xticks(np.arange(len(cats)), [c if c != "all" else "all (150)" for c in cats])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title.lower())
        ax.set_title(f"{title} on held-out test scenarios (50 per category, deterministic)")
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save(fig, f"eval_{metric}")


def load_traj(label: str) -> dict | None:
    f = EVAL / "trajectories" / f"{label}.json"
    return json.loads(f.read_text()) if f.exists() else None


def plot_trajectory_grid(labels: dict[str, str], scenario_index: int = 0) -> None:
    cfg = load_env_config()
    eps = {k: pd.read_csv(EVAL / "episodes" / f"{k}.csv") for k in labels
           if (EVAL / "episodes" / f"{k}.csv").exists()}
    if not eps:
        return
    cats = ["open", "indoor", "narrow"]
    fig, axes = plt.subplots(len(cats), len(eps), figsize=(2.6 * len(eps), 2.6 * len(cats)), squeeze=False)
    for r, cat in enumerate(cats):
        seed = 2_000_000 + scenario_index
        sc = generate_scenario(cat, seed, cfg)
        for c, (lab, df) in enumerate(eps.items()):
            row = df[df.scenario_id == sc.scenario_id].iloc[0]
            draw_scenario(axes[r, c], sc, load_traj(lab)[sc.scenario_id], cfg,
                          title=f"{labels[lab]} | {sc.scenario_id}\n{row.outcome}, "
                                f"{int(row.episode_length)} steps", outcome=row.outcome)
            axes[r, c].tick_params(labelsize=6)
            axes[r, c].set_xlabel("")
            axes[r, c].set_ylabel("")
    fig.tight_layout()
    save(fig, "trajectories_grid")


def plot_examples(label: str, display: str, fname_prefix: str) -> None:
    """First success and first failure in fixed test order (selection rule stated in title)."""
    f = EVAL / "episodes" / f"{label}.csv"
    if not f.exists():
        return
    cfg = load_env_config()
    df = pd.read_csv(f)
    traj = load_traj(label)
    picks = []
    for cat in ("narrow", "indoor", "open"):
        d = df[(df.category == cat) & (df.outcome == "success")]
        if len(d):
            picks.append(("success", d.iloc[0]))
            break
    fails = df[df.outcome != "success"]
    if len(fails):
        picks.append(("failure", fails.iloc[0]))
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (kind, row) in zip(axes, picks):
        sc = generate_scenario(row.category, int(row.scenario_seed), cfg)
        draw_scenario(ax, sc, traj[sc.scenario_id], cfg, outcome=row.outcome,
                      title=f"{display}: first {kind} in test order\n{sc.scenario_id} -> "
                            f"{row.outcome} ({int(row.episode_length)} steps, "
                            f"path {row.path_length:.2f} m)")
    axes[0].legend(loc="best", fontsize=6, framealpha=0.85)
    fig.tight_layout()
    save(fig, f"{fname_prefix}_examples")


def main() -> None:
    for label, total in (("controlled", 50000), ("extended", 200000)):
        plot_training(label, "episode_return", "episode return (rolling 100)",
                      f"training_return_{label}", total)
        plot_training_comparison(label, total)
        plot_single_algo("ppo", label, total)
        plot_single_algo("sac", label, total)
        plot_validation(label)
    plot_eval_bars({"random": "random", "heuristic": "heuristic",
                    "ppo_extended_final": "ppo", "sac_extended_final": "sac"})
    plot_trajectory_grid({"heuristic": "Heuristic", "ppo_extended_seed0_final": "PPO seed 0",
                          "sac_extended_seed0_final": "SAC seed 0"})
    plot_examples("ppo_extended_seed0_final", "PPO (seed 0)", "ppo")
    plot_examples("sac_extended_seed0_final", "SAC (seed 0)", "sac")
    plot_examples("random", "Random", "random")


if __name__ == "__main__":
    main()
