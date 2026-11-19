"""Evaluate locked Week 10 policies and paired footprint sensitivity."""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.utils import check_for_correct_spaces

from robot_aplus_rl import Navigation2DEnv
from robot_aplus_rl.baselines import GoalSeekingHeuristic, RandomPolicy
from robot_aplus_rl.config import load_env_config
from robot_aplus_rl.evaluation import evaluate_on_scenarios, save_trajectories, summarise
from robot_aplus_rl.geometry import segment_is_free
from robot_aplus_rl.planning import path_length, reference_path
from train import ALGOS
from week10 import checked_config, digest, load_scenarios, read_json, write_json

METRICS = ["success_rate", "collision_rate", "timeout_rate", "mean_return", "mean_length",
           "mean_path_length", "mean_time_to_goal_success", "spl_approx"]


def valid_reference(scenario, cfg):
    path = reference_path(scenario.start[:2], scenario.goal, scenario.obstacles,
                          cfg.world_size, cfg.robot_radius, cfg.planning_margin,
                          cfg.planning_resolution)
    if path and all(segment_is_free(a, b, cfg.robot_radius, scenario.obstacles, cfg.world_size)
                    for a, b in zip(path, path[1:])):
        return path
    return None


def reference_lengths(scenarios, cfg):
    lengths = {}
    for scenario in scenarios:
        reference = valid_reference(scenario, cfg)
        if reference is not None:
            lengths[scenario.scenario_id] = path_length(reference)
    return lengths


def model_from(path: Path, algo: str, cfg):
    model = ALGOS[algo].load(path, device="cpu")
    check_for_correct_spaces(Navigation2DEnv(cfg), model.observation_space, model.action_space)
    return model


def evaluate_item(output: Path, suite: str, model, scenarios, cfg, label: str,
                  algo: str, group: str, seed, steps, wall_time, references, checkpoint):
    rows, traj = evaluate_on_scenarios(model, scenarios, cfg, label, algo, seed, algo != "random", references)
    ep = output / "evaluation" / suite / "episodes" / f"{label}.csv"
    tr = ep.parent.parent / "trajectories" / f"{label}.json"
    ep.parent.mkdir(parents=True, exist_ok=True)
    tr.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ep, index=False)
    save_trajectories(tr, traj)
    stats = []
    for category, selection in [("all", rows), *rows.groupby("category")]:
        stats.append({
            "policy": label, "algorithm": algo, "group": group, "train_seed": seed,
            "steps": steps, "training_wall_s": wall_time, "category": category,
            "radius_m": cfg.robot_radius, "checkpoint_sha256": checkpoint,
            "spl_reference_episodes": int(selection.reference_path_length.notna().sum()),
            **summarise(selection),
        })
    print(f"{suite} {label}: {len(rows)} episodes; success={rows.success.mean():.3f}, "
          f"collision={rows.collision.mean():.3f}, timeout={rows.timeout.mean():.3f}", flush=True)
    return stats, rows, traj


def write_comparison(path: Path, stats) -> pd.DataFrame:
    frame = pd.DataFrame(stats)
    frame.to_csv(path, index=False)
    aggregate = (frame.groupby(["algorithm", "group", "category", "radius_m"], dropna=False)
                 .agg(policies=("policy", "size"), training_seeds=("train_seed", "nunique"),
                      **{f"{m}_{suffix}": (m, fn) for m in METRICS
                         for suffix, fn in (("mean", "mean"), ("sd", "std"))}))
    aggregate.to_csv(path.with_name(f"{path.stem}_aggregate.csv"))
    return frame


def narrow_diagnostics(rows, trajectories, by_id, cfg):
    for row in rows[rows.category.eq("narrow")].itertuples():
        if row.outcome == "success":
            continue
        sc = by_id[row.scenario_id]
        trajectory = np.asarray(trajectories[row.scenario_id])
        axis = int(sc.meta["horizontal"])
        plane = sc.meta["wall_pos"]
        posts = sc.obstacles[np.isclose(sc.obstacles[:, axis], plane)]
        gap = np.argsort(posts[:, 1-axis])
        posts = posts[gap]
        across = posts[:, 1-axis]
        doorway = int(np.argmax(np.diff(across)))
        final_xy = trajectory[-1, :2]
        endpoint = float(final_xy[1-axis])
        if row.outcome == "collision":
            index = int(np.argmin(np.linalg.norm(sc.obstacles[:, :2] - final_xy, axis=1) - sc.obstacles[:, 2]))
            hit = sc.obstacles[index]
            if min(*final_xy, *(cfg.world_size - final_xy)) <= cfg.robot_radius + 1e-5:
                reason = "world boundary contact"
            elif any(np.allclose(hit, post) for post in posts[doorway:doorway+2]):
                reason = "doorway post contact"
            elif abs(hit[axis] - plane) <= 1e-5:
                reason = "partition contact"
            else:
                reason = "clutter contact"
        else:
            tail = trajectory[-51:]
            tail_dist = float(np.linalg.norm(np.diff(tail[:, :2], axis=0), axis=1).sum())
            tail_rotation = float(np.abs(np.diff(np.unwrap(tail[:, 2]))).sum())
            if tail_dist <= 0.2 and tail_rotation >= math.pi:
                reason = "low translation with rotation"
            elif tail_dist <= 0.2 and abs(final_xy[axis] - plane) <= 0.55 and across[doorway] <= endpoint <= across[doorway+1]:
                reason = "stopped near opening"
            elif tail_dist <= 0.2:
                reason = "low translation elsewhere"
            else:
                reason = "moving at timeout"
        yield {"policy": row.policy, "scenario_id": row.scenario_id, "outcome": row.outcome,
               "episode_length": row.episode_length, "final_distance": row.final_distance, "reason": reason}


def inherited_runs(handoff: Path, algo: str):
    for seed in (0, 1, 2):
        folder = handoff / "results" / algo / f"extended_seed{seed}"
        info = read_json(folder / "run_info.json")
        path = folder / "final_model.zip"
        if info["status"] != "completed":
            raise ValueError(f"Incomplete historical run: {folder.name}")
        yield seed, path, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--handoff", type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(1)
    cfg = load_env_config(checked_config(output, "environment.yaml"))
    locked_runs = {}
    for algo in ALGOS:
        selection = output / "tuning" / f"{algo}_selection.json"
        manifest = read_json(output / "manifests" / f"{algo}_confirmation.json")
        if digest(selection) != manifest["selection_sha256"]:
            raise ValueError(f"{algo} selection changed after confirmation")
        locked_runs[algo] = manifest["runs"]
        for run in locked_runs[algo].values():
            if digest(output / run["checkpoint"]) != run["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed after confirmation: {run['checkpoint']}")
    scenarios = load_scenarios(output, "test")
    by_id = {sc.scenario_id: sc for sc in scenarios}
    refs = reference_lengths(scenarios, cfg)
    stats, diagnostic = [], []
    for algo in ALGOS:
        for tag, run in locked_runs[algo].items():
            mode = tag.split("-")[0]
            model = model_from(output / run["checkpoint"], algo, cfg)
            summary, rows, traj = evaluate_item(output, "test", model, scenarios, cfg,
                f"{algo}-{tag}", algo, mode, run["seed"], run["executed_steps"],
                run["wall_time_s"], refs, run["checkpoint_sha256"])
            stats.extend(summary)
            diagnostic.extend(narrow_diagnostics(rows, traj, by_id, cfg))
        for seed, path, run in inherited_runs(args.handoff, algo):
            model = model_from(path, algo, cfg)
            summary, rows, traj = evaluate_item(output, "test", model, scenarios, cfg,
                f"{algo}-week9-seed{seed}", algo, "historical", seed, run["executed_timesteps"],
                run["wall_time_s"], refs, digest(path))
            stats.extend(summary)
            diagnostic.extend(narrow_diagnostics(rows, traj, by_id, cfg))
    for label, policy in (("random", RandomPolicy(0)),
                          ("heuristic", GoalSeekingHeuristic(cfg.num_rays, cfg.max_range))):
        summary, rows, traj = evaluate_item(output, "test", policy, scenarios, cfg,
            label, label, "reference", None, 0, None, refs, None)
        stats.extend(summary)
        diagnostic.extend(narrow_diagnostics(rows, traj, by_id, cfg))
    comparisons = output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    frame = write_comparison(comparisons / "final_metrics.csv", stats)
    pd.DataFrame(diagnostic).to_csv(comparisons / "narrow_failures.csv", index=False)
    keys = ["algorithm", "train_seed", "category"]
    groups = frame[frame.group.isin(["baseline", "tuned"])].set_index(keys + ["group"])
    delta = groups.xs("tuned", level="group")[METRICS] - groups.xs("baseline", level="group")[METRICS]
    delta.to_csv(comparisons / "paired_tuned_minus_baseline.csv")

    larger = replace(cfg, robot_radius=0.25)
    larger_refs = reference_lengths(scenarios, larger)
    paired = [sc for sc in scenarios if sc.scenario_id in larger_refs]
    write_json(output / "evaluation/footprint/paired_scenario_ids.json", [sc.scenario_id for sc in paired])
    footprint = []
    for current, ref in ((cfg, refs), (larger, larger_refs)):
        label_radius = f"r{round(current.robot_radius * 100)}"
        for algo in ALGOS:
            for seed, path, run in inherited_runs(args.handoff, algo):
                model = model_from(path, algo, current)
                summary, _, _ = evaluate_item(output, "footprint", model, paired, current,
                    f"{algo}-week9-seed{seed}-{label_radius}", algo, "historical",
                    seed, run["executed_timesteps"], run["wall_time_s"], ref, digest(path))
                footprint.extend(summary)
        heuristic = GoalSeekingHeuristic(current.num_rays, current.max_range)
        summary, _, _ = evaluate_item(output, "footprint", heuristic, paired, current,
            f"heuristic-{label_radius}", "heuristic", "reference", None, 0, None, ref, None)
        footprint.extend(summary)
    write_comparison(comparisons / "footprint_metrics.csv", footprint)


if __name__ == "__main__":
    main()
