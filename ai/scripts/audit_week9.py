"""Recalculate Week 9 results and replay fixed historical trajectories from the handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robot_aplus_rl.config import load_env_config
from robot_aplus_rl.evaluation import evaluate_on_scenarios, save_trajectories
from robot_aplus_rl.navigation_env import Navigation2DEnv
from robot_aplus_rl.scenarios import scenario_set
from train import ALGOS


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("handoff", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = args.handoff.resolve()
    old_eval = root / "results/evaluation"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_env_config(root / "configs/environment.yaml")
    full_pool = scenario_set("test", 50, cfg)
    probes = [sc for sc in full_pool if sc.seed < 2_000_003]
    env = Navigation2DEnv(cfg)
    inventories, probes_seen, historical = [], [], []
    for folder in (old_eval / "episodes", old_eval / "robustness/episodes"):
        for path in sorted(folder.glob("*.csv")):
            frame = pd.read_csv(path)
            assert len(frame) == frame.scenario_id.nunique() == 150
            assert frame.groupby("category").size().to_dict() == dict.fromkeys(("open", "indoor", "narrow"), 50)
            assert frame[["success", "collision", "timeout"]].sum(axis=1).eq(1).all()
            for category, group in [("all", frame), *frame.groupby("category")]:
                historical.append({
                    "policy": path.stem, "category": category, "episodes": len(group),
                    **{f"{k}_rate": group[k].mean() for k in ("success", "collision", "timeout")},
                    "source": path.relative_to(root).as_posix(), "sha256": sha256(path),
                })
    pd.DataFrame(historical).to_csv(output / "week9_historical_metrics.csv", index=False)
    metrics = pd.DataFrame(historical)
    metrics = metrics[metrics.category.eq("all")]
    groups = {
        "PPO final": r"ppo_extended_seed[0-2]_final",
        "SAC interim": r"sac_interim100k_seed[0-2]_ckpt",
        "SAC final": r"sac_extended_seed[0-2]_final",
    }
    for group, pattern in groups.items():
        data = metrics[metrics.policy.str.fullmatch(pattern)].filter(like="_rate") * 100
        print(group, "mean %", data.mean().round(3).to_dict(),
              "population SD pp", data.std(ddof=0).round(3).to_dict(),
              "sample SD pp", data.std().round(3).to_dict())

    for algo, constructor in ALGOS.items():
        for path in sorted((root / "results" / algo).glob("*/*.zip")) + sorted(
            (root / "results" / algo).glob("*/checkpoints/*.zip")
        ):
            model = constructor.load(path, device="cpu", print_system_info=False)
            assert model.observation_space == env.observation_space
            assert model.action_space == env.action_space
            obs, _ = env.reset(options={"scenario": probes[0]})
            action, _ = model.predict(obs, deterministic=True)
            assert env.action_space.contains(action)
            inventories.append({
                "algorithm": algo, "path": path.relative_to(root).as_posix(),
                "timesteps": model.num_timesteps, "sha256": sha256(path),
            })
            run = path.parent.name
            label = None
            if path.name == "final_model.zip" and run.startswith("extended_"):
                label = f"{algo}_{run}_final"
            if algo == "sac" and path.name == "sac_100000_steps.zip" and path.parent.parent.name.startswith("extended_"):
                run = path.parent.parent.name
                label = f"sac_interim100k_{run.removeprefix('extended_')}_ckpt"
            if label:
                rows, trajectories = evaluate_on_scenarios(model, probes, cfg, label, algo, model.seed, True)
                original = pd.read_csv(old_eval / "episodes" / f"{label}.csv").set_index("scenario_id")
                old_trajectories = json.loads((old_eval / "trajectories" / f"{label}.json").read_text())
                rows.to_csv(output / f"{label}_replayed.csv", index=False)
                save_trajectories(output / f"{label}_replayed.json", trajectories)
                for row in rows.itertuples():
                    actual = np.round(trajectories[row.scenario_id], 4)
                    expected = np.asarray(old_trajectories[row.scenario_id])
                    probes_seen.append({
                        "policy": label, "scenario_id": row.scenario_id, "outcome": row.outcome,
                        "historical_outcome": original.loc[row.scenario_id, "outcome"],
                        "same_length": len(actual) == len(expected),
                        "max_pose_delta_rounded": float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None,
                    })
    pd.DataFrame(inventories).to_csv(output / "week9_checkpoint_inventory.csv", index=False)
    frame = pd.DataFrame(probes_seen)
    frame.to_csv(output / "week9_trajectory_replay.csv", index=False)
    print(f"Loaded {len(inventories)} policy ZIPs with compatible spaces and valid deterministic actions.")
    print(f"Outcomes matched on {(frame.outcome == frame.historical_outcome).sum()}/{len(frame)} replayed episodes.")
    print(f"Largest rounded trajectory difference among equal-length replays: {frame.max_pose_delta_rounded.max():.6g}")
    assert frame.outcome.eq(frame.historical_outcome).all()


if __name__ == "__main__":
    main()
