"""Evaluate a policy on the fixed held-out TEST scenarios (seeds 2_000_000+).

Examples:
    python scripts/evaluate.py --policy random --seed 0
    python scripts/evaluate.py --policy heuristic
    python scripts/evaluate.py --policy ppo --model results/ppo/extended_seed0/final_model.zip \
        --label ppo_extended_seed0_final --train-seed 0

Writes results/evaluation/episodes/<label>.csv (one row per episode) and
results/evaluation/trajectories/<label>.json. Every policy sees exactly the same scenario list.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed

from robot_aplus_rl.baselines import GoalSeekingHeuristic, RandomPolicy
from robot_aplus_rl.config import DEFAULT_ENV_CONFIG, PROJECT_ROOT, load_env_config
from robot_aplus_rl.evaluation import (
    evaluate_on_scenarios,
    reference_lengths,
    save_trajectories,
    summarise,
)
from robot_aplus_rl.scenarios import scenario_set

EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"


def load_test_set(cfg, per_category: int, name: str = "test"):
    scenarios = scenario_set(name, per_category, cfg)
    ref_file = EVAL_DIR / f"{name}_set_{per_category}_per_category.json"
    if ref_file.exists():
        refs = {d["scenario_id"]: d["reference_path_length"]
                for d in json.loads(ref_file.read_text(encoding="utf-8"))}
    else:
        refs = reference_lengths(scenarios, cfg)
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(json.dumps(
            [{**s.to_dict(), "reference_path_length": refs[s.scenario_id]} for s in scenarios],
            indent=1), encoding="utf-8")
    return scenarios, refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", choices=["random", "heuristic", "ppo", "sac"], required=True)
    ap.add_argument("--model", help="checkpoint .zip for ppo/sac")
    ap.add_argument("--label", help="output name (default: policy name)")
    ap.add_argument("--train-seed", type=int)
    ap.add_argument("--seed", type=int, default=0, help="seed for random/stochastic actions")
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--scenario-set", default="test", choices=["test", "validation"])
    ap.add_argument("--stochastic", action="store_true", help="sample actions (default deterministic)")
    ap.add_argument("--env-config", default=str(DEFAULT_ENV_CONFIG))
    ap.add_argument("--robot-radius", type=float,
                    help="robustness test: evaluate with a different collision radius on the SAME "
                         "scenarios (generated with the base config); output goes to "
                         "results/evaluation/robustness/ and SPL is not computed")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = load_env_config(args.env_config)
    set_random_seed(args.seed)
    if args.policy == "random":
        policy = RandomPolicy(seed=args.seed)
    elif args.policy == "heuristic":
        policy = GoalSeekingHeuristic(cfg.num_rays, cfg.max_range)
    else:
        if not args.model:
            ap.error("--model is required for ppo/sac")
        policy = {"ppo": PPO, "sac": SAC}[args.policy].load(args.model, device="cpu")
    label = args.label or args.policy
    deterministic = not args.stochastic

    scenarios, refs = load_test_set(cfg, args.per_category, args.scenario_set)
    out_dir = EVAL_DIR
    if args.robot_radius is not None:
        cfg = dataclasses.replace(cfg, robot_radius=args.robot_radius)
        refs = {}
        out_dir = EVAL_DIR / "robustness"
    t0 = time.time()
    df, trajs = evaluate_on_scenarios(policy, scenarios, cfg, label, args.policy, args.train_seed,
                                      deterministic, refs)
    df["model_path"] = args.model or ""
    df["scenario_set"] = args.scenario_set
    df["robot_radius"] = cfg.robot_radius
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episodes" / f"{label}.csv", index=False)
    save_trajectories(out_dir / "trajectories" / f"{label}.json", trajs)
    summary = {"label": label, "command": " ".join([Path(sys.executable).name] + sys.argv),
               "deterministic": deterministic, "wall_time_s": round(time.time() - t0, 1),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in summarise(df).items()}}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
