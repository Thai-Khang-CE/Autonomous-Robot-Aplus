"""Train PPO or SAC on Navigation2DEnv with full experiment bookkeeping.

Example:
    python scripts/train.py --algo ppo --timesteps 50000 --seed 0 --label controlled
    python scripts/train.py --algo sac --timesteps 3000 --seed 0 --label smoke \
        --set learning_starts=500

Outputs (in --output-dir, default results/<algo>/<label>_seed<seed>):
    run_info.json          command, versions, configs, seed, executed timesteps, checks, status
    training_episodes.csv  one row per finished training episode (global timestep, return, outcome)
    monitor/               SB3 Monitor CSV files (one per environment copy)
    logs/progress.csv      SB3 training diagnostics (losses, entropy, ...); logs/ has TensorBoard
    evaluations.npz        periodic validation (held-out VALIDATION scenarios, not the test set)
    best_model.zip         checkpoint with the best validation mean reward
    final_model.zip        checkpoint at the end of training
    checkpoints/           periodic checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from robot_aplus_rl import Navigation2DEnv
from robot_aplus_rl.config import DEFAULT_ENV_CONFIG, PROJECT_ROOT, load_env_config, load_yaml
from robot_aplus_rl.scenarios import scenario_set

ALGOS = {"ppo": PPO, "sac": SAC}
ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}
VALIDATION_PER_CATEGORY = 10  # 30 validation scenarios, seeds 1_000_000+


class EpisodeLogCallback(BaseCallback):
    """Write every finished training episode with the global timestep at which it ended."""

    FIELDS = ["timestep", "env_index", "episode_return", "episode_length", "outcome",
              "is_success", "category", "scenario_id", "wall_time_s"]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._t0 = time.time()

    def _on_training_start(self) -> None:
        new = not self.path.exists()
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        if new:
            self._writer.writeheader()

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            if "episode" in info:  # added by Monitor at episode end
                self._writer.writerow({
                    "timestep": self.num_timesteps,
                    "env_index": i,
                    "episode_return": round(float(info["episode"]["r"]), 6),
                    "episode_length": int(info["episode"]["l"]),
                    "outcome": info.get("outcome"),
                    "is_success": int(bool(info.get("is_success", False))),
                    "category": info.get("category"),
                    "scenario_id": info.get("scenario_id"),
                    "wall_time_s": round(time.time() - self._t0, 2),
                })
        return True

    def _on_training_end(self) -> None:
        self._fh.close()


def parse_override(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def versions() -> dict:
    pk = ["numpy", "torch", "gymnasium", "stable-baselines3", "matplotlib", "pandas"]
    return {"python": sys.version.split()[0], **{p: metadata.version(p) for p in pk}}


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu().clone() for p in model.policy.parameters()])


def reload_check(model, path: Path, algo_cls, obs_batch: np.ndarray) -> dict:
    loaded = algo_cls.load(path, device="cpu")
    a_mem, _ = model.predict(obs_batch, deterministic=True)
    a_disk, _ = loaded.predict(obs_batch, deterministic=True)
    diff = float(np.abs(a_mem - a_disk).max())
    return {"passed": diff == 0.0, "max_abs_action_diff": diff, "n_obs": int(len(obs_batch)),
            "loaded_num_timesteps": int(loaded.num_timesteps)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--algo", choices=sorted(ALGOS), required=True)
    ap.add_argument("--config", help="algorithm YAML (default configs/<algo>.yaml)")
    ap.add_argument("--env-config", default=str(DEFAULT_ENV_CONFIG))
    ap.add_argument("--timesteps", type=int, help="override total_timesteps from the config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run", help="smoke | controlled | extended | ...")
    ap.add_argument("--output-dir")
    ap.add_argument("--resume", help="path to a .zip checkpoint to continue training from")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--eval-freq", type=int, default=10000, help="validation every N env steps")
    ap.add_argument("--checkpoint-freq", type=int, default=50000)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override algorithm hyperparameters")
    args = ap.parse_args()

    algo_cfg_path = Path(args.config or PROJECT_ROOT / "configs" / f"{args.algo}.yaml")
    algo_cfg = load_yaml(algo_cfg_path)
    env_cfg = load_env_config(args.env_config)
    hp = dict(algo_cfg.get("hyperparameters", {}))
    overrides = parse_override(args.set)
    hp.update(overrides)
    pk = dict(algo_cfg.get("policy_kwargs", {}))
    if "activation_fn" in pk:
        pk["activation_fn"] = ACTIVATIONS[pk["activation_fn"]]
    total = int(args.timesteps or algo_cfg["total_timesteps"])
    n_envs = int(algo_cfg.get("n_envs", 1))

    out = Path(args.output_dir or PROJECT_ROOT / "results" / args.algo / f"{args.label}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed)

    info: dict = {
        "run_name": out.name,
        "algorithm": args.algo,
        "label": args.label,
        "status": "running",
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "requested_timesteps": total,
        "n_envs": n_envs,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "algo_config_file": str(algo_cfg_path.relative_to(PROJECT_ROOT)),
        "algo_config": algo_cfg,
        "hyperparameter_overrides": overrides,
        "env_config": env_cfg.to_dict(),
        "validation_set": f"validation, {VALIDATION_PER_CATEGORY} per category",
        "versions": versions(),
        "resume_from": args.resume,
    }
    info_path = out / "run_info.json"
    info_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    t0 = time.time()
    try:
        venv = make_vec_env(
            lambda: Navigation2DEnv(env_cfg), n_envs=n_envs, seed=args.seed,
            vec_env_cls=DummyVecEnv, monitor_dir=str(out / "monitor"),
            monitor_kwargs={"info_keywords": ("is_success", "outcome")},
        )
        val_pool = scenario_set("validation", VALIDATION_PER_CATEGORY, env_cfg)
        eval_env = DummyVecEnv([lambda: Monitor(Navigation2DEnv(env_cfg, scenario_pool=val_pool))])
        algo_cls = ALGOS[args.algo]
        if args.resume:
            model = algo_cls.load(args.resume, env=venv, device=args.device, **hp)
        else:
            model = algo_cls(algo_cfg.get("policy", "MlpPolicy"), venv, seed=args.seed,
                             device=args.device, policy_kwargs=pk, verbose=0, **hp)
        model.set_logger(configure(str(out / "logs"), ["csv", "tensorboard"]))
        params_before = flat_params(model)

        callbacks = [
            EpisodeLogCallback(out / "training_episodes.csv"),
            EvalCallback(eval_env, n_eval_episodes=len(val_pool),
                         eval_freq=max(args.eval_freq // n_envs, 1), log_path=str(out),
                         best_model_save_path=str(out), deterministic=True, verbose=0),
            CheckpointCallback(save_freq=max(args.checkpoint_freq // n_envs, 1),
                               save_path=str(out / "checkpoints"), name_prefix=args.algo),
        ]
        model.learn(total_timesteps=total, callback=callbacks, reset_num_timesteps=not args.resume,
                    progress_bar=False)
        final_path = out / "final_model.zip"
        model.save(final_path)

        # Evidence that learning actually updated the network and the checkpoint is usable.
        delta = flat_params(model) - params_before
        rng = np.random.default_rng(0)
        probe = Navigation2DEnv(env_cfg)
        obs_batch = []
        for s in val_pool[:16]:
            o, _ = probe.reset(options={"scenario": s})
            obs_batch.append(o)
            for _ in range(3):
                o, *_ = probe.step(rng.uniform(-1, 1, 2))
                obs_batch.append(o)
        info.update(
            status="completed",
            executed_timesteps=int(model.num_timesteps),
            n_gradient_updates=int(model._n_updates),
            policy_param_l2_change=float(delta.norm()),
            policy_param_changed=bool(delta.abs().max() > 0),
            reload_check=reload_check(model, final_path, algo_cls, np.asarray(obs_batch)),
            best_model_exists=(out / "best_model.zip").exists(),
        )
    except Exception as exc:  # keep the real error as evidence
        info.update(status="failed", error=repr(exc))
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        info["wall_time_s"] = round(time.time() - t0, 1)
        info["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        info_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("run_name", "status", "executed_timesteps",
                                           "n_gradient_updates", "policy_param_l2_change",
                                           "reload_check", "wall_time_s")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
