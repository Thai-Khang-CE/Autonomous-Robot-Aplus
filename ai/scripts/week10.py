"""Freeze, screen and confirm Week 10 PPO/SAC configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import torch
from filelock import FileLock
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from robot_aplus_rl import Navigation2DEnv
from robot_aplus_rl.config import PROJECT_ROOT, load_env_config, load_yaml
from robot_aplus_rl.evaluation import evaluate_on_scenarios, summarise
from robot_aplus_rl.scenarios import CATEGORIES, TRAIN_SEED_LIMIT, Scenario, generate_scenario
from train import ACTIVATIONS, ALGOS, EpisodeLogCallback, flat_params, reload_check, versions

CONFIGS = PROJECT_ROOT / "configs"
PROTOCOL = CONFIGS / "week10/protocol.yaml"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scratch.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(root: Path) -> dict:
    return read_json(root / "manifests/protocol.json")


def checked_config(root: Path, name: str) -> Path:
    path = root / "configs" / name
    if digest(path) != snapshot(root)["files"][name]:
        raise ValueError(f"Frozen file changed: {path}")
    return path


def freeze(root: Path) -> None:
    files = {
        "protocol.yaml": PROTOCOL,
        **{f"{key}.yaml": CONFIGS / f"{key}.yaml" for key in ("environment", *ALGOS)},
    }
    plan = load_yaml(PROTOCOL)
    target = root / "manifests/protocol.json"
    if target.exists():
        for name, source in files.items():
            if digest(checked_config(root, name)) != digest(source):
                raise ValueError(f"Source differs from frozen protocol: {source}")
        for split in ("validation", "test"):
            checked_config(root, f"{split}_scenarios.json")
        return
    validation, test = (plan[k] for k in ("validation", "test"))
    if not TRAIN_SEED_LIMIT <= validation["offset"] < validation["offset"] + validation["per_category"] <= test["offset"]:
        raise ValueError("Training, validation and final-test seed intervals must be disjoint")
    for name, source in files.items():
        dest = root / "configs" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    cfg = load_env_config(files["environment.yaml"])
    hashes = {name: digest(root / "configs" / name) for name in files}
    for split in ("validation", "test"):
        spec = plan[split]
        scenarios = [
            generate_scenario(category, spec["offset"] + i, cfg)
            for category in CATEGORIES for i in range(spec["per_category"])
        ]
        name = f"{split}_scenarios.json"
        path = root / "configs" / name
        write_json(path, [{
            "scenario_id": sc.scenario_id, "category": sc.category, "seed": sc.seed,
            "obstacles": sc.obstacles.tolist(), "start": sc.start, "goal": sc.goal,
            "meta": sc.meta,
        } for sc in scenarios])
        hashes[name] = digest(path)
    write_json(target, {"files": hashes, "versions": {**versions(), "optuna": optuna.__version__}})
    print(f"Frozen validation and final test: {target}", flush=True)


def load_scenarios(root: Path, split: str) -> list[Scenario]:
    return [
        Scenario(**{**r, "obstacles": np.asarray(r["obstacles"], dtype=np.float64).reshape(-1, 3),
                    "start": tuple(r["start"]), "goal": tuple(r["goal"])})
        for r in read_json(checked_config(root, f"{split}_scenarios.json"))
    ]


def navigation_score(success: float, collision: float, weight: float) -> float:
    return success - weight * collision


def validate_budget(algo: str, settings: dict, count: int, steps: int) -> None:
    if algo == "ppo":
        rollout = settings["n_steps"] * count
        if rollout < 2 or settings["batch_size"] < 2 or rollout % settings["batch_size"] or steps % rollout:
            raise ValueError("PPO needs whole rollouts and equal minibatches for a matched transition budget")
    elif count != 1 or not 0 <= settings["learning_starts"] < steps <= settings["buffer_size"]:
        raise ValueError("SAC requires one environment and learning starts below the training budget")


def run_training(root: Path, algo: str, run_id: str, section: str, seed: int, steps: int,
                 overrides: dict) -> tuple[dict, object]:
    cfg = load_env_config(checked_config(root, "environment.yaml"))
    algo_cfg = load_yaml(checked_config(root, f"{algo}.yaml"))
    hp = {**algo_cfg["hyperparameters"], **overrides}
    kwargs = dict(algo_cfg["policy_kwargs"])
    kwargs["activation_fn"] = ACTIVATIONS[kwargs["activation_fn"]]
    count = algo_cfg["n_envs"]
    validate_budget(algo, hp, count, steps)
    folder = root / "logs" / section / algo / run_id
    folder.mkdir(parents=True, exist_ok=False)
    path = root / "checkpoints" / section / algo / run_id / "final_model.zip"
    path.parent.mkdir(parents=True, exist_ok=False)
    envs = make_vec_env(lambda: Navigation2DEnv(cfg), n_envs=count, seed=seed)
    log = configure(str(folder), ["csv"])
    started = time.perf_counter()
    try:
        model = ALGOS[algo](algo_cfg["policy"], envs, seed=seed, device="cpu", policy_kwargs=kwargs, **hp)
        model.set_logger(log)
        before = flat_params(model)
        model.learn(total_timesteps=steps, callback=EpisodeLogCallback(folder / "training_episodes.csv"))
        duration = time.perf_counter() - started
        changed = float((flat_params(model) - before).norm())
        model.save(path)
        # Probe only the training seed namespace. Final-test scenarios are never opened here.
        probe = Navigation2DEnv(cfg)
        obs = np.asarray([probe.reset(seed=seed + offset)[0] for offset in range(4)])
        check = reload_check(model, path, ALGOS[algo], obs)
        if model.num_timesteps != steps or changed <= 0 or not check["passed"]:
            raise RuntimeError(f"Invalid training or checkpoint for {run_id}: {check}")
        info = {
            "algorithm": algo, "run_id": run_id, "seed": seed, "requested_steps": steps,
            "executed_steps": int(model.num_timesteps), "wall_time_s": duration, "n_envs": count,
            "gradient_updates": int(model._n_updates), "parameter_l2_change": changed,
            "hyperparameters": hp, "overrides": overrides, "reload": check,
            "checkpoint": path.relative_to(root).as_posix(), "checkpoint_sha256": digest(path),
        }
        write_json(folder / "run_info.json", info)
        print(f"{section} {algo} {run_id}: {steps} transitions in {duration:.1f} s", flush=True)
        return info, model
    finally:
        envs.close()
        log.close()


def smoke(root: Path, algo: str) -> None:
    overrides = {"n_steps": 128, "batch_size": 128} if algo == "ppo" else {"learning_starts": 128}
    run_training(root, algo, "smoke", "smoke", 99, 1024, overrides)


def screen(root: Path, algo: str, plan: dict) -> None:
    study_dir = root / "tuning"
    study_dir.mkdir(parents=True, exist_ok=True)
    journal = str(study_dir / f"{algo}.journal")
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    candidates = plan["candidates"][algo]
    cfg = load_env_config(checked_config(root, "environment.yaml"))
    validation = load_scenarios(root, "validation")
    study = optuna.create_study(study_name=f"week10-{algo}", storage=storage,
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.RandomSampler(seed=plan["screen_seed"]))
    protocol_hash = digest(root / "manifests/protocol.json")
    saved_hash = study.user_attrs.get("protocol_sha256")
    if saved_hash not in (None, protocol_hash):
        raise ValueError("Study belongs to a different frozen protocol")
    study.set_user_attr("protocol_sha256", protocol_hash)

    def objective(trial: optuna.Trial) -> float:
        name = trial.suggest_categorical("configuration", list(candidates))
        run_id = f"trial-{trial.number:02d}-{name}"
        info, model = run_training(root, algo, run_id, "screen", plan["screen_seed"],
                                   plan["screen_steps"], candidates[name])
        rows, _ = evaluate_on_scenarios(model, validation, cfg, run_id, algo, info["seed"], True)
        ep = root / "evaluation/validation" / algo / f"{run_id}.csv"
        ep.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(ep, index=False)
        stats = summarise(rows)
        for key in ("success_rate", "collision_rate", "timeout_rate", "mean_return"):
            trial.set_user_attr(key, float(stats[key]))
        trial.set_user_attr("run", info)
        return navigation_score(stats["success_rate"], stats["collision_rate"], plan["collision_weight"])

    for trial in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
    complete = {t.params["configuration"] for t in study.get_trials(states=(TrialState.COMPLETE,))}
    queued = len(study.get_trials(states=(TrialState.WAITING,)))
    if queued:
        study.optimize(objective, n_trials=queued)
        complete = {t.params["configuration"] for t in study.get_trials(states=(TrialState.COMPLETE,))}
    pending = [name for name in candidates if name not in complete]
    try:
        for name in pending:
            study.enqueue_trial({"configuration": name})
        if pending:
            study.optimize(objective, n_trials=len(pending))
    finally:
        study.trials_dataframe().to_csv(study_dir / f"{algo}_trials.csv", index=False)
    eligible = [t for t in study.get_trials(states=(TrialState.COMPLETE,)) if t.params["configuration"] != "baseline"]
    best = min(eligible, key=lambda t: (-t.value, t.user_attrs["collision_rate"], t.number))
    write_json(study_dir / f"{algo}_selection.json", {
        "trial": best.number, "configuration": best.params["configuration"], "validation_score": best.value,
        "overrides": candidates[best.params["configuration"]], "protocol_sha256": protocol_hash,
        "baseline_trials": [t.number for t in study.trials if t.state == TrialState.COMPLETE
                            and t.params["configuration"] == "baseline"],
    })


def confirm_seed(root: Path, algo: str, steps: int, seed: int, tuned: dict) -> dict:
    torch.set_num_threads(1)
    done = {}
    for label, overrides in (("baseline", {}), ("tuned", tuned)):
        tag = f"{label}-seed{seed}"
        folder = root / "logs/confirmation" / algo
        previous = sorted(folder.glob(f"{tag}-attempt*/run_info.json"))
        if previous:
            info = read_json(previous[-1])
            if (info["overrides"] != overrides or info["executed_steps"] != steps
                    or digest(root / info["checkpoint"]) != info["checkpoint_sha256"]):
                raise ValueError(f"Existing run does not match confirmation protocol: {tag}")
        else:
            attempt = len(list(folder.glob(f"{tag}-attempt*"))) + 1
            info, _ = run_training(root, algo, f"{tag}-attempt{attempt:02d}", "confirmation",
                                   seed, steps, overrides)
        done[tag] = info
    return done


def confirm(root: Path, algo: str, plan: dict, workers: int) -> None:
    selection = read_json(root / "tuning" / f"{algo}_selection.json")
    if selection["protocol_sha256"] != digest(root / "manifests/protocol.json"):
        raise ValueError("Selection does not match the frozen protocol")
    jobs = [(root, algo, plan["confirmation_steps"], seed, selection["overrides"])
            for seed in plan["confirmation_seeds"]]
    if workers == 1:
        runs = [confirm_seed(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(confirm_seed, *job) for job in jobs]
            runs = [future.result() for future in futures]
    write_json(root / "manifests" / f"{algo}_confirmation.json", {
        "runs": {key: info for run in runs for key, info in run.items()},
        "selection_sha256": digest(root / "tuning" / f"{algo}_selection.json"),
    })


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "smoke", "screen", "confirm"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--algo", choices=ALGOS)
    cli.add_argument("--workers", type=int, choices=(1, 2, 3), default=3,
                     help="independent CPU processes for confirmation seeds")
    args = cli.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.phase == "freeze":
        with FileLock(str(root / ".freeze.lock"), timeout=0):
            freeze(root)
        return
    if args.algo is None:
        cli.error("--algo is required for this phase")
    plan = load_yaml(checked_config(root, "protocol.yaml"))
    with FileLock(str(root / f".{args.algo}.lock"), timeout=0):
        if args.phase == "smoke":
            smoke(root, args.algo)
        elif args.phase == "screen":
            screen(root, args.algo, plan)
        else:
            confirm(root, args.algo, plan, args.workers)


if __name__ == "__main__":
    main()
