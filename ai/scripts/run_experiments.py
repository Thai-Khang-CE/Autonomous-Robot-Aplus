"""End-to-end Week 9 pipeline. Each stage can be run separately with --stages.

    python scripts/run_experiments.py --stages all
    python scripts/run_experiments.py --stages evaluate,summarize,plot,manifest

Stages:
    tests      pytest (results/test_report.txt)
    check      scripts/check_environment.py
    baselines  random + heuristic on the held-out test set
    train      PPO/SAC controlled (50k, seed 0) and extended (200k, seeds 0-2), in parallel
    evaluate   final + best-validation checkpoints of every completed run on the test set
    summarize  scripts/summarize.py
    plot       scripts/plot_results.py
    manifest   scripts/build_manifest.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN_PLAN = [("ppo", "controlled", 0, 50000), ("sac", "controlled", 0, 50000)] + [
    (algo, "extended", seed, 200000) for seed in (0, 1, 2) for algo in ("ppo", "sac")
]
STAGES = ["tests", "check", "baselines", "train", "evaluate", "summarize", "plot", "manifest"]


def run(cmd: list[str], log: Path | None = None) -> int:
    print(">>", " ".join(cmd), flush=True)
    if log is None:
        return subprocess.call(cmd, cwd=ROOT)
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)


def stage_tests() -> None:
    rc = run([PY, "-m", "pytest", "-v", "-p", "no:cacheprovider"], ROOT / "results" / "test_report.txt")
    print("pytest exit code", rc)


def stage_train(max_parallel: int) -> None:
    procs = []
    for algo, label, seed, steps in TRAIN_PLAN:
        out = ROOT / "results" / algo / f"{label}_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        while len([p for p in procs if p.poll() is None]) >= max_parallel:
            procs[0].wait() if procs[0].poll() is None else procs.pop(0)
        cmd = [PY, "scripts/train.py", "--algo", algo, "--label", label, "--seed", str(seed),
               "--timesteps", str(steps), "--torch-threads", "1"]
        print(">>", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=open(out / "stdout.log", "w"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()


def completed_runs():
    for info in sorted((ROOT / "results").glob("*/*_seed*/run_info.json")):
        d = json.loads(info.read_text(encoding="utf-8"))
        if d.get("status") == "completed" and d["label"] != "smoke":
            yield info.parent, d


def stage_evaluate() -> None:
    for run_dir, d in completed_runs():
        for which in ("final", "best"):
            ckpt = run_dir / f"{which}_model.zip"
            if not ckpt.exists():
                continue
            label = f"{d['algorithm']}_{d['label']}_seed{d['seed']}_{which}"
            run([PY, "scripts/evaluate.py", "--policy", d["algorithm"], "--model",
                 str(ckpt.relative_to(ROOT)), "--label", label, "--train-seed", str(d["seed"])],
                ROOT / "results" / "evaluation" / "logs" / f"{label}.log")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--max-parallel", type=int, default=8)
    args = ap.parse_args()
    stages = STAGES if args.stages == "all" else args.stages.split(",")
    (ROOT / "results" / "evaluation" / "logs").mkdir(parents=True, exist_ok=True)
    for st in stages:
        print(f"===== stage: {st}", flush=True)
        if st == "tests":
            stage_tests()
        elif st == "check":
            run([PY, "scripts/check_environment.py"])
        elif st == "baselines":
            run([PY, "scripts/evaluate.py", "--policy", "random", "--seed", "0"])
            run([PY, "scripts/evaluate.py", "--policy", "heuristic"])
        elif st == "train":
            stage_train(args.max_parallel)
        elif st == "evaluate":
            stage_evaluate()
        elif st in ("summarize", "plot"):
            run([PY, f"scripts/{'summarize' if st == 'summarize' else 'plot_results'}.py"])
        elif st == "manifest":
            run([PY, "scripts/build_manifest.py"])
        else:
            raise SystemExit(f"unknown stage {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
