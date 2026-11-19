# Robot Aplus – Week 9: PPO vs SAC for goal-directed navigation (simplified 2D simulator)

HCMUT capstone HK261-DAGD1-227 · Advisor: Dr. Võ Tuấn Bình · AI/RL work package: **Nguyễn Văn Viết**

This repository contains a custom Gymnasium environment (`Navigation2DEnv`) for a differential-
drive disc robot with an 8-ray range sensor, unit tests, Stable-Baselines3 PPO and SAC training
scripts, random and heuristic baselines, an evaluation protocol on held-out scenarios, figures,
and the Week 9 report source.

> **Scope warning.** All results come from a simplified **numerical 2D simulator**. They are
> **not** Gazebo results and **not** physical-robot results. ROS 2 / Gazebo integration is an
> outstanding dependency (see `docs/integration_contract.md`).

## Layout

```
configs/            environment.yaml, ppo.yaml, sac.yaml
src/robot_aplus_rl/ navigation_env, geometry, sensors, rewards, scenarios, planning,
                    baselines, evaluation, visualization, config
scripts/            check_environment, train, evaluate, summarize, plot_results,
                    build_manifest, run_experiments, run_training_batch.sh
tests/              45 pytest tests (geometry, sensors, reward, env contract, reproducibility)
results/            real run outputs (see RUN_MANIFEST.md)
docs/               implementation_notes, experiments, integration_contract, learning_guide_vi
reports/week09/     viet_section.tex, main.tex, references.bib, figures/, tables/
```

## Setup (Windows, Git Bash; tested 2026-11-13)

Python 3.12.10, CPU only. `uv` was used to create the virtual environment.

```bash
cd ~/projects/robot-aplus-week9-rl
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe -e .
```

On Linux/WSL replace `.venv/Scripts/python.exe` by `.venv/bin/python`. Without `uv`:
`python -m venv .venv && .venv/bin/pip install ...`. The exact transitive versions are in
`requirements-lock.txt`.

## Verify

```bash
PY=.venv/Scripts/python.exe
$PY -m pytest                          # unit tests incl. Gymnasium + SB3 env checkers
$PY scripts/check_environment.py       # machine info, versions, env contract -> results/environment_report.json
```

## Reproduce the Week 9 experiment

```bash
PY=.venv/Scripts/python.exe
# 1. smoke tests (pipeline check only)
$PY scripts/train.py --algo ppo --timesteps 4096 --seed 0 --label smoke --eval-freq 2048
$PY scripts/train.py --algo sac --timesteps 3000 --seed 0 --label smoke --eval-freq 1500 --set learning_starts=500
# 2. baselines on the held-out test set (150 scenarios)
$PY scripts/evaluate.py --policy random --seed 0
$PY scripts/evaluate.py --policy heuristic
# 3. training: controlled (50k, seed 0) + extended (200k, seeds 0,1,2), PPO and SAC, in parallel
bash scripts/run_training_batch.sh
# 4. evaluate all checkpoints, aggregate, plot, write RUN_MANIFEST.md
$PY scripts/run_experiments.py --stages evaluate,summarize,plot,manifest,tests
```

Or everything: `$PY scripts/run_experiments.py --stages all`.
Wall-clock on the reference laptop (i7-1355U, 8 runs in parallel): see `RUN_MANIFEST.md`
(`wall_time_min`). SAC is the bottleneck (one gradient step per environment step).

Single runs:

```bash
$PY scripts/train.py --algo sac --timesteps 100000 --seed 3 --label extra --torch-threads 1
$PY scripts/train.py --algo ppo --resume results/ppo/extended_seed0/final_model.zip --timesteps 50000 --label resumed
$PY scripts/evaluate.py --policy sac --model results/sac/extended_seed0/final_model.zip --label sac_test --train-seed 0
```

## Where is the evidence?

| Evidence | Location |
|---|---|
| Run status, commands, timesteps, update counts, reload checks | `RUN_MANIFEST.md`, `results/run_manifest.json`, `results/<algo>/<run>/run_info.json` |
| Training episodes (global timestep, return, outcome) | `results/<algo>/<run>/training_episodes.csv`, `monitor/` |
| SB3 diagnostics (losses, entropy, ...) + TensorBoard | `results/<algo>/<run>/logs/` |
| Validation during training | `results/<algo>/<run>/evaluations.npz` |
| Checkpoints | `final_model.zip`, `best_model.zip`, `checkpoints/` |
| Test-set episodes / trajectories | `results/evaluation/episodes/*.csv`, `results/evaluation/trajectories/*.json` |
| Aggregated metrics | `results/evaluation/summary.md`, `summary_by_label.csv`, `summary_by_group.csv` |
| Figures | `results/figures/` (PNG + PDF) |
| Test output | `results/test_report.txt` |

## Results

See `docs/experiments.md` for the protocol, the measured results and their interpretation, and
`reports/week09/` for the report.

## Documentation

* `docs/implementation_notes.md` – model, geometry, sensor, reward, scenarios, metrics.
* `docs/experiments.md` – budgets, protocol, results, limitations.
* `docs/integration_contract.md` – ROS 2 / Gazebo interface (not implemented yet).
* `docs/learning_guide_vi.md` – Vietnamese learning guide based on this code.
