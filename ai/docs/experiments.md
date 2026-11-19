# Week 9 experiments - protocol and status

All numbers: `results/evaluation/summary.md` (generated) and `reports/week09/tables/` (generated
from the raw CSV/JSON files by `scripts/make_report_assets.py`). Simulator: simplified 2D
Gymnasium environment, NOT Gazebo.

## Budgets actually executed (2026-11-13)

| Run | Steps | Seeds | Status |
|---|---|---|---|
| PPO smoke / SAC smoke | 4,096 / 3,000 | 0 | completed (pipeline check only) |
| PPO controlled | 51,200 (rollout multiple of 2048) | 0 | completed |
| SAC controlled | 50,000 | 0 | completed |
| PPO extended | 200,704 | 0, 1, 2 | completed |
| SAC extended | 200,000 | 0, 1, 2 | completed (about 265 min each); the 100k intermediate checkpoints were also evaluated ("SAC 100k ckpt") |

Reason for the SAC delay: one gradient step per environment step costs ~16 ms on this CPU when
run alone, and 9-16 env steps/s when 8 runs shared the CPU under memory pressure.

## Protocol

* Training scenarios: random seeds in [0, 1e6) drawn by `env.np_random`.
* Validation (checkpoint monitoring, `best_model.zip`): 30 fixed scenarios, seeds 1,000,000+.
* Test (final comparison only): 150 fixed scenarios (50 open, 50 indoor, 50 narrow), seeds 2,000,000+.
* Deterministic policy actions; random baseline seeded.
* Metrics: success / collision / timeout rate, return, length, path length, time to goal,
  final distance, SPL_approx (approximate reference path, see implementation_notes.md).
* Robustness: same test scenarios with robot radius 0.25 m (URDF circumscribed radius).

## Regenerating everything

```bash
PY=.venv/Scripts/python.exe
$PY scripts/run_experiments.py --stages evaluate,summarize,plot,manifest
$PY scripts/make_report_assets.py
$PY scripts/build_evidence.py
```

Robustness runs (radius 0.25 m) use `scripts/evaluate.py --robot-radius 0.25` (commands in RUN_MANIFEST.md / evaluation logs).
