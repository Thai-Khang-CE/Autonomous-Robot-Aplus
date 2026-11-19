# Week 10 hyperparameter optimization

The Week 9 environment, YAML defaults, policy architecture and outcome definitions are preserved. Week 10 adds a fixed, resumable Optuna search, independent multi-seed training, locked final evaluation and a paired footprint comparison. The experiments use a numerical Gymnasium environment.

## Environment

Run commands from the repository's `ai/` directory. Use Python 3.12.10 and a new virtual environment. The experiment used Torch 2.15.0 CPU and Optuna 5.1.0. The inherited lock contains Windows package versions. For the full resolved environment, see `logs/environment.json` in the Week 10 evidence bundle.

```powershell
python -m venv .venv
$py = (Resolve-Path .venv/Scripts/python.exe).Path
uv pip install --python $py --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r requirements-week10.txt
& $py -m pip check
$env:PYTHONPATH = (Resolve-Path src).Path
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $py -m pytest -p no:cacheprovider -v
```

Set `$handoff` to an isolated extraction of the supplied Week 9 ZIP, with its `configs/` and `results/` directories directly inside. The prior virtual environment is unnecessary. Consult [the handoff audit](WEEK9_HANDOFF_AUDIT.md) for input hashes and recovered model status. To reproduce the full runs, use a fresh evidence directory.

```powershell
$handoff = 'D:/path/to/recovered-week9'
$out = 'D:/path/to/results/week10'
& $py scripts/check_environment.py --output "$out/logs/environment.json"
& $py scripts/audit_week9.py $handoff "$out/comparisons/week9"
& $py scripts/week10.py freeze --output $out
foreach ($algo in 'ppo','sac') {
    & $py scripts/week10.py smoke --algo $algo --output $out
    & $py scripts/week10.py screen --algo $algo --output $out
    & $py scripts/week10.py confirm --algo $algo --workers 3 --output $out
}
& $py scripts/evaluate_week10.py --output $out --handoff $handoff
& $py scripts/plot_week10.py --output $out
```

The runner uses CPU and one Torch intra-op thread per worker. Pass `--workers 1` to confirmation on memory-constrained systems. PowerShell operators can add `2>&1 | Tee-Object ...` to each command to persist its console output. To regenerate numerical tables for the supplied LaTeX report tree, run `scripts/report_week10_tables.py --output <evidence-root> --report <report-source-root>`.

## Frozen design

| Phase | Seeds and budget | Evidence |
| --- | --- | --- |
| Smoke | Seed 99, 1,024 transitions per algorithm, shorter warm-up for SAC | Training episodes, save/reload and checkpoint |
| Screening | Seed 100, 8,192 transitions per candidate, five configurations per algorithm | Checkpoints, candidate parameters, Optuna journals, raw validation episodes |
| Confirmation | Seeds 201, 202, 203, 16,384 transitions per configuration and seed | A fresh inherited-configuration run and a fresh selected-configuration run at every seed |
| Final | 50 scenarios per category, 150 total | Deterministic predictions from locked learned policies, reactive heuristic and seeded uniform random |
| Footprint | Matched feasible subset of the final placements at 0.18 m and 0.25 m | Separate replay of historical finals and heuristic |

Training scenario seeds are below 1,000,000. Week 10 validation uses 2,100,000 through 2,100,009 per category and final test uses 3,000,000 through 3,000,049. The inherited test namespace between them is audit-only. The test simulator geometry is serialized at full precision and hashed before screening. Candidate evaluation reads only validation scenarios. The final evaluator requires selected configurations, confirmation manifests and verified checkpoint hashes for both algorithms before opening the test suite.

Screening maximizes validation success minus 0.5 times validation collision rate. Equal scores prefer lower collision rate, then earlier trial number. The best alternative advances alongside the inherited configuration, regardless of how the original configuration ranked. Configurations and all overrides are in `configs/week10/protocol.yaml`. PPO checks that the transition budget is a multiple of the vectorized rollout and that the batch divides the rollout. SAC retains the inherited replay capacity and validates the warm-up against the train budget.

Each study uses a separate file-backed Optuna journal. Start one screening driver per algorithm. Rerun the same phase against the same evidence root to resume completed trial records and attempt interrupted candidates again. A partially trained model is never used as the starting point for a replacement trial. Confirmation similarly verifies completed runs and restarts unfinished attempts with fresh models. The journal backend was chosen after the Windows application-control policy blocked SQLAlchemy's native extension before any SQLite trial ran.

## Metric and artifact conventions

- Per-policy episode files carry scenario identifiers, outcomes, return, duration, path length, goal time and approximate SPL contributions where a checked geometric path exists.
- The per-group table reports sample standard deviation across training seeds. Replays of a policy across scenarios do not increase the count of independent training seeds. Random policy uses action RNG seed 0. Goal time uses completed successes.
- The historical PPO and SAC finals provide context at 200,704 and 200,000 transitions respectively. The direct tuning comparison is the paired 16,384-transition baseline versus selected group.
- In the footprint replay, identical scene geometry is used for both radii. The 0.25 m radius is a sensitivity case from the legacy URDF box. Only placements with a verified swept-disc path at 0.25 m enter this comparison.
- Figures, tables and tables per training seed are generated from raw episode/trial CSVs. Full-resolution trajectories, checkpoints and source logs are distributed as a standalone evidence bundle. No HPO importance is reported for this small prespecified search.
