# Week 9 handoff audit

## Provenance and repository reconciliation

- Primary archive: `robot-aplus-week9-rl (2).zip`, 614,282,604 bytes. SHA-256 `7635486224ef33e3524a5e20fc2baaacbd423441487569f55d9c19c81894eedb`. The user supplied the downloaded archive and the clean report template locally. The connected Drive folder was checked, but its listing exposed only the inputs subfolder.
- An isolated extraction validated 27,958 ZIP entries for path traversal, non-regular entries, case-insensitive name collisions and Windows-invalid paths. It recovered 797 files (375,869,968 bytes), excluding the inherited virtual environment, pytest cache and bytecode. The archive and the previously extracted handoff were preserved.
- The standalone 33-page report PDF (SHA-256 `7c9428a04c3620dc1dd58c8332135914fe15c1fa8769426016b21961e4a5f54f`) covers interim SAC. The 33-page PDF embedded in the archive (SHA-256 `d3d95a87eac3178d18524a2112bd984e9de72c4533842be5493580c8514348bd`) is a distinct revision and contains final SAC results. Both are historical documents.
- Recovered source, YAMLs, tests, reports, reference results, model ZIPs and evidence manifests were inspected. The handoff Git HEAD was `e3f6bed`, and its working tree included delivered evaluation changes absent from that commit. The import uses the ZIP working tree.
- The private team repo cloned with authenticated login `nvviet05` and API-confirmed push access at `main` revision `bce7c54e2325531a0b546d5dfac440a7bdf0555b`. Remote branches `main`, `khangn/feature/implement_urdf` and `khangn/feature/set_up_ci` contained no Week 9 RL package. The portal reported zero open PRs and successful C++ formatting CI on the integration base. The older snapshot lacked the current Xacro, launch and RViz additions. AI code was imported under `ai/`, with the ROS 2 tree retained.

## Execution and software

The Windows 11 x64 host used an i7-13700HX (24 logical CPUs), 15.75 GiB visible RAM and an RTX 4070 Laptop GPU (8188 MiB) with driver 560.94. CPU wheels and one Torch intra-op thread per training worker were used. WSL Ubuntu was present. ROS 2 and Gazebo were absent from the Windows PATH and were outside the numerical experiment.

A clean Python 3.12.10 environment installed the inherited lock. Core versions match historical records: NumPy 2.5.4, Torch 2.15.0+cpu, Gymnasium 1.4.0, Stable-Baselines3 2.9.2, pandas 3.1.1 and Matplotlib 3.11.2. Optuna 5.1.0 was added for Week 10. The lock passed `pip check`. The package imported, all 45 inherited tests passed, and Gymnasium and SB3 environment checks passed without warnings. The log and environment JSON record exact observed versions and space bounds.

Source and YAML agree on 11 float32 observation values, two actions in [−1, 1], eight range rays, a 6 m square world, 0.18 m footprint, 0.35 m/s forward ceiling, 1.5 rad/s yaw ceiling, 0.2 s control step, 250-step timeout and 0.25 m success tolerance. The inherited reward is distance progress ×10, goal bonus +20, collision −20 and a −0.05 step cost. These are simulator settings. The 0.4 × 0.3 m chassis from the legacy URDF yields 0.25 m circumscribed radius. The current display launch consumes a different Xacro whose lower deck is 0.340 × 0.240 m. The 0.25 m test is a legacy chassis box sensitivity case, with no claim of footprint validation for the full Xacro or physical robot.

## Recalculated historical measurements

Every recovered raw CSV contained 150 distinct scenarios with 50 per category, and mutually exclusive outcome flags. Report entries use the population standard deviation (denominator three). Confirmation analyses use the sample standard deviation and label the distinction explicitly.

| Historical policy | Seeds | Success % (population SD, pp) | Collision % (population SD, pp) | Timeout % |
| --- | ---: | ---: | ---: | ---: |
| PPO final, 200,704 transitions | 3 | 69.1 (2.6) | 17.1 (3.8) | 13.8 |
| SAC interim, 100,000 transitions | 3 | 76.2 (3.3) | 4.9 (1.7) | 18.9 |
| SAC final, 200,000 transitions | 3 | 76.0 (1.4) | 5.6 (1.7) | 18.4 |

The first two rows agree with the standalone PDF to its displayed precision. The archive contains three PPO final ZIPs with 200,704 transitions, and three SAC final ZIPs with 200,000 transitions and completed run metadata dated November 13, 2026. The SAC ZIPs and completed logs are independent evidence for the later report revision. No serialized SAC replay buffer was recovered. Exact training continuation from policy-only ZIPs is unavailable.

All 46 unique canonical policy ZIP paths were loaded on CPU (23 per algorithm, excluding evidence directory copies). Observation and action spaces matched by equality and each model produced a valid deterministic action. The three PPO final, three SAC interim and three SAC final policies each replayed the first three cases per category from Week 9. All 81 outcomes and saved trajectories rounded to four decimals matched. These replays are baseline checks on historical scenarios. Week 10 candidate selection uses the new validation scenarios exclusively.

## Evidence

Under `results/week10/`, see `logs/dependencies.log`, `logs/week9-pytest.log`, `logs/environment.json` and `logs/week9-audit.log`. The `comparisons/week9/` directory contains original-file paths and SHA-256 checksums alongside the recalculated statistics, checkpoint inventory and full replay comparisons. The two PDF revisions and the checksums above trace the source of historical claims. Training artifacts and software were recovered locally, and the original worker processes were never assumed to still be running.
