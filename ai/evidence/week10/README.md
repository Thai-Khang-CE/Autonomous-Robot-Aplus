# Week 10 result extracts

The matched comparison uses three fresh seeds and 16,384 transitions per policy. On 150 new test scenarios, PPO success increased from 57.3% to 61.6% and collisions fell from 42.7% to 38.4%. SAC success increased from 61.8% to 66.9%, mean collisions fell from 15.3% to 13.3%, and timeouts fell from 22.9% to 19.8%. The per-seed SAC collision changes were −6.0, +10.0 and −10.0 percentage points. The longer historical PPO and SAC finals reached 65.1% and 75.6% success. The heuristic reached 68.7%.

- `*_trials.csv` and `*_selection.json` contain the validation results and frozen choices. Each trial trained for 8,192 transitions.
- `final_metrics.csv` holds per-policy and per-category rates, budget, host runtime, and checkpoint hashes. `final_metrics_aggregate.csv` reports group means and sample standard deviations across seeds.
- `paired_tuned_minus_baseline.csv` reports within-seed changes. `footprint_metrics*.csv` summarize the evaluation-only comparison over 142 shared feasible placements.
- The full local bundle is `deliverables/Robot_Aplus_Week10_Evidence.zip`. It includes raw validation and test CSVs, Optuna journals, screenshots, model ZIPs and a per-file SHA-256 ledger. Large generated artifacts are distributed outside ordinary Git history.

See [reproduction commands](../../docs/WEEK10_REPRODUCIBILITY.md) and [handoff audit](../../docs/WEEK9_HANDOFF_AUDIT.md).
