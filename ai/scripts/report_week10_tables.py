"""Build LaTeX result tables directly from Week 10 metrics and Optuna exports."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from week10 import read_json


def table(dst: Path, caption: str, label: str, spec: str, headers, rows) -> None:
    newline = chr(10)
    slash = chr(92)
    row_end = f" {slash}{slash}"
    tabular = [r"\begin{table}[!htbp]", r"\centering\small", f"{slash}caption{{{caption}}}",
               f"{slash}label{{tab:{label}}}", f"{slash}begin{{tabular}}{{@{{}}{spec}@{{}}}}",
               r"\toprule", " & ".join(headers) + row_end, r"\midrule"]
    tabular += [" & ".join(map(str, row)) + row_end for row in rows]
    tabular += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    dst.write_text(newline.join(tabular) + newline, encoding="utf-8")


def pct(x, precision=1):
    return f"{100*x:.{precision}f}"


def number(x, decimals=1):
    return f"{x:.{decimals}f}" if pd.notna(x) else "--"


def small_rates(row, metric):
    sd = getattr(row, f"{metric}_sd")
    mean = pct(getattr(row, f"{metric}_mean"))
    return f"{mean} ({pct(sd)})" if pd.notna(sd) else mean


def readable(algo, group):
    if group == "reference":
        return algo.capitalize()
    phrase = {"baseline": "inherited config.", "tuned": "selected config.",
              "historical": "historical final"}[group]
    return f"{algo.upper()} {phrase}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--screen-only", action="store_true")
    args = ap.parse_args()
    root = args.output
    dst = args.report / "tables"
    dst.mkdir(parents=True, exist_ok=True)
    results = []
    for algo in ("ppo", "sac"):
        selection = read_json(root / "tuning" / f"{algo}_selection.json")
        runs = pd.read_csv(root / "tuning" / f"{algo}_trials.csv")
        for run in runs[runs.state.eq("COMPLETE")].itertuples():
            label = run.params_configuration.replace("_", " ").capitalize()
            if run.number == selection["trial"]:
                label += r"\textsuperscript{*}"
            results.append([
                algo.upper(), label, pct(run.user_attrs_success_rate),
                pct(run.user_attrs_collision_rate), pct(run.user_attrs_timeout_rate),
                number(run.value, 3), number(run.user_attrs_mean_return),
            ])
    table(dst / "screen.tex", r"Validation outcomes at 8,192 transitions. The asterisk denotes the selected alternative.",
          "screen", "llrrrrr", ("Method", "Configuration", r"Success (\%)", r"Collision (\%)",
                              r"Timeout (\%)", r"$J$", "Return"), results)
    if args.screen_only:
        return

    aggregate = pd.read_csv(root / "comparisons/final_metrics_aggregate.csv")
    aggregate = aggregate[aggregate.category.eq("all")]
    full = pd.read_csv(root / "comparisons/final_metrics.csv")
    groups = [("ppo", g) for g in ("baseline", "tuned", "historical")]
    groups += [("sac", g) for g in ("baseline", "tuned", "historical")]
    groups += [("heuristic", "reference"), ("random", "reference")]
    by_key = {(row.algorithm, row.group): row for row in aggregate.itertuples()}
    final, efficiency = [], []
    for algo, group in groups:
        row = by_key[algo, group]
        selected = full[full.algorithm.eq(algo) & full.group.eq(group) & full.category.eq("all")]
        steps = selected.steps.iloc[0]
        label = readable(algo, group)
        final.append([label, f"{steps:,}" if steps else "--", row.training_seeds or "--",
                      *[small_rates(row, f"{metric}_rate") for metric in ("success", "collision", "timeout")]])
        efficiency.append([label, number(row.mean_return_mean), number(row.mean_length_mean, 0),
                           number(row.mean_path_length_mean, 2), number(row.mean_time_to_goal_success_mean),
                           number(row.spl_approx_mean, 3), number(selected.training_wall_s.mean())])
    table(dst / "final.tex",
          r"Outcome rates on 150 independent test scenarios at 0.18 m. Parentheses contain the sample standard deviation across training seeds, in percentage points.",
          "final", "lrrrrr", ("Policy group", "Steps", "Seeds", r"Success (\%)", r"Collision (\%)",
                             r"Timeout (\%)"), final)
    table(dst / "efficiency.tex",
          r"Episode means on the final suite. Goal time includes successes only. Training time is the mean wall-clock time per seed, in seconds. Historical timings come from the original machine.",
          "efficiency", "lrrrrrr", ("Policy group", "Return", "Length", "Path (m)", "Goal (s)",
                                    "SPL", "Train (s)"), efficiency)

    deltas = pd.read_csv(root / "comparisons/paired_tuned_minus_baseline.csv")
    deltas = deltas[deltas.category.eq("all")]
    paired = []
    for algo in ("ppo", "sac"):
        a = deltas[deltas.algorithm.eq(algo)]
        for row in a.itertuples():
            paired.append([algo.upper(), int(row.train_seed), *[pct(getattr(row, f"{metric}_rate"))
                for metric in ("success", "collision", "timeout")]])
        paired.append([algo.upper(), "Mean", *[pct(a[f"{metric}_rate"].mean())
            for metric in ("success", "collision", "timeout")]])
    table(dst / "paired.tex", r"Paired final-suite outcome differences (selected minus inherited configuration), in percentage points.",
          "paired", "llrrr", ("Method", "Seed", r"$\Delta$ Success", r"$\Delta$ Collision",
                              r"$\Delta$ Timeout"), paired)

    fp = pd.read_csv(root / "comparisons/footprint_metrics_aggregate.csv")
    fp = fp[fp.category.eq("all")]
    ids = read_json(root / "evaluation/footprint/paired_scenario_ids.json")
    rows = [[(row.algorithm.upper() if row.algorithm != "heuristic" else "Heuristic"),
             f"{row.radius_m:.2f}", *[small_rates(row, f"{metric}_rate")
             for metric in ("success", "collision", "timeout")]] for row in fp.itertuples()]
    table(dst / "footprint.tex", f"Footprint evaluation on {len(ids)} paired scenarios. "
          "Standard deviations over training seeds appear in parentheses.",
          "footprint", "lrrrr", ("Policy", "Radius (m)", r"Success (\%)", r"Collision (\%)",
                                 r"Timeout (\%)"), rows)


if __name__ == "__main__":
    main()
