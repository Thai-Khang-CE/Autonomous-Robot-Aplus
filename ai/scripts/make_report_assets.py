"""Generate LaTeX tables and number macros for the Week 9 report directly from raw results.

Every number printed in the report's tables and prose comes from:
    results/evaluation/episodes/*.csv            (test-set episodes)
    results/evaluation/robustness/episodes/*.csv (footprint robustness episodes)
    results/<algo>/<run>/run_info.json            (training runs)
    results/<algo>/<run>/evaluations.npz          (validation during training)
    results/test_report.txt                       (pytest)
Outputs: reports/week09/tables/*.tex and reports/week09/tables/numbers.tex, plus
results/evaluation/report_numbers.json (same values, machine-readable, for cross-checking).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from robot_aplus_rl.config import PROJECT_ROOT
from robot_aplus_rl.evaluation import summarise

RES = PROJECT_ROOT / "results"
EVAL = RES / "evaluation"
TAB = PROJECT_ROOT / "reports" / "week09" / "tables"

GROUPS = {  # group key -> (macro stem, display name)
    "random": ("Rand", "Random policy"),
    "heuristic": ("Heur", "Goal-seeking heuristic"),
    "ppo_controlled_final": ("PPOctrl", "PPO 50k, final (1 seed)"),
    "sac_controlled_final": ("SACctrl", "SAC 50k, final (1 seed)"),
    "ppo_extended_final": ("PPOext", "PPO 200k, final (3 seeds)"),
    "sac_extended_final": ("SACext", "SAC 200k, final (3 seeds)"),
    "sac_interim100k_ckpt": ("SACmid", "SAC 100k ckpt of 200k runs (3 seeds)"),
    "ppo_extended_best": ("PPObest", "PPO 200k, best-val. (3 seeds)"),
    "sac_extended_best": ("SACbest", "SAC 200k, best-val. (3 seeds)"),
}
CATS = {"open": "Open", "indoor": "Indoor", "narrow": "Narrow", "all": "All"}
SEEDL = "ABC"
macros: dict[str, str] = {}


def pct(x: float) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def num(x: float, d: int = 2) -> str:
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def macro(name: str, value) -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = str(value)


def group_stats(paths) -> dict[str, pd.DataFrame]:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["group"] = df["policy"].str.replace(r"_seed\d+", "", regex=True)
    return df


def per_seed_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, lab), d in df.groupby(["group", "policy"], sort=False):
        for cat in CATS:
            dc = d if cat == "all" else d[d.category == cat]
            rows.append({"group": g, "label": lab, "category": cat, **summarise(dc)})
    return pd.DataFrame(rows)


def agg(t: pd.DataFrame, g: str, cat: str, col: str):
    v = t[(t.group == g) & (t.category == cat)][col]
    return v.mean(), v.std(ddof=0), v.min(), v.max(), len(v)


def cell(t, g, cat, col):
    m, s, lo, hi, n = agg(t, g, cat, col)
    return pct(m) if n <= 1 else f"{pct(m)} $\\pm$ {pct(s)}"


def training_runs() -> list[dict]:
    runs = []
    for f in sorted(RES.glob("*/*_seed*/run_info.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["label"] == "smoke" or d["status"] != "completed":
            if d["label"] != "smoke":
                runs.append(d)
            continue
        ep = pd.read_csv(f.parent / "training_episodes.csv")
        last = ep.tail(500)
        d["n_train_episodes"] = len(ep)
        d["last500_success"] = last["is_success"].mean()
        d["last500_return"] = last["episode_return"].mean()
        ev = f.parent / "evaluations.npz"
        if ev.exists():
            z = np.load(ev)
            sr = z["successes"].mean(axis=1)
            d["val_final_success"] = float(sr[-1])
            d["val_best_success"] = float(sr.max())
            d["val_best_step"] = int(z["timesteps"][int(sr.argmax())])
        runs.append(d)
    return runs


def write(name: str, lines: list[str]) -> None:
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote tables/" + name)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------- tests
    rep = (RES / "test_report.txt").read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed", rep)
    macro("NTestsPassed", m.group(1) if m else "0")
    f = re.search(r"(\d+) failed", rep)
    macro("NTestsFailed", f.group(1) if f else "0")

    # ------------------------------------------------------------- training runs
    runs = training_runs()
    lines = [r"\begin{tabular}{llrrrrrc}", r"\toprule",
             r"Algorithm & Run & Seed & Executed steps & Grad.\ updates & Train episodes "
             r"& Wall time [min] & Reload \\", r"\midrule"]
    for d in runs:
        if d["status"] != "completed":
            prog = pd.read_csv(RES / d["algorithm"] / d["run_name"] / "logs" / "progress.csv")
            done_steps = int(prog["time/total_timesteps"].max())
            lines.append(f"{d['algorithm'].upper()} & {d['label']} & {d['seed']} & "
                         f"\\multicolumn{{5}}{{l}}{{status {d['status']}: {done_steps:,} of "
                         f"{d['requested_timesteps']:,} steps logged at report time}} \\\\")
            continue
        ok = "pass" if d["reload_check"]["passed"] else "FAIL"
        lines.append(f"{d['algorithm'].upper()} & {d['label']} & {d['seed']} & "
                     f"{d['executed_timesteps']:,} & {d['n_gradient_updates']:,} & "
                     f"{d['n_train_episodes']:,} & {d['wall_time_s'] / 60:.1f} & {ok} \\\\")
        stem = f"{d['algorithm'].upper()}{d['label'][:3]}{SEEDL[d['seed']]}"
        macro(f"Steps{stem}", f"{d['executed_timesteps']:,}".replace(",", "{,}"))
        macro(f"Upd{stem}", f"{d['n_gradient_updates']:,}".replace(",", "{,}"))
        macro(f"Wall{stem}", f"{d['wall_time_s'] / 60:.0f}")
        macro(f"ValFinal{stem}", pct(d.get("val_final_success", np.nan)))
        macro(f"ValBest{stem}", pct(d.get("val_best_success", np.nan)))
        macro(f"TrainSucc{stem}", pct(d["last500_success"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("training_runs.tex", lines)
    done = [d for d in runs if d["status"] == "completed"]
    for algo in ("ppo", "sac"):
        ext = [d for d in done if d["algorithm"] == algo and d["label"] == "extended"]
        if ext:
            macro(f"NSeeds{algo.upper()}ext", len(ext))
            macro(f"MaxWall{algo.upper()}ext", f"{max(d['wall_time_s'] for d in ext) / 60:.0f}")

    # ------------------------------------------------------------- test-set evaluation
    df = group_stats(sorted((EVAL / "episodes").glob("*.csv")))
    t = per_seed_table(df)
    present = [g for g in GROUPS if g in set(t.group)]
    macro("NEvalEpisodes", int(t[(t.group == "random") & (t.category == "all")]["episodes"].iloc[0]))
    for g in present:
        stem = GROUPS[g][0]
        for col, key in (("success_rate", "SR"), ("collision_rate", "CR"), ("timeout_rate", "TR")):
            mean, std, lo, hi, n = agg(t, g, "all", col)
            macro(f"{key}{stem}", pct(mean))
            if n > 1:
                macro(f"{key}std{stem}", pct(std))
                macro(f"{key}min{stem}", pct(lo))
                macro(f"{key}max{stem}", pct(hi))
            for cat in ("open", "indoor", "narrow"):
                macro(f"{key}{stem}{CATS[cat]}", pct(agg(t, g, cat, col)[0]))
        for col, key, dgt in (("mean_return", "Ret", 1), ("mean_length", "Len", 0),
                              ("mean_path_length", "Path", 2), ("mean_time_to_goal_success", "TTG", 1),
                              ("mean_final_distance", "Fdist", 2), ("spl_approx", "SPL", 2)):
            macro(f"{key}{stem}", num(agg(t, g, "all", col)[0], dgt))
        if agg(t, g, "all", "success_rate")[4] == 1:
            r = t[(t.group == g) & (t.category == "all")].iloc[0]
            macro(f"SRlo{stem}", pct(r.success_ci95_low))
            macro(f"SRhi{stem}", pct(r.success_ci95_high))

    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Policy & SR [\%] $\uparrow$ & CR [\%] $\downarrow$ & TR [\%] & Return & Steps "
             r"& Path [m] & SPL$_{\approx}$ \\", r"\midrule"]
    for g in present:
        lines.append(" & ".join([
            GROUPS[g][1], cell(t, g, "all", "success_rate"), cell(t, g, "all", "collision_rate"),
            pct(agg(t, g, "all", "timeout_rate")[0]), num(agg(t, g, "all", "mean_return")[0], 1),
            num(agg(t, g, "all", "mean_length")[0], 0), num(agg(t, g, "all", "mean_path_length")[0]),
            num(agg(t, g, "all", "spl_approx")[0])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("overall.tex", lines)

    lines = [r"\begin{tabular}{lccc|ccc|ccc}", r"\toprule",
             r" & \multicolumn{3}{c|}{Success [\%]} & \multicolumn{3}{c|}{Collision [\%]} "
             r"& \multicolumn{3}{c}{Timeout [\%]} \\",
             r"Policy & open & indoor & narrow & open & indoor & narrow & open & indoor & narrow \\",
             r"\midrule"]
    for g in present:
        cells = [pct(agg(t, g, c, col)[0]) for col in ("success_rate", "collision_rate", "timeout_rate")
                 for c in ("open", "indoor", "narrow")]
        lines.append(" & ".join([GROUPS[g][1]] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("by_category.tex", lines)

    # ------------------------------------------------------------- footprint robustness
    rob = sorted((EVAL / "robustness" / "episodes").glob("*.csv"))
    if rob:
        rt = per_seed_table(group_stats(rob))
        rt["group"] = rt["group"].str.replace("_r025", "", regex=False)
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r" & \multicolumn{3}{c}{$r = 0.18$ m (training radius)} & "
                 r"\multicolumn{3}{c}{$r = 0.25$ m (URDF circumscribed)} \\",
                 r"Policy & SR [\%] & CR [\%] & TR [\%] & SR [\%] & CR [\%] & TR [\%] \\", r"\midrule"]
        for g in [g for g in GROUPS if g in set(rt.group)]:
            stem = GROUPS[g][0]
            row = [GROUPS[g][1]]
            for tt in (t, rt):
                row += [cell(tt, g, "all", "success_rate"), cell(tt, g, "all", "collision_rate"),
                        pct(agg(tt, g, "all", "timeout_rate")[0])]
            lines.append(" & ".join(row) + r" \\")
            macro(f"SRrob{stem}", pct(agg(rt, g, "all", "success_rate")[0]))
            macro(f"CRrob{stem}", pct(agg(rt, g, "all", "collision_rate")[0]))
        lines += [r"\bottomrule", r"\end{tabular}"]
        write("robustness.tex", lines)

    out = ["% Auto-generated by scripts/make_report_assets.py from raw result files. Do not edit."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (TAB / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    (EVAL / "report_numbers.json").write_text(json.dumps(macros, indent=1), encoding="utf-8")
    print(f"wrote tables/numbers.tex ({len(macros)} macros)")


if __name__ == "__main__":
    main()
