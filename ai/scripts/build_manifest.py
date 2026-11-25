"""Write RUN_MANIFEST.md (and results/run_manifest.json) from the artifacts actually on disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from robot_aplus_rl.config import PROJECT_ROOT

RES = PROJECT_ROOT / "results"


def main() -> None:
    runs = []
    for f in sorted(RES.glob("*/*_seed*/run_info.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rd = f.parent
        runs.append({
            "run": f"{d['algorithm']}/{rd.name}",
            "label": d["label"],
            "status": d["status"],
            "seed": d["seed"],
            "requested_timesteps": d["requested_timesteps"],
            "executed_timesteps": d.get("executed_timesteps"),
            "gradient_updates": d.get("n_gradient_updates"),
            "param_l2_change": round(d["policy_param_l2_change"], 3) if "policy_param_l2_change" in d else None,
            "reload_check": (d.get("reload_check") or {}).get("passed"),
            "wall_time_min": round(d.get("wall_time_s", 0) / 60, 1),
            "training_episodes": (sum(1 for _ in open(rd / "training_episodes.csv")) - 1)
            if (rd / "training_episodes.csv").exists() else 0,
            "final_model": (rd / "final_model.zip").exists(),
            "best_model": (rd / "best_model.zip").exists(),
            "command": d["command"],
            "error": d.get("error"),
        })
    evals = []
    for f in sorted((RES / "evaluation" / "episodes").glob("*.csv")):
        df = pd.read_csv(f)
        evals.append({"label": f.stem, "episodes": len(df),
                      "scenario_set": df.get("scenario_set", pd.Series(["test"])).iloc[0],
                      "deterministic": bool(df["deterministic"].iloc[0]),
                      "success_rate": round(df["success"].mean(), 4),
                      "collision_rate": round(df["collision"].mean(), 4),
                      "file": str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")})
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RES / "run_manifest.json").write_text(json.dumps({"generated_utc": stamp, "training_runs": runs,
                                                       "evaluations": evals}, indent=2),
                                           encoding="utf-8")
    tr = pd.DataFrame(runs)
    md = [
        "# RUN_MANIFEST - Robot Aplus Week 9",
        "",
        f"Generated {stamp} by `scripts/build_manifest.py` from files on disk. Every row below",
        "corresponds to a real run directory; nothing is entered by hand.",
        "",
        "Simulator: simplified 2D Gymnasium environment (`Navigation2DEnv`). NOT Gazebo, NOT a physical robot.",
        "",
        "## Training runs",
        "",
        "`smoke` = short pipeline check only (not an experiment); `controlled` = 50k-step single-seed run;",
        "`extended` = 200k-step runs with three training seeds.",
        "",
        tr.drop(columns=["command", "error"]).to_markdown(index=False) if len(tr) else "(none)",
        "",
        "### Exact commands",
        "",
    ]
    md += [f"- `{r['run']}`: `{r['command']}`" + (f"  - ERROR: {r['error']}" if r["error"] else "")
           for r in runs]
    md += ["", "## Evaluations (held-out test set unless stated)", "",
           pd.DataFrame(evals).to_markdown(index=False) if evals else "(none)", "",
           "## Other evidence", "",
           "- `results/test_report.txt` - pytest output",
           "- `results/environment_report.json` - machine, versions, env checker results, pip freeze",
           "- `results/evaluation/summary.md` - aggregated metrics",
           "- `results/figures/` - all figures (PNG + PDF)", ""]
    (PROJECT_ROOT / "RUN_MANIFEST.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
