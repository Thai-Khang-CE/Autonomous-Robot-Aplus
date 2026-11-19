"""Check the completeness and integrity of a generated Week 10 evidence tree."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import yaml

from week10 import checked_config, digest, read_json

OUTCOMES = ["success", "collision", "timeout"]


def verify_episodes(path: Path, scenarios: set[str]) -> int:
    frame = pd.read_csv(path)
    if len(frame) != len(scenarios) or set(frame.scenario_id) != scenarios:
        raise ValueError(f"Scenario coverage failed: {path}")
    if (not frame[OUTCOMES].isin([0, 1]).all().all()
            or not frame[OUTCOMES].sum(axis=1).eq(1).all()
            or not frame[OUTCOMES].idxmax(axis=1).eq(frame.outcome).all()):
        raise ValueError(f"Outcome ledger failed: {path}")
    return len(frame)


def verify(root: Path) -> None:
    plan = yaml.safe_load(checked_config(root, "protocol.yaml").read_text())
    scenario_ids = {
        split: {s["scenario_id"] for s in read_json(checked_config(root, f"{split}_scenarios.json"))}
        for split in ("test", "validation")
    }
    if scenario_ids["validation"] & scenario_ids["test"]:
        raise ValueError("The validation and final test overlap")
    footprint_ids = set(read_json(root / "evaluation/footprint/paired_scenario_ids.json"))
    if not footprint_ids or not footprint_ids <= scenario_ids["test"]:
        raise ValueError("The paired footprint suite is invalid")
    verified_checkpoints = 0
    validation_episodes = 0
    for algo, variants in plan["candidates"].items():
        completed = pd.read_csv(root / "tuning" / f"{algo}_trials.csv")
        completed = completed[completed.state.eq("COMPLETE")]
        if sorted(completed.params_configuration) != sorted(variants):
            raise ValueError(f"The trial matrix is incomplete for {algo}")
        for run in completed.itertuples():
            trial_name = f"trial-{run.number:02d}-{run.params_configuration}"
            validation_episodes += verify_episodes(root / "evaluation/validation" / algo / f"{trial_name}.csv",
                                                   scenario_ids["validation"])
            info = read_json(root / "logs/screen" / algo / trial_name / "run_info.json")
            if (info["seed"] != plan["screen_seed"] or info["executed_steps"] != plan["screen_steps"]
                    or not info["reload"]["passed"]
                    or digest(root / info["checkpoint"]) != info["checkpoint_sha256"]):
                raise ValueError(f"Screen checkpoint failed: {trial_name}")
            verified_checkpoints += 1
        confirmation = read_json(root / "manifests" / f"{algo}_confirmation.json")
        selection = root / "tuning" / f"{algo}_selection.json"
        if confirmation["selection_sha256"] != digest(selection):
            raise ValueError(f"Selection differs from the locked confirmation: {algo}")
        runs = confirmation["runs"]
        expected_ids = {f"{group}-seed{seed}" for group in ("baseline", "tuned") for seed in plan["confirmation_seeds"]}
        if set(runs) != expected_ids:
            raise ValueError(f"Unexpected confirmation runs: {algo}")
        for run_id, run in runs.items():
            if (run["executed_steps"] != plan["confirmation_steps"] or not run["reload"]["passed"]
                    or digest(root / run["checkpoint"]) != run["checkpoint_sha256"]):
                raise ValueError(f"Confirmation checkpoint failed: {run_id}")
            verified_checkpoints += 1
    final_files = sorted((root / "evaluation/test/episodes").glob("*.csv"))
    footprint_files = sorted((root / "evaluation/footprint/episodes").glob("*.csv"))
    if len(final_files) != 20 or len(footprint_files) != 14:
        raise ValueError("The evaluation policy matrix is incomplete")
    final_count = sum(verify_episodes(p, scenario_ids["test"]) for p in final_files)
    footprint_count = sum(verify_episodes(p, footprint_ids) for p in footprint_files)
    print(f"Verified {verified_checkpoints} screening and confirmation checkpoints.")
    print(f"Verified {validation_episodes} validation, {final_count} final and {footprint_count} footprint episodes.")


def write_manifest(root: Path) -> None:
    manifest = root / "manifests/sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != manifest and not path.name.endswith((".lock", ".tmp")):
                writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, digest(path)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()
    verify(args.output)
    if args.manifest:
        write_manifest(args.output)


if __name__ == "__main__":
    main()
