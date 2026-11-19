"""Record the machine / software environment and verify the Gymnasium environment contract.

Usage:  python scripts/check_environment.py [--output results/environment_report.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "torch", "gymnasium", "stable-baselines3", "matplotlib", "pandas",
            "pytest", "tensorboard", "PyYAML"]


def software_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "NOT INSTALLED"
    return out


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "is_wsl": "microsoft" in platform.release().lower(),
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        info["torch_cuda_available"] = None
    for tool in ("ros2", "gz", "pdflatex", "git"):
        info[f"{tool}_on_path"] = shutil.which(tool) is not None
    return info


def check_env_contract() -> dict:
    from gymnasium.utils.env_checker import check_env as gym_check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    from robot_aplus_rl import Navigation2DEnv

    result = {}
    for name, fn in (("gymnasium_check_env", lambda e: gym_check_env(e, skip_render_check=True)),
                     ("sb3_check_env", lambda e: sb3_check_env(e, warn=True))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fn(Navigation2DEnv())
                result[name] = {"passed": True, "warnings": [str(w.message) for w in caught]}
            except Exception as exc:  # noqa: BLE001 - report every failure
                result[name] = {"passed": False, "error": repr(exc)}
    env = Navigation2DEnv()
    result["observation_space"] = str(env.observation_space)
    result["action_space"] = str(env.action_space)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "results" / "environment_report.json"))
    args = ap.parse_args()
    report = {"machine": machine_info(), "software": software_versions(),
              "env_contract": check_env_contract()}
    try:
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True).stdout
    report["pip_freeze"] = [ln for ln in freeze.splitlines() if ln and not ln.startswith("-e")]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("machine", "software", "env_contract")}, indent=2))
    ok = all(v.get("passed") for k, v in report["env_contract"].items() if isinstance(v, dict))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
