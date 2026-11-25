#!/usr/bin/env bash
# Week 9 training batch: controlled (50k, seed 0) + extended (200k, seeds 0-2) for PPO and SAC.
# All runs use the same configs/environment.yaml. Each process uses 1 torch thread (fastest on
# the reference CPU, see docs/experiments.md). Runs execute in parallel; stdout goes to
# results/<algo>/<run>/stdout.log.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=.venv/bin/python

launch() {  # algo label seed timesteps
  local out="results/$1/$2_seed$3"
  mkdir -p "$out"
  "$PY" scripts/train.py --algo "$1" --label "$2" --seed "$3" --timesteps "$4" \
    --torch-threads 1 > "$out/stdout.log" 2>&1 &
}

launch ppo controlled 0 50000
launch sac controlled 0 50000
for s in 0 1 2; do
  launch ppo extended "$s" 200000
  launch sac extended "$s" 200000
done
wait
echo "batch finished"
