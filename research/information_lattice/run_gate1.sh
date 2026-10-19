#!/usr/bin/env bash
# Run every Gate 1 block in order from a clean checkout. Usage: run_gate1.sh <output-dir> [python]
# Run from a dedicated worktree of a pushed commit so edits elsewhere cannot change running code.
set -euo pipefail
out=$1
python=${2:-.venv/bin/python}
config=research/information_lattice/configs/gate1.toml
cpus=$(python3 -c 'import os; print(os.cpu_count())')
mkdir -p "$out/logs"
export PYTHONPATH=src:.
run() {  # block learner workers threads
  "$python" -m research.information_lattice.gate1 --config "$config" --block "$1" --learner "$2" \
    --output "$out" --workers "$3" --threads "$4" > "$out/logs/$1-$2.log" 2>&1
}
for block in validity power scale; do run "$block" lp $((cpus - 1)) 1; done
run validity mr $((cpus - 1)) 1
run validity-cnn cnn $((cpus / 2 - 1)) 2
for block in power scale; do run "$block" mr $((cpus - 1)) 1; done
echo "gate1 complete" > "$out/logs/DONE"
