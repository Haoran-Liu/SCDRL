#!/usr/bin/env bash
set -euo pipefail

# Run each (seed, percentage) in a fresh Python process.
# This prevents memory (e.g., PyTorch caches / large arrays) from accumulating
# across the full sweep, which can trigger Slurm OOM kills (exit code 137).

seeds=(9 19 29 39 49 59 69 79 89 99)
percentages=(0.05 0.15 0.25)

for percentage in "${percentages[@]}"; do
	for seed in "${seeds[@]}"; do
		echo "[biolord_haniffa] seed=${seed} percentage=${percentage}"
		python competing_methods/biolord_haniffa.py --seed "${seed}" --percentage "${percentage}"
	done
done
