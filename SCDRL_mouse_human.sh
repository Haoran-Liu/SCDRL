#!/usr/bin/env bash
set -euo pipefail

# Run each (seed, percentage) in a fresh Python process to avoid memory
# accumulation across the full sweep.

seeds=(9 19 29 39 49 59 69 79 89 99)
percentages=(0.05 0.15 0.25)

for percentage in "${percentages[@]}"; do
	for seed in "${seeds[@]}"; do
		echo "[SCDRL_mouse_human] seed=${seed} percentage=${percentage}"
		python SCDRL_mouse_human.py --seed "${seed}" --percentage "${percentage}" --log_dir log_dir --experiment mouse_human
	done
done
