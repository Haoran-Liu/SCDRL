#!/usr/bin/env bash
set -euo pipefail

# Run Seurat on simulation data across seeds + percentages.
# Note: percentage isn't used inside Seurat itself; it's used here to
# produce per-(percentage,seed) output filenames that match the rest of the pipeline.

SEEDS=(9 19 29 39 49 59 69 79 89 99)
PCTS=(0.05 0.15 0.25)

NPZ_PATH="SCDRL_data/simulation_data.npz"
OUT_DIR="results/seurat"

for PCT in "${PCTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "[Seurat] percentage=${PCT} seed=${SEED}"

    conda run -n utopia --no-capture-output Rscript competing_methods/Seurat_simulation.r \
      --seed "${SEED}" \
      --percentage "${PCT}" \
      --npz_path "${NPZ_PATH}" \
      --out_dir "${OUT_DIR}" \
      --no_write_base
  done
done
