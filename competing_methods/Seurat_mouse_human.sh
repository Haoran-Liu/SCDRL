#!/usr/bin/env bash
set -euo pipefail

# Run Seurat on mouse_human data across seeds + percentages.
# percentage isn't used for splitting inside Seurat; it's used only for per-(percentage,seed) filenames.

SEEDS=(9 19 29 39 49 59 69 79 89 99)
PCTS=(0.05 0.15 0.25)

NPZ_PATH="SCDRL_data/mouse_human.npz"
OUT_DIR="results/seurat"

for PCT in "${PCTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "[Seurat_mouse_human] percentage=${PCT} seed=${SEED}"

    conda run -n utopia --no-capture-output Rscript competing_methods/Seurat_mouse_human.r \
      --seed "${SEED}" \
      --percentage "${PCT}" \
      --npz_path "${NPZ_PATH}" \
      --out_dir "${OUT_DIR}" \
      --no_write_base
  done
done
