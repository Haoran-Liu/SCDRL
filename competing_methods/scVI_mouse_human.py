import argparse
import os
from pathlib import Path
import sys
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import scanpy as sc
import scvi
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disentanglement_metrics import dci_score, hungarian_alignment


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_copy_matrix(x: object) -> object:
    copy_fn = getattr(x, "copy", None)
    if callable(copy_fn):
        try:
            return copy_fn()
        except TypeError:
            pass
    return np.array(x, copy=True)


def _load_base_data() -> tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(ROOT / "SCDRL_data" / "mouse_human.npz")
    print(npzfile.files)
    return npzfile["counts"], npzfile["factors"]


def _make_permutation(seed: int, num_cells: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    random_idx = np.arange(num_cells)
    rng.shuffle(random_idx)
    return random_idx


def _make_test_idx(seed: int, num_cells: int, percentage: float) -> np.ndarray:
    if not (0.0 < float(percentage) <= 1.0):
        raise ValueError(f"percentage must be in (0,1], got {percentage}")
    rng = np.random.RandomState(seed)
    random_idx = np.arange(num_cells)
    rng.shuffle(random_idx)
    num_labels = int(num_cells * float(percentage))
    labeled_idx = rng.choice(num_cells, num_labels, replace=False)
    test_idx = np.setdiff1d(np.arange(num_cells), labeled_idx)
    return test_idx


def _spearman_correlations(latent: np.ndarray, factors: np.ndarray) -> np.ndarray:
    spearman_correlations: list[float] = []
    for idx in range(factors.shape[1]):
        y = factors[:, idx]
        for i in range(latent.shape[1]):
            corr, _p = spearmanr(latent[:, i], y)
            corr_val = float(np.asarray(corr).ravel()[0])
            spearman_correlations.append(corr_val)
    return np.asarray(spearman_correlations).reshape(factors.shape[1], -1)


def run_for_seed(
    *,
    seed: int,
    n_latent: int,
    n_top_genes: int,
    leiden_resolution: float,
    max_epochs: int,
    batch_size: int,
    early_stopping: bool,
    early_stopping_patience: int,
    check_val_every_n_epoch: int,
    precision: str,
    results_dir: str,
    counts_base: np.ndarray,
    factors_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed
    print("scvi-tools version:", scvi.__version__)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    num_cells = counts_base.shape[0]
    random_idx = _make_permutation(seed, num_cells)
    counts = counts_base[random_idx]
    factors = factors_base[random_idx]

    adata = ad.AnnData(counts)
    adata.layers["counts"] = _safe_copy_matrix(adata.X)

    adata.obs["system"] = factors[:, 0].astype(str)
    adata.obs["cell_type"] = factors[:, 1].astype(str)

    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=int(n_top_genes),
        layer="counts",
        batch_key="system",
        subset=True,
    )

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="system")
    model = scvi.model.SCVI(
        adata, n_layers=2, n_latent=int(n_latent), gene_likelihood="nb"
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision_norm = str(precision).strip().lower()

    if precision_norm in {"16", "16-mixed", "bf16", "bf16-mixed", "32"}:
        model.train(
            accelerator=accelerator,
            devices=1,
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            early_stopping=bool(early_stopping),
            early_stopping_patience=int(early_stopping_patience),
            check_val_every_n_epoch=int(check_val_every_n_epoch),
            precision=precision_norm,
        )
    else:
        model.train(
            accelerator=accelerator,
            devices=1,
            max_epochs=int(max_epochs),
            batch_size=int(batch_size),
            early_stopping=bool(early_stopping),
            early_stopping_patience=int(early_stopping_patience),
            check_val_every_n_epoch=int(check_val_every_n_epoch),
        )

    latent_all = np.asarray(model.get_latent_representation())
    np.save(
        os.path.join(results_dir, f"scVI_mouse_human_latent_seed{seed}.npy"),
        latent_all,
    )

    adata.obsm["X_scVI"] = latent_all
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.leiden(adata, resolution=float(leiden_resolution))
    leiden_all = adata.obs["leiden"].astype(str).to_numpy()

    import pandas as pd

    pd.DataFrame({"leiden": leiden_all}).to_csv(
        os.path.join(results_dir, f"scVI_mouse_human_seed{seed}.csv"), index=False
    )

    return latent_all, factors, np.asarray(leiden_all)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--percentage", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--percentages", type=float, nargs="*", default=None)
    parser.add_argument("--n_latent", type=int, default=19)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--leiden_resolution", type=float, default=0.35)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--early_stopping", dest="early_stopping", action="store_true")
    parser.add_argument(
        "--no_early_stopping", dest="early_stopping", action="store_false"
    )
    parser.set_defaults(early_stopping=True)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed" if torch.cuda.is_available() else "32",
        help="Trainer precision: 32 | 16-mixed | bf16-mixed",
    )
    parser.add_argument("--results_dir", type=str, default="results/scVI")
    args = parser.parse_args(argv)

    seeds: Iterable[int] = args.seeds if args.seeds else [args.seed]
    percentages: Iterable[float] = (
        args.percentages if args.percentages else [args.percentage]
    )

    _ensure_dir(args.results_dir)
    counts_base, factors_base = _load_base_data()

    last_seed: Optional[int] = None
    last_percentage: Optional[float] = None
    last_leiden_all: Optional[np.ndarray] = None
    last_latent_all: Optional[np.ndarray] = None
    last_random_idx: Optional[np.ndarray] = None

    for seed in seeds:
        print(f"[scVI_mouse_human] training seed={seed}")
        latent_all, factors_perm, leiden_all = run_for_seed(
            seed=int(seed),
            n_latent=int(args.n_latent),
            n_top_genes=int(args.n_top_genes),
            leiden_resolution=float(args.leiden_resolution),
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            early_stopping=bool(args.early_stopping),
            early_stopping_patience=int(args.early_stopping_patience),
            check_val_every_n_epoch=int(args.check_val_every_n_epoch),
            precision=str(args.precision),
            results_dir=str(args.results_dir),
            counts_base=counts_base,
            factors_base=factors_base,
        )

        num_cells = factors_perm.shape[0]
        random_idx = _make_permutation(int(seed), num_cells)
        for percentage in percentages:
            print(f"[scVI_mouse_human] eval seed={seed} percentage={percentage}")
            test_idx = _make_test_idx(int(seed), num_cells, float(percentage))

            latent_test = latent_all[test_idx]
            factors_test = factors_perm[test_idx]
            cor = _spearman_correlations(latent_test, factors_test)

            z_for = np.asarray(latent_test)
            y_for = np.asarray(factors_test)
            max_samples = 5000
            if z_for.shape[0] > max_samples:
                rng = np.random.default_rng(int(seed))
                keep = rng.choice(z_for.shape[0], size=max_samples, replace=False)
                z_for = z_for[keep]
                y_for = y_for[keep]

            dci_res = dci_score(z_for, y_for, random_state=int(seed))
            hung_res = hungarian_alignment(z_for, y_for, random_state=int(seed))

            dci_informativeness = np.concatenate(
                [
                    dci_res.informativeness_per_factor,
                    np.array([dci_res.informativeness], dtype=np.float32),
                ]
            )
            dci_disentanglement = np.concatenate(
                [
                    dci_res.disentanglement_per_latent,
                    np.array([dci_res.disentanglement], dtype=np.float32),
                ]
            )
            dci_completeness = np.concatenate(
                [
                    dci_res.completeness_per_factor,
                    np.array([dci_res.completeness], dtype=np.float32),
                ]
            )

            hungarian_matched = np.concatenate(
                [
                    hung_res.matched_scores,
                    np.array([hung_res.matched_mean], dtype=np.float32),
                ]
            )
            hungarian_leakage = np.concatenate(
                [
                    hung_res.leakage_per_factor,
                    np.array([hung_res.leakage_mean], dtype=np.float32),
                ]
            )

            np.save(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_latent_{percentage}_{seed}.npy",
                ),
                latent_test,
            )
            np.savetxt(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_cor_{percentage}_{seed}.csv",
                ),
                cor,
                delimiter=",",
            )

            np.savetxt(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_dci_informativeness_{percentage}_{seed}.csv",
                ),
                dci_informativeness,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_dci_disentanglement_{percentage}_{seed}.csv",
                ),
                dci_disentanglement,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_dci_completeness_{percentage}_{seed}.csv",
                ),
                dci_completeness,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_hungarian_matched_{percentage}_{seed}.csv",
                ),
                hungarian_matched,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_hungarian_leakage_{percentage}_{seed}.csv",
                ),
                hungarian_leakage,
                delimiter=",",
            )

            np.savez_compressed(
                os.path.join(
                    args.results_dir,
                    f"scVI_mouse_human_{percentage}_{seed}.npz",
                ),
                seed=int(seed),
                percentage=float(percentage),
                random_idx=random_idx,
                test_idx=test_idx,
                latent_test=latent_test,
                factors_test=factors_test,
                leiden_test=leiden_all[test_idx],
                dci_informativeness=dci_informativeness,
                dci_disentanglement=dci_disentanglement,
                dci_completeness=dci_completeness,
                dci_importance_matrix=dci_res.importance_matrix,
                hungarian_association_matrix=hung_res.association_matrix,
                hungarian_matched=hungarian_matched,
                hungarian_leakage=hungarian_leakage,
                hungarian_matched_latent_idx=hung_res.matched_latent_idx,
                hungarian_matched_factor_idx=hung_res.matched_factor_idx,
                hungarian_leakage_ratio_mean=np.array(
                    [hung_res.leakage_ratio_mean], dtype=np.float32
                ),
            )

            last_seed = int(seed)
            last_percentage = float(percentage)
            last_leiden_all = leiden_all
            last_latent_all = latent_all
            last_random_idx = random_idx

    if last_seed is not None and last_percentage is not None:
        _ensure_dir("results")
        if last_latent_all is not None:
            np.save("results/scVI_mouse_human_latent.npy", last_latent_all)
        if last_leiden_all is not None:
            import pandas as pd

            pd.DataFrame({"leiden": last_leiden_all}).to_csv(
                "results/scVI_mouse_human.csv", index=False
            )
        if last_random_idx is not None:
            np.save("results/scVI_mouse_human_random_idx.npy", last_random_idx)


if __name__ == "__main__":
    main()
