import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import yaml

import anndata as ad
import scanpy as sc
from scipy.stats import spearmanr

from training import Model
from disentanglement_metrics import dci_score, hungarian_alignment, mig_score, sap_score

ROOT = Path(__file__).resolve().parent


def _build_log_dirs(log_root_dir: str, experiment: str) -> tuple[str, str, str]:
    base_dir = os.path.join(log_root_dir, experiment)
    model_dir = os.path.join(base_dir, "model_dir")
    tensorboard_dir = os.path.join(base_dir, "tensorboard_dir")
    eval_dir = os.path.join(base_dir, "eval_dir")
    return model_dir, tensorboard_dir, eval_dir


def delete_directory_contents(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exc:
            print(f"Failed to delete {file_path}. Reason: {exc}")


def _ensure_results_dir(results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)


def _load_base_data_and_normalize() -> tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(ROOT / "SCDRL_data" / "simulation_data.npz")
    print(npzfile.files)
    counts = npzfile["counts"]
    factors = npzfile["factors"]

    adata = ad.AnnData(counts)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    counts_normalized = adata.X
    del adata

    return counts_normalized, factors


def _build_config(seed: int, n_samples: int, n_genes: int, n_factors: int) -> dict:
    with open(ROOT / "default.yaml", "r", encoding="utf-8") as config_fp:
        config = yaml.safe_load(config_fp)

    config["seed"] = seed
    config["n_samples"] = n_samples
    config["n_genes"] = n_genes
    config["n_factors"] = n_factors
    config["factor_names"] = ["batch", "condition 1", "condition 2", "cell_type"]
    config["factor_sizes"] = [2, 2, 2, 16]
    config["factor_dim"] = 1
    config["residual_dim"] = 8
    config["train"]["n_epochs"] = 20

    return config


def run_experiment(
    *,
    seed: int,
    percentage: float,
    counts_normalized_base: np.ndarray,
    factors_base: np.ndarray,
    model_dir: str,
    tensorboard_dir: str,
    eval_dir: str,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
    val_split: float = 0.0,
    clean_log_dirs: bool = True,
    results_dir: str = "results/SCDRL",
) -> None:
    if clean_log_dirs:
        delete_directory_contents(model_dir)
        delete_directory_contents(tensorboard_dir)
        delete_directory_contents(eval_dir)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    _ensure_results_dir(results_dir)

    np.random.seed(seed)
    num_cells = counts_normalized_base.shape[0]
    num_labels = int(num_cells * percentage)

    random_idx = np.arange(num_cells)
    np.random.shuffle(random_idx)

    counts_normalized = counts_normalized_base[random_idx]
    factors = factors_base[random_idx]

    labeled_idx = np.random.choice(num_cells, num_labels, replace=False)
    test_idx = np.setdiff1d(np.arange(num_cells), labeled_idx)

    label_masks = np.zeros_like(factors, dtype=bool)
    label_masks[labeled_idx] = True

    config = _build_config(
        seed=seed,
        n_samples=counts_normalized.shape[0],
        n_genes=counts_normalized.shape[1],
        n_factors=factors.shape[1],
    )

    config.setdefault("train", {})
    config["train"]["val_split"] = float(val_split)
    config["train"].setdefault("early_stopping", {})
    config["train"]["early_stopping"].update(
        {
            "enabled": bool(early_stopping),
            "patience": int(patience),
            "min_delta": float(min_delta),
        }
    )

    model = Model(config)
    model.train_latent_model(
        counts_normalized,
        factors,
        label_masks,
        model_dir,
        tensorboard_dir,
    )

    # Evaluation
    import torch
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics.cluster import adjusted_rand_score

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    test_data = torch.tensor(
        counts_normalized[test_idx], dtype=torch.float32, device=device
    )
    test_factors = torch.tensor(factors[test_idx], device=device)

    predictions = np.empty([test_factors.shape[0], test_factors.shape[1]])
    latent_representation = torch.empty(test_factors.shape[0], 0, device=device)
    with torch.no_grad():
        for idx in range(config["n_factors"]):
            factor_model = model.latent_model.factor_model
            fm = (
                factor_model.module if hasattr(factor_model, "module") else factor_model
            )
            logits = fm.factor_classifiers[idx](test_data)
            latent_representation = torch.cat((latent_representation, logits), dim=1)
            predictions[:, idx] = logits.argmax(dim=1).cpu().numpy()

    # Disentanglement evaluation: use inferred factor_codes (n_factors * factor_dim)
    # instead of concatenated per-factor logits (sum(factor_sizes)).
    with torch.no_grad():
        factor_model = model.latent_model.factor_model
        fm = factor_model.module if hasattr(factor_model, "module") else factor_model
        factor_model_out = fm(test_data)
        factor_codes_test = factor_model_out["factor_codes"]

    logits_test = latent_representation.detach().cpu().numpy()
    z_test = factor_codes_test.detach().cpu().numpy()
    y_test = test_factors.detach().cpu().numpy()
    mig_res = mig_score(z_test, y_test, random_state=seed)
    sap_res = sap_score(z_test, y_test, random_state=seed)

    dci_res = dci_score(z_test, y_test, random_state=seed)
    hung_res = hungarian_alignment(z_test, y_test, random_state=seed)

    mig = np.concatenate(
        [mig_res.per_factor, np.array([mig_res.mig], dtype=np.float32)]
    )
    sap = np.concatenate(
        [sap_res.per_factor, np.array([sap_res.sap], dtype=np.float32)]
    )

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
        [hung_res.matched_scores, np.array([hung_res.matched_mean], dtype=np.float32)]
    )
    hungarian_leakage = np.concatenate(
        [
            hung_res.leakage_per_factor,
            np.array([hung_res.leakage_mean], dtype=np.float32),
        ]
    )

    np.savetxt("results/SCDRL_simulation_mig.csv", mig, delimiter=",")
    np.savetxt(
        os.path.join(results_dir, f"SCDRL_simulation_mig_{percentage}_{seed}.csv"),
        mig,
        delimiter=",",
    )
    np.savetxt("results/SCDRL_simulation_sap.csv", sap, delimiter=",")
    np.savetxt(
        os.path.join(results_dir, f"SCDRL_simulation_sap_{percentage}_{seed}.csv"),
        sap,
        delimiter=",",
    )

    np.savetxt(
        "results/SCDRL_simulation_dci_informativeness.csv",
        dci_informativeness,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            results_dir,
            f"SCDRL_simulation_dci_informativeness_{percentage}_{seed}.csv",
        ),
        dci_informativeness,
        delimiter=",",
    )
    np.savetxt(
        "results/SCDRL_simulation_dci_disentanglement.csv",
        dci_disentanglement,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            results_dir,
            f"SCDRL_simulation_dci_disentanglement_{percentage}_{seed}.csv",
        ),
        dci_disentanglement,
        delimiter=",",
    )
    np.savetxt(
        "results/SCDRL_simulation_dci_completeness.csv",
        dci_completeness,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            results_dir,
            f"SCDRL_simulation_dci_completeness_{percentage}_{seed}.csv",
        ),
        dci_completeness,
        delimiter=",",
    )

    np.savetxt(
        "results/SCDRL_simulation_hungarian_matched.csv",
        hungarian_matched,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            results_dir,
            f"SCDRL_simulation_hungarian_matched_{percentage}_{seed}.csv",
        ),
        hungarian_matched,
        delimiter=",",
    )
    np.savetxt(
        "results/SCDRL_simulation_hungarian_leakage.csv",
        hungarian_leakage,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(
            results_dir,
            f"SCDRL_simulation_hungarian_leakage_{percentage}_{seed}.csv",
        ),
        hungarian_leakage,
        delimiter=",",
    )

    spearman_correlations = []
    spearman_p_values = []
    mi_scores = []
    for idx in range(config["n_factors"]):
        ground_truth_labels_tmp = test_factors[:, idx].cpu().numpy()
        mi = mutual_info_classif(
            latent_representation.cpu().numpy(), ground_truth_labels_tmp
        )
        mi_scores.append(mi)
        for i in range(latent_representation.shape[1]):
            corr, p_value = spearmanr(
                latent_representation[:, i].cpu().numpy(), ground_truth_labels_tmp
            )
            spearman_correlations.append(corr)
            spearman_p_values.append(p_value)

    spearman_correlations = np.array(spearman_correlations).reshape(
        config["n_factors"], -1
    )
    spearman_p_values = np.array(spearman_p_values).reshape(config["n_factors"], -1)

    np.savetxt("results/SCDRL_simulation_cor.csv", spearman_correlations, delimiter=",")
    np.savetxt(
        os.path.join(results_dir, f"SCDRL_simulation_cor_{percentage}_{seed}.csv"),
        spearman_correlations,
        delimiter=",",
    )

    spearman_correlations_abs = np.abs(spearman_correlations)
    cg_batch = np.mean(spearman_correlations_abs[0][:2]) - np.mean(
        spearman_correlations_abs[0][2:]
    )
    cg_con1 = np.mean(spearman_correlations_abs[1][2:4]) - np.mean(
        np.concatenate(
            (spearman_correlations_abs[1][0:2], spearman_correlations_abs[1][4:]),
            axis=0,
        )
    )
    cg_con2 = np.mean(spearman_correlations_abs[2][4:6]) - np.mean(
        np.concatenate(
            (spearman_correlations_abs[2][0:4], spearman_correlations_abs[2][6:]),
            axis=0,
        )
    )
    cg_cell_type = np.mean(spearman_correlations_abs[3][6:]) - np.mean(
        spearman_correlations_abs[3][:6]
    )
    cg_mean = (cg_batch + cg_con1 + cg_con2 + cg_cell_type) / 4
    cg = [cg_batch, cg_con1, cg_con2, cg_cell_type, cg_mean]

    np.savetxt("results/SCDRL_simulation_cg.csv", cg, delimiter=",")
    np.savetxt(
        os.path.join(results_dir, f"SCDRL_simulation_cg_{percentage}_{seed}.csv"),
        cg,
        delimiter=",",
    )

    performance = np.empty([3, config["n_factors"]])
    for idx in range(config["n_factors"]):
        ground_truth_labels = test_factors[:, idx].cpu().numpy()
        accuracy = accuracy_score(ground_truth_labels, predictions[:, idx])
        f1 = f1_score(ground_truth_labels, predictions[:, idx], average="macro")
        ari = adjusted_rand_score(ground_truth_labels, predictions[:, idx])
        performance[0, idx] = accuracy
        performance[1, idx] = f1
        performance[2, idx] = ari

    print(seed, percentage)
    print(performance)

    path = os.path.join(results_dir, f"SCDRL_simulation_{percentage}_{seed}.npz")
    np.savez(
        path,
        predictions=predictions,
        performance=performance,
        random_idx=random_idx,
        labeled_idx=labeled_idx,
        test_idx=test_idx,
        latent_test=z_test,
        logits_test=logits_test,
        factors_test=y_test,
        spearman_correlations=spearman_correlations,
        spearman_p_values=spearman_p_values,
        mig=mig,
        sap=sap,
        mig_per_factor=mig_res.per_factor,
        sap_per_factor=sap_res.per_factor,
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


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentage", type=float, default=0.05)

    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--percentages", type=float, nargs="*", default=None)
    parser.add_argument(
        "--no_clean_log_dirs",
        action="store_true",
        help="Do not clear log_dir/* between runs.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/SCDRL",
        help="Directory to write per-run results.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="log_dir",
        help="Root directory for logs/checkpoints.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="simulation",
        help="Experiment name (subdirectory under --log_dir).",
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping during training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement in loss to reset patience.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Validation split fraction for early stopping (0 disables).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    model_dir, tensorboard_dir, eval_dir = _build_log_dirs(
        log_root_dir=args.log_dir,
        experiment=args.experiment,
    )

    counts_normalized_base, factors_base = _load_base_data_and_normalize()

    seeds: Iterable[int] = args.seeds if args.seeds else [args.seed]
    percentages: Iterable[float] = (
        args.percentages if args.percentages else [args.percentage]
    )

    for percentage in percentages:
        for seed in seeds:
            run_experiment(
                seed=seed,
                percentage=percentage,
                counts_normalized_base=counts_normalized_base,
                factors_base=factors_base,
                model_dir=model_dir,
                tensorboard_dir=tensorboard_dir,
                eval_dir=eval_dir,
                early_stopping=args.early_stopping,
                patience=args.patience,
                min_delta=args.min_delta,
                val_split=args.val_split,
                clean_log_dirs=not args.no_clean_log_dirs,
                results_dir=args.results_dir,
            )


if __name__ == "__main__":
    main()
