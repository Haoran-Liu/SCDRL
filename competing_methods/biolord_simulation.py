import argparse
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import pandas as pd

import scanpy as sc
import warnings

import biolord

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import anndata as ad

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.feature_selection import mutual_info_classif


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_base_data() -> tuple[np.ndarray, np.ndarray]:
    npzfile = np.load(ROOT / "SCDRL_data" / "simulation_data.npz")
    print(npzfile.files)
    return npzfile["counts"], npzfile["factors"]


def _mig_from_mi_scores(mi: np.ndarray, block: slice) -> float:
    """Compute a simple block-vs-rest MI gap (old Biolord script behavior)."""
    mi = np.asarray(mi)
    in_block = mi[block]
    if block.start == 0 and block.stop == mi.shape[0]:
        return float(np.mean(in_block))
    out_block = np.concatenate([mi[: block.start], mi[block.stop :]], axis=0)
    return float(np.mean(in_block) - np.mean(out_block))


def run_experiment(
    *, seed: int, percentage: float, counts_base: np.ndarray, factors_base: np.ndarray
) -> None:
    np.random.seed(seed)

    num_cells = counts_base.shape[0]
    num_labels = int(num_cells * percentage)

    random_idx = np.arange(num_cells)
    np.random.shuffle(random_idx)

    counts = counts_base[random_idx]
    factors = factors_base[random_idx]

    labeled_idx = np.random.choice(num_cells, num_labels, replace=False)

    train_mask = np.zeros(num_cells, dtype=bool)
    train_mask[labeled_idx] = True
    test_mask = ~train_mask

    adata = ad.AnnData(counts)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    adata.obs["batch"] = factors[:, 0].astype(str)
    adata.obs["condition 1"] = factors[:, 1].astype(str)
    adata.obs["condition 2"] = factors[:, 2].astype(str)
    adata.obs["cell_type"] = factors[:, 3].astype(str)

    adata.obs.loc[test_mask, "batch"] = "Unknown"
    adata.obs.loc[test_mask, "condition 1"] = "Unknown"
    adata.obs.loc[test_mask, "condition 2"] = "Unknown"
    adata.obs.loc[test_mask, "cell_type"] = "Unknown"

    adata.obs["split"] = "train"
    adata.obs.loc[test_mask, "split"] = "test"

    biolord.Biolord.setup_anndata(
        adata,
        categorical_attributes_keys=[
            "batch",
            "condition 1",
            "condition 2",
            "cell_type",
        ],
        categorical_attributes_missing={
            "batch": "Unknown",
            "condition 1": "Unknown",
            "condition 2": "Unknown",
            "cell_type": "Unknown",
        },
    )

    module_params = {
        "decoder_width": 512,
        "decoder_depth": 4,
        "attribute_nn_width": 512,
        "attribute_nn_depth": 4,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "unknown_attribute_noise_param": 1e-1,
        "seed": seed,
        "n_latent_attribute_ordered": 16,
        "n_latent_attribute_categorical": 4,
        "gene_likelihood": "normal",
        "loss_regression": "normal",
        "reconstruction_penalty": 1e1,
        "unknown_attribute_penalty": 1e2,
        "attribute_dropout_rate": 0.05,
        "eval_r2_ordered": False,
        "classifier_penalty": 1e1,
        "classification_penalty": 0,
        "classify_all": False,
        "classifier_dropout_rate": 0.05,
    }

    model = biolord.Biolord(
        adata=adata,
        n_latent=32,
        model_name="simulation",
        train_classifiers=True,
        module_params=module_params,
        split_key="split",
    )

    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-3,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 90,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }

    model.train(
        max_epochs=200,
        batch_size=256,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=10,
        # num_workers=1,
        enable_checkpointing=False,
    )

    # ----- Classifier evaluation on the held-out (test) cells -----
    dataset = model.get_dataset(adata[test_mask])
    module = cast(Any, model.module)
    classification = module.classify(dataset["X"])

    factor_list = ["batch", "condition 1", "condition 2", "cell_type"]

    predictions_df = pd.DataFrame(index=np.arange(dataset["X"].shape[0]))
    latent_representation = np.empty((dataset["X"].shape[0], 0), dtype=np.float32)
    blocks: list[slice] = []
    col = 0
    for name in factor_list:
        logits = classification[name].detach().cpu().numpy()
        predictions_df[name] = logits.argmax(axis=1)
        latent_representation = np.concatenate((latent_representation, logits), axis=1)
        blocks.append(slice(col, col + int(logits.shape[1])))
        col += int(logits.shape[1])

    predictions = predictions_df.to_numpy()

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics.cluster import adjusted_rand_score

    performance = np.empty([3, len(factor_list)])
    for idx in range(len(factor_list)):
        ground_truth_labels = factors[:, idx][test_mask]
        accuracy = accuracy_score(ground_truth_labels, predictions[:, idx])
        f1 = f1_score(ground_truth_labels, predictions[:, idx], average="macro")
        ari = adjusted_rand_score(ground_truth_labels, predictions[:, idx])
        performance[0, idx] = accuracy
        performance[1, idx] = f1
        performance[2, idx] = ari

    print(seed, percentage)
    print(performance)

    # ----- Correlation/MIG on the classifier-derived representation (original behavior) -----
    latent_test = latent_representation
    logits_test = latent_representation
    latent_source = "classifier_logits"

    spearman_correlations = []
    spearman_p_values = []
    for idx in range(len(factor_list)):
        ground_truth_labels_tmp = factors[:, idx][test_mask]
        for i in range(latent_test.shape[1]):
            corr, p_value = spearmanr(latent_test[:, i], ground_truth_labels_tmp)
            spearman_correlations.append(corr)
            spearman_p_values.append(p_value)

    spearman_correlations = np.array(spearman_correlations).reshape(
        len(factor_list), -1
    )
    spearman_p_values = np.array(spearman_p_values).reshape(len(factor_list), -1)

    _ensure_dir("results")
    _ensure_dir("results/biolord")

    np.savetxt(
        "results/biolord_simulation_cor.csv", spearman_correlations, delimiter=","
    )
    np.savetxt(
        f"results/biolord/biolord_simulation_cor_{percentage}_{seed}.csv",
        spearman_correlations,
        delimiter=",",
    )

    y_test = factors[test_mask]

    mi_scores: list[np.ndarray] = []
    mig_per_factor: list[float] = []
    for idx in range(len(factor_list)):
        ground_truth_labels_tmp = factors[:, idx][test_mask]
        mi = mutual_info_classif(latent_test, ground_truth_labels_tmp)
        mi_scores.append(mi)
        mig_per_factor.append(_mig_from_mi_scores(mi, blocks[idx]))

    mig_mean = float(np.mean(mig_per_factor)) if mig_per_factor else float("nan")
    mig = np.asarray([*mig_per_factor, mig_mean], dtype=np.float32)

    np.savetxt("results/biolord_simulation_mig.csv", mig, delimiter=",")
    np.savetxt(
        f"results/biolord/biolord_simulation_mig_{percentage}_{seed}.csv",
        mig,
        delimiter=",",
    )

    _ensure_dir("results/biolord")
    path = f"results/biolord/biolord_simulation_{percentage}_{seed}.npz"
    np.savez(
        path,
        predictions=predictions,
        performance=performance,
        random_idx=random_idx,
        labeled_idx=labeled_idx,
        test_idx=np.flatnonzero(test_mask),
        latent_test=latent_test,
        logits_test=logits_test,
        latent_source=latent_source,
        factors_test=y_test,
        spearman_correlations=spearman_correlations,
        spearman_p_values=spearman_p_values,
        mig=mig,
        mi_scores=np.stack(mi_scores, axis=0) if mi_scores else np.empty((0, 0)),
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentage", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--percentages", type=float, nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    counts_base, factors_base = _load_base_data()

    seeds: Iterable[int] = args.seeds if args.seeds else [args.seed]
    percentages: Iterable[float] = (
        args.percentages if args.percentages else [args.percentage]
    )

    for percentage in percentages:
        for seed in seeds:
            run_experiment(
                seed=seed,
                percentage=percentage,
                counts_base=counts_base,
                factors_base=factors_base,
            )


if __name__ == "__main__":
    main()
