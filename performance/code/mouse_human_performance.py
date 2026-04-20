import argparse
import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_method_csvs(
    dataset: str, percentage: float, seed: int, df: pd.DataFrame
) -> None:
    for method in df.columns:
        out_dir = os.path.join("performance", str(method))
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{dataset}_{percentage}_{seed}.csv")
        mdf = df[[method]].copy()
        keep_always = {
            "MIG_mean",
            "SAP_mean",
            "DCI_informativeness_mean",
            "DCI_disentanglement_mean",
            "DCI_completeness_mean",
            "Hungarian_matched_mean",
            "Hungarian_leakage_mean",
            "Hungarian_leakage_ratio_mean",
        }
        always_mask = mdf.index.astype(str).isin(keep_always)
        mdf = mdf.loc[always_mask | mdf[method].notna()]
        mdf.to_csv(out_path, na_rep="NA")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disentanglement_metrics import dci_score, hungarian_alignment, mig_score, sap_score


def _load_scvi_npz(*, percentage: float, seed: int):
    path = os.path.join("results", "scVI", f"scVI_mouse_human_{percentage}_{seed}.npz")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


SEURAT_DIR = os.path.join("results", "seurat")


def _load_seurat_predictions(*, percentage: float, seed: int) -> pd.DataFrame:
    tagged = os.path.join(SEURAT_DIR, f"Seurat_mouse_human_{percentage}_{seed}.csv")
    base_new = os.path.join(SEURAT_DIR, "Seurat_mouse_human.csv")
    base_legacy = os.path.join("results", "Seurat_mouse_human.csv")
    if os.path.exists(tagged):
        return pd.read_csv(tagged)
    if os.path.exists(base_new):
        return pd.read_csv(base_new)
    return pd.read_csv(base_legacy)


def _load_seurat_pca(*, percentage: float, seed: int):
    tagged = os.path.join(SEURAT_DIR, f"Seurat_mouse_human_pca_{percentage}_{seed}.npy")
    base_new = os.path.join(SEURAT_DIR, "Seurat_mouse_human_pca.npy")
    base_legacy = os.path.join("results", "Seurat_mouse_human_pca.npy")
    try:
        if os.path.exists(tagged):
            return np.load(tagged)
        if os.path.exists(base_new):
            return np.load(base_new)
        return np.load(base_legacy)
    except FileNotFoundError:
        return None


def _clusters_to_onehot(labels: pd.Series) -> np.ndarray:
    df = pd.get_dummies(labels.astype(str), dtype=np.float32)
    return df.to_numpy()


def _compute_mig_sap_from_latent(
    z: np.ndarray, y: np.ndarray, *, seed: int
) -> tuple[float, float]:
    if z is None or y is None:
        return (np.nan, np.nan)
    z = np.asarray(z)
    y = np.asarray(y)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.shape[0] != y.shape[0] or z.shape[0] == 0:
        return (np.nan, np.nan)
    max_samples = 5000
    if z.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(z.shape[0], size=max_samples, replace=False)
        z = z[keep]
        y = y[keep]

    try:
        mig_val = float(mig_score(z, y, random_state=seed).mig)
        sap_val = float(sap_score(z, y, random_state=seed).sap)
    except Exception:
        return (np.nan, np.nan)

    return (mig_val, sap_val)


def _compute_mig_sap_from_npz(
    npz: "np.lib.npyio.NpzFile", *, seed: int
) -> tuple[float, float]:
    mig_arr = npz.get("mig", None)
    sap_arr = npz.get("sap", None)

    # Prefer precomputed metrics saved by each method script.
    if mig_arr is not None and sap_arr is not None:
        mig_mean = float(np.ravel(mig_arr)[-1])
        sap_mean = float(np.ravel(sap_arr)[-1])
        return (mig_mean, sap_mean)

    # Fallback: compute from saved latent/factors.
    z = npz.get("latent_test", None)
    y = npz.get("factors_test", None)
    if z is not None and y is not None:
        return _compute_mig_sap_from_latent(z, y, seed=seed)

    return (np.nan, np.nan)


def _compute_dci_hungarian_from_latent(
    z: np.ndarray, y: np.ndarray, *, seed: int
) -> tuple[float, float, float, float, float, float]:
    if z is None or y is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    z = np.asarray(z)
    y = np.asarray(y)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.shape[0] != y.shape[0] or z.shape[0] == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    max_samples = 5000
    if z.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(z.shape[0], size=max_samples, replace=False)
        z = z[keep]
        y = y[keep]

    try:
        d = dci_score(z, y, random_state=seed)
        h = hungarian_alignment(z, y, random_state=seed)
        return (
            float(d.informativeness),
            float(d.disentanglement),
            float(d.completeness),
            float(h.matched_mean),
            float(h.leakage_mean),
            float(h.leakage_ratio_mean),
        )
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def _compute_dci_hungarian_from_npz(
    npz: "np.lib.npyio.NpzFile", *, seed: int
) -> tuple[float, float, float, float, float, float]:
    inf_arr = npz.get("dci_informativeness", None)
    dis_arr = npz.get("dci_disentanglement", None)
    comp_arr = npz.get("dci_completeness", None)
    hung_m = npz.get("hungarian_matched", None)
    hung_l = npz.get("hungarian_leakage", None)
    hung_r = npz.get("hungarian_leakage_ratio_mean", None)

    if (
        inf_arr is not None
        and dis_arr is not None
        and comp_arr is not None
        and hung_m is not None
        and hung_l is not None
    ):
        inf_mean = float(np.ravel(inf_arr)[-1])
        dis_mean = float(np.ravel(dis_arr)[-1])
        comp_mean = float(np.ravel(comp_arr)[-1])
        hung_match = float(np.ravel(hung_m)[-1])
        hung_leak = float(np.ravel(hung_l)[-1])
        hung_ratio = float(np.ravel(hung_r)[0]) if hung_r is not None else np.nan
        return (inf_mean, dis_mean, comp_mean, hung_match, hung_leak, hung_ratio)

    z = npz.get("latent_test", None)
    y = npz.get("factors_test", None)
    if z is not None and y is not None:
        return _compute_dci_hungarian_from_latent(z, y, seed=seed)

    return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--percentage", type=float, default=0.05)
parser.add_argument("--seeds", type=int, nargs="*", default=None)
parser.add_argument("--percentages", type=float, nargs="*", default=None)
args = parser.parse_args()

SEEDS = args.seeds if args.seeds else [args.seed]
PERCENTAGES = args.percentages if args.percentages else [args.percentage]

npzfile = np.load(os.path.join(REPO_ROOT, "SCDRL_data", "mouse_human.npz"))
print(npzfile.files)
factors_base = npzfile["factors"]

try:
    scvi_latent_all = np.load("results/scVI_mouse_human_latent.npy")
except FileNotFoundError:
    scvi_latent_all = None


for PERCENTAGE in PERCENTAGES:
    for SEED in SEEDS:
        np.random.seed(SEED)

        num_cells = factors_base.shape[0]
        num_labels = int(num_cells * PERCENTAGE)

        random_idx = np.arange(num_cells)
        np.random.shuffle(random_idx)
        factors = factors_base[random_idx]

        labeled_idx = np.random.choice(num_cells, num_labels, replace=False)
        test_idx = np.setdiff1d(np.arange(num_cells), labeled_idx)

        # SCDRL
        PATH = f"results/SCDRL/SCDRL_mouse_human_{PERCENTAGE}_{SEED}.npz"
        if not os.path.exists(PATH):
            print(
                f"[warn] missing {PATH}; skipping percentage={PERCENTAGE} seed={SEED}"
            )
            continue
        npz_sc = np.load(PATH)
        SCDRL_predictions = npz_sc["predictions"]
        SCDRL_mig_mean, SCDRL_sap_mean = _compute_mig_sap_from_npz(npz_sc, seed=SEED)
        (
            SCDRL_dci_informativeness_mean,
            SCDRL_dci_disentanglement_mean,
            SCDRL_dci_completeness_mean,
            SCDRL_hungarian_matched_mean,
            SCDRL_hungarian_leakage_mean,
            SCDRL_hungarian_leakage_ratio_mean,
        ) = _compute_dci_hungarian_from_npz(npz_sc, seed=SEED)

        SCDRL_accuracy_system = accuracy_score(
            factors[test_idx, 0], SCDRL_predictions[:, 0]
        )
        SCDRL_accuracy_cell_type = accuracy_score(
            factors[test_idx, 1], SCDRL_predictions[:, 1]
        )
        SCDRL_f1_system = f1_score(
            factors[test_idx, 0], SCDRL_predictions[:, 0], average="macro"
        )
        SCDRL_f1_cell_type = f1_score(
            factors[test_idx, 1], SCDRL_predictions[:, 1], average="macro"
        )
        SCDRL_ARI_system = adjusted_rand_score(
            factors[test_idx, 0], SCDRL_predictions[:, 0]
        )
        SCDRL_ARI_cell_type = adjusted_rand_score(
            factors[test_idx, 1], SCDRL_predictions[:, 1]
        )

        # biolord
        PATH = f"results/biolord/biolord_mouse_human_{PERCENTAGE}_{SEED}.npz"
        if os.path.exists(PATH):
            npz_bl = np.load(PATH)
            biolord_predictions = npz_bl["predictions"]
            biolord_mig_mean, biolord_sap_mean = _compute_mig_sap_from_npz(
                npz_bl, seed=SEED
            )
            (
                biolord_dci_informativeness_mean,
                biolord_dci_disentanglement_mean,
                biolord_dci_completeness_mean,
                biolord_hungarian_matched_mean,
                biolord_hungarian_leakage_mean,
                biolord_hungarian_leakage_ratio_mean,
            ) = _compute_dci_hungarian_from_npz(npz_bl, seed=SEED)

            biolord_accuracy_system = accuracy_score(
                factors[test_idx, 0], biolord_predictions[:, 0]
            )
            biolord_accuracy_cell_type = accuracy_score(
                factors[test_idx, 1], biolord_predictions[:, 1]
            )
            biolord_f1_system = f1_score(
                factors[test_idx, 0], biolord_predictions[:, 0], average="macro"
            )
            biolord_f1_cell_type = f1_score(
                factors[test_idx, 1], biolord_predictions[:, 1], average="macro"
            )
            biolord_ARI_system = adjusted_rand_score(
                factors[test_idx, 0], biolord_predictions[:, 0]
            )
            biolord_ARI_cell_type = adjusted_rand_score(
                factors[test_idx, 1], biolord_predictions[:, 1]
            )
        else:
            print(
                f"[warn] missing {PATH}; writing NaNs for biolord at percentage={PERCENTAGE} seed={SEED}"
            )
            biolord_mig_mean = np.nan
            biolord_sap_mean = np.nan
            biolord_dci_informativeness_mean = np.nan
            biolord_dci_disentanglement_mean = np.nan
            biolord_dci_completeness_mean = np.nan
            biolord_hungarian_matched_mean = np.nan
            biolord_hungarian_leakage_mean = np.nan
            biolord_hungarian_leakage_ratio_mean = np.nan
            biolord_accuracy_system = np.nan
            biolord_accuracy_cell_type = np.nan
            biolord_f1_system = np.nan
            biolord_f1_cell_type = np.nan
            biolord_ARI_system = np.nan
            biolord_ARI_cell_type = np.nan

        scvi_npz = _load_scvi_npz(percentage=PERCENTAGE, seed=SEED)
        if scvi_npz is not None and "leiden_test" in scvi_npz.files:
            scVI_ARI_cell_type = adjusted_rand_score(
                factors[test_idx, 1], scvi_npz["leiden_test"]
            )
        else:
            scVI_predictions_base = pd.read_csv("results/scVI_mouse_human.csv")
            scVI_predictions = scVI_predictions_base.iloc[random_idx].reset_index()
            scVI_ARI_cell_type = adjusted_rand_score(
                factors[test_idx, 1], scVI_predictions.loc[test_idx, "leiden"]
            )

        Seurat_predictions_base = _load_seurat_predictions(
            percentage=PERCENTAGE, seed=SEED
        )
        Seurat_predictions = Seurat_predictions_base.iloc[random_idx].reset_index()
        Seurat_ARI_cell_type = adjusted_rand_score(
            factors[test_idx, 1], Seurat_predictions.loc[test_idx, "seurat_clusters"]
        )

        if scvi_npz is not None:
            scvi_mig, scvi_sap = _compute_mig_sap_from_npz(scvi_npz, seed=SEED)
            (
                scvi_dci_informativeness_mean,
                scvi_dci_disentanglement_mean,
                scvi_dci_completeness_mean,
                scvi_hungarian_matched_mean,
                scvi_hungarian_leakage_mean,
                scvi_hungarian_leakage_ratio_mean,
            ) = _compute_dci_hungarian_from_npz(scvi_npz, seed=SEED)
        elif scvi_latent_all is not None:
            scvi_latent = scvi_latent_all[random_idx][test_idx]
            scvi_y = factors[test_idx]
            scvi_mig, scvi_sap = _compute_mig_sap_from_latent(
                scvi_latent, scvi_y, seed=SEED
            )
            (
                scvi_dci_informativeness_mean,
                scvi_dci_disentanglement_mean,
                scvi_dci_completeness_mean,
                scvi_hungarian_matched_mean,
                scvi_hungarian_leakage_mean,
                scvi_hungarian_leakage_ratio_mean,
            ) = _compute_dci_hungarian_from_latent(scvi_latent, scvi_y, seed=SEED)
        else:
            scvi_mig = np.nan
            scvi_sap = np.nan
            scvi_dci_informativeness_mean = np.nan
            scvi_dci_disentanglement_mean = np.nan
            scvi_dci_completeness_mean = np.nan
            scvi_hungarian_matched_mean = np.nan
            scvi_hungarian_leakage_mean = np.nan
            scvi_hungarian_leakage_ratio_mean = np.nan

        seurat_pca_all = _load_seurat_pca(percentage=PERCENTAGE, seed=SEED)
        if seurat_pca_all is not None:
            seurat_latent = seurat_pca_all[random_idx][test_idx]
        else:
            seurat_latent = _clusters_to_onehot(
                Seurat_predictions.loc[:, "seurat_clusters"]
            )
            seurat_latent = seurat_latent[test_idx]
        seurat_y = factors[test_idx]
        seurat_mig, seurat_sap = _compute_mig_sap_from_latent(
            seurat_latent, seurat_y, seed=SEED
        )
        (
            seurat_dci_informativeness_mean,
            seurat_dci_disentanglement_mean,
            seurat_dci_completeness_mean,
            seurat_hungarian_matched_mean,
            seurat_hungarian_leakage_mean,
            seurat_hungarian_leakage_ratio_mean,
        ) = _compute_dci_hungarian_from_latent(seurat_latent, seurat_y, seed=SEED)

        performance = {
            "SCDRL": {
                "accuracy_system": SCDRL_accuracy_system,
                "accuracy_cell_type": SCDRL_accuracy_cell_type,
                "f1_system": SCDRL_f1_system,
                "f1_cell_type": SCDRL_f1_cell_type,
                "ARI_system": SCDRL_ARI_system,
                "ARI_cell_type": SCDRL_ARI_cell_type,
                "MIG_mean": float(SCDRL_mig_mean),
                "SAP_mean": float(SCDRL_sap_mean),
                "DCI_informativeness_mean": float(SCDRL_dci_informativeness_mean),
                "DCI_disentanglement_mean": float(SCDRL_dci_disentanglement_mean),
                "DCI_completeness_mean": float(SCDRL_dci_completeness_mean),
                "Hungarian_matched_mean": float(SCDRL_hungarian_matched_mean),
                "Hungarian_leakage_mean": float(SCDRL_hungarian_leakage_mean),
                "Hungarian_leakage_ratio_mean": float(
                    SCDRL_hungarian_leakage_ratio_mean
                ),
            },
            "biolord": {
                "accuracy_system": biolord_accuracy_system,
                "accuracy_cell_type": biolord_accuracy_cell_type,
                "f1_system": biolord_f1_system,
                "f1_cell_type": biolord_f1_cell_type,
                "ARI_system": biolord_ARI_system,
                "ARI_cell_type": biolord_ARI_cell_type,
                "MIG_mean": float(biolord_mig_mean),
                "SAP_mean": float(biolord_sap_mean),
                "DCI_informativeness_mean": float(biolord_dci_informativeness_mean),
                "DCI_disentanglement_mean": float(biolord_dci_disentanglement_mean),
                "DCI_completeness_mean": float(biolord_dci_completeness_mean),
                "Hungarian_matched_mean": float(biolord_hungarian_matched_mean),
                "Hungarian_leakage_mean": float(biolord_hungarian_leakage_mean),
                "Hungarian_leakage_ratio_mean": float(
                    biolord_hungarian_leakage_ratio_mean
                ),
            },
            "scVI": {
                "ARI_cell_type": scVI_ARI_cell_type,
                "MIG_mean": float(scvi_mig),
                "SAP_mean": float(scvi_sap),
                "DCI_informativeness_mean": float(scvi_dci_informativeness_mean),
                "DCI_disentanglement_mean": float(scvi_dci_disentanglement_mean),
                "DCI_completeness_mean": float(scvi_dci_completeness_mean),
                "Hungarian_matched_mean": float(scvi_hungarian_matched_mean),
                "Hungarian_leakage_mean": float(scvi_hungarian_leakage_mean),
                "Hungarian_leakage_ratio_mean": float(
                    scvi_hungarian_leakage_ratio_mean
                ),
            },
            "Seurat": {
                "ARI_cell_type": Seurat_ARI_cell_type,
                "MIG_mean": float(seurat_mig),
                "SAP_mean": float(seurat_sap),
                "DCI_informativeness_mean": float(seurat_dci_informativeness_mean),
                "DCI_disentanglement_mean": float(seurat_dci_disentanglement_mean),
                "DCI_completeness_mean": float(seurat_dci_completeness_mean),
                "Hungarian_matched_mean": float(seurat_hungarian_matched_mean),
                "Hungarian_leakage_mean": float(seurat_hungarian_leakage_mean),
                "Hungarian_leakage_ratio_mean": float(
                    seurat_hungarian_leakage_ratio_mean
                ),
            },
        }

        performance = pd.DataFrame(performance)

        _write_method_csvs(
            dataset="mouse_human", percentage=PERCENTAGE, seed=SEED, df=performance
        )
