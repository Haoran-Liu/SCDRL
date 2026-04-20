"""Disentanglement metrics (MIG, SAP) for categorical ground-truth factors.

This repo typically represents latents as real-valued vectors (e.g., logits or
continuous embeddings) and factors as integer-coded categorical variables.

We implement:
- MIG (Mutual Information Gap):
  For each factor k, compute MI(z_j; y_k) across latent dims j, then
  (top1 - top2) / H(y_k). Score is mean over factors.

- SAP (Separated Attribute Predictability):
  For each factor k and latent dim j, train a 1D classifier z_j -> y_k and
  measure test accuracy. SAP is mean over factors of (best - second_best).

These definitions match common usage in disentanglement literature for
categorical factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _as_2d_float_array(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError(
            f"Expected z to be 2D (n_samples, n_latents), got shape {z.shape}"
        )
    if not np.issubdtype(z.dtype, np.floating):
        z = z.astype(np.float32, copy=False)
    return z


def _as_2d_int_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError(
            f"Expected y to be 1D or 2D (n_samples, n_factors), got shape {y.shape}"
        )
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.int64, copy=False)
    return y


def _entropy_discrete(labels: np.ndarray) -> float:
    """Empirical entropy H(Y) in nats for integer-coded labels."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    if values.size <= 1:
        return 0.0
    p = counts.astype(np.float64) / float(labels.size)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@dataclass(frozen=True)
class MIGResult:
    mig: float
    per_factor: np.ndarray  # (n_factors,)
    mi_matrix: np.ndarray  # (n_factors, n_latents)
    entropies: np.ndarray  # (n_factors,)


def mig_score(
    z: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    n_neighbors: int = 3,
) -> MIGResult:
    """Compute MIG for categorical factors.

    Args:
        z: (n_samples, n_latents) float array.
        y: (n_samples, n_factors) int array.
        random_state: Passed to sklearn MI estimator.
        n_neighbors: Passed to mutual_info_classif.

    Returns:
        MIGResult with per-factor gaps and global mean MIG.
    """

    z = _as_2d_float_array(z)
    y = _as_2d_int_array(y)
    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"z and y must have same n_samples, got {z.shape[0]} vs {y.shape[0]}"
        )

    from sklearn.feature_selection import mutual_info_classif

    n_factors = y.shape[1]
    n_latents = z.shape[1]

    mi_matrix = np.zeros((n_factors, n_latents), dtype=np.float64)
    entropies = np.zeros((n_factors,), dtype=np.float64)

    for k in range(n_factors):
        yk = y[:, k]
        entropies[k] = _entropy_discrete(yk)
        if entropies[k] <= 0:
            continue
        # sklearn returns MI in nats.
        mi = mutual_info_classif(
            z,
            yk,
            discrete_features=False,
            random_state=int(random_state),
            n_neighbors=int(n_neighbors),
        )
        mi_matrix[k] = mi

    per_factor = np.zeros((n_factors,), dtype=np.float64)
    valid = entropies > 0
    for k in range(n_factors):
        if not valid[k]:
            per_factor[k] = 0.0
            continue
        row = np.sort(mi_matrix[k])[::-1]
        top1 = float(row[0]) if row.size >= 1 else 0.0
        top2 = float(row[1]) if row.size >= 2 else 0.0
        per_factor[k] = (top1 - top2) / float(entropies[k])

    mig = float(per_factor[valid].mean()) if np.any(valid) else 0.0
    return MIGResult(
        mig=mig,
        per_factor=per_factor.astype(np.float32),
        mi_matrix=mi_matrix.astype(np.float32),
        entropies=entropies.astype(np.float32),
    )


@dataclass(frozen=True)
class SAPResult:
    sap: float
    per_factor: np.ndarray  # (n_factors,)
    score_matrix: np.ndarray  # (n_factors, n_latents)


def sap_score(
    z: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    test_size: float = 0.3,
    max_iter: int = 300,
) -> SAPResult:
    """Compute SAP for categorical factors using 1D multinomial logistic regression.

    For each factor k and each latent dimension j, fit a classifier on z[:, j]
    to predict y[:, k]. Score is accuracy on a held-out split.

    Args:
        z: (n_samples, n_latents) float array.
        y: (n_samples, n_factors) int array.
        random_state: Reproducible data splits.
        test_size: Held-out fraction.
        max_iter: Max iterations for logistic regression.

    Returns:
        SAPResult with per-factor gaps and global mean SAP.
    """

    z = _as_2d_float_array(z)
    y = _as_2d_int_array(y)
    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"z and y must have same n_samples, got {z.shape[0]} vs {y.shape[0]}"
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit

    n_factors = y.shape[1]
    n_latents = z.shape[1]

    score_matrix = np.zeros((n_factors, n_latents), dtype=np.float64)

    for k in range(n_factors):
        yk = y[:, k]
        # If the factor is constant, SAP gap is 0.
        if np.unique(yk).size <= 1:
            continue

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(test_size),
            random_state=int(random_state) + 1000 * int(k),
        )
        (train_idx, test_idx) = next(splitter.split(z, yk))

        y_train = yk[train_idx]
        y_test = yk[test_idx]

        for j in range(n_latents):
            x_train = z[train_idx, j].reshape(-1, 1)
            x_test = z[test_idx, j].reshape(-1, 1)

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=int(max_iter),
                n_jobs=None,
            )

            try:
                clf.fit(x_train, y_train)
                score = float(clf.score(x_test, y_test))
            except Exception:
                score = 0.0

            score_matrix[k, j] = score

    per_factor = np.zeros((n_factors,), dtype=np.float64)
    valid = np.array(
        [np.unique(y[:, k]).size > 1 for k in range(n_factors)], dtype=bool
    )
    for k in range(n_factors):
        if not valid[k]:
            per_factor[k] = 0.0
            continue
        row = np.sort(score_matrix[k])[::-1]
        top1 = float(row[0]) if row.size >= 1 else 0.0
        top2 = float(row[1]) if row.size >= 2 else 0.0
        per_factor[k] = top1 - top2

    sap = float(per_factor[valid].mean()) if np.any(valid) else 0.0
    return SAPResult(
        sap=sap,
        per_factor=per_factor.astype(np.float32),
        score_matrix=score_matrix.astype(np.float32),
    )


def _safe_normalize_rows(mat: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise normalize a non-negative matrix."""
    mat = np.asarray(mat, dtype=np.float64)
    denom = mat.sum(axis=1, keepdims=True)
    denom = np.where(denom > eps, denom, 1.0)
    return mat / denom


def _entropy_from_prob_rows(p: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


@dataclass(frozen=True)
class DCIResult:
    informativeness: float
    disentanglement: float
    completeness: float

    informativeness_per_factor: np.ndarray  # (n_factors,)
    disentanglement_per_latent: np.ndarray  # (n_latents,)
    completeness_per_factor: np.ndarray  # (n_factors,)

    importance_matrix: np.ndarray  # (n_latents, n_factors)


def dci_score(
    z: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    cv_splits: int = 3,
    max_iter: int = 500,
    C: float = 1.0,
) -> DCIResult:
    """Evaluate disentanglement using a DCI-style probe framework.

    This implementation follows the spirit of the DCI framework:
    - Train supervised probes to predict each factor from the full latent vector.
    - Use cross-validated prediction performance as "informativeness".
    - Use absolute probe coefficients as feature importances to compute:
        - disentanglement: each latent should matter for only one factor
        - completeness: each factor should be explained by only one latent

    Notes:
        - This assumes categorical integer-coded factors (common in this repo).
        - For multi-class logistic regression, coefficient importances are
          aggregated across classes via mean(|coef|).
    """

    z = _as_2d_float_array(z)
    y = _as_2d_int_array(y)
    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"z and y must have same n_samples, got {z.shape[0]} vs {y.shape[0]}"
        )

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_latents = z.shape
    n_factors = y.shape[1]

    informativeness_per_factor = np.zeros((n_factors,), dtype=np.float64)
    importance = np.zeros((n_latents, n_factors), dtype=np.float64)

    for k in range(n_factors):
        yk = y[:, k]
        if np.unique(yk).size <= 1:
            continue

        # Informativeness: cross-validated accuracy.
        cv = StratifiedKFold(
            n_splits=int(cv_splits),
            shuffle=True,
            random_state=int(random_state) + 1000 * int(k),
        )
        scores: list[float] = []
        for train_idx, test_idx in cv.split(z, yk):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver="lbfgs",
                    multi_class="auto",
                    max_iter=int(max_iter),
                    C=float(C),
                ),
            )
            try:
                clf.fit(z[train_idx], yk[train_idx])
                scores.append(float(clf.score(z[test_idx], yk[test_idx])))
            except Exception:
                scores.append(0.0)
        informativeness_per_factor[k] = float(np.mean(scores)) if scores else 0.0

        # Importance matrix: fit on full data and use absolute coefficients.
        clf_full = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                multi_class="auto",
                max_iter=int(max_iter),
                C=float(C),
            ),
        )
        try:
            clf_full.fit(z, yk)
            lr = clf_full.named_steps["logisticregression"]
            coef = np.asarray(lr.coef_, dtype=np.float64)
            # coef: (n_classes, n_latents) or (1, n_latents)
            imp_k = np.mean(np.abs(coef), axis=0)
            if imp_k.shape[0] == n_latents:
                importance[:, k] = imp_k
        except Exception:
            # leave zeros
            pass

    # Disentanglement: per-latent distribution over factors.
    p_latent = _safe_normalize_rows(importance)
    ent_latent = _entropy_from_prob_rows(p_latent)
    denom_latent = np.log(float(n_factors)) if n_factors > 1 else 1.0
    disentanglement_per_latent = 1.0 - (ent_latent / denom_latent)
    disentanglement_per_latent = np.clip(disentanglement_per_latent, 0.0, 1.0)

    latent_weights = importance.sum(axis=1)
    if float(latent_weights.sum()) > 0:
        disentanglement = float(
            np.sum(latent_weights * disentanglement_per_latent) / latent_weights.sum()
        )
    else:
        disentanglement = 0.0

    # Completeness: per-factor distribution over latents.
    importance_T = importance.T  # (n_factors, n_latents)
    p_factor = _safe_normalize_rows(importance_T)
    ent_factor = _entropy_from_prob_rows(p_factor)
    denom_factor = np.log(float(n_latents)) if n_latents > 1 else 1.0
    completeness_per_factor = 1.0 - (ent_factor / denom_factor)
    completeness_per_factor = np.clip(completeness_per_factor, 0.0, 1.0)

    factor_weights = importance.sum(axis=0)
    if float(factor_weights.sum()) > 0:
        completeness = float(
            np.sum(factor_weights * completeness_per_factor) / factor_weights.sum()
        )
    else:
        completeness = 0.0

    # Informativeness: average across factors (ignoring constants).
    valid_factors = np.array([np.unique(y[:, k]).size > 1 for k in range(n_factors)])
    informativeness = (
        float(informativeness_per_factor[valid_factors].mean())
        if np.any(valid_factors)
        else 0.0
    )

    return DCIResult(
        informativeness=float(informativeness),
        disentanglement=float(disentanglement),
        completeness=float(completeness),
        informativeness_per_factor=informativeness_per_factor.astype(np.float32),
        disentanglement_per_latent=disentanglement_per_latent.astype(np.float32),
        completeness_per_factor=completeness_per_factor.astype(np.float32),
        importance_matrix=importance.astype(np.float32),
    )


@dataclass(frozen=True)
class HungarianAlignmentResult:
    matched_mean: float
    leakage_mean: float
    leakage_ratio_mean: float

    association_matrix: np.ndarray  # (n_latents, n_factors)
    matched_latent_idx: np.ndarray  # (n_matches,)
    matched_factor_idx: np.ndarray  # (n_matches,)
    matched_scores: np.ndarray  # (n_matches,)
    leakage_per_factor: np.ndarray  # (n_factors,)


def hungarian_alignment(
    z: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 0,
    n_neighbors: int = 3,
) -> HungarianAlignmentResult:
    """One-to-one latent/label alignment via Hungarian matching on MI.

    Builds an association matrix A where A[j, k] = MI(z_j; y_k), then finds
    the assignment that maximizes the total matched association.

    Reports:
        - matched_mean: average MI over matched pairs
        - leakage_mean: average off-diagonal MI for each factor (others -> y_k)
        - leakage_ratio_mean: leakage_mean normalized by matched pair MI
    """

    z = _as_2d_float_array(z)
    y = _as_2d_int_array(y)
    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"z and y must have same n_samples, got {z.shape[0]} vs {y.shape[0]}"
        )

    from sklearn.feature_selection import mutual_info_classif
    from scipy.optimize import linear_sum_assignment

    n_latents = z.shape[1]
    n_factors = y.shape[1]

    association = np.zeros((n_latents, n_factors), dtype=np.float64)
    for k in range(n_factors):
        yk = y[:, k]
        if np.unique(yk).size <= 1:
            continue
        mi = mutual_info_classif(
            z,
            yk,
            discrete_features=False,
            random_state=int(random_state) + 1000 * int(k),
            n_neighbors=int(n_neighbors),
        )
        association[:, k] = mi

    # Maximize association by minimizing negative association.
    row_ind, col_ind = linear_sum_assignment(-association)
    matched_scores = association[row_ind, col_ind]
    matched_mean = float(np.mean(matched_scores)) if matched_scores.size else 0.0

    # Leakage per factor: mean association from *other* latents to that factor.
    leakage_per_factor = np.zeros((n_factors,), dtype=np.float64)
    matched_latent_for_factor = {int(f): int(l) for l, f in zip(row_ind, col_ind)}
    for k in range(n_factors):
        if k not in matched_latent_for_factor:
            # Unmatched factors can happen when n_latents < n_factors.
            vals = association[:, k]
            leakage_per_factor[k] = float(np.mean(vals)) if vals.size else 0.0
            continue
        j_star = matched_latent_for_factor[k]
        vals = np.delete(association[:, k], j_star, axis=0)
        leakage_per_factor[k] = float(np.mean(vals)) if vals.size else 0.0

    leakage_mean = (
        float(np.mean(leakage_per_factor)) if leakage_per_factor.size else 0.0
    )

    # Leakage ratio: off-diagonal vs matched, averaged over matched factors.
    eps = 1e-12
    ratios: list[float] = []
    for j, k in zip(row_ind, col_ind):
        diag = float(association[j, k])
        leak = float(leakage_per_factor[int(k)])
        ratios.append(leak / (diag + eps))
    leakage_ratio_mean = float(np.mean(ratios)) if ratios else 0.0

    return HungarianAlignmentResult(
        matched_mean=matched_mean,
        leakage_mean=leakage_mean,
        leakage_ratio_mean=leakage_ratio_mean,
        association_matrix=association.astype(np.float32),
        matched_latent_idx=np.asarray(row_ind, dtype=np.int64),
        matched_factor_idx=np.asarray(col_ind, dtype=np.int64),
        matched_scores=np.asarray(matched_scores, dtype=np.float32),
        leakage_per_factor=leakage_per_factor.astype(np.float32),
    )
