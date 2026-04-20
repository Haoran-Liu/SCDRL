"""Recompute Biolord MIG/SAP from saved NPZ artifacts.

Why this exists:
Biolord encodes each categorical attribute in a small vector (commonly 4 dims)
per factor. Standard per-dimension MIG can look near-zero even when Biolord is
cleanly factor-aligned, because MI is tied across the 4 dims within each block.

This script updates results/biolord/biolord_*_{pct}_{seed}.npz in-place:
- computes MIG/SAP on a reduced latent with 1 dim per factor (first dim per block)
- writes back `latent_test_for_metrics`, `latent_metrics_source`, `mig`, `sap`,
  `mig_per_factor`, `sap_per_factor`

After running this, regenerate performance CSVs (e.g. simulation_performance.py)
so performance/*/{dataset}_{pct}_{seed}.csv pick up the updated MIG/SAP.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure we can import the repo's metrics implementation.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disentanglement_metrics import mig_score, sap_score  # noqa: E402


def _reduce_blocks(
    z: np.ndarray, *, n_factors: int, block_size: int
) -> Tuple[np.ndarray, str]:
    z = np.asarray(z)
    if z.ndim != 2 or z.size == 0:
        return (z, "as_is")
    if n_factors <= 0 or block_size <= 0:
        return (z, "as_is")
    expected = n_factors * block_size
    if z.shape[1] != expected:
        # Robust fallback: if the latent dimensionality is divisible by the
        # number of factors, assume contiguous per-factor blocks and infer size.
        if z.shape[1] % n_factors != 0:
            return (z, "as_is")
        inferred = int(z.shape[1] // n_factors)
        if inferred <= 1:
            return (z, "as_is")
        keep_cols = [i * inferred for i in range(n_factors)]
        return (z[:, keep_cols], f"block_first(k={block_size}->infer{inferred})")

    keep_cols = [i * block_size for i in range(n_factors)]
    return (z[:, keep_cols], f"block_first(k={block_size})")


def _expected_index_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] == 0:
        return np.empty((logits.shape[0], 0), dtype=np.float32)
    x = logits.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(x)
    p = p / np.sum(p, axis=1, keepdims=True)
    idx = np.arange(x.shape[1], dtype=np.float32)[None, :]
    return np.sum(p * idx, axis=1, keepdims=True)


def _latent_from_logits_expected_index(
    logits_concat: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, str]:
    """Build a 1D code per factor from concatenated classifier logits.

    We infer per-factor class counts from `y` (assumes labels are 0..K-1).
    """

    logits_concat = np.asarray(logits_concat)
    y = np.asarray(y)
    if logits_concat.ndim != 2 or y.ndim != 2:
        return (np.asarray(logits_concat), "stage1_logits_expected_index(as_is)")

    # Desired class sizes from y (assumes labels are 0..K-1), but Biolord's
    # classifier output dimensionality can differ if some categories were absent
    # from the anndata setup for that run. To stay robust, we clamp early factor
    # sizes and allocate the remaining logits to the last factor(s).
    desired_sizes: List[int] = [int(y[:, i].max() + 1) for i in range(y.shape[1])]

    logits_dim = int(logits_concat.shape[1])
    n_factors = int(y.shape[1])
    class_sizes: List[int] = []
    remaining = logits_dim
    for i in range(n_factors):
        if i < n_factors - 1:
            want = int(desired_sizes[i])
            # leave at least 1 logit for each remaining factor
            max_allowed = remaining - (n_factors - i - 1)
            k = max(1, min(want, max_allowed))
        else:
            k = remaining
        class_sizes.append(int(k))
        remaining -= int(k)

    if remaining != 0 or sum(class_sizes) != logits_dim:
        return (np.asarray(logits_concat), "stage1_logits_expected_index(as_is;bad_split)")

    zs: List[np.ndarray] = []
    start = 0
    for k in class_sizes:
        sl = slice(start, start + k)
        start += k
        zs.append(_expected_index_from_logits(logits_concat[:, sl]))

    tag = f"stage1_logits_expected_index(sizes={class_sizes})"
    return (np.concatenate(zs, axis=1), tag)


def _update_npz(
    path: str, *, seed: int, block_size: int, mode: str, dry_run: bool
) -> Tuple[float, float]:
    d = np.load(path, allow_pickle=True)
    obj: Dict[str, Any] = {k: d[k] for k in d.files}

    y = obj.get("factors_test")
    if y is None:
        raise KeyError("missing factors_test")

    y = np.asarray(y)
    n_factors = int(y.shape[1]) if y.ndim == 2 else 1

    if mode == "latent_blocks":
        z = obj.get("latent_test")
        if z is None:
            raise KeyError("missing latent_test")
        z_for, reduce_tag = _reduce_blocks(
            np.asarray(z), n_factors=n_factors, block_size=block_size
        )
    elif mode == "logits_expected_index":
        logits = obj.get("logits_test")
        if logits is None:
            raise KeyError("missing logits_test")
        z_for, reduce_tag = _latent_from_logits_expected_index(np.asarray(logits), y)
    else:
        raise ValueError(f"unknown mode: {mode}")

    m = mig_score(z_for, y, random_state=seed)
    s = sap_score(z_for, y, random_state=seed)

    obj["latent_test_for_metrics"] = np.asarray(z_for, dtype=np.float32)
    obj["latent_metrics_source"] = np.array(str(obj.get("latent_source", "")) + "|" + reduce_tag)
    obj["mig"] = np.concatenate([m.per_factor, np.array([m.mig], dtype=np.float32)])
    obj["sap"] = np.concatenate([s.per_factor, np.array([s.sap], dtype=np.float32)])
    obj["mig_per_factor"] = m.per_factor
    obj["sap_per_factor"] = s.per_factor

    if not dry_run:
        np.savez(path, **obj)

    return (float(obj["mig"][-1]), float(obj["sap"][-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="results/biolord/biolord_*_*.npz",
        help="Glob for biolord NPZ files to update",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4,
        help="Per-factor latent block size (Biolord default is typically 4)",
    )
    parser.add_argument(
        "--mode",
        choices=["latent_blocks", "logits_expected_index"],
        default="latent_blocks",
        help=(
            "How to derive the latent used for MIG/SAP. "
            "latent_blocks uses latent_test with per-factor block reduction; "
            "logits_expected_index builds 1D-per-factor codes from logits_test."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and print, but do not rewrite NPZ files",
    )
    parser.add_argument(
        "--seed_fallback",
        type=int,
        default=0,
        help="Seed used when it cannot be parsed from filename",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    updated = 0
    skipped = 0
    failed = 0

    for path in paths:
        base = os.path.basename(path)
        # filenames look like: biolord_{dataset}_{pct}_{seed}.npz
        seed = args.seed_fallback
        try:
            seed = int(os.path.splitext(base)[0].split("_")[-1])
        except Exception:
            pass

        try:
            mig, sap = _update_npz(
                path,
                seed=seed,
                block_size=int(args.block_size),
                mode=str(args.mode),
                dry_run=bool(args.dry_run),
            )
            action = "DRY" if args.dry_run else "OK"
            print(f"[{action}] {path}: MIG_mean={mig:.6f} SAP_mean={sap:.6f}")
            updated += 1
        except KeyError:
            print(f"[SKIP] {path}: no latent_test/factors_test in NPZ")
            skipped += 1
        except Exception as e:
            print(f"[FAIL] {path}: {e}")
            failed += 1

    print(f"done: updated={updated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
