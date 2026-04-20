#!/usr/bin/env python3
"""Collect disentanglement metrics from per-method performance CSVs and plot them.

This script is intentionally decoupled from training: it reads files like
    performance/{method}/{dataset}_{percentage}_{seed}.csv
and extracts metric rows (by default MIG/SAP + DCI + Hungarian).

Outputs:
- performance/summary_tables/disentanglement_summary.csv (tidy table)
- plot/line/disentanglement_{dataset}.png
- plot/line/disentanglement_all.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("simulation", "mouse_human", "haniffa")
DEFAULT_METRICS = (
    "MIG_mean",
    "SAP_mean",
    "DCI_informativeness_mean",
    "DCI_disentanglement_mean",
    "DCI_completeness_mean",
    "Hungarian_matched_mean",
    "Hungarian_leakage_mean",
    "Hungarian_leakage_ratio_mean",
)
PLOT_KINDS = ("auto", "line", "bar")


@dataclass(frozen=True)
class RunKey:
    dataset: str
    percentage: float
    seed: int


def _parse_perf_filename(path: str) -> RunKey | None:
    base = os.path.basename(path)
    m = re.fullmatch(r"(simulation|mouse_human|haniffa)_([0-9.]+)_([0-9]+)\.csv", base)
    if not m:
        return None
    dataset, pct_s, seed_s = m.group(1), m.group(2), m.group(3)
    try:
        return RunKey(dataset=dataset, percentage=float(pct_s), seed=int(seed_s))
    except ValueError:
        return None


def _read_metric_rows(path: str) -> pd.DataFrame:
    """Read a performance CSV and return a frame indexed by metric name."""
    df = pd.read_csv(path, index_col=0)
    # Normalize column names (methods)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    return df


def collect_disentanglement(
    perf_dir: str, datasets: Iterable[str], metrics: Iterable[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    metrics_set = set(metrics)

    # Preferred layout: performance/{method}/{dataset}_{percentage}_{seed}.csv
    for csv_path in sorted(glob.glob(os.path.join(perf_dir, "*", "*.csv"))):
        method = os.path.basename(os.path.dirname(csv_path))
        key = _parse_perf_filename(csv_path)
        if key is None or key.dataset not in set(datasets):
            continue

        df = _read_metric_rows(csv_path)
        # Per-method files should contain exactly one column; tolerate mismatches.
        if df.shape[1] == 1:
            series = df.iloc[:, 0]
        elif method in df.columns:
            series = df[method]
        else:
            continue

        for metric in metrics_set:
            if metric not in series.index:
                continue
            v = pd.to_numeric(series.loc[metric], errors="coerce")
            rows.append(
                {
                    "dataset": key.dataset,
                    "percentage": key.percentage,
                    "seed": key.seed,
                    "method": method,
                    "metric": metric,
                    "value": float(v) if pd.notna(v) else np.nan,
                    "source_csv": os.path.relpath(csv_path, start=os.getcwd()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Keep only disentanglement metrics
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["percentage"] = out["percentage"].astype(float)
    out["seed"] = out["seed"].astype(int)
    out["value"] = out["value"].astype(float)
    return out


def _apply_filters(
    df: pd.DataFrame, seed: int | None, percentage: float | None
) -> pd.DataFrame:
    if df.empty:
        return df
    if seed is not None:
        df = df[df["seed"] == int(seed)]
    if percentage is not None:
        df = df[np.isclose(df["percentage"].astype(float), float(percentage))]
    return df.reset_index(drop=True)


def _infer_plot_kind(df: pd.DataFrame, plot_kind: str) -> str:
    plot_kind = str(plot_kind).lower().strip()
    if plot_kind not in PLOT_KINDS:
        raise ValueError(f"plot_kind must be one of {PLOT_KINDS}, got {plot_kind!r}")
    if plot_kind != "auto":
        return plot_kind
    if df.empty:
        return "line"
    unique_pcts = sorted(df["percentage"].astype(float).unique())
    return "bar" if len(unique_pcts) <= 1 else "line"


def _plot_dataset_bar(
    df: pd.DataFrame,
    dataset: str,
    out_dir: str,
    metrics: list[str],
    suffix: str = "",
) -> str | None:
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return None

    unique_pcts = sorted(sub["percentage"].astype(float).unique())
    pct_label = ""
    if len(unique_pcts) == 1:
        pct_label = f" (pct={unique_pcts[0]:g})"

    ncols = max(1, len(metrics))
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = np.array([axes])

    for ax, metric in zip(axes, metrics):
        mdf = sub[sub["metric"] == metric]
        if mdf.empty:
            ax.set_axis_off()
            continue

        # mean±std across seeds per method (percentage is fixed)
        agg = (
            mdf.groupby(["method"], as_index=False)["value"]
            .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
            .sort_values("mean", ascending=False)
        )

        methods = agg["method"].astype(str).tolist()
        y = agg["mean"].astype(float).to_numpy()
        n = agg["n"].astype(int).to_numpy()
        yerr = agg["std"].astype(float).to_numpy()
        yerr = np.where(n >= 2, yerr, 0.0)

        x = np.arange(len(methods), dtype=float)
        ax.bar(x, y, yerr=yerr, capsize=3, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Disentanglement metrics on {dataset}{pct_label}{suffix}")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"disentanglement_{dataset}{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all_bar(
    df: pd.DataFrame, out_dir: str, metrics: list[str], suffix: str = ""
) -> str | None:
    if df.empty:
        return None

    datasets = [d for d in DATASETS if d in df["dataset"].unique()]
    nrows = len(datasets)
    ncols = len(metrics)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), sharex=False
    )
    if nrows == 1:
        axes = np.array([axes])

    for r, dataset in enumerate(datasets):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            if sub.empty:
                ax.set_axis_off()
                continue

            agg = (
                sub.groupby(["method"], as_index=False)["value"]
                .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
                .sort_values("mean", ascending=False)
            )
            methods = agg["method"].astype(str).tolist()
            y = agg["mean"].astype(float).to_numpy()
            n = agg["n"].astype(int).to_numpy()
            yerr = agg["std"].astype(float).to_numpy()
            yerr = np.where(n >= 2, yerr, 0.0)

            x = np.arange(len(methods), dtype=float)
            ax.bar(x, y, yerr=yerr, capsize=3, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=25, ha="right")

            ax.set_title(f"{dataset} — {metric}")
            ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"disentanglement_all{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_dataset(
    df: pd.DataFrame,
    dataset: str,
    out_dir: str,
    metrics: list[str],
    suffix: str = "",
) -> str | None:
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return None

    ncols = max(1, len(metrics))
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4), sharex=True)
    if ncols == 1:
        axes = np.array([axes])

    for ax, metric in zip(axes, metrics):
        mdf = sub[sub["metric"] == metric]
        if mdf.empty:
            ax.set_axis_off()
            continue

        # mean±std across seeds per (percentage, method)
        agg = (
            mdf.groupby(["percentage", "method"], as_index=False)["value"]
            .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
            .reset_index()
        )

        methods = sorted(agg["method"].unique())
        x = np.array(sorted(agg["percentage"].unique()), dtype=float)

        for method in methods:
            mm = agg[agg["method"] == method].set_index("percentage")
            y = np.array(
                [mm.loc[p, "mean"] if p in mm.index else np.nan for p in x], dtype=float
            )
            yerr = np.array(
                [mm.loc[p, "std"] if p in mm.index else np.nan for p in x], dtype=float
            )

            ax.plot(x, y, marker="o", linewidth=1.8, label=method)
            # only draw errorbars if we have >=2 samples
            n = np.array(
                [mm.loc[p, "n"] if p in mm.index else 0 for p in x], dtype=float
            )
            yerr = np.where(n >= 2, yerr, np.nan)
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

        ax.set_title(metric)
        ax.set_xlabel("Label percentage")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Disentanglement metrics on {dataset}{suffix}")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"disentanglement_{dataset}{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all(
    df: pd.DataFrame, out_dir: str, metrics: list[str], suffix: str = ""
) -> str | None:
    if df.empty:
        return None

    # Facet-like: rows = datasets, cols = metrics
    datasets = [d for d in DATASETS if d in df["dataset"].unique()]
    nrows = len(datasets)
    ncols = len(metrics)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), sharex=False
    )
    if nrows == 1:
        axes = np.array([axes])

    for r, dataset in enumerate(datasets):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            if sub.empty:
                ax.set_axis_off()
                continue

            agg = (
                sub.groupby(["percentage", "method"], as_index=False)["value"]
                .agg([("mean", "mean"), ("std", "std"), ("n", "count")])
                .reset_index()
            )
            methods = sorted(agg["method"].unique())
            x = np.array(sorted(agg["percentage"].unique()), dtype=float)

            for method in methods:
                mm = agg[agg["method"] == method].set_index("percentage")
                y = np.array(
                    [mm.loc[p, "mean"] if p in mm.index else np.nan for p in x],
                    dtype=float,
                )
                yerr = np.array(
                    [mm.loc[p, "std"] if p in mm.index else np.nan for p in x],
                    dtype=float,
                )
                n = np.array(
                    [mm.loc[p, "n"] if p in mm.index else 0 for p in x], dtype=float
                )
                yerr = np.where(n >= 2, yerr, np.nan)

                ax.plot(x, y, marker="o", linewidth=1.8, label=method)
                ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, linewidth=1)

            ax.set_title(f"{dataset} — {metric}")
            ax.set_xlabel("Label percentage")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"disentanglement_all{suffix}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--perf_dir",
        default="performance",
        help="Directory containing per-run performance CSVs",
    )
    ap.add_argument(
        "--out_csv", default="performance/summary_tables/disentanglement_summary.csv"
    )
    ap.add_argument("--plot_dir", default="plot/line")
    ap.add_argument(
        "--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS)
    )
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help=(
            "Metric row names to plot (must match the index in performance CSVs). "
            "Default: MIG/SAP + DCI + Hungarian means."
        ),
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="Optional seed filter (e.g., 9)"
    )
    ap.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="Optional label percentage filter (e.g., 0.05)",
    )
    ap.add_argument(
        "--plot_kind",
        default="auto",
        choices=list(PLOT_KINDS),
        help="Plot kind: 'line' for sweeps over percentages, 'bar' for single-percentage comparisons, 'auto' chooses based on data.",
    )
    args = ap.parse_args()

    metrics = [str(m).strip() for m in (args.metrics or []) if str(m).strip()]
    df = collect_disentanglement(args.perf_dir, args.datasets, metrics)
    df = _apply_filters(df, args.seed, args.percentage)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    if df.empty:
        print("No requested metric rows found in performance CSVs.")
        return

    suffix = ""
    if args.seed is not None:
        suffix += f"_seed{int(args.seed)}"
    if args.percentage is not None:
        suffix += f"_pct{args.percentage:g}"

    out_csv = args.out_csv
    if suffix:
        root, ext = os.path.splitext(args.out_csv)
        out_csv = f"{root}{suffix}{ext or '.csv'}"

    df.to_csv(out_csv, index=False)
    print(f"Wrote summary: {out_csv}")

    plot_kind = _infer_plot_kind(df, args.plot_kind)

    for ds in args.datasets:
        if plot_kind == "bar":
            out = _plot_dataset_bar(
                df, ds, args.plot_dir, metrics=metrics, suffix=suffix
            )
        else:
            out = _plot_dataset(df, ds, args.plot_dir, metrics=metrics, suffix=suffix)
        if out:
            print(f"Wrote plot: {out}")

    out_all = (
        plot_all_bar(df, args.plot_dir, metrics=metrics, suffix=suffix)
        if plot_kind == "bar"
        else plot_all(df, args.plot_dir, metrics=metrics, suffix=suffix)
    )
    if out_all:
        print(f"Wrote plot: {out_all}")


if __name__ == "__main__":
    main()
