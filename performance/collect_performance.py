#!/usr/bin/env python3
"""Aggregate per-run performance CSVs into tidy tables.

This repo writes one CSV per (dataset, percentage, seed) at:
  performance/{dataset}_{percentage}_{seed}.csv

Each CSV looks like a metric-by-method table (index=metric, columns=methods).

This script scans those per-run CSVs and writes:
- performance/performance_long.csv: tidy rows (dataset, percentage, seed, metric, method, value)
- performance/performance_summary.csv: mean/std/n over seeds for each (dataset, percentage, metric, method)

It is designed to be robust to missing methods/metrics (e.g., scVI still running).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


DATASETS = ("simulation", "mouse_human", "haniffa")
DISENTANGLEMENT_METRICS = (
    "MIG_mean",
    "SAP_mean",
    "DCI_informativeness_mean",
    "DCI_disentanglement_mean",
    "DCI_completeness_mean",
    "Hungarian_matched_mean",
    "Hungarian_leakage_mean",
    "Hungarian_leakage_ratio_mean",
)


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


def _read_perf(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    return df


def collect_long(perf_dir: str, datasets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []

    # Preferred layout: performance/{method}/{dataset}_{percentage}_{seed}.csv
    for csv_path in sorted(glob.glob(os.path.join(perf_dir, "*", "*.csv"))):
        method = os.path.basename(os.path.dirname(csv_path))
        key = _parse_perf_filename(csv_path)
        if key is None or key.dataset not in set(datasets):
            continue

        df = _read_perf(csv_path)
        # Per-method files should contain exactly one column. If not, fall back
        # to reading any matching method column.
        if df.shape[1] == 1:
            col = df.columns[0]
            series = df[col]
        elif method in df.columns:
            series = df[method]
        else:
            # Unknown format; skip
            continue

        for metric, value in series.items():
            v = pd.to_numeric(value, errors="coerce")
            rows.append(
                {
                    "dataset": key.dataset,
                    "percentage": float(key.percentage),
                    "seed": int(key.seed),
                    "metric": str(metric),
                    "method": str(method),
                    "value": float(v) if pd.notna(v) else np.nan,
                    "source_csv": os.path.relpath(csv_path, start=os.getcwd()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["percentage"] = out["percentage"].astype(float)
    out["seed"] = out["seed"].astype(int)
    out["value"] = out["value"].astype(float)
    return out


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    grp = long_df.dropna(subset=["value"]).groupby(
        ["dataset", "percentage", "metric", "method"], as_index=False
    )["value"]

    return grp.agg([("mean", "mean"), ("std", "std"), ("n", "count")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf_dir", default="performance")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    ap.add_argument(
        "--out_long", default="performance/summary_tables/performance_long.csv"
    )
    ap.add_argument(
        "--out_summary", default="performance/summary_tables/performance_summary.csv"
    )
    ap.add_argument(
        "--out_perf_only_long",
        default="performance/summary_tables/performance_only_long.csv",
        help="Long table excluding MIG_mean/SAP_mean",
    )
    ap.add_argument(
        "--out_perf_only_summary",
        default="performance/summary_tables/performance_only_summary.csv",
        help="Summary table excluding MIG_mean/SAP_mean",
    )
    ap.add_argument(
        "--out_tables_dir",
        default="performance/summary_tables",
        help="Base directory for summary tables",
    )
    args = ap.parse_args()

    long_df = collect_long(args.perf_dir, list(args.datasets))
    if long_df.empty:
        print("No per-run performance CSVs found.")
        return

    os.makedirs(os.path.dirname(args.out_long) or ".", exist_ok=True)
    long_df.to_csv(args.out_long, index=False)
    print(f"Wrote: {args.out_long}")

    summary_df = summarize(long_df)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    summary_df.to_csv(args.out_summary, index=False)
    print(f"Wrote: {args.out_summary}")

    # Separate: non-disentanglement performance only
    perf_only_long = long_df[~long_df["metric"].isin(DISENTANGLEMENT_METRICS)]
    perf_only_summary = summarize(perf_only_long)
    os.makedirs(os.path.dirname(args.out_perf_only_long) or ".", exist_ok=True)
    perf_only_long.to_csv(args.out_perf_only_long, index=False)
    print(f"Wrote: {args.out_perf_only_long}")
    os.makedirs(os.path.dirname(args.out_perf_only_summary) or ".", exist_ok=True)
    perf_only_summary.to_csv(args.out_perf_only_summary, index=False)
    print(f"Wrote: {args.out_perf_only_summary}")

    # Convenience outputs: for each (dataset, percentage) write tables like the per-run CSV
    # but aggregated over seeds.
    perf_tables_dir = os.path.join(args.out_tables_dir, "performance")
    disent_tables_dir = os.path.join(args.out_tables_dir, "disentanglement")
    os.makedirs(perf_tables_dir, exist_ok=True)
    os.makedirs(disent_tables_dir, exist_ok=True)
    if not summary_df.empty:
        for (dataset, pct), sub in summary_df.groupby(["dataset", "percentage"]):
            pct_s = f"{float(pct):g}"

            sub_dis = sub[sub["metric"].isin(DISENTANGLEMENT_METRICS)]
            sub_perf = sub[~sub["metric"].isin(DISENTANGLEMENT_METRICS)]

            if not sub_perf.empty:
                mean_tbl = sub_perf.pivot(
                    index="metric", columns="method", values="mean"
                )
                std_tbl = sub_perf.pivot(index="metric", columns="method", values="std")
                mean_path = os.path.join(perf_tables_dir, f"{dataset}_{pct_s}_mean.csv")
                std_path = os.path.join(perf_tables_dir, f"{dataset}_{pct_s}_std.csv")
                mean_tbl.to_csv(mean_path)
                std_tbl.to_csv(std_path)

            if not sub_dis.empty:
                mean_tbl = sub_dis.pivot(
                    index="metric", columns="method", values="mean"
                )
                std_tbl = sub_dis.pivot(index="metric", columns="method", values="std")
                mean_path = os.path.join(
                    disent_tables_dir, f"{dataset}_{pct_s}_mean.csv"
                )
                std_path = os.path.join(disent_tables_dir, f"{dataset}_{pct_s}_std.csv")
                mean_tbl.to_csv(mean_path)
                std_tbl.to_csv(std_path)

        print(f"Wrote tables under: {perf_tables_dir} and {disent_tables_dir}")


if __name__ == "__main__":
    main()
