from pathlib import Path

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent

counts = pd.read_csv(HERE / "counts_gene_by_cell.txt", sep="\t", header=0)
counts = counts.T # transpose the counts matrix
factors = pd.read_csv(HERE / "metadata.txt", sep="\t", header=0, index_col=0)

counts = counts.values
factors = factors.values

adata = ad.AnnData(counts)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

counts_normalized = adata.X

np.savez(
    HERE / "simulation_data.npz",
    counts=counts,
    counts_normalized=counts_normalized,
    factors=factors,
)
