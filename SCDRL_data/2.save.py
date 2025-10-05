# cd /Users/haoran/Documents/SCDRL/data/SCDRL_data
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np

counts = pd.read_csv("/Users/haoran/Documents/SCDRL/data/SCDRL_data/counts_gene_by_cell.txt", sep = "\t", header = 0)
counts = counts.T # transpose the counts matrix
factors = pd.read_csv("/Users/haoran/Documents/SCDRL/data/SCDRL_data/metadata.txt", sep = "\t", header = 0, index_col = 0)

counts = counts.values
factors = factors.values

adata = ad.AnnData(counts)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

counts_normalized = adata.X

np.savez("/Users/haoran/Documents/SCDRL/data/SCDRL_data/simulation_data.npz",
            counts=counts, counts_normalized=counts_normalized, factors=factors)

# scp /Users/haoran/Documents/SCDRL/data/SCDRL_data/simulation_data.npz hl425@jupiter.hpcnet.campus.njit.edu:/ssd/haoran/SCDRL/data/simulation_data
