import torch
import scvi
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np

scvi.settings.seed = 0
print("scvi-tools version:", scvi.__version__)

torch.set_float32_matmul_precision("high")

npzfile = np.load('/Users/haoran/Documents/SCDRL/data/SCDRL_data/simulation_data.npz')
print(npzfile.files)
counts = npzfile['counts']
factors = npzfile['factors']

#
adata = ad.AnnData(counts)
adata.layers["counts"] = adata.X.copy()

adata.obs["batch"] = factors[:, 0].astype(str)
adata.obs["condition 1"] = factors[:, 1].astype(str)
adata.obs["condition 2"] = factors[:, 2].astype(str)
adata.obs["cell_type"] = factors[:, 3].astype(str)

adata.raw = adata
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=2000,
    layer="counts",
    batch_key="batch",
    subset=True,
)

scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")

model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")

model.train(accelerator='mps')

SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(adata, resolution=1.12) # 16 clusters
print(adata.obs["leiden"])

adata.obs["leiden"].to_csv("/Users/haoran/Documents/SCDRL/results/scVI_simulation.csv", index = False)
