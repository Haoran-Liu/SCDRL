import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from pathlib import Path

HERE = Path(__file__).resolve().parent

# "Status", "initial_clustering": 2, 18
adata = sc.read(
    "data/haniffa_tutorial_subset.h5ad",
    backup_url="https://figshare.com/ndownloader/files/46017615")

counts = adata.X.toarray()

factors = pd.DataFrame({
    "status": adata.obs["Status"].cat.codes.to_numpy(),
    "cell_type": adata.obs["initial_clustering"].cat.codes.to_numpy()
})

mapping = pd.DataFrame(enumerate(adata.obs['initial_clustering'].cat.categories),
                        columns=['index', 'cell_type'])
mapping_index = mapping['index'].to_numpy()
mapping_cell_type = mapping['cell_type'].tolist()

np.savez(
    HERE / "haniffa.npz",
    counts=counts,
    factors=factors,
    mapping_index=mapping_index,
    mapping_cell_type=mapping_cell_type,
)

# "system", "cell_type_eval": 2, 17
adata = sc.read(
    "data/mouse-human_pancreas_subset10000.h5ad",
    backup_url="https://github.com/theislab/cross_system_integration/raw/main/tutorials/data/mouse-human_pancreas_subset10000.h5ad",
)

counts = adata.layers["counts"].toarray()

factors = pd.DataFrame({
    "system": adata.obs["system"].to_numpy(),
    "cell_type": adata.obs["cell_type_eval"].cat.codes.to_numpy()
})

mapping = pd.DataFrame(enumerate(adata.obs['cell_type_eval'].cat.categories),
                        columns=['index', 'cell_type'])
mapping_index = mapping['index'].to_numpy()
mapping_cell_type = mapping['cell_type'].tolist()

np.savez(
    HERE / "mouse_human.npz",
    counts=counts,
    factors=factors,
    mapping_index=mapping_index,
    mapping_cell_type=mapping_cell_type,
)
