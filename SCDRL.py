import os
import yaml
import argparse

import sys
sys.path.append("/Users/haoran/Documents/SCDRL/code") # path to SCDRL code
from training import Model

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc

#
parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--percentage", type=float, default=0.05)

args = parser.parse_args()

SEED = args.seed
PERCENTAGE = args.percentage

#
model_dir = '/Users/haoran/Documents/SCDRL/log_dir/model_dir'
tensorboard_dir = '/Users/haoran/Documents/SCDRL/log_dir/tensorboard_dir'
eval_dir = '/Users/haoran/Documents/SCDRL/log_dir/eval_dir'

# clean the folder
import shutil

def delete_directory_contents(folder_path):
    """Deletes all files and subdirectories within a specified folder.
    
    Args:
        folder_path: The path to the folder to be cleared.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

delete_directory_contents(model_dir)
delete_directory_contents(tensorboard_dir)
delete_directory_contents(eval_dir)

#
npzfile = np.load('/Users/haoran/Documents/SCDRL/data/SCDRL_data/simulation_data.npz')
print(npzfile.files)
counts = npzfile['counts']
factors = npzfile['factors']

#
with open("/Users/haoran/Documents/SCDRL/code/default.yaml", 'r') as config_fp:
    config = yaml.safe_load(config_fp)

config['seed'] = SEED

config['n_samples'] = counts.shape[0]
config['n_genes'] = counts.shape[1]
config['n_factors'] = factors.shape[1]
config["factor_names"] = ["batch", "condition 1", "condition 2", "cell_type"]
config['factor_sizes'] = [2, 2, 2, 16]

config['factor_dim'] = 1
config['residual_dim'] = 8

config['train']['n_epochs'] = 10

##########
np.random.seed(SEED)

num_cells = counts.shape[0]
num_labels = int(num_cells * PERCENTAGE)

# randomly shuffle the dataset
random_idx = np.arange(num_cells)
np.random.shuffle(random_idx)

counts = counts[random_idx]
factors = factors[random_idx]

#
labeled_idx = np.random.choice(num_cells, num_labels, replace=False)
test_idx = np.setdiff1d(np.arange(num_cells), labeled_idx)

# label_masks
label_masks = np.zeros_like(factors, dtype=bool)
label_masks[labeled_idx] = True

##########
adata = ad.AnnData(counts)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

counts_normalized = adata.X
del counts
del adata



##########
model = Model(config)

model.train_latent_model(
    counts_normalized, factors, label_masks,
    model_dir, tensorboard_dir
)

# scores = {} # accuracy
# for f, factor_name in enumerate(config['factor_names']):
#     scores[factor_name] = model._Model__eval_factor_classification(
#         counts_normalized[test_idx], factors[test_idx], f
#     )

# print(scores)



##############
# Evaluation #
##############
import torch
from utils import CustomDataset
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# test
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score

test_data = torch.tensor(counts_normalized[test_idx], dtype=torch.float32, device=device)
test_factors = torch.tensor(factors[test_idx], device=device)

predictions = np.empty([test_factors.shape[0], test_factors.shape[1]])
with torch.no_grad():
    for idx in range(config['n_factors']):
        logits = model.latent_model.factor_model.module.factor_classifiers[idx](test_data)
        predictions[:, idx] = logits.argmax(dim=1).cpu().numpy()

performance = np.empty([3, config['n_factors']])
for idx in range(config['n_factors']):
    ground_truth_labels = test_factors[:, idx].cpu().numpy()
    accuracy = accuracy_score(ground_truth_labels, predictions[:, idx])
    f1 = f1_score(ground_truth_labels, predictions[:, idx], average='macro')
    ARI = adjusted_rand_score(ground_truth_labels, predictions[:, idx])
    performance[0, idx] = accuracy
    performance[1, idx] = f1
    performance[2, idx] = ARI

print(SEED, PERCENTAGE)
print(performance)

# save results
PATH = "/Users/haoran/Documents/SCDRL/results/SCDRL/SCDRL_simulation_" + \
    str(PERCENTAGE) + "_" + \
    str(SEED) + \
    ".npz"

np.savez(PATH, predictions=predictions, performance=performance,
            random_idx=random_idx, labeled_idx=labeled_idx, test_idx=test_idx)
