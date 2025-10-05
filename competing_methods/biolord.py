import pandas as pd
import numpy as np

import scanpy as sc
from scipy.stats import ttest_rel
import seaborn as sns
import warnings

import biolord

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import anndata as ad
import torch

#
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--percentage", type=float, default=0.05)

args = parser.parse_args()

SEED = args.seed
PERCENTAGE = args.percentage

#
npzfile = np.load('/Users/haoran/Documents/SCDRL/data/SCDRL_data/simulation_data.npz')
print(npzfile.files)
counts = npzfile['counts']
factors = npzfile['factors']

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
label_masks = np.zeros(num_cells, dtype=bool)
label_masks[labeled_idx] = True

##########
adata = ad.AnnData(counts)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

#
adata.obs["batch"] = factors[:, 0].astype(str)
adata.obs["condition 1"] = factors[:, 1].astype(str)
adata.obs["condition 2"] = factors[:, 2].astype(str)
adata.obs["cell_type"] = factors[:, 3].astype(str)

label_masks = ~label_masks
adata.obs["batch"][label_masks] = "Unknown"
adata.obs["condition 1"][label_masks] = "Unknown"
adata.obs["condition 2"][label_masks] = "Unknown"
adata.obs["cell_type"][label_masks] = "Unknown"

adata.obs['split'] = "train"
adata.obs['split'][label_masks] = "test"

#
biolord.Biolord.setup_anndata(
    adata,
    categorical_attributes_keys=["batch", "condition 1", "condition 2", "cell_type"],
    categorical_attributes_missing={"batch": "Unknown", "condition 1": "Unknown", "condition 2": "Unknown", "cell_type": "Unknown"},
)
adata

module_params = {
    "decoder_width": 512,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 4,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "unknown_attribute_noise_param": 1e-1,
    "seed": 42,
    "n_latent_attribute_ordered": 16,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "normal",
    "loss_regression": "normal",
    "reconstruction_penalty": 1e1,
    "unknown_attribute_penalty": 1e2,
    "attribute_dropout_rate": 0.05,
    "eval_r2_ordered": False,
    "classifier_penalty": 1e1,
    "classification_penalty": 0,
    "classify_all": False,
    "classifier_dropout_rate": 0.05,
}

model = biolord.Biolord(
    adata=adata,
    n_latent=32,
    model_name="simulation",
    train_classifiers=True,
    module_params=module_params,
    split_key="split",
)

trainer_params = {
    "n_epochs_warmup": 0,
    "latent_lr": 1e-3,
    "latent_wd": 1e-4,
    "decoder_lr": 1e-4,
    "decoder_wd": 1e-4,
    "attribute_nn_lr": 1e-2,
    "attribute_nn_wd": 4e-8,
    "step_size_lr": 90,
    "cosine_scheduler": True,
    "scheduler_final_lr": 1e-5,
}

model.train(
    max_epochs=200,
    batch_size=256,
    plan_kwargs=trainer_params,
    early_stopping=True,
    early_stopping_patience=20,
    check_val_every_n_epoch=10,
    # num_workers=1,
    enable_checkpointing=False,
)

#
dataset = model.get_dataset(adata[label_masks])
classification = model.module.classify(dataset["X"])

predictions = pd.DataFrame()
for idx in ["batch", "condition 1", "condition 2", "cell_type"]:
    predictions[idx] = classification[idx].argmax(dim=-1).cpu().numpy()

predictions = predictions.to_numpy()

#
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score

factor_list = ["batch", "condition 1", "condition 2", "cell_type"]
performance = np.empty([3, len(factor_list)])
for idx in range(len(factor_list)):
    ground_truth_labels = factors[:, idx][label_masks]
    accuracy = accuracy_score(ground_truth_labels, predictions[:, idx])
    f1 = f1_score(ground_truth_labels, predictions[:, idx], average='macro')
    ARI = adjusted_rand_score(ground_truth_labels, predictions[:, idx])
    performance[0, idx] = accuracy
    performance[1, idx] = f1
    performance[2, idx] = ARI

print(SEED, PERCENTAGE)
print(performance)

# save results
PATH = "/Users/haoran/Documents/SCDRL/results/biolord/biolord_simulation_" + \
    str(PERCENTAGE) + "_" + \
    str(SEED) + \
    ".npz"

np.savez(PATH, predictions=predictions, performance=performance)
