# model.latent_model.factor_model.module.factor_embeddings
# model.latent_model.residual_embeddings.module
import torch
import numpy as np
from utils import CustomDataset
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tmp_data = torch.tensor(counts_norm[train_idx], dtype=torch.float32, device=device)
tmp_factors = torch.tensor(factors[train_idx], device=device)
tmp_label_masks = torch.tensor(label_masks, device=device)

data = dict(
    counts_norm=torch.from_numpy(counts_norm).to(torch.float32),
    counts_norm_id=torch.from_numpy(np.arange(counts_norm.shape[0])),
    factors=torch.from_numpy(factors.astype(np.int64)),
    label_masks=torch.from_numpy(label_masks.astype(bool))
)

dataset = CustomDataset(data)

#
model.latent_model.eval()

factor_idx = 3
with torch.no_grad():
    logits = model.latent_model.factor_model.module.factor_classifiers[factor_idx](tmp_data)

tmp_predictions = logits.argmax(dim=1)
tmp_predictions = tmp_predictions.cpu().numpy()

true_labels = tmp_factors[:, factor_idx].cpu().numpy()
np.mean(true_labels == tmp_predictions)





latent_factors = model._Model__embed_factors(dataset)

tmp = model.latent_model.factor_model(tmp_data, tmp_factors, tmp_label_masks)['factor_codes']

# plot
random = np.random.RandomState(seed=config['seed'])
img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
batch = dataset[img_idx]
batch = {name: tensor.to(device) for name, tensor in batch.items()}

latent_model.eval()
batch['factor_codes'] = model.latent_model.factor_model(batch['counts_norm'], batch['factors'], batch['label_masks'])['factor_codes']
batch['residual_code'] = model.latent_model.residual_embeddings(batch['counts_norm_id'])

generator = model.latent_model.generator

latent_code = torch.cat((batch['factor_codes'], batch['residual_code']), dim=1)
img_reconstructed = generator(latent_code)

figure = torch.cat([
    torch.cat(list(batch['counts_norm']), dim=0), # dim
    torch.cat(list(img_reconstructed), dim=0) # dim
], dim=0)




# DCI
tmp_factors = torch.tensor(factors[train_idx], device=device)

with torch.no_grad():
    tmp = model.latent_model.factor_model(tmp_data, tmp_factors, tmp_label_masks)['factor_codes']

import numpy as np
import scipy

from sklearn import ensemble
from sklearn.model_selection import train_test_split


tmp = tmp.cpu().numpy()
tmp_factors = tmp_factors.cpu().numpy()

x_train, x_test, y_train, y_test = train_test_split(tmp, tmp_factors, test_size=0.2, random_state=1337)

latent_dim = x_train.shape[1]
n_factors = y_train.shape[1]

importance_matrix = np.zeros(shape=[latent_dim, n_factors], dtype=np.float64)
acc_train = []
acc_test = []

i = 3

tmp_model = ensemble.GradientBoostingClassifier()
tmp_model.fit(x_train.reshape(x_train.shape[0], -1), y_train[:, i])

importance_matrix[:, i] = np.sum(tmp_model.feature_importances_.reshape(latent_dim, -1), axis=-1)

np.mean(tmp_model.predict(x_train.reshape(x_train.shape[0], -1)) == y_train[:, i])
