# main.py/training.py       np.bool → bool
# modules.py                h.view() → h.reshape()
# shapes3d.yaml             n_epochs: 10

# batch['img']              batch['counts_norm']
# batch['img_id']           batch['counts_norm_ids']
# config['img_shape']       config['n_genes']
# n_channels                n_genes
# residual_factors          none
# imgs                      counts_norm
# img                       counts_norm or data
# CustomDataset({'img': torch.from_numpy(imgs).permute(0, 3, 1, 2)})
# CustomDataset({'counts_norm': torch.from_numpy(counts_norm).to(torch.float32)})
