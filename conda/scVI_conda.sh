mamba create -n scVI -c pytorch python=3.11.10 numpy=1.26.4 \
notebook ipywidgets matplotlib \
pytorch::pytorch torchvision torchaudio torchmetrics==0.11.4 \
pandas scikit-learn \
scanpy python-igraph leidenalg seaborn scvi-tools=1.2.2 flax=0.10.2

pip install scikit-misc