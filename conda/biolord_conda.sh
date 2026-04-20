mamba create -n biolord -c conda-forge -c pytorch -c bioconda \
notebook ipywidgets ipywidgets \
r-seurat r-devtools r-hdf5r r-presto bioconductor-singler \
r-biocmanager bioconductor-celldex \
bioconductor-singlecellexperiment bioconductor-spatialexperiment bioconductor-splatter bioconductor-spatiallibd \
pytorch::pytorch torchvision torchaudio \
cuda-toolkit=12.4.1 pandas scikit-learn opencv scanpy scvi-tools=1.2.2

pip install biolord

# Apple silicon
mamba create -n biolord -c conda-forge -c pytorch -c bioconda python=3.11.10 numpy=1.26.4 \
notebook ipywidgets ipywidgets \
pytorch::pytorch torchvision torchaudio \
pandas scikit-learn scanpy scvi-tools=1.2.2

pip install biolord