mamba create -n SCDRL -c conda-forge -c pytorch \
notebook ipywidgets matplotlib \
tensorflow=2.16.1 pytorch::pytorch torchvision torchaudio \
cuda-toolkit=12.4.1 pandas scikit-learn opencv \
tqdm imageio ninja \
scanpy=1.10.4
