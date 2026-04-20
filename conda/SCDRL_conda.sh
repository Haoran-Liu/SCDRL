# Optional on shared systems:
# export TMPDIR="$HOME/tmp"

# mamba create -n SCDRL -c conda-forge -c pytorch \
# notebook ipywidgets matplotlib \
# tensorflow=2.16.1 pytorch::pytorch torchvision torchaudio \
# cuda-toolkit=12.4.1 pandas scikit-learn opencv \
# tqdm imageio ninja \
# scanpy=1.10.4


conda create -n SCDRL -y python=3.10
conda activate SCDRL

python -m pip install -U uv

uv pip install -U \
  notebook ipywidgets matplotlib \
  pandas scikit-learn opencv-python \
  tqdm imageio ninja \
  scanpy==1.10.4
uv pip install tensorflow==2.16.1
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
