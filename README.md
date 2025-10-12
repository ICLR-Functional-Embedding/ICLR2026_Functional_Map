# ICLR2026_Functional_Map
This repo contains all the codes and models used in the submitted ICLR2026 titled:"Functional Embeddings Enable Aggregation of Multi-Area SEEG Recordings Over Subjects and Sessions"

# Quick Start
Tested on: Python 3.11.8, PyTorch 2.5.1 (cu121).
Tip: users should install PyTorch that matches their hardware.
1) Create & activate a fresh env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
2) Install PyTorch
•	Use your exact build (works on CUDA 12.1 systems):
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
•	Or pick what matches your machine (CPU/CUDA): https://pytorch.org/get-started/locally/
3) Install repo dependencies
pip install --upgrade pip
pip install -r requirements.txt          # for full requirement file
# (optional) pip install -r requirements-minimal.txt   # for minimal version
# (optional) pip install -r requirements-viz.txt       # brain plots/3D viz
# (optional) pip install -r requirements-notebooks.txt # Jupyter tooling
