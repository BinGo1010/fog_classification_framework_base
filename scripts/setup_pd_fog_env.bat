@echo off
setlocal

REM Run from the project root. This installs the runtime dependencies into
REM the existing conda environment named pd_fog without relying on activation.
conda run -n pd_fog python -m pip install --upgrade pip setuptools wheel
conda run -n pd_fog python -m pip install PyYAML tqdm matplotlib scikit-learn

REM CUDA 12.1 PyTorch build. For CPU-only installation, replace the next line
REM with: conda run -n pd_fog python -m pip install torch
conda run -n pd_fog python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

conda run -n pd_fog python -c "import torch, yaml, tqdm, matplotlib, sklearn; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('env ok')"
