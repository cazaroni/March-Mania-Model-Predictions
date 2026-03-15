#!/bin/bash
# Phase 6 Setup: PyTorch Geometric and GNN Dependencies
# 
# Usage on YUCA cluster:
#   bash setup_phase6.sh
#
# This script installs PyTorch and PyTorch Geometric in the ncaa-pipeline environment.
# Defaults target ROCm (MI210). Override TORCH_INDEX_URL for CUDA clusters.

set -euo pipefail

echo "Setting up Phase 6 dependencies (PyTorch + PyTorch Geometric)..."

# Check if conda environment exists
if ! conda env list | grep -q ncaa-pipeline; then
    echo "ERROR: ncaa-pipeline environment not found. Run 'conda env create -f environment.yml' first."
    exit 1
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ncaa-pipeline

# Prevent picking packages from ~/.local and force installs into active env.
export PYTHONNOUSERSITE=1

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.1}"

echo "Installing PyTorch from index: ${TORCH_INDEX_URL}"
python -m pip install --no-user torch --index-url "${TORCH_INDEX_URL}"

echo "Installing PyTorch Geometric dependencies..."

# Try to install prebuilt wheels that match the installed torch version.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_BUILD=$(python -c "import torch; print((torch.__version__.split('+')[1] if '+' in torch.__version__ else 'cpu'))")
PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_BUILD}.html"

echo "Attempting prebuilt PyG wheels from: ${PYG_WHL_URL}"
if python -m pip install --no-user pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "${PYG_WHL_URL}"; then
    python -m pip install --no-user torch-geometric
else
    echo "Prebuilt PyG extension wheels unavailable for this torch build; trying non-isolated build fallback..."
    if python -m pip install --no-user --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv; then
        python -m pip install --no-user torch-geometric
    else
        echo "WARNING: Optional PyG C++/HIP extensions failed to install."
        echo "Installing torch-geometric only (works for many layers, possibly slower)."
        python -m pip install --no-user torch-geometric
    fi
fi

echo "Verifying installation..."
python -c "import torch; import torch_geometric; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Phase 6 setup complete!"
echo "You can now run the pipeline with:"
echo "  sbatch run_pipeline_phase6.slurm      # GPU version"
echo "  sbatch run_pipeline_phase6_cpu.slurm  # CPU-only version"
