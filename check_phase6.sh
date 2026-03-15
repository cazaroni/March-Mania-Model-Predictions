#!/bin/bash
# Quick verification that Phase 6 dependencies are properly installed
# Run this BEFORE submitting to SLURM, or to debug installation issues

set -euo pipefail

echo "=== NCAA Phase 6 Dependency Check ==="
echo ""

# Check conda environment
echo "[1/5] Checking conda environment..."
if ! conda env list | grep -q ncaa-pipeline; then
    echo "ERROR: ncaa-pipeline environment not found"
    echo "Run: conda env create -f environment.yml"
    exit 1
fi
echo "✓ ncaa-pipeline environment found"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ncaa-pipeline
export PYTHONNOUSERSITE=1

# Check Python
echo "[2/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Check PyTorch
echo "[3/5] Checking PyTorch..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || {
    echo "ERROR: PyTorch not installed"
    echo "Run: bash setup_phase6.sh"
    exit 1
}

# Check PyTorch Geometric
echo "[4/5] Checking PyTorch Geometric..."
python -c "import torch_geometric; print(f'✓ PyTorch Geometric {torch_geometric.__version__}')" || {
    echo "ERROR: PyTorch Geometric not installed"
    echo "Run: bash setup_phase6.sh"
    exit 1
}

# Optional fast-extension check (non-fatal)
echo "    Checking optional PyG extensions (non-fatal)..."
python -c "
mods = ['pyg_lib', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv']
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print('    ⚠ Missing optional extensions:', ', '.join(missing))
else:
    print('    ✓ Optional extensions available')
"

# Check accelerator availability
echo "[5/5] Checking accelerator support..."
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$cuda_available" = "True" ]; then
    accel_ver=$(python -c "import torch; print(torch.version.hip if torch.version.hip else torch.version.cuda)")
    accel_name=$(python -c "import torch; print('ROCm/HIP' if torch.version.hip else 'CUDA')")
    echo "✓ $accel_name $accel_ver available (GPU acceleration enabled)"
else
    echo "⚠ No GPU backend available (will use CPU, slower but functional)"
fi

echo ""
echo "=== All checks passed! ==="
echo ""
echo "You can now run the pipeline:"
echo "  sbatch run_pipeline_phase6.slurm      # GPU version (recommended)"
echo "  sbatch run_pipeline_phase6_cpu.slurm  # CPU-only version"
