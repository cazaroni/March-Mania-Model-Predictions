#!/bin/bash
# Setup script for YUCA supercomputer NCAA pipeline
# Run this once to initialize your environment

set -e

echo "=========================================="
echo "YUCA NCAA Pipeline Setup"
echo "=========================================="
echo ""

# Load required modules
echo "Step 1: Loading base modules..."
module purge
module load gcc/11.2.0
module load cuda/12.1
module load cudnn/8.6.0-cuda12
module load nccl/2.16.5-cuda12
module load anaconda3/2023.09

echo "✓ Modules loaded"
echo ""

# Create conda environment
echo "Step 2: Creating conda environment..."
if conda env list | grep -q ncaa-pipeline; then
    echo "Environment 'ncaa-pipeline' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ncaa-pipeline -y
        conda env create -f environment.yml
    fi
else
    conda env create -f environment.yml
fi

echo "✓ Environment created"
echo ""

# Verify installation
echo "Step 3: Verifying installation..."
conda activate ncaa-pipeline
python -c "import pandas; import numpy; import sklearn; import xgboost; print('✓ All core packages verified')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  module load anaconda3/2023.09"
echo "  conda activate ncaa-pipeline"
echo ""
echo "To submit a job:"
echo "  sbatch run_pipeline.slurm"
echo ""
