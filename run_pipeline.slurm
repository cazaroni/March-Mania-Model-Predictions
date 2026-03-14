#!/bin/bash
#SBATCH --job-name=ncaa-pipeline
#SBATCH --output=logs/ncaa_pipeline_%j.log
#SBATCH --error=logs/ncaa_pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@unison.mx

# Set up environment for YUCA supercomputer
cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo ""

# Load YUCA-specific modules
echo "Loading YUCA modules..."
module purge
module load gcc/11.2.0
module load anaconda3/2023.09

# Activate conda environment
echo "Activating ncaa-pipeline environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ncaa-pipeline

# Ensure writable TMPDIR for sklearn/joblib and avoid slurmstepd TMPDIR warnings
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"

# Threading controls for CPU-bound sklearn training
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Keep default run lightweight; set to 1 only when you explicitly want slower extra models
export NCAA_ENABLE_EXTRA_MODELS=${NCAA_ENABLE_EXTRA_MODELS:-0}

# Run the training pipeline
echo "Starting NCAA baseline pipeline..."
python src/train_baseline.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo "Outputs saved to:"
    echo "  - features/team_ratings_m.parquet"
    echo "  - features/team_ratings_w.parquet"
    echo "  - oof/ (out-of-fold predictions)"
    echo "  - submissions/ (submission files)"
else
    echo "Pipeline failed with exit code $?"
    exit 1
fi

echo ""
echo "Job completed at $(date)"
