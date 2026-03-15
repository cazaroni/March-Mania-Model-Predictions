#!/bin/bash
#SBATCH --job-name=ncaa-pipeline-cpu-heavy
#SBATCH --output=logs/ncaa_pipeline_cpu_heavy_%j.log
#SBATCH --error=logs/ncaa_pipeline_cpu_heavy_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --account=default
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@unison.mx

set -euo pipefail

# Set up environment for YUCA supercomputer
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
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

# Ensure writable tmp directories for sklearn/joblib
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
export JOBLIB_TEMP_FOLDER="$TMPDIR"

# Threading controls for CPU-bound training
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

# Keep Python output live in logs
export PYTHONUNBUFFERED=1

# Phase 4/5 defaults; can still override via sbatch --export=ALL,...
export NCAA_ENABLE_EXTRA_MODELS="${NCAA_ENABLE_EXTRA_MODELS:-1}"
export NCAA_STACK_TOURNEY_WEIGHT="${NCAA_STACK_TOURNEY_WEIGHT:-1.0}"
export NCAA_CAL_METHODS="${NCAA_CAL_METHODS:-platt,temperature}"
export NCAA_CAL_SCOPES="${NCAA_CAL_SCOPES:-all,tournament_only}"
export NCAA_CAL_SHRINKS="${NCAA_CAL_SHRINKS:-0.0,0.02,0.04,0.06}"

echo "Run configuration:"
echo "  NCAA_ENABLE_EXTRA_MODELS=$NCAA_ENABLE_EXTRA_MODELS"
echo "  NCAA_STACK_TOURNEY_WEIGHT=$NCAA_STACK_TOURNEY_WEIGHT"
echo "  NCAA_CAL_METHODS=$NCAA_CAL_METHODS"
echo "  NCAA_CAL_SCOPES=$NCAA_CAL_SCOPES"
echo "  NCAA_CAL_SHRINKS=$NCAA_CAL_SHRINKS"
echo ""

echo "Starting NCAA Phase 0-5 pipeline..."
python src/train_baseline.py

echo ""
echo "Pipeline completed successfully!"
echo "Outputs saved to:"
echo "  - features/team_ratings_m.parquet"
echo "  - features/team_ratings_w.parquet"
echo "  - oof/ (base, stack, and calibrated OOF predictions)"
echo "  - eval/ (fold metrics + calibration bins)"
echo "  - submissions/submission_template.csv"
echo ""
echo "Job completed at $(date)"
