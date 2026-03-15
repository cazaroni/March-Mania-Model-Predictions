#!/bin/bash
#SBATCH --job-name=ncaa-pipeline-phase6-cpu
#SBATCH --output=logs/ncaa_pipeline_phase6_cpu_%A_%a.log
#SBATCH --error=logs/ncaa_pipeline_phase6_cpu_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --partition=cpu

set -euo pipefail

# Set up environment for YUCA supercomputer
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"
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
export XDG_CACHE_HOME="$TMPDIR/.cache"
mkdir -p "$XDG_CACHE_HOME"

# Threading controls for CPU-bound training
export NCAA_CPU_THREADS="${NCAA_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export OMP_NUM_THREADS="$NCAA_CPU_THREADS"
export MKL_NUM_THREADS="$NCAA_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$NCAA_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$NCAA_CPU_THREADS"

# Keep Python output live in logs
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Phase 4/5 defaults; can still override via sbatch --export=ALL,...
export NCAA_ENABLE_EXTRA_MODELS="${NCAA_ENABLE_EXTRA_MODELS:-1}"
export NCAA_STACK_TOURNEY_WEIGHT="${NCAA_STACK_TOURNEY_WEIGHT:-1.0}"
export NCAA_CAL_METHODS="${NCAA_CAL_METHODS:-platt,temperature}"
export NCAA_CAL_SCOPES="${NCAA_CAL_SCOPES:-all,tournament_only}"
export NCAA_CAL_SHRINKS="${NCAA_CAL_SHRINKS:-0.0,0.02,0.04,0.06}"
export NCAA_OUTPUT_SUFFIX="${NCAA_OUTPUT_SUFFIX:-}"

# Phase 6 settings
export NCAA_PHASE6_ENABLE="${NCAA_PHASE6_ENABLE:-1}"
export NCAA_PHASE6_EMBEDDING_DIM="${NCAA_PHASE6_EMBEDDING_DIM:-64}"
export NCAA_PHASE6_GNN_LAYERS="${NCAA_PHASE6_GNN_LAYERS:-2}"
export NCAA_PHASE6_EPOCHS="${NCAA_PHASE6_EPOCHS:-30}"
export NCAA_PHASE6_LR="${NCAA_PHASE6_LR:-0.001}"
export NCAA_PHASE6_INCLUDE_TOURNEY_EDGES="${NCAA_PHASE6_INCLUDE_TOURNEY_EDGES:-0}"
export NCAA_PHASE6_USE_GRAPH_WINPROB="${NCAA_PHASE6_USE_GRAPH_WINPROB:-0}"

# Optional split mode for two-node parallel CPU runs:
# - Submit with: sbatch --array=0-1 run_pipeline_phase6_cpu.slurm
# - Task 0 runs men (m), task 1 runs women (w)
if [ -z "${NCAA_GENDER_FILTER:-}" ]; then
	if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ]; then
		export NCAA_GENDER_FILTER="m"
	elif [ "${SLURM_ARRAY_TASK_ID:-}" = "1" ]; then
		export NCAA_GENDER_FILTER="w"
	else
		export NCAA_GENDER_FILTER="both"
	fi
fi

echo "Run configuration:"
echo "  NCAA_CPU_THREADS=$NCAA_CPU_THREADS"
echo "  NCAA_GENDER_FILTER=$NCAA_GENDER_FILTER"
echo "  NCAA_ENABLE_EXTRA_MODELS=$NCAA_ENABLE_EXTRA_MODELS"
echo "  NCAA_STACK_TOURNEY_WEIGHT=$NCAA_STACK_TOURNEY_WEIGHT"
echo "  NCAA_CAL_METHODS=$NCAA_CAL_METHODS"
echo "  NCAA_CAL_SCOPES=$NCAA_CAL_SCOPES"
echo "  NCAA_CAL_SHRINKS=$NCAA_CAL_SHRINKS"
echo "  NCAA_OUTPUT_SUFFIX=$NCAA_OUTPUT_SUFFIX"
echo "  NCAA_PHASE6_ENABLE=$NCAA_PHASE6_ENABLE"
echo "  NCAA_PHASE6_EMBEDDING_DIM=$NCAA_PHASE6_EMBEDDING_DIM"
echo "  NCAA_PHASE6_GNN_LAYERS=$NCAA_PHASE6_GNN_LAYERS"
echo "  NCAA_PHASE6_EPOCHS=$NCAA_PHASE6_EPOCHS"
echo "  NCAA_PHASE6_LR=$NCAA_PHASE6_LR"
echo "  NCAA_PHASE6_INCLUDE_TOURNEY_EDGES=$NCAA_PHASE6_INCLUDE_TOURNEY_EDGES"
echo "  NCAA_PHASE6_USE_GRAPH_WINPROB=$NCAA_PHASE6_USE_GRAPH_WINPROB"
echo ""

echo "Starting NCAA Phase 0-6 pipeline (CPU mode, R-GCN embeddings and SSL)..."
python src/train_baseline_cpu.py

echo ""
echo "Pipeline completed successfully!"
echo "Outputs saved to:"
echo "  - features/team_ratings_m.parquet"
echo "  - features/team_ratings_w.parquet"
echo "  - features/graph_features_m.csv"
echo "  - features/graph_features_w.csv"
echo "  - oof/ (base, stack, and calibrated OOF predictions)"
echo "  - eval/ (fold metrics + calibration bins)"
echo "  - submissions/submission_template.csv"
echo ""
echo "Job completed at $(date)"
