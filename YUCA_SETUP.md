# NCAA Pipeline - YUCA Supercomputer Setup

Quick start guide for running the NCAA pipeline on the YUCA supercomputer at UniSon.

## Initial Setup (One-time)

1. **Clone/transfer your project to YUCA**
   ```bash
   # If not already done
   git clone <your-repo> ncaa
   cd ncaa
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup_yuca.sh
   ./setup_yuca.sh
   ```
   This will:
   - Load required YUCA modules
   - Create the conda environment
   - Verify all dependencies

## Running the Pipeline

### Interactive Session (for testing)
```bash
salloc --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=01:00:00
module load anaconda3/2023.09
conda activate ncaa-pipeline
python src/train_baseline.py
```

### Batch Job (for production runs)
```bash
sbatch run_pipeline.slurm
```

**Monitor your job:**
```bash
squeue -u $USER                          # Check job status
tail -f logs/ncaa_pipeline_*.log        # Watch output
scancel <job-id>                         # Cancel a job
```

## Environment Details

- **Python**: 3.11
- **GPU**: V100 (2x per job, configurable)
- **CPU**: 16 cores, 64GB RAM (configurable in SLURM script)
- **Key packages**: pandas, scikit-learn, xgboost, lightgbm, catboost, cuda/cudnn

## Customizing the SLURM Script

Edit `run_pipeline.slurm` to adjust:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--time` | 2:00:00 | Max job duration |
| `--cpus-per-task` | 16 | CPU cores |
| `--mem` | 64G | Memory per node |
| `--gres=gpu` | gpu:v100:2 | GPU type and count |
| `--partition` | gpu | Use `cpu` if no GPU needed |

## Common Issues

**Module not found**: Run `module avail` to check available versions
**No GPU**: Check partition availability with `sinfo` or switch to CPU-only jobs
**Out of memory**: Increase `--mem` and `--cpus-per-task` in SLURM script

## Getting Help

For YUCA-specific issues, contact:
- YUCA Support: support@unison.mx
- HPC Documentation: https://www.unison.mx/yuca

---
Last updated: March 2026
