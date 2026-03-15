# Phase 6 Quick Reference Card

## Setup (One-time, ~5 min)

```bash
# 1. Install dependencies
bash setup_phase6.sh

# 2. Verify
bash check_phase6.sh
# Expected: ✓ All checks passed!
```

## Run (12-14 hours)

### Standard (GPU)
```bash
sbatch run_pipeline_phase6.slurm
```

### CPU-Only
```bash
sbatch run_pipeline_phase6_cpu.slurm
```

### Custom Settings
```bash
sbatch --export=ALL,NCAA_PHASE6_EMBEDDING_DIM=128,NCAA_PHASE6_EPOCHS=50 \
  run_pipeline_phase6.slurm
```

## Monitor

```bash
# Job status
squeue -u $USER

# Real-time log
tail -f logs/ncaa_pipeline_phase6_JOBID.log

# Success check
grep "\[PIPELINE\] gender=M integrated" logs/ncaa_pipeline_phase6_JOBID.log
```

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `NCAA_PHASE6_EMBEDDING_DIM` | 64 | Larger = richer but slower (try 32, 128) |
| `NCAA_PHASE6_GNN_LAYERS` | 2 | Depth of graph convolutions (1-3) |
| `NCAA_PHASE6_EPOCHS` | 30 | Training iterations (30-60) |
| `NCAA_PHASE6_LR` | 0.001 | Learning rate (0.0001-0.01) |

## Features Generated

18 new features per matchup, auto-selected:

| Category | Features |
|----------|----------|
| Geometry | `EmbedCosSim`, `EmbedDist`, `EmbedDot`, `EmbedDiffMean`, `EmbedDiffStd`, `EmbedProdMean`, `EmbedProdStd` |
| Strength | `EmbedNorm_T1`, `EmbedNorm_T2`, `EmbedNormDiff`, `EmbedNormRatio` |
| Archetypes | `Cluster_T1`, `Cluster_T2`, `ClusterMatch` |
| Network | `NeighborStrength_T1`, `NeighborStrength_T2`, `NeighborStrengthDiff` |

## Output Files

```
features/
├── graph_features_m.csv       # Automatically merged
└── graph_features_w.csv       # into training data

(All other outputs unchanged)
oof/stack_m.csv               # Now includes graph signal
eval/fold_metrics_m.csv       # Updated metrics
```

## Log Markers

### Success Path
```
[PHASE6] starting graph embedding phase...
[GRAPH] gender=M building graph...
[GRAPH] nodes=352 edges=13045              ✓ Graph built
[GRAPH] gender=M training R-GCN + SSL...
[GRAPH] gender=M epoch=30/30 loss=0.3421   ✓ Training done
[GRAPH] gender=M completed. Generated 352 embeddings.
[FEATURES] gender=M extracted 13045 game features  ✓ Features done
[PIPELINE] gender=M integrated 18 graph features   ✓ SUCCESS!
[PIPELINE] gender=M model=logreg starting
```

### Error Path
```
[PHASE6] WARNING: graph embedding failed (error message)
[PIPELINE] continuing without graph features    ⚠ Graceful degradation

(Pipeline continues without Phase 6, using existing phases)
```

## Troubleshooting Checklist

- [ ] Data files exist: `data/M*.csv`
- [ ] Environment created: `conda activate ncaa-pipeline`
- [ ] Dependencies installed: `bash setup_phase6.sh`
- [ ] Validation passed: `bash check_phase6.sh`
- [ ] Logs accessible: `tail logs/ncaa_pipeline_phase6_*.log`
- [ ] SLURM email configured: Edit `run_pipeline_phase6.slurm` line with your email

## Quick Stats

| Metric | Value |
|--------|-------|
| Graph nodes (men) | ~352 |
| Graph edges (men) | ~13,000 |
| R-GCN parameters | ~200k |
| Features per game | 18 |
| Time per gender (GPU) | ~5-10 min |
| Time per gender (CPU) | ~15-25 min |
| Expected accuracy gain | +0.5-2% Brier |

## Feature Selection

Graph features automatically included via prefix matching in `feature_columns_for_training()`:

```python
# This automatically selects all:
keep_prefixes = ("T1_", "T2_", "Diff_", "Interact_", "Rating_", 
                 "Embed", "Cluster", "Neighbor")  # NEW!
```

No manual configuration needed.

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ImportError: torch_geometric` | `bash setup_phase6.sh` |
| `CUDA out of memory` | Use `run_pipeline_phase6_cpu.slurm` |
| `0 edges in graph` | Check data files; verify team IDs match |
| `SSL loss = NaN` | Reduce `NCAA_PHASE6_LR` or check data normalization |
| `Features all zeros` | Verify embeddings trained correctly; check logs |

## File Locations

```
Entry point:          src/train_baseline.py
Graph code:           src/graph_embed.py
Feature selection:    src/matchups.py
Output features:      features/graph_features_*.csv
Logs:                 logs/ncaa_pipeline_phase6_*.log
SLURM launcher:       run_pipeline_phase6.slurm or run_pipeline_phase6_cpu.slurm
Setup/verify:         setup_phase6.sh, check_phase6.sh
Documentation:        PHASE6_README.md, PHASE6_IMPLEMENTATION.md, PHASE6_SUMMARY.md
This card:            PHASE6_QUICKREF.md
```

## Next Steps After Completion

1. **Evaluate:** Compare `fold_metrics_*.csv` with Phase 5 baseline
2. **Analyze:** Look at which graph features matter most (feature importance)
3. **Iterate:** Try different embedding dimensions or learning rates
4. **Extend:** Add custom SSL tasks or heterogeneous graph layers

## Quick Commands

```bash
# Full setup
bash setup_phase6.sh && bash check_phase6.sh

# Submit job
JOB_ID=$(sbatch run_pipeline_phase6.slurm | awk '{print $4}')
swatch -j $JOB_ID

# Monitor specific job
watch -n 10 "tail -20 logs/ncaa_pipeline_phase6_$JOB_ID.log"

# Check success
grep "integrated.*graph" logs/ncaa_pipeline_phase6_*.log

# Compare metrics (after completion)
head -5 eval/fold_metrics_m.csv | cut -d, -f1,2,3,4

# Extract graph features only (if needed)
python -c "
import pandas as pd
df = pd.read_csv('features/graph_features_m.csv')
print(f'Extracted {len(df)} games, {len(df.columns)} features')
print(df.columns.tolist())
"
```

## Architecture at a Glance

```
Input: Competition graph (teams as nodes, games as weighted edges)
  ↓ R-GCN(in=40, hidden=64, out=64, layers=2)
Output: Embeddings z_i ∈ ℝ^64 per team
  ↓ SSL Tasks: predict win/loss and margin bucket
Output: 18 matchup features from embedding pairs
  ↓ Merge into DataFrame
Result: 18 new columns in train_df
  ↓ Feature selection auto-includes (prefix "Embed", "Cluster", "Neighbor")
Result: Used by all downstream models
```

---

**Last Updated:** Phase 6 Implementation  
**For full details:** See PHASE6_README.md or PHASE6_IMPLEMENTATION.md
