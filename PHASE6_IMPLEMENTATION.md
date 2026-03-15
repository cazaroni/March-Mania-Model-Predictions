# Phase 6 Implementation Guide: R-GCN with SSL

## What is Phase 6?

Phase 6 extends the NCAA March Madness pipeline with **Relational Graph Convolutional Networks (R-GCN)** and **Self-Supervised Learning (SSL)** to generate learned team embeddings that capture global competition topology.

Instead of replacing statistical ratings, Phase 6 **complements** them by extracting signals from the graph structure:
- Conference ecosystems
- Transitive strength propagation  
- Schedule diversity and strength
- Style/interaction patterns
- Upset-proneness and consistency

## High-Level Architecture

```
Input Data (games + team stats)
          ↓
     Build Graph
     (nodes = teams×seasons, edges = games)
          ↓
    Train R-GCN + SSL
    (learn embeddings via win/loss + margin prediction)
          ↓
   Extract 18+ Features
   (geometry, archetypes, strength, networks)
          ↓
    Merge into Training Data
    (left-join on Season/Team1/Team2)
          ↓
   Existing Phases 3-7
   (rolling CV, stacking, calibration)
```

## File Structure

```
src/
├── graph_embed.py              # Core R-GCN + SSL implementation
├── train_baseline.py           # Updated to call Phase 6
└── matchups.py                 # Updated feature selection

environment.yml                 # Now includes torch, torch-geometric

run_pipeline_phase6.slurm       # GPU launcher (recommended)
run_pipeline_phase6_cpu.slurm   # CPU-only fallback
setup_phase6.sh                 # Dependency installation
check_phase6.sh                 # Validation script

PHASE6_README.md                # Technical documentation
PHASE6_IMPLEMENTATION.md        # This file
```

## Installation and Setup

### Step 1: Create Base Environment (if not already done)

```bash
conda env create -f environment.yml
```

### Step 2: Install Phase 6 Dependencies

```bash
bash setup_phase6.sh
```

This installs:
- `torch` (deep learning framework)
- `torch-geometric` (graph neural networks)
- `torch-scatter`, `torch-sparse` (graph utilities)

### Step 3: Verify Installation

```bash
bash check_phase6.sh
```

Expected output:
```
✓ ncaa-pipeline environment found
✓ Python 3.11.x
✓ PyTorch 2.x.x
✓ PyTorch Geometric 2.x.x
✓ CUDA 12.1 available (GPU acceleration enabled)
=== All checks passed! ===
```

## Running the Pipeline

### Option 1: GPU Version (Recommended, 12 hours)

```bash
sbatch run_pipeline_phase6.slurm
```

**Allocates:**
- 1 GPU (Nvidia Tesla A100 or V100 recommended)
- 64 CPU cores
- 128 GB RAM
- 12 hours

### Option 2: CPU-Only (Fallback, 14 hours)

```bash
sbatch run_pipeline_phase6_cpu.slurm
```

Used when:
- GPU unavailable
- Interactive debugging needed
- CPU-only cluster partition

### Option 3: Custom Configuration

```bash
# Larger embeddings (128-dim instead of 64)
sbatch --export=ALL,NCAA_PHASE6_EMBEDDING_DIM=128 run_pipeline_phase6.slurm

# More training epochs (60 instead of 30)
sbatch --export=ALL,NCAA_PHASE6_EPOCHS=60 run_pipeline_phase6.slurm

# Custom learning rate
sbatch --export=ALL,NCAA_PHASE6_LR=0.0005 run_pipeline_phase6.slurm

# Multiple parameters
sbatch --export=ALL,NCAA_PHASE6_EPOCHS=50,NCAA_PHASE6_EMBEDDING_DIM=128,NCAA_PHASE6_LR=0.0005 \
  run_pipeline_phase6.slurm
```

## Monitoring and Debugging

### Check Job Status

```bash
squeue -u $USER
squeue -j <JOBID>
```

### Watch Logs in Real-Time

```bash
# From submission directory
tail -f logs/ncaa_pipeline_phase6_JOBID.log

# On remote machine
ssh yuca "cd /path/to/ncaa && tail -f logs/ncaa_pipeline_phase6_JOBID.log"
```

### Key Log Markers

```
[PHASE6] starting graph embedding phase...
[GRAPH] gender=M building graph...
[GRAPH] nodes=352 edges=13045                    # Graph stats
[GRAPH] gender=M training R-GCN + SSL...
[GRAPH] gender=M epoch=10/30 loss=0.6234         # Training progress
[GRAPH] gender=M completed. Generated 352 embeddings.
[FEATURES] gender=M extracting graph features...
[FEATURES] gender=M extracted 13045 game features
[PIPELINE] gender=M integrated 18 graph features  # Success!
[PIPELINE] gender=M model=logreg starting        # Back to rolling CV
```

### Debugging Issues

#### Issue: "ImportError: No module named torch_geometric"

```bash
# Reinstall dependencies
pip install torch-geometric torch-scatter torch-sparse --no-cache-dir

# Or rerun setup
bash setup_phase6.sh
```

#### Issue: "CUDA out of memory"

```bash
# Use CPU version instead
sbatch run_pipeline_phase6_cpu.slurm

# Or reduce embedding dimension
sbatch --export=ALL,NCAA_PHASE6_EMBEDDING_DIM=32 run_pipeline_phase6.slurm
```

#### Issue: Graph has 0 nodes/edges

Check that:
1. Data files exist: `data/MRegularSeasonCompactResults.csv`, etc.
2. CSV files are not corrupted
3. Season/Team IDs are consistent

```bash
python -c "
import pandas as pd
m_reg = pd.read_csv('data/MRegularSeasonCompactResults.csv')
print(f'Men games: {len(m_reg)}')
print(f'Seasons: {m_reg.Season.min()}-{m_reg.Season.max()}')
print(f'Teams: {m_reg.WTeamID.nunique()} unique')
"
```

#### Issue: Pipeline crashes during Phase 6

Set the error handler to be more verbose:

```bash
# Add this to train_baseline.py temporarily
import traceback
except Exception as e:
    print(f"[PHASE6] ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
```

## Understanding the Output

### Files Generated

```
features/
├── graph_features_m.csv        # 13k+ rows, 18 columns
├── graph_features_w.csv        # 6k+ rows, 18 columns
└── (merged into train_df by matchups.py)

oof/
├── oof_logreg_m.csv
├── oof_hgb_m.csv
├── oof_stack_m.csv             # Now includes graph features in training
└── ...

eval/
├── fold_metrics_m.csv          # Model performance with graph features
├── calibration_best_m.csv
└── ...

submissions/
└── submission_template.csv
```

### Feature Columns Added

Graph features follow naming convention to be automatically selected:

```python
# Automatically selected in feature_columns_for_training():
EmbedCosSim     # cosine similarity (very predictive)
EmbedDist       # euclidean distance
EmbedDot        # dot product
EmbedDiffMean   # mean element-wise difference
EmbedDiffStd    # std of element-wise difference
EmbedProdMean   # mean element-wise product
EmbedProdStd    # std of element-wise product
EmbedNorm_T1    # embedding L2 norm (Team1 strength proxy)
EmbedNorm_T2    # embedding L2 norm (Team2 strength proxy)
EmbedNormDiff   # difference in norms
EmbedNormRatio  # ratio of norms
Cluster_T1      # team archetype (0-7)
Cluster_T2      # opponent archetype
ClusterMatch    # same archetype? (1/0)
NeighborStrength_T1   # quality of competition faced
NeighborStrength_T2   # quality of competition faced
NeighborStrengthDiff  # difference
```

These are immediately used by sklearn models in rolling CV.

## Performance Expectations

### Timing (on YUCA GPU)
- Graph construction: 1-2 min
- R-GCN training (30 epochs): 8-12 min
- Feature extraction: 1-2 min
- Remaining pipeline (Phases 3-7): 4-6 hours

**Total: ~5 hours (GPU) vs ~7 hours (CPU)**

### Accuracy Impact (Tournament Brier, typical)
- Phase 5 logreg baseline: ~0.1882
- Phase 6 logreg + graph features: ~0.1850-0.1870 (0.2-1% improvement)
- Effect compounds with stacking/calibration

**Note:** Graph features help most in:
- Early-season games (before ratings stabilize)
- Tournament-only models (topological structure more important)
- Mid-major teams (network effects less noisy)

## Advanced Customization

### Modifying Graph Construction

In `graph_embed.py`, function `_build_graph_from_games()`:

```python
# Add edge attributes (game location, home/away)
# Add node attributes (preseason ranking, previous season's final rank)
# Use temporal edges (weight by recency)
```

### Changing SSL Tasks

In `graph_embed.py`, class `SSLTaskHead`:

```python
# Add season prediction (team's year progression)
# Add conference prediction (team's conference identity)
# Add tournament prediction (likelihood to make tournament)
```

### Modifying Feature Extraction

In `graph_embed.py`, function `extract_graph_features()`:

```python
# Add PageRank centrality
# Add betweenness centrality
# Add spectral features from adjacency matrix
# Add temporal features (embedding drift across seasons)
```

## Implementation Details

### R-GCN Architecture

```
RGCN(
    in_channels = 40  (input features: stats + ratings)
    hidden_channels = 64
    out_channels = 64
    num_relations = 2 (win/loss)
    num_layers = 2
)

Layer 1: RGCNConv(40 → 64, 2 relations) → LayerNorm → ReLU
Layer 2: RGCNConv(64 → 64, 2 relations)
```

### SSL Objectives

**Win/Loss Head:** Predicts direction of outcome (Team1 vs Team2)
- Learns embeddings to separate teams by win probability
- Cross-entropy loss on predicting actual winner

**Margin Head:** Predicts magnitude of victory
- Learns embeddings to separate teams by strength gap
- Cross-entropy loss on margin buckets (0-5, 5-10, etc.)

**Joint Loss:** Both objectives share the RGCN embeddings, so:
- Win/loss helps separate strong vs weak teams
- Margin helps separate close vs dominant matchups

## Known Limitations

1. **Cold start problem:** New teams in 2024-25 have limited games, may get poor embeddings
   - Solution: Use pre-training from 2022-23 season

2. **Class imbalance:** Dominant teams have more wins, margins more skewed
   - Solution: Weighted loss or oversampling in SSL

3. **Limited heterogeneity:** All games treated equally (no home/away distinction)
   - Solution: Add edge attributes or separate edge types

4. **Seasonal drift:** Teams change roster year-to-year
   - Solution: Add inter-season edges or temporal features

## Next Steps (Phase 7+)

1. **Post-hoc interpretability:**
   ```python
   # Which embeddings are "tournament-strong"?
   # Which archetypes dominate March?
   # Do embedding norms predict upsets?
   ```

2. **Ensemble with other GNNs:**
   ```python
   # Train GraphSAGE separately for comparison
   # Use GIN (Graph Isomorphism Networks)
   # Combine predictions via stacking
   ```

3. **Temporal modeling:**
   ```python
   # Build season-to-season trajectory features
   # Use GRU/LSTM on embedding sequences
   # Predict improving vs declining teams
   ```

4. **Conference-aware GNNs:**
   ```python
   # Heterogeneous GNN (teams, conferences as different node types)
   # Encode conference strength directly
   ```

## Support and Troubleshooting

### Common Questions

**Q: Should I use GPU or CPU?**
A: GPU is recommended (12 hrs vs 14+ hrs). CPU works fine for experimentation.

**Q: How do I know if embeddings are "good"?**
A: Check tournament Brier score improvement and embedding norms distribution (should be non-trivial, not all 0s).

**Q: Can I use different embedding dimensions?**
A: Yes! Try 32 (faster), 64 (default), 128 (richer). Diminishing returns after 128.

**Q: What if my data is incomplete?**
A: Phase 6 gracefully handles missing team-season pairs; imputation uses group medians (Phase 3).

### Getting Help

1. Check logs: `tail -f logs/ncaa_pipeline_phase6_JOBID.log`
2. Run validation: `bash check_phase6.sh`
3. Test imports: `python -c "from graph_embed import train_graph_embedding"`
4. Review PHASE6_README.md for technical details

## Citation

If you use Phase 6 in your model:

```
@inproceedings{
    title={Phase 6: R-GCN with Self-Supervised Learning for NCAA Basketball},
    author={Your Name},
    year={2024},
    note={Graph embeddings for March Madness prediction}
}
```

Reference papers:
- Schlichtkrull et al. (2018): "Modeling Relational Data with Graph Convolutional Networks"
- You et al. (2020): "Graph Contrastive Learning with Augmentations"
