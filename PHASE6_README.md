# Phase 6: R-GCN with Self-Supervised Learning

## Overview

Phase 6 implements a **Relational Graph Convolutional Network (R-GCN)** with **Self-Supervised Learning (SSL)** to generate rich graph-based embeddings that capture the global competition topology of NCAA basketball.

Rather than predicting with a GNN, Phase 6 uses graphs to generate **latent team representations** that encode:
- Global competition topology (conference ecosystems, strength propagation)
- Style and interaction patterns between teams
- Network centrality and competitive positioning

These embeddings are then featurized and integrated into the existing pipeline as an additional signal layer on top of ratings and matchup features.

## Architecture

### 1. Graph Construction

**Nodes:** (Season, TeamID) pairs - representing a team in a specific season
**Edges:** Games - where an edge from Team1 to Team2 indicates a matchup
**Edge Types:** 
- Type 0: Team1 lost to Team2
- Type 1: Team1 beat Team2
**Edge Attributes:** Point margin

### 2. R-GCN Model

```
Input Layer: Team features (basic stats + ratings) → node embeddings
RGCN Layer 1: Relational convolution with type-specific transformations
BatchNorm + ReLU
RGCN Layer 2: Output embeddings (default 64-dim)
```

**Why R-GCN?** Handles multiple edge types (win/loss) with separate transformation matrices per relation, allowing the model to learn type-specific propagation patterns.

### 3. Self-Supervised Learning (SSL) Tasks

After GCN encoding, two SSL prediction heads operate on edge pairs:

#### Task 1: Win/Loss Prediction
- **Input:** Concatenated embeddings of two teams [z₁, z₂]
- **Output:** Binary classification (1 = Team1 won, 0 = Team1 lost)
- **Loss:** CrossEntropyLoss

#### Task 2: Margin Bucketing
- **Input:** Concatenated embeddings [z₁, z₂]
- **Output:** 5-class classification of point margin
  - Bucket 0: |margin| < 5
  - Bucket 1: 5 ≤ |margin| < 10
  - Bucket 2: 10 ≤ |margin| < 15
  - Bucket 3: 15 ≤ |margin| < 20
  - Bucket 4: |margin| ≥ 20
- **Loss:** CrossEntropyLoss

**Joint Loss:** `Loss_total = Loss_wl + Loss_margin`

### 4. Feature Extraction

From the learned embeddings, 18+ features are derived per matchup:

#### Matchup Geometry (Core)
- `EmbedCosSim`: Cosine similarity between embeddings
- `EmbedDist`: Euclidean distance
- `EmbedDot`: Dot product
- `EmbedDiffMean/Std`: Mean/std of element-wise difference
- `EmbedProdMean/Std`: Mean/std of element-wise product

#### Strength Proxies
- `EmbedNorm_T1/T2`: L2 norm of each embedding
- `EmbedNormDiff`: Difference in embedding norms
- `EmbedNormRatio`: Ratio of norms (learned strength proxy)

#### Archetypes
- `Cluster_T1/T2`: K-means cluster assignment (k=8)
- `ClusterMatch`: Binary indicator of same cluster
- (Can compute archetype win rates post-hoc)

#### Network Context
- `NeighborStrength_T1/T2`: Average norm of k-nearest neighbors (k=5)
- `NeighborStrengthDiff`: Difference in neighborhood quality

## Integration with Existing Pipeline

**Phase 6 inputs:**
- Basic team features (from Phase 1)
- Ratings (from Phase 2)
- Game results (regular + tournament)

**Phase 6 outputs:**
- `features/graph_features_m.csv`: Graph features for men
- `features/graph_features_w.csv`: Graph features for women

**Pipeline flow:**
```
Phase 0-2: Load data, build ratings
Phase 3: Build match-up matrix
↓
Phase 6: Generate graph embeddings + extract features [NEW]
↓
Merge graph features into training DataFrame with left-join on (Season, Team1, Team2)
↓
Phase 4: Feature selection includes Embed*, Cluster*, Neighbor* prefixes
Phase 5: Rolling CV with base models (logreg, HGB, etc.) using augmented features
Phase 6-7: Stacking + Calibration (unchanged)
```

## Configuration Parameters

Set via environment variables or SLURM `--export`:

```bash
# Graph embedding hyperparameters
NCAA_PHASE6_EMBEDDING_DIM=64          # Embedding dimension (default: 64)
NCAA_PHASE6_GNN_LAYERS=2              # Number of GCN layers (default: 2)
NCAA_PHASE6_EPOCHS=30                 # Training epochs (default: 30)
NCAA_PHASE6_LR=0.001                  # Learning rate (default: 0.001)

# Existing Phase 4-5 settings (still apply)
NCAA_ENABLE_EXTRA_MODELS=1
NCAA_STACK_TOURNEY_WEIGHT=1.0
NCAA_CAL_METHODS=platt,temperature
NCAA_CAL_SCOPES=all,tournament_only
NCAA_CAL_SHRINKS=0.0,0.02,0.04,0.06
```

## Running on YUCA

### Setup (first time only)

```bash
# 1. Create environment with base dependencies
conda env create -f environment.yml

# 2. Install Phase 6 dependencies
bash setup_phase6.sh
```

### Execution

#### GPU Version (recommended)
```bash
sbatch run_pipeline_phase6.slurm
```

Allocates:
- 1 GPU (Tesla V100/A100)
- 64 CPU cores
- 128 GB RAM
- 12 hours

#### CPU-Only Version (fallback)
```bash
sbatch run_pipeline_phase6_cpu.slurm
```

Allocates:
- 64 CPU cores
- 128 GB RAM
- 14 hours (longer due to CPU embedding training)

### Custom Hyperparameters

```bash
sbatch --export=ALL,NCAA_PHASE6_EPOCHS=50,NCAA_PHASE6_EMBEDDING_DIM=128 run_pipeline_phase6.slurm
```

### Monitor Progress

```bash
# Watch log in real-time
tail -f logs/ncaa_pipeline_phase6_JOBID.log

# Check job status
squeue -u $USER
```

## Expected Output

```
[GRAPH] gender=M building graph...
[GRAPH] nodes=352 edges=13045
[GRAPH] gender=M training R-GCN + SSL...
[GRAPH] gender=M epoch=10/30 loss=0.6234
[GRAPH] gender=M epoch=20/30 loss=0.4156
[GRAPH] gender=M epoch=30/30 loss=0.3421
[GRAPH] gender=M completed. Generated 352 embeddings.
[FEATURES] gender=M extracting graph features...
[FEATURES] gender=M extracted 13045 game features
[PIPELINE] gender=M integrated 18 graph features
[PIPELINE] gender=M model=logreg starting
...
```

## Key Files

- `src/graph_embed.py`: R-GCN model, SSL trainer, feature extraction
- `src/train_baseline.py`: Phase 6 integration point (calls graph_embed before rolling CV)
- `src/matchups.py`: Updated feature_columns_for_training to include graph feature prefixes
- `features/graph_features_m.csv`, `features/graph_features_w.csv`: Output embeddings → features
- `run_pipeline_phase6.slurm`: GPU-accelerated launcher
- `run_pipeline_phase6_cpu.slurm`: CPU-only launcher
- `setup_phase6.sh`: Dependency installation script

## Performance Expectations

- **Embedding time:** 5-15 min per gender (GPU) / 30-45 min per gender (CPU)
- **Feature extraction:** <5 min per gender
- **Impact on rolling CV:** ~5-10% slowdown (additional 18 features per game)
- **Expected tournament Brier improvement:** +0.5-2% over Phase 5 (from additional signal)

## Troubleshooting

### "ImportError: No module named torch"
```bash
# Reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "ImportError: No module named torch_geometric"
```bash
# Reinstall PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

### Out of Memory on GPU
- Reduce batch size in graph construction (modify `_build_graph_from_games`)
- Switch to CPU version: `sbatch run_pipeline_phase6_cpu.slurm`
- Reduce embedding dimension: `--export=NCAA_PHASE6_EMBEDDING_DIM=32`

### Graph has too few edges / isolated nodes
- Check data loading (ensure games loaded correctly)
- Verify team IDs match between games and team_features
- Try reducing val_season_max to ensure more training data

## Future Enhancements

1. **Heterogeneous GNN**: Separate node types for conferences
2. **Temporal graphs**: Use season embeddings to model team trajectory
3. **Multi-head attention**: Replace convolutions with attention mechanism
4. **Edge attributes**: Incorporate game scores/margins as edge features
5. **Downstream tasks**: Fine-tune embeddings with supervised margin prediction

## References

- Relational GCN: Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (ESWC 2018)
- Self-supervised learning in GNNs: You et al., "Graph Contrastive Learning with Augmentations" (NeurIPS 2020)
- NCAA March Madness context: Competition graphs naturally encode strength propagation
