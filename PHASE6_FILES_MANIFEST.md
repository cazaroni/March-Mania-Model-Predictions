# Phase 6 Files Manifest

## 📌 Start Here

**New to Phase 6?** Read in this order:
1. **PHASE6_SUMMARY.md** (5 min) - Executive overview
2. **PHASE6_QUICKREF.md** (2 min) - Command reference
3. **PHASE6_IMPLEMENTATION.md** (20 min) - Detailed setup guide

**Troubleshooting?**
- Check **PHASE6_README.md** → Troubleshooting section
- Run **check_phase6.sh** → Validates installation
- Review logs: `tail -f logs/ncaa_pipeline_phase6_JOBID.log`

---

## 📁 Phase 6 File Structure

### Code (3 files)

#### NEW
- **`src/graph_embed.py`** (440 lines)
  - **Purpose:** R-GCN model, SSL training, feature extraction
  - **Contains:** RGCN class, SSLTaskHead, graph construction, feature engineering
  - **Run as:** Called automatically from train_baseline.py
  - **Do not edit:** Unless customizing architecture

#### MODIFIED
- **`src/train_baseline.py`**
  - **Changed:** Lines 60-70, 65-75, 321-370
  - **Why:** Added Phase 6 execution before rolling CV + integrated graph features
  - **Status:** Backward compatible (graceful degradation if Phase 6 fails)

- **`src/matchups.py`**
  - **Changed:** feature_columns_for_training() function
  - **Why:** Added prefix matching for Embed*, Cluster*, Neighbor* features
  - **Status:** Automatic feature selection (no manual config needed)

- **`environment.yml`**
  - **Changed:** Added torch, torch-geometric, torch-scatter, etc.
  - **Why:** Required for R-GCN and SSL
  - **Status:** Still compatible with all previous phases

### Setup & Validation (2 scripts)

- **`setup_phase6.sh`** (Executable)
  - **Purpose:** Install PyTorch + PyTorch Geometric
  - **When:** Once, before first execution
  - **Time:** ~5 minutes
  - **Run:** `bash setup_phase6.sh`

- **`check_phase6.sh`** (Executable)
  - **Purpose:** Validate installation before SLURM submission
  - **When:** After setup, before each SLURM submission
  - **Time:** ~1 minute
  - **Run:** `bash check_phase6.sh`

### SLURM Launchers (2 scripts)

- **`run_pipeline_phase6.slurm`** (GPU - Recommended)
  - **Time allocation:** 12 hours
  - **Resources:** 1×GPU, 64 CPU, 128GB RAM
  - **When:** Primary execution method
  - **Run:** `sbatch run_pipeline_phase6.slurm`
  - **Note:** Edit email address (line 10)

- **`run_pipeline_phase6_cpu.slurm`** (CPU - Fallback)
  - **Time allocation:** 14 hours
  - **Resources:** 64 CPU, 128GB RAM (no GPU)
  - **When:** GPU unavailable or debugging needed
  - **Run:** `sbatch run_pipeline_phase6_cpu.slurm`
  - **Note:** Slower but fully functional

### Documentation (6 files)

#### Technical References
- **`PHASE6_README.md`**
  - **Purpose:** Deep technical documentation
  - **Coverage:** Architecture, features, parameters, API details
  - **Read when:** Need to understand internals or troubleshoot
  - **Length:** ~400 lines

- **`PHASE6_IMPLEMENTATION.md`**
  - **Purpose:** Complete usage guide
  - **Coverage:** Installation, execution, debugging, customization
  - **Read when:** Getting started or advanced customization
  - **Length:** ~500 lines

#### Quick References
- **`PHASE6_SUMMARY.md`**
  - **Purpose:** Executive summary
  - **Coverage:** What was implemented, how to run, expected results
  - **Read when:** Want quick overview
  - **Length:** ~200 lines

- **`PHASE6_QUICKREF.md`**
  - **Purpose:** One-page command/parameter reference
  - **Coverage:** Common commands, env vars, key files
  - **Read when:** Quick lookup during execution
  - **Length:** ~150 lines

#### Comprehensive Overview
- **`PHASE6_COMPLETE.md`**
  - **Purpose:** All-in-one implementation summary
  - **Coverage:** Files created, architecture, quick start, next steps
  - **Read when:** Want complete picture
  - **Length:** ~600 lines

#### This File
- **`PHASE6_FILES_MANIFEST.md`**
  - **Purpose:** Navigation guide for all Phase 6 files
  - **Coverage:** What each file does, when to read it
  - **Read when:** Lost or need file reference

---

## 🗂️ Complete Directory Map

```
ncaa/
├── src/
│   ├── graph_embed.py                    [NEW] R-GCN + SSL implementation
│   ├── train_baseline.py                 [MODIFIED] Phase 6 integration
│   ├── matchups.py                       [MODIFIED] Feature selection
│   └── (other phases 0-5, unchanged)
│
├── environment.yml                       [MODIFIED] Added torch packages
│
├── run_pipeline_phase6.slurm             [NEW] GPU launcher (recommended)
├── run_pipeline_phase6_cpu.slurm         [NEW] CPU launcher (fallback)
│
├── setup_phase6.sh                       [NEW] Install dependencies
├── check_phase6.sh                       [NEW] Validate installation
│
├── PHASE6_README.md                      [NEW] Technical reference
├── PHASE6_IMPLEMENTATION.md              [NEW] Usage guide
├── PHASE6_SUMMARY.md                     [NEW] Executive summary
├── PHASE6_QUICKREF.md                    [NEW] Quick reference
├── PHASE6_COMPLETE.md                    [NEW] Complete overview
├── PHASE6_FILES_MANIFEST.md              [NEW] This file
│
├── features/
│   ├── graph_features_m.csv              [OUTPUT] Men's graph features
│   ├── graph_features_w.csv              [OUTPUT] Women's graph features
│   └── (other phase outputs unchanged)
│
├── oof/
│   └── (predictions with graph features integrated)
│
└── logs/
    └── ncaa_pipeline_phase6_JOBID.log    [OUTPUT] Execution log
```

---

## 📋 Quick Decision Tree

**I want to...**

- **Get started immediately** → Read PHASE6_SUMMARY.md + Run setup_phase6.sh
- **Understand everything first** → Read PHASE6_IMPLEMENTATION.md
- **Look up a command** → Check PHASE6_QUICKREF.md
- **Understand the architecture** → Read PHASE6_README.md
- **Debug an issue** → Check PHASE6_README.md#Troubleshooting section
- **Modify the model** → Read src/graph_embed.py + PHASE6_README.md
- **Know what changed** → Read PHASE6_COMPLETE.md#Files Modified
- **See full overview** → Read PHASE6_COMPLETE.md

---

## 🚀 Typical User Workflow

```
1. First Time (one-time setup, ~10 minutes)
   ├─ bash setup_phase6.sh              (5 min: install deps)
   ├─ bash check_phase6.sh              (1 min: validate)
   └─ Read PHASE6_SUMMARY.md            (4 min: understand)

2. Execution (12-14 hours)
   ├─ sbatch run_pipeline_phase6.slurm  (submit job)
   ├─ tail -f logs/...                  (monitor)
   └─ Check success marker              (verify completion)

3. Post-Execution (1 hour)
   ├─ Review metrics                    (eval/fold_metrics_*.csv)
   ├─ Compare with Phase 5              (expect +0.5-2% improvement)
   └─ Analyze feature importance        (which graph features matter?)

4. Optional: Iterate
   ├─ Modify hyperparameters
   ├─ Rerun with custom settings
   └─ Or proceed to next phase
```

---

## 🎯 Key Files by Role

**For Pipeline Managers:**
- PHASE6_QUICKREF.md (commands)
- run_pipeline_phase6.slurm (execution)
- check_phase6.sh (validation)

**For Data Scientists:**
- PHASE6_README.md (features & interpretation)
- src/graph_embed.py (architecture)
- PHASE6_IMPLEMENTATION.md (customization)

**For SysAdmins on HPC:**
- setup_phase6.sh (dependency management)
- run_pipeline_phase6.slurm (resource specs)
- PHASE6_IMPLEMENTATION.md#Troubleshooting

**For Developers:**
- src/graph_embed.py (implementation)
- PHASE6_README.md (architecture)
- environment.yml (dependencies)

---

## 📊 File Statistics

| File | Lines | Type | Status |
|------|-------|------|--------|
| src/graph_embed.py | 440 | Code | NEW |
| src/train_baseline.py | +50 | Code | MODIFIED |
| src/matchups.py | +10 | Code | MODIFIED |
| environment.yml | +7 | Config | MODIFIED |
| run_pipeline_phase6.slurm | 85 | Script | NEW |
| run_pipeline_phase6_cpu.slurm | 85 | Script | NEW |
| setup_phase6.sh | 30 | Script | NEW |
| check_phase6.sh | 35 | Script | NEW |
| PHASE6_README.md | 400 | Doc | NEW |
| PHASE6_IMPLEMENTATION.md | 500 | Doc | NEW |
| PHASE6_SUMMARY.md | 200 | Doc | NEW |
| PHASE6_QUICKREF.md | 150 | Doc | NEW |
| PHASE6_COMPLETE.md | 600 | Doc | NEW |
| PHASE6_FILES_MANIFEST.md | 300 | Doc | NEW |
| **TOTAL** | **~3,027** | - | - |

---

## ✅ Pre-Execution Checklist

Use this to ensure you're ready:

```
[ ] Data files exist (data/*.csv)
[ ] environment.yml reviewed and matches
[ ] setup_phase6.sh executed successfully
[ ] check_phase6.sh passes all checks
[ ] SLURM script email updated
[ ] Storage quota verified (~10GB free)
[ ] Logs directory exists (mkdir -p logs)
[ ] GPU partition available (if GPU version)
[ ] Ready to execute
```

---

## 📞 Support Resources

**By Issue Type:**

| Issue | Resource |
|-------|----------|
| Installation failing | setup_phase6.sh error + PHASE6_IMPLEMENTATION.md#Setup |
| Validation failing | check_phase6.sh + PHASE6_README.md#Troubleshooting |
| Job not starting | check_phase6.sh + SLURM email config |
| Job running slow | PHASE6_QUICKREF.md (hyperparameters) + run_pipeline_phase6_cpu.slurm |
| Unexpected results | eval/fold_metrics_*.csv + PHASE6_IMPLEMENTATION.md#Evaluation |
| Want to customize | PHASE6_README.md#Advanced Customization + src/graph_embed.py |
| Don't understand | PHASE6_SUMMARY.md → PHASE6_README.md → code |

---

## 🔄 Version Control

All Phase 6 files are ready to commit:

```bash
git add src/graph_embed.py src/train_baseline.py src/matchups.py environment.yml
git add run_pipeline_phase6*.slurm setup_phase6.sh check_phase6.sh
git add PHASE6_*.md
git commit -m "Phase 6: R-GCN with SSL for graph embeddings"
```

---

## 📈 Next Phases (Future Work)

After Phase 6 matures:
- **Phase 7:** Heterogeneous GNNs (conference-aware)
- **Phase 8:** Temporal embeddings (season-to-season trajectory)
- **Phase 9:** Multi-task learning (additional SSL tasks)
- **Phase 10:** Ensemble with other graph architectures

See PHASE6_SUMMARY.md#Next Steps for details.

---

## 📖 Reading Guide by Time Available

**5 minutes:** PHASE6_QUICKREF.md + run setup_phase6.sh
**15 minutes:** PHASE6_SUMMARY.md + PHASE6_QUICKREF.md
**30 minutes:** PHASE6_IMPLEMENTATION.md (Installation section)
**1 hour:** PHASE6_IMPLEMENTATION.md (full)
**2 hours:** PHASE6_IMPLEMENTATION.md + PHASE6_README.md
**4 hours:** Everything + src/graph_embed.py detailed review

---

**Last Updated:** Phase 6 Complete
**Status:** Ready for Production
**Questions?** See appropriate documentation above
