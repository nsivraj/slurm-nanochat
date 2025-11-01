# Documentation Index

**Quick navigation to all nanochat documentation organized by topic and platform.**

Last Updated: 2025-11-01

---

## 🚀 **I'M NEW HERE - WHERE DO I START?**

### For Ptolemy/SLURM (HPC Cluster)
👉 **[ptolemy_slurm_docs/RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md)** ⭐ **START HERE**

Or if completely new to setup:
👉 **[ptolemy_slurm_docs/PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md)**

### For Local CPU (Laptop/Desktop)
👉 **[local_cpu_docs/START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md)** ⭐ **START HERE**

Or for quick setup:
👉 **[local_cpu_docs/LOCAL_CPU_QUICKSTART.md](local_cpu_docs/LOCAL_CPU_QUICKSTART.md)**

---

## 📚 Documentation Folders

### [`ptolemy_slurm_docs/`](ptolemy_slurm_docs/)
Complete documentation for running on **Ptolemy HPC cluster** with SLURM job submission.

**Best for:**
- ✅ Full-scale training (560M parameters)
- ✅ Production results (~12 hours)
- ✅ Assignment submission
- ✅ Working ChatGPT-style model

**Key files:**
- [RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md) - Quick start
- [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md) - Complete setup
- [QUICK_RERUN_GUIDE.md](ptolemy_slurm_docs/QUICK_RERUN_GUIDE.md) - Commands only
- [WANDB_RUN_FIX_SUMMARY.md](ptolemy_slurm_docs/WANDB_RUN_FIX_SUMMARY.md) - Critical fix info
- [Full index →](ptolemy_slurm_docs/README.md)

### [`local_cpu_docs/`](local_cpu_docs/)
Complete documentation for running **locally on CPU** for development and testing.

**Best for:**
- ✅ Quick experimentation (5-30 min)
- ✅ Learning the code
- ✅ No GPU access needed
- ✅ Free (no compute costs)

**Key files:**
- [START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md) - Beginner guide
- [LOCAL_CPU_QUICKSTART.md](local_cpu_docs/LOCAL_CPU_QUICKSTART.md) - Quick setup
- [LOCAL_CPU_TRAINING.md](local_cpu_docs/LOCAL_CPU_TRAINING.md) - Training guide
- [TRAINING_COMPARISON.md](local_cpu_docs/TRAINING_COMPARISON.md) - CPU vs GPU
- [Full index →](local_cpu_docs/README.md)

---

## 📄 Top-Level Documentation

### General
- **[README.md](README.md)** - Main project README
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Overall project status
- **[ASSIGNMENT_README.md](ASSIGNMENT_README.md)** - Assignment context and guidance

### Reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - General troubleshooting
- **[SUMMARY_OF_CHANGES.md](SUMMARY_OF_CHANGES.md)** - Summary of modifications
- **[QUICK_RESUME.md](QUICK_RESUME.md)** - Quick resume guide

### Generated
- **[report.md](report.md)** - Training report (generated after runs)

---

## 🎯 Common Tasks - Quick Links

### Ptolemy/SLURM Tasks

#### Submit Training Job
```bash
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```
📖 [QUICK_RERUN_GUIDE.md](ptolemy_slurm_docs/QUICK_RERUN_GUIDE.md)

#### Monitor Training
```bash
squeue -u $USER
tail -f logs/nanochat_speedrun_*.out
```
📖 [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md#6-monitor-progress)

#### Chat with Model
```bash
python -m scripts.chat_cli
```
📖 [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md#chat-with-your-model-interactive-session)

### Local CPU Tasks

#### Train Tiny Model (5 min)
```bash
python -m scripts.local_cpu_train --depth=4 --width=256 --vocab_size=512
```
📖 [LOCAL_CPU_QUICKSTART.md](local_cpu_docs/LOCAL_CPU_QUICKSTART.md)

#### Train Small Model (30 min)
```bash
python -m scripts.local_cpu_train --depth=6 --width=384 --vocab_size=2048
```
📖 [LOCAL_CPU_TRAINING.md](local_cpu_docs/LOCAL_CPU_TRAINING.md)

#### Analyze Results
```bash
python -m scripts.local_cpu_analysis
```
📖 [LOCAL_CPU_ANALYSIS_GUIDE.md](local_cpu_docs/LOCAL_CPU_ANALYSIS_GUIDE.md)

---

## ⚠️ Critical Information

### Ptolemy/SLURM
- **MUST SET WANDB_RUN** when submitting jobs (or midtraining/SFT will be skipped!)
- **No internet on GPU nodes** (must pre-download all data)
- **12-hour time limit** (QOS maximum for class-cse8990)
- 📖 [WANDB_RUN_FIX_SUMMARY.md](ptolemy_slurm_docs/WANDB_RUN_FIX_SUMMARY.md)

### Local CPU
- **Model size matters** (start small, 5M params recommended)
- **CPU is ~30x slower** than GPU (but free!)
- **Good for learning**, not production
- 📖 [TRAINING_COMPARISON.md](local_cpu_docs/TRAINING_COMPARISON.md)

---

## 🔍 Find Documentation By Topic

### Setup & Installation
| Topic | Ptolemy/SLURM | Local CPU |
|-------|---------------|-----------|
| **Complete Setup** | [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md) | [START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md) |
| **Quick Setup** | [QUICK_RERUN_GUIDE.md](ptolemy_slurm_docs/QUICK_RERUN_GUIDE.md) | [LOCAL_CPU_QUICKSTART.md](local_cpu_docs/LOCAL_CPU_QUICKSTART.md) |
| **Resume Guide** | [RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md) | - |

### Training
| Topic | Ptolemy/SLURM | Local CPU |
|-------|---------------|-----------|
| **Training Guide** | [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md) | [LOCAL_CPU_TRAINING.md](local_cpu_docs/LOCAL_CPU_TRAINING.md) |
| **Monitoring** | [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md#6-monitor-progress) | [LOCAL_CPU_TRAINING.md](local_cpu_docs/LOCAL_CPU_TRAINING.md) |
| **Comparison** | - | [TRAINING_COMPARISON.md](local_cpu_docs/TRAINING_COMPARISON.md) |

### Analysis & Usage
| Topic | Ptolemy/SLURM | Local CPU |
|-------|---------------|-----------|
| **Chat with Model** | [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md#chat-with-your-model-interactive-session) | - |
| **Analysis** | - | [LOCAL_CPU_ANALYSIS_GUIDE.md](local_cpu_docs/LOCAL_CPU_ANALYSIS_GUIDE.md) |
| **Reports** | [report.md](report.md) | - |

### Status & History
| Topic | Ptolemy/SLURM | Local CPU |
|-------|---------------|-----------|
| **Current Status** | [PTOLEMY_SESSION_STATUS.md](ptolemy_slurm_docs/PTOLEMY_SESSION_STATUS.md) | [SESSION_SUMMARY_LOCAL_CPU.md](local_cpu_docs/SESSION_SUMMARY_LOCAL_CPU.md) |
| **Recent Changes** | [SESSION_2025_11_01_SUMMARY.md](ptolemy_slurm_docs/SESSION_2025_11_01_SUMMARY.md) | - |
| **Overall Status** | [PROJECT_STATUS.md](PROJECT_STATUS.md) | [PROJECT_STATUS.md](PROJECT_STATUS.md) |

### Troubleshooting
| Topic | Ptolemy/SLURM | Local CPU |
|-------|---------------|-----------|
| **Common Issues** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | [TROUBLESHOOTING_LOCAL_CPU.md](local_cpu_docs/TROUBLESHOOTING_LOCAL_CPU.md) |
| **WANDB_RUN Issue** | [WANDB_RUN_FIX_SUMMARY.md](ptolemy_slurm_docs/WANDB_RUN_FIX_SUMMARY.md) | - |
| **Fixes Applied** | [TRAINING_FIXES_APPLIED.md](ptolemy_slurm_docs/TRAINING_FIXES_APPLIED.md) | - |

---

## 🎓 For MSU CSE8990 Students

### Recommended Path

1. **Start with Local CPU** (optional, but recommended for learning):
   - Train tiny model in 5 minutes → [LOCAL_CPU_QUICKSTART.md](local_cpu_docs/LOCAL_CPU_QUICKSTART.md)
   - Understand the code and architecture
   - Experiment with hyperparameters

2. **Move to Ptolemy for Full Training**:
   - Complete setup → [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md)
   - Submit training job → [QUICK_RERUN_GUIDE.md](ptolemy_slurm_docs/QUICK_RERUN_GUIDE.md)
   - Wait ~12 hours for results
   - Chat with your model!

3. **Analysis & Assignment**:
   - Review report.md
   - Compare local vs cluster results
   - Complete assignment → [ASSIGNMENT_README.md](ASSIGNMENT_README.md)

---

## 📊 Documentation Statistics

### Ptolemy/SLURM Docs
- **12 files** in `ptolemy_slurm_docs/`
- **Topics:** Setup, training, monitoring, fixes, status
- **Platform:** Ptolemy HPC cluster with SLURM

### Local CPU Docs
- **8 files** in `local_cpu_docs/`
- **Topics:** Setup, training, analysis, comparison
- **Platform:** Any laptop/desktop with Python

### Top-Level Docs
- **7 files** in root directory
- **Topics:** General project info, troubleshooting, status

**Total:** 27 documentation files (well-organized! 🎉)

---

## 🔄 Recent Updates (2025-11-01)

### What Changed Today
- ✅ Fixed WANDB_RUN issue (midtraining/SFT were being skipped)
- ✅ Added validation to `speedrun.slurm`
- ✅ Updated all Ptolemy documentation
- ✅ Created comprehensive status files
- ✅ Organized documentation into folders

### Key New Files
- [RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md) - Quick start for resuming
- [WANDB_RUN_FIX_SUMMARY.md](ptolemy_slurm_docs/WANDB_RUN_FIX_SUMMARY.md) - Fix details
- [PTOLEMY_SESSION_STATUS.md](ptolemy_slurm_docs/PTOLEMY_SESSION_STATUS.md) - Comprehensive status
- [SESSION_2025_11_01_SUMMARY.md](ptolemy_slurm_docs/SESSION_2025_11_01_SUMMARY.md) - Today's work

---

## 💡 Tips for Navigation

1. **Use the folder indexes:**
   - [ptolemy_slurm_docs/README.md](ptolemy_slurm_docs/README.md)
   - [local_cpu_docs/README.md](local_cpu_docs/README.md)

2. **Bookmark the "START HERE" files:**
   - Ptolemy: [RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md)
   - Local CPU: [START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md)

3. **Use search:**
   - All docs are markdown
   - Search for keywords across all files
   - File names are descriptive

4. **Follow the links:**
   - Documents cross-reference each other
   - "See: filename.md" links throughout

---

## 🤝 Contributing

When adding new documentation:

1. **Determine platform:**
   - Ptolemy/SLURM → `ptolemy_slurm_docs/`
   - Local CPU → `local_cpu_docs/`
   - General → root directory

2. **Update indexes:**
   - Add to folder README.md
   - Update this file (DOCUMENTATION_INDEX.md)

3. **Cross-reference:**
   - Link to related docs
   - Use relative paths

4. **Keep organized:**
   - Descriptive filenames
   - Clear section headers
   - Table of contents for long docs

---

**Happy training! 🚀**

Questions? Start with the appropriate "START HERE" file for your platform!
