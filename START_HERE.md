# 🚀 nanochat - Start Here

**Training ChatGPT-style language models from scratch**

Last Updated: 2025-11-01

---

## ⚡ Quick Navigation

### I want to train on Ptolemy HPC cluster
👉 **[ptolemy_slurm_docs/RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md)** ⭐

### I want to train locally on CPU
👉 **[local_cpu_docs/START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md)** ⭐

### I want to browse all documentation
👉 **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** 📚

---

## 📁 Project Structure

```
slurm-nanochat/
├── 📄 START_HERE.md              ← You are here!
├── 📄 DOCUMENTATION_INDEX.md     ← Complete doc navigation
├── 📄 README.md                  ← Original project README
├── 📄 PROJECT_STATUS.md          ← Overall project status
├── 📄 TROUBLESHOOTING.md         ← General troubleshooting
│
├── 📂 ptolemy_slurm_docs/        ← Ptolemy HPC documentation (13 files)
│   ├── README.md                 ← Folder index
│   ├── RESUME_HERE.md            ← ⭐ Start here for Ptolemy
│   ├── PTOLEMY_SETUP.md          ← Complete setup guide
│   ├── QUICK_RERUN_GUIDE.md      ← Quick commands
│   └── ...                       ← Status, fixes, history
│
├── 📂 local_cpu_docs/            ← Local CPU documentation (9 files)
│   ├── README.md                 ← Folder index
│   ├── START_HERE_LOCAL_CPU.md   ← ⭐ Start here for local CPU
│   ├── LOCAL_CPU_QUICKSTART.md   ← Quick setup
│   ├── LOCAL_CPU_TRAINING.md     ← Training guide
│   └── ...                       ← Analysis, comparison, troubleshooting
│
├── 📂 scripts/                   ← Training and analysis scripts
│   ├── speedrun.slurm            ← SLURM job script (Ptolemy)
│   ├── local_cpu_train.py        ← Local CPU training
│   ├── chat_cli.py               ← Chat interface
│   └── ...
│
├── 📂 nanochat/                  ← Core codebase
│   ├── gpt.py                    ← Transformer model
│   ├── engine.py                 ← Training engine
│   └── ...
│
└── 📂 logs/                      ← Training logs (generated)
```

---

## 🎯 What is nanochat?

**nanochat** is a complete implementation of a ChatGPT-style language model:
- ✅ **Full pipeline:** Tokenization → Pretraining → Midtraining → SFT
- ✅ **Production-ready:** Web UI, CLI chat, evaluations
- ✅ **Educational:** Clean, hackable codebase
- ✅ **Scalable:** 1M params (CPU) to 560M params (GPU cluster)

### This Project

This is a fork adapted for **MSU CSE8990** course with:
- ✅ Ptolemy HPC cluster support (SLURM)
- ✅ Local CPU training for development
- ✅ Comprehensive documentation
- ✅ All fixes and optimizations applied

---

## ⚡ Quick Start Guide

### Option 1: Ptolemy HPC (Recommended for Assignment)

**What you get:**
- Full 560M parameter model
- ~12 hours training time
- Professional ChatGPT-style results
- All pipeline stages

**Quick command:**
```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```

**Full guide:** [ptolemy_slurm_docs/RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md)

---

### Option 2: Local CPU (Great for Learning)

**What you get:**
- 1M-50M parameter models
- 5 min - 12 hours training time
- Perfect for experimentation
- Learn the code

**Quick command:**
```bash
# Tiny model (5 minutes)
python -m scripts.local_cpu_train --depth=4 --width=256 --vocab_size=512

# Small model (30 minutes)
python -m scripts.local_cpu_train --depth=6 --width=384 --vocab_size=2048
```

**Full guide:** [local_cpu_docs/START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md)

---

## 📚 Documentation Organization

### By Platform

**Ptolemy/SLURM Cluster** → [`ptolemy_slurm_docs/`](ptolemy_slurm_docs/)
- Complete HPC cluster documentation
- SLURM job submission
- Full-scale training (560M params)
- 13 documentation files
- 📖 [Index](ptolemy_slurm_docs/README.md)

**Local CPU Training** → [`local_cpu_docs/`](local_cpu_docs/)
- Complete local training documentation
- No GPU required
- Development and experimentation
- 9 documentation files
- 📖 [Index](local_cpu_docs/README.md)

### General

**Top Level** → Root directory
- [README.md](README.md) - Original project README
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Overall status
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General issues
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Complete navigation

---

## ⚠️ Critical Information

### For Ptolemy Users

**MUST SET WANDB_RUN!**
```bash
# ✅ CORRECT
WANDB_RUN=my_run sbatch scripts/speedrun.slurm

# ❌ WRONG - Will skip midtraining and SFT!
sbatch scripts/speedrun.slurm
```

Without `WANDB_RUN`, midtraining and SFT phases are SKIPPED!

📖 [WANDB_RUN_FIX_SUMMARY.md](ptolemy_slurm_docs/WANDB_RUN_FIX_SUMMARY.md)

### For Local CPU Users

**Start Small!**
- Tiny model (1M params) = 5 minutes ✅
- Small model (5M params) = 30 minutes ✅
- Medium model (15M params) = 2-3 hours ⚠️
- Large model (50M params) = 8+ hours ⚠️

📖 [TRAINING_COMPARISON.md](local_cpu_docs/TRAINING_COMPARISON.md)

---

## 🎓 For MSU CSE8990 Students

### Assignment Context
- **Course:** CSE8990 - Final Project
- **Goal:** Train and analyze Transformer model
- **Platform:** Ptolemy HPC or local CPU
- **Deliverable:** Understanding of Transformer architecture

### Recommended Approach

**Week 1: Setup & Learning**
1. Read this file (START_HERE.md)
2. Train tiny model locally (5 min)
3. Understand the codebase
4. Review [ASSIGNMENT_README.md](ASSIGNMENT_README.md)

**Week 2: Experimentation**
1. Train small model locally (30 min)
2. Experiment with hyperparameters
3. Analyze results
4. Read [LOCAL_CPU_ANALYSIS_GUIDE.md](local_cpu_docs/LOCAL_CPU_ANALYSIS_GUIDE.md)

**Week 3: Full Training**
1. Submit Ptolemy job
2. Wait ~12 hours
3. Analyze results
4. Compare with local CPU results

**Week 4: Assignment**
1. Write analysis
2. Generate visualizations
3. Complete assignment
4. Submit

---

## 🔧 Common Tasks

### Training

**Ptolemy HPC:**
```bash
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```

**Local CPU (Tiny):**
```bash
python -m scripts.local_cpu_train --depth=4 --width=256 --vocab_size=512
```

**Local CPU (Small):**
```bash
python -m scripts.local_cpu_train --depth=6 --width=384 --vocab_size=2048
```

### Monitoring

**Ptolemy:**
```bash
squeue -u $USER
tail -f logs/nanochat_speedrun_*.out
```

**Local CPU:**
```bash
# Watch the console output in real-time
```

### Chat

**Ptolemy (SFT model):**
```bash
python -m scripts.chat_cli
```

**Ptolemy (Base model):**
```bash
python -m scripts.chat_cli -i mid
```

### Analysis

**Local CPU:**
```bash
python -m scripts.local_cpu_analysis
```

---

## 📊 Platform Comparison

| Feature | Ptolemy HPC | Local CPU |
|---------|-------------|-----------|
| **Model Size** | 560M params | 1M-50M params |
| **Training Time** | ~12 hours | 5 min - 12 hours |
| **Cost** | ~$150-200 | Free |
| **Setup Time** | 1-2 hours | 10 minutes |
| **Queue Time** | 0-1 hour | Immediate |
| **Quality** | Production | Experimental |
| **Best For** | Final results | Learning, testing |

---

## 🆘 Getting Help

### Documentation

1. **Quick answers:** Check appropriate "START HERE" file
2. **Comprehensive guide:** Review platform-specific docs
3. **All docs:** Browse [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

### Troubleshooting

1. **General issues:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Ptolemy issues:** [ptolemy_slurm_docs/](ptolemy_slurm_docs/README.md)
3. **Local CPU issues:** [local_cpu_docs/TROUBLESHOOTING_LOCAL_CPU.md](local_cpu_docs/TROUBLESHOOTING_LOCAL_CPU.md)

### Platform-Specific

**Ptolemy:**
- WANDB_RUN not set → [WANDB_RUN_FIX_SUMMARY.md](ptolemy_slurm_docs/WANDB_RUN_FIX_SUMMARY.md)
- Chat not working → [PTOLEMY_SETUP.md](ptolemy_slurm_docs/PTOLEMY_SETUP.md#chat-with-your-model-interactive-session)
- Job failing → [QUICK_RERUN_GUIDE.md](ptolemy_slurm_docs/QUICK_RERUN_GUIDE.md)

**Local CPU:**
- Out of memory → [TROUBLESHOOTING_LOCAL_CPU.md](local_cpu_docs/TROUBLESHOOTING_LOCAL_CPU.md)
- Too slow → Use smaller model
- Can't find script → Check you're in project root

---

## 📝 Recent Updates (2025-11-01)

### What's New
- ✅ **Fixed WANDB_RUN issue** (critical for Ptolemy)
- ✅ **Organized documentation** into folders
- ✅ **Created comprehensive indexes** for easy navigation
- ✅ **Updated all Ptolemy docs** with latest fixes

### What Changed
- All Ptolemy/SLURM docs → `ptolemy_slurm_docs/`
- All local CPU docs → `local_cpu_docs/`
- Top-level docs cleaned up (8 files remain)
- New navigation: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

### Key New Files
- [START_HERE.md](START_HERE.md) - This file!
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Complete navigation
- [ptolemy_slurm_docs/RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md) - Ptolemy quick start
- [local_cpu_docs/START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md) - Local CPU quick start

---

## 🎉 Ready to Begin!

Choose your path:

### 🖥️ I have Ptolemy access
👉 [ptolemy_slurm_docs/RESUME_HERE.md](ptolemy_slurm_docs/RESUME_HERE.md)

### 💻 I want to use my laptop/desktop
👉 [local_cpu_docs/START_HERE_LOCAL_CPU.md](local_cpu_docs/START_HERE_LOCAL_CPU.md)

### 📚 I want to explore documentation
👉 [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

---

**Questions?** Browse the documentation or check the troubleshooting guides!

**Good luck with your training! 🚀**
