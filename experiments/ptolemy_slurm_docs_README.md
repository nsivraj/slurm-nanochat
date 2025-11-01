# Ptolemy/SLURM Documentation Index

This folder contains all documentation specific to running nanochat on the **Ptolemy HPC cluster** using **SLURM** for job submission.

---

## 🚀 Quick Start

### New to Ptolemy Setup?
1. **[PTOLEMY_SETUP.md](PTOLEMY_SETUP.md)** ⭐ **START HERE** - Complete setup guide from scratch

### Resuming After a Break?
1. **[RESUME_HERE.md](RESUME_HERE.md)** ⭐ **QUICK START** - TL;DR for resuming work

### Need Quick Commands?
1. **[QUICK_RERUN_GUIDE.md](QUICK_RERUN_GUIDE.md)** - Just the commands, minimal explanation

---

## 📚 Documentation by Purpose

### Setup & Configuration
- **[PTOLEMY_SETUP.md](PTOLEMY_SETUP.md)** - Complete setup guide with WANDB_RUN requirements
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Initial setup completion status
- **[SETUP_FIXES_SUMMARY.md](SETUP_FIXES_SUMMARY.md)** - Summary of setup fixes applied
- **[SCRATCH_STORAGE_VERIFICATION.md](SCRATCH_STORAGE_VERIFICATION.md)** - Storage verification guide

### Running Training
- **[QUICK_RERUN_GUIDE.md](QUICK_RERUN_GUIDE.md)** - Quick reference for running training
- **[RESUME_HERE.md](RESUME_HERE.md)** - Quick start guide for resuming work
- **[IMPORTANT_PTOLEMY_NOTES.md](IMPORTANT_PTOLEMY_NOTES.md)** - Critical notes about Ptolemy

### Status & History
- **[PTOLEMY_SESSION_STATUS.md](PTOLEMY_SESSION_STATUS.md)** - Comprehensive training status and history
- **[SESSION_STATUS.md](SESSION_STATUS.md)** - Original session status (historical)
- **[SESSION_2025_11_01_SUMMARY.md](SESSION_2025_11_01_SUMMARY.md)** - Nov 1 session summary

### Issues & Fixes
- **[WANDB_RUN_FIX_SUMMARY.md](WANDB_RUN_FIX_SUMMARY.md)** - WANDB_RUN issue diagnosis and fix
- **[TRAINING_FIXES_APPLIED.md](TRAINING_FIXES_APPLIED.md)** - Training pipeline fixes

---

## 🎯 Common Tasks

### Submit Training Job
```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm
```

See: [QUICK_RERUN_GUIDE.md](QUICK_RERUN_GUIDE.md)

### Monitor Training
```bash
squeue -u $USER
tail -f logs/nanochat_speedrun_*.out
```

See: [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md#6-monitor-progress)

### Chat with Model
```bash
# SFT model (default)
python -m scripts.chat_cli

# Base model (fallback)
python -m scripts.chat_cli -i mid
```

See: [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md#chat-with-your-model-interactive-session)

---

## 📊 Training History

### Run 1 (Job 76322) - Oct 30
- ✅ Base model trained
- ❌ Tokenizer eval failed (no internet)
- ❌ Midtraining failed (no internet)

### Run 2 (Job 76324) - Oct 31
- ✅ Base model trained
- ✅ Tokenizer eval succeeded
- ⚠️  Midtraining ran in dummy mode
- ⚠️  SFT ran in dummy mode

### Run 3 (Planned)
- ✅ All fixes applied
- ✅ WANDB_RUN validation in place
- ⏳ Ready to run with full training

See: [PTOLEMY_SESSION_STATUS.md](PTOLEMY_SESSION_STATUS.md)

---

## ⚠️ Critical Information

### WANDB_RUN Requirement
**You MUST set WANDB_RUN when submitting jobs**, or midtraining and SFT will be skipped!

```bash
# ✅ CORRECT
WANDB_RUN=my_run sbatch scripts/speedrun.slurm

# ❌ WRONG
sbatch scripts/speedrun.slurm
```

See: [WANDB_RUN_FIX_SUMMARY.md](WANDB_RUN_FIX_SUMMARY.md)

### No Internet on GPU Nodes
GPU compute nodes have **no internet access**. All data must be pre-downloaded.

See: [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md#4-download-required-data)

### Time Limit
Maximum job time: **12 hours** (class-cse8990 QOS limit)

See: [QUICK_RERUN_GUIDE.md](QUICK_RERUN_GUIDE.md)

---

## 🔧 Troubleshooting

### Common Issues
1. **FileNotFoundError: chatsft_checkpoints**
   - Use `-i mid` flag to chat with base model
   - Or re-run training with correct WANDB_RUN

2. **Job fails immediately**
   - Check WANDB_RUN is set
   - Check all data is downloaded

3. **Training times out at 12 hours**
   - This is expected - QOS limit
   - Can run SFT separately if needed

See: Main [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

---

## 📁 File Organization

### This Folder (ptolemy_slurm_docs/)
All Ptolemy/SLURM-specific documentation

### Parent Directory
- `README.md` - Main project README
- `PROJECT_STATUS.md` - Overall project status
- `TROUBLESHOOTING.md` - General troubleshooting
- `report.md` - Training report (generated)

### Other Docs
- `local_cpu_docs/` - Local CPU training documentation

---

## 💡 Tips

1. **Always check WANDB_RUN** before submitting jobs
2. **Download data first** on ptolemy-devel-1 (has internet)
3. **Monitor with email** notifications (configured in .env.local)
4. **Use -i mid flag** to test base model while waiting for SFT

---

**Last Updated:** 2025-11-01
**Status:** All fixes applied, ready for next training run ✅
