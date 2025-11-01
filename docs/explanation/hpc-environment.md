# HPC Environment: Ptolemy Cluster Details

This document explains the unique constraints and features of the Ptolemy HPC environment at Mississippi State University.

---

## Overview

Ptolemy is MSU's HPC cluster with specialized GPU nodes. Understanding its architecture and constraints is essential for successful training.

---

## Critical Constraint: No Internet on GPU Nodes

### The Problem

**GPU compute nodes on Ptolemy do NOT have internet access.**

This is a standard HPC security practice, but it requires a modified workflow.

### Why This Design?

**HPC Security Model:**
- Compute nodes are isolated from internet for security
- Only development/login nodes have internet access
- Prevents compromised jobs from exfiltrating data
- Standard practice across most HPC facilities

### Two-Phase Workflow Required

#### Phase 1: Data Download (on devel node, ~30-60 min)
```bash
# SSH to devel node (has internet)
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# Download all data
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh
```

Downloads:
- 240 dataset shards (~24GB)
- Evaluation bundle (~162MB)
- Identity conversations (~2.3MB)
- GPT-2/GPT-4 tokenizers
- SmolTalk dataset

#### Phase 2: Training (on GPU nodes, ~12 hours)
```bash
# Submit from any login node
cd /scratch/ptolemy/users/$USER/slurm-nanochat
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```

The job:
- Verifies all data exists (fails fast if not)
- Runs training with pre-downloaded data only
- No network operations during training

### What Happens if You Skip Phase 1?

The job fails immediately with:
```
❌ ERROR: Required data not found!

GPU compute nodes do NOT have internet access.
All data must be downloaded BEFORE submitting this job.

[Instructions provided in error message]
```

---

## Node Types

### Development Nodes (ptolemy-devel-1, ptolemy-devel-2)

**Purpose:** Setup, data download, development

**Features:**
- ✅ Internet access
- ✅ SSH access
- ✅ Python 3.12.5 available
- ✅ Rust/Cargo can be installed
- ❌ No GPUs

**Use for:**
- Running `setup_environment.sh`
- Running `download_data.sh`
- Installing dependencies
- Testing scripts (without GPU)

### Login Nodes (ptolemy-login.arc.msstate.edu)

**Purpose:** Job submission, file management

**Features:**
- ✅ SSH access
- ✅ SLURM commands
- ✅ Access to scratch storage
- ❌ No internet (usually)
- ❌ No GPUs
- ❌ Not for compute tasks

**Use for:**
- Submitting SLURM jobs
- Checking job status
- Managing files
- Downloading results

### GPU Compute Nodes (allocated via SLURM)

**Purpose:** Training, inference

**Features:**
- ✅ 8x A100 80GB GPUs
- ✅ High CPU count
- ✅ Large RAM
- ❌ **NO INTERNET ACCESS**
- ❌ No direct SSH (via SLURM only)

**Access via:**
- SLURM batch jobs (`sbatch`)
- Interactive jobs (`srun`)

---

## Storage Configuration

### Critical: Use /scratch, NOT Home Directory

**Why /scratch?**
- Large quotas (~5TB vs ~5GB)
- No backups (saves space)
- Faster I/O
- Designed for temporary large datasets

### Storage Layout

```
/scratch/ptolemy/users/$USER/
├── slurm-nanochat/              # Repository
├── nanochat-venv/               # Virtual environment (~2-3GB)
├── nanochat-cache/              # Training data (~30-50GB)
│   ├── base_data/               # 240 shards (~24GB)
│   ├── models/                  # Checkpoints (~2-5GB)
│   ├── tokenizer/               # Tokenizer (~100MB)
│   ├── eval_bundle/             # Eval data (~162MB)
│   └── identity_conversations.jsonl
└── cache/                       # Various caches (~5-10GB)
    ├── huggingface/
    ├── torch/
    ├── uv/
    └── pip/
```

**Total:** ~35-50GB (well within scratch quotas)

### Home Directory

Keep minimal in `~`:
- `.env.local` (config file)
- `.bashrc` (shell config)
- SSH keys
- **NO LARGE DATA**

**Quota:** Usually ~5GB (varies)

### Environment Variables Set

All scripts configure:
```bash
NANOCHAT_BASE_DIR=/scratch/ptolemy/users/$USER/nanochat-cache
HF_HOME=/scratch/ptolemy/users/$USER/cache/huggingface
TORCH_HOME=/scratch/ptolemy/users/$USER/cache/torch
CARGO_HOME=/scratch/ptolemy/users/$USER/.cargo
RUSTUP_HOME=/scratch/ptolemy/users/$USER/.rustup
UV_CACHE_DIR=/scratch/ptolemy/users/$USER/cache/uv
PIP_CACHE_DIR=/scratch/ptolemy/users/$USER/cache/pip
```

### Verification

Check storage usage:
```bash
# Scratch usage (should be ~35-50GB)
du -sh /scratch/ptolemy/users/$USER/

# Home usage (should be < 100MB)
du -sh ~

# Check quotas
quota -s
```

---

## Resource Allocation

### SLURM Configuration

```bash
#SBATCH --partition=gpu-a100      # GPU partition
#SBATCH --nodes=1                 # Single node
#SBATCH --gres=gpu:8              # 8 A100 GPUs
#SBATCH --cpus-per-task=16        # 16 CPUs
#SBATCH --mem=256G                # 256GB RAM
#SBATCH --time=12:00:00           # 12-hour limit
#SBATCH --account=class-cse8990   # Course allocation
```

### QOS Limits

**class-cse8990 QOS:**
- Maximum time: 12 hours
- Maximum nodes: 1
- GPUs: A100 80GB
- Shared allocation among students

### Time Expectations

```
Tokenizer eval:    ~5 minutes
Base pretraining:  ~7-8 hours
Midtraining:       ~2-3 hours
SFT:               ~1-2 hours
Report:            ~5 minutes
Total:             ~10-12 hours
```

**Note:** A100s are slower than H100s (production uses H100s → ~4 hours total)

---

## Python Environment

### Version Management

**System Python:** 3.9.14 (too old!)

**Module Python:** 3.12.5 ✅
```bash
module load python/3.12.5
```

All scripts automatically load Python 3.12.5.

### Virtual Environment Location

**Created in scratch:**
```bash
/scratch/ptolemy/users/$USER/nanochat-venv/
```

**Not** in the repo directory (would be too large).

### Dependencies

Installed via:
```bash
pip install -e '.[gpu]'
```

Includes:
- PyTorch with CUDA support
- All nanochat requirements
- GPU-specific packages

---

## Email Notifications

### Configuration

Set in `.env.local`:
```bash
EMAIL=your_email@msstate.edu
```

### When You'll Receive Emails

- **BEGIN:** Job starts on compute node
- **END:** Job completes successfully
- **FAIL:** Job fails

### Email Contents

Includes:
- Job ID
- Node name
- Exit status
- Basic timing info

---

## Common Workflows

### Initial Setup (One-Time)

```bash
# 1. SSH to devel node
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# 2. Navigate to scratch
cd /scratch/ptolemy/users/$USER

# 3. Clone repo
git clone <repo-url> slurm-nanochat
cd slurm-nanochat

# 4. Configure email
cp .env.local.example .env.local
nano .env.local  # Set email

# 5. Setup environment
bash scripts/setup_environment.sh

# 6. Download all data
bash scripts/download_data.sh  # ~30-60 min
```

### Submitting Training Job

```bash
# 1. SSH to any login node
ssh <username>@ptolemy-login.arc.msstate.edu

# 2. Navigate to project
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# 3. Submit job (CRITICAL: Set WANDB_RUN)
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm
```

### Monitoring Job

```bash
# Check status
squeue -u $USER

# Watch output
tail -f logs/nanochat_speedrun_*.out

# Check errors
tail -f logs/nanochat_speedrun_*.err
```

### Interactive GPU Session

```bash
# Request 1 GPU for 1 hour
srun --account=class-cse8990 --partition=gpu-a100 \
     --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash

# Activate environment
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"

# Chat with model
python -m scripts.chat_cli
```

---

## Unique Ptolemy Constraints

### 1. No Internet on GPU Nodes
**Solution:** Two-phase workflow (download → train)

### 2. Limited Time (12 hours max)
**Impact:** Training must complete within limit
**Mitigation:** A100 training usually finishes in ~10-12 hours

### 3. Shared Resource Allocation
**Impact:** Queue times vary
**Mitigation:** Submit jobs during off-peak hours

### 4. Storage Quotas
**Impact:** Must use /scratch, not home
**Mitigation:** All scripts configured for /scratch

### 5. Module System
**Impact:** Need to load Python 3.12.5
**Mitigation:** Scripts automatically load modules

---

## Comparison with Other Environments

| Feature | Ptolemy HPC | Production GPU | Local CPU |
|---------|-------------|----------------|-----------|
| **Internet** | Devel only | ✅ Yes | ✅ Yes |
| **Setup Complexity** | High | Medium | Low |
| **Job Submission** | SLURM | Direct | Direct |
| **Time Limit** | 12 hours | None | None |
| **Storage** | /scratch required | Any | Any |
| **Queue Time** | Variable | None | None |
| **Cost** | Free | ~$100 | Free |

---

## Best Practices

### Before First Use

- [ ] Understand two-phase workflow
- [ ] Know difference between devel, login, and compute nodes
- [ ] Configure email notifications
- [ ] Verify storage locations

### Before Each Job

- [ ] Ensure all data downloaded (on devel node)
- [ ] Set WANDB_RUN environment variable
- [ ] Check scratch disk space
- [ ] Verify virtual environment exists

### During Training

- [ ] Monitor via email notifications
- [ ] Occasionally check `squeue -u $USER`
- [ ] Don't worry if SSH disconnects (job keeps running)

### After Training

- [ ] Download report.md
- [ ] Test chat functionality
- [ ] Consider cleaning up large files if done

---

## Quick Reference

### Node Access

```bash
# Development (internet, setup, downloads)
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# Login (job submission, monitoring)
ssh <username>@ptolemy-login.arc.msstate.edu

# Compute (via SLURM only, no direct SSH)
```

### Essential Commands

```bash
# Submit job
WANDB_RUN=name sbatch scripts/speedrun.slurm

# Check status
squeue -u $USER

# Cancel job
scancel <JOBID>

# Interactive session
srun --account=class-cse8990 --partition=gpu-a100 \
     --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
```

### Storage Paths

```bash
# Repo
/scratch/ptolemy/users/$USER/slurm-nanochat/

# Virtual environment
/scratch/ptolemy/users/$USER/nanochat-venv/

# Training data
/scratch/ptolemy/users/$USER/nanochat-cache/
```

---

## Troubleshooting HPC-Specific Issues

**Job fails immediately:**
→ Check `logs/nanochat_speedrun_*.err`
→ Usually WANDB_RUN or missing data

**Can't download data:**
→ Must be on ptolemy-devel-1 or devel-2
→ Check `hostname`

**Out of disk space:**
→ Using home instead of scratch
→ Verify environment variables

**Python version wrong:**
→ Module not loaded
→ Run `setup_environment.sh`

For complete troubleshooting:
→ [Troubleshoot Common Issues](../how-to/troubleshoot-common-issues.md)

---

**Summary:** Ptolemy HPC provides free GPU access but requires understanding its unique constraints (no internet on GPU nodes, storage configuration, SLURM workflow). Following the two-phase workflow and using /scratch storage ensures successful training.
