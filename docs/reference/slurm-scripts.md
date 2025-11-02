# Reference: SLURM Scripts

Complete reference for all SLURM job scripts used in this project.

---

## Overview

This project includes two main SLURM scripts for training on Ptolemy HPC:

| Script | Purpose | Training Time | Use When |
|--------|---------|---------------|----------|
| `speedrun.slurm` | Complete training pipeline | ~10-12 hours | First training run or starting from scratch |
| `resume_mid_sft.slurm` | Resume from midtraining | ~4-6 hours | Base training completed, resume mid+SFT |

---

## `scripts/speedrun.slurm`

### Purpose

Runs the complete training pipeline:
1. Tokenizer evaluation
2. Base pretraining (~7 hours)
3. Evaluation
4. Midtraining (~2-3 hours)
5. Evaluation
6. Supervised finetuning (~1-2 hours)
7. Evaluation
8. Report generation

### Usage

```bash
# CRITICAL: Always set WANDB_RUN
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm

# ❌ WRONG - Will fail immediately
sbatch scripts/speedrun.slurm

# ❌ WRONG - Will skip midtraining and SFT
WANDB_RUN=dummy sbatch scripts/speedrun.slurm
```

### SLURM Parameters

```bash
#SBATCH --account=class-cse8990        # Class account
#SBATCH --partition=gpu-a100           # A100 GPU partition
#SBATCH --gres=gpu:8                   # Request 8 GPUs
#SBATCH --nodes=1                      # Single node
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=48             # 48 CPU cores
#SBATCH --mem=0                        # All memory on node
#SBATCH --time=12:00:00                # 12 hour time limit (QOS max)
#SBATCH --output=logs/nanochat_speedrun_%j.out
#SBATCH --error=logs/nanochat_speedrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # Email notifications
```

### Environment Setup

```bash
# Load Python module
module load python/3.12.5

# Activate virtual environment
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate

# Set cache directory
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"

# Configure WandB offline mode (GPU nodes have no internet)
export WANDB_MODE=offline
export WANDB_DIR="$NANOCHAT_BASE_DIR/wandb"
```

### Validation Checks

The script performs critical validation before training:

```bash
# 1. Check WANDB_RUN is set
if [ -z "$WANDB_RUN" ]; then
    echo "ERROR: WANDB_RUN environment variable is not set!"
    exit 1
fi

# 2. Check WANDB_RUN is not "dummy"
if [ "$WANDB_RUN" = "dummy" ]; then
    echo "ERROR: WANDB_RUN is set to 'dummy'!"
    exit 1
fi

# 3. Verify base training data exists
if [ ! -d "$NANOCHAT_BASE_DIR/base_data" ]; then
    echo "ERROR: Required data not found!"
    exit 1
fi
```

### Training Commands

**Tokenizer Evaluation:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.tok_eval
```

**Base Pretraining:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --run=$WANDB_RUN \
    --num_iterations=21400 \
    --eval_interval=500 \
    --device_batch_size=32 \
    --total_batch_size=4194304
```

**Base Evaluation:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

**Midtraining:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --run=$WANDB_RUN
```

**Mid Evaluation:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

**Supervised Finetuning:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --run=$WANDB_RUN
```

**SFT Evaluation:**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**Report Generation:**
```bash
python -m scripts.report
```

### Prerequisites

Before submitting this job:

1. **Download base training data** (on devel node):
   ```bash
   bash scripts/download_data.sh
   ```

2. **Download midtraining/SFT datasets** (on devel node):
   ```bash
   bash scripts/download_after_basetraining.sh
   ```

3. **Configure email** (optional):
   ```bash
   cp .env.local.example .env.local
   nano .env.local  # Set your email
   ```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output log
tail -f logs/nanochat_speedrun_<JOBID>.out

# Check error log
tail -f logs/nanochat_speedrun_<JOBID>.err
```

### Expected Timeline

```
00:00 - Job starts
00:00 - Validation checks pass
00:05 - Tokenizer evaluation complete
07:00 - Base pretraining complete (CORE ~0.21)
09:30 - Midtraining complete
11:30 - Supervised finetuning complete
11:36 - Report generated, job complete
```

---

## `scripts/resume_mid_sft.slurm`

### Purpose

Resume training from midtraining phase when base training has already completed successfully.

**Saves ~7 hours** by skipping base training.

### When to Use

Use this script when:
- ✅ Base training completed successfully
- ✅ Base checkpoint exists at `base_checkpoints/d20/`
- ❌ Midtraining failed or didn't run
- ❌ SFT failed or didn't run

**Common scenarios:**
- Job timed out after base training
- Dataset download errors during midtraining
- WANDB_RUN was set to "dummy" (skipped mid/SFT)

### Usage

```bash
# 1. Download required datasets (if not already done)
ssh <username>@ptolemy-devel-1.arc.msstate.edu
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_after_basetraining.sh

# 2. Submit resume job
WANDB_RUN=my_resume_run sbatch scripts/resume_mid_sft.slurm
```

### SLURM Parameters

Same as `speedrun.slurm` except:

```bash
#SBATCH --time=06:00:00                # 6 hours (reduced from 12)
#SBATCH --output=logs/nanochat_resume_mid_sft_%j.out
#SBATCH --error=logs/nanochat_resume_mid_sft_%j.err
```

### Validation Checks

Additional validation specific to resuming:

```bash
# 1. Standard WANDB_RUN checks (same as speedrun.slurm)

# 2. Verify base model checkpoint exists
BASE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
if [ ! -d "$BASE_CHECKPOINT_DIR" ]; then
    echo "❌ ERROR: Base model checkpoint not found!"
    echo "Base training must complete before resuming."
    echo "Expected checkpoint at: $BASE_CHECKPOINT_DIR"
    exit 1
fi
```

### Training Commands

Runs only midtraining + SFT pipeline:

```bash
# Midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --run=$WANDB_RUN

# Mid evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# Supervised finetuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --run=$WANDB_RUN

# SFT evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Report generation
python -m scripts.report
```

### Expected Timeline

```
00:00 - Job starts
00:00 - Validation checks pass (including base checkpoint)
00:00 - Skips base training (already done)
02:30 - Midtraining complete
04:30 - Supervised finetuning complete
04:35 - Report generated, job complete
```

Total: ~4-6 hours (vs. ~10-12 hours for full pipeline)

### Prerequisites

1. **Base training completed**:
   - Checkpoint must exist at `base_checkpoints/d20/`
   - Run `ls $NANOCHAT_BASE_DIR/base_checkpoints/d20/` to verify

2. **Datasets downloaded** (on devel node):
   ```bash
   bash scripts/download_after_basetraining.sh
   ```

   Required datasets:
   - MMLU (auxiliary_train)
   - GSM8K (main)
   - SmolTalk (default)
   - ARC (ARC-Easy)

---

## Common Issues

### Issue 1: WANDB_RUN Not Set

**Error:**
```
ERROR: WANDB_RUN environment variable is not set!
```

**Solution:**
```bash
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```

### Issue 2: WANDB_RUN is "dummy"

**Error:**
```
ERROR: WANDB_RUN is set to 'dummy' which will skip midtraining and SFT!
```

**Solution:**
```bash
WANDB_RUN=my_actual_run sbatch scripts/speedrun.slurm
```

### Issue 3: Base Data Not Found

**Error:**
```
ERROR: Required data not found!
```

**Solution:**
```bash
# SSH to devel node
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# Download base data
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh
```

### Issue 4: Midtraining/SFT Dataset Errors

**Error:**
```
ConnectionError: Couldn't reach 'cais/mmlu' on the Hub (LocalEntryNotFoundError)
```

**Solution:**
```bash
# SSH to devel node
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# Download midtraining/SFT datasets
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_after_basetraining.sh
```

### Issue 5: Base Checkpoint Not Found (Resume Script)

**Error:**
```
ERROR: Base model checkpoint not found!
```

**Solution:**

You must complete base training first:
```bash
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```

Or verify the checkpoint exists:
```bash
ls /scratch/ptolemy/users/$USER/nanochat-cache/base_checkpoints/d20/
```

### Issue 6: Job Times Out

**Symptom:** Job reaches 12-hour limit before completion

**Check:**
```bash
tail -100 logs/nanochat_speedrun_*.out
```

**Expected progress:**
- Base training: ~7 hours
- If base training is taking >8 hours, something is wrong

**Solutions:**
- Check GPU utilization in logs
- Contact HPC support if abnormal
- Consider using resume script if base training completed

### Issue 7: Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**

Edit the SLURM script to reduce batch size:
```bash
--device_batch_size=16  # Instead of 32
```

---

## File Locations

### Input Data

```
/scratch/ptolemy/users/$USER/nanochat-cache/
├── base_data/                      # Base training data (~24GB)
│   ├── shard_00000.parquet
│   ├── shard_00001.parquet
│   └── ... (240 shards total)
│
├── eval_bundle/                    # Evaluation data (~162MB)
│   └── ...
│
└── cache/huggingface/datasets/     # HuggingFace datasets
    ├── cais___mmlu/                # MMLU (midtraining)
    ├── openai___gsm8k/             # GSM8K (mid + SFT)
    ├── HuggingFaceTB___smol-smoltalk/  # SmolTalk (mid + SFT)
    └── allenai___ai2_arc/          # ARC (SFT)
```

### Output Files

```
/scratch/ptolemy/users/$USER/nanochat-cache/
├── tokenizer/
│   └── tokenizer.pkl               # BPE tokenizer
│
├── base_checkpoints/
│   └── d20/                        # Base model checkpoint
│
├── mid_checkpoints/
│   └── d20/                        # Midtrained model
│
├── chatsft_checkpoints/
│   └── d20/                        # SFT model (final)
│
├── wandb/                          # WandB logs (offline mode)
│   └── ...
│
└── report/                         # Training logs
    └── ...
```

### Logs

```
/scratch/ptolemy/users/$USER/slurm-nanochat/logs/
├── nanochat_speedrun_<JOBID>.out   # Standard output
├── nanochat_speedrun_<JOBID>.err   # Error output
├── nanochat_resume_mid_sft_<JOBID>.out
└── nanochat_resume_mid_sft_<JOBID>.err
```

### Report

```
/scratch/ptolemy/users/$USER/slurm-nanochat/report.md
```

---

## Environment Variables

### Required

**`WANDB_RUN`** - Run identifier (MUST be set)
```bash
export WANDB_RUN=my_training_run
```

### Automatically Set by Scripts

**`NANOCHAT_BASE_DIR`** - Cache directory
```bash
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
```

**`WANDB_MODE`** - WandB offline mode (GPU nodes have no internet)
```bash
export WANDB_MODE=offline
```

**`WANDB_DIR`** - WandB log directory
```bash
export WANDB_DIR="$NANOCHAT_BASE_DIR/wandb"
```

**`HF_HOME`** - HuggingFace cache directory
```bash
export HF_HOME="/scratch/ptolemy/users/$USER/cache/huggingface"
```

---

## Performance Tuning

### Batch Size Tuning

**Device batch size** - Per-GPU batch size
```bash
--device_batch_size=32  # Default (8 GPUs × 32 = 256 samples per step)
```

Reduce if OOM:
```bash
--device_batch_size=16  # 8 GPUs × 16 = 128 samples per step
```

**Total batch size** - Global batch size (all GPUs)
```bash
--total_batch_size=4194304  # Total tokens per optimization step
```

This is the **effective batch size** after gradient accumulation.

### Iteration Tuning

**Base training:**
```bash
--num_iterations=21400  # Default (~11.2B tokens)
```

**Evaluation interval:**
```bash
--eval_interval=500  # Run evaluation every 500 steps
```

### Memory Optimization

**Reduce sequence length** (if OOM):
```bash
--max_seq_len=1024  # Instead of 2048
```

**Enable gradient checkpointing** (not currently used):
```bash
--gradient_checkpointing=True
```

---

## Quick Reference

### Submit Full Training

```bash
# 1. Download all data (devel node)
bash scripts/download_data.sh
bash scripts/download_after_basetraining.sh

# 2. Submit job
WANDB_RUN=my_run sbatch scripts/speedrun.slurm

# 3. Monitor
squeue -u $USER
tail -f logs/nanochat_speedrun_*.out
```

### Resume from Midtraining

```bash
# 1. Verify base checkpoint exists
ls /scratch/ptolemy/users/$USER/nanochat-cache/base_checkpoints/d20/

# 2. Download datasets (if needed)
bash scripts/download_after_basetraining.sh

# 3. Submit resume job
WANDB_RUN=my_resume sbatch scripts/resume_mid_sft.slurm

# 4. Monitor
squeue -u $USER
tail -f logs/nanochat_resume_mid_sft_*.out
```

### Check Results

```bash
# View report
cat /scratch/ptolemy/users/$USER/slurm-nanochat/report.md

# Chat with model
srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
python -m scripts.chat_cli
```

---

## See Also

- [How-To: Run a Training Job](../how-to/run-a-training-job.md)
- [Tutorial: HPC First Job](../tutorials/02-hpc-first-job.md)
- [How-To: Troubleshoot Common Issues](../how-to/troubleshoot-common-issues.md)
- [Explanation: HPC Environment Details](../explanation/hpc-environment.md)
