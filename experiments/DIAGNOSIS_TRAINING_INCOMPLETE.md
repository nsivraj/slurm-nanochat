# Training Incomplete - Missing SFT Checkpoint

## Problem Summary

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '/scratch/ptolemy/users/ncj79/nanochat-cache/chatsft_checkpoints'`

**What happened**: The training pipeline only completed the **base model** phase and started midtraining but never completed the full training pipeline through SFT (Supervised Fine-Tuning).

## Evidence from Report

### ✅ What Completed Successfully

1. **Tokenizer Training** ✅
   - Timestamp: 2025-11-01 12:32:33
   - Completed successfully

2. **Base Model Training** ✅
   - Timestamp: 2025-11-01 19:34:45 (completed)
   - Duration: ~7 hours (12:29:31 → 19:34:45)
   - Final validation: 0.8123 bpb
   - Model saved: `/scratch/ptolemy/users/ncj79/nanochat-cache/base_checkpoints/d20/`
   - Parameters: 560M
   - Iterations: 21,400

3. **Base Model Evaluation** ✅
   - Timestamp: 2025-11-01 19:41:05
   - CORE metric: 0.2042

### ❌ What Did NOT Complete

From the report.md summary table:

```markdown
| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2042   | -        | -        | -        |
```

**Missing**:
- ❌ Midtraining (MID) - Started but did not complete
- ❌ Supervised Fine-Tuning (SFT) - Never ran
- ❌ Reinforcement Learning (RL) - Never ran

### Error Log Evidence

From `nanochat_speedrun_76356.err`:

```
FileNotFoundError: [Errno 2] No such file or directory:
'/scratch/ptolemy/users/ncj79/nanochat-cache/chatsft_checkpoints'
```

This error occurred when trying to evaluate the chat model, but the SFT checkpoint directory was never created because SFT training never completed.

## Root Cause Analysis

The training job **crashed or was interrupted** during the midtraining phase:

1. Base training completed successfully (7 hours)
2. Midtraining started but report shows no completion timestamp
3. The job appears to have failed during midtraining or SFT evaluation
4. No SFT checkpoints were ever created

**Likely causes**:
- Job time limit exceeded (SLURM job timeout)
- Out of memory during midtraining
- Node failure or preemption
- Script error during midtraining phase

## Current State

### What You Have

✅ **Base model checkpoint**: `/scratch/ptolemy/users/ncj79/nanochat-cache/base_checkpoints/d20/`

You can use the base model with:
```bash
python -m scripts.chat_cli --source=base -p "Why is the sky blue?"
```

### What You're Missing

❌ **SFT (chat) model**: `/scratch/ptolemy/users/ncj79/nanochat-cache/chatsft_checkpoints/`

The default `chat_cli` tries to load from `source=sft` which doesn't exist.

## Solution Options

### Option 1: Use the Base Model (Quick Fix)

The base model exists and can be used:

```bash
# On HPC
python -m scripts.chat_cli --source=base -p "Why is the sky blue?"
```

**Note**: The base model won't be as good at chat/instruction-following since it hasn't been fine-tuned.

### Option 2: Complete the Training Pipeline

Re-run the speedrun script, but this time:

1. **Check SLURM time limits**: The base training took 7 hours. You need at least 12-15 hours for the full pipeline.

2. **Increase job time**:
   ```bash
   # In your SLURM script
   #SBATCH --time=20:00:00  # 20 hours instead of default
   ```

3. **Check the speedrun script**:
   ```bash
   # Verify which phases will run
   bash scripts/speedrun.sh --help
   ```

4. **Resume from midtraining** (if possible):
   Check if there's a way to skip base training and start from midtraining since you already have the base checkpoint.

### Option 3: Run Just Midtraining + SFT

If your base model is good, you might be able to run just the remaining phases:

```bash
# Check for individual training scripts
python -m scripts.mid_train  # Midtraining
python -m scripts.chat_sft   # SFT
```

## Checking Your HPC Environment

Run these commands on HPC to diagnose:

```bash
# 1. Check what checkpoints you have
ls -lh /scratch/ptolemy/users/ncj79/nanochat-cache/*/

# 2. Check base model checkpoint
ls -lh /scratch/ptolemy/users/ncj79/nanochat-cache/base_checkpoints/d20/

# 3. Check if midtraining started
ls -lh /scratch/ptolemy/users/ncj79/nanochat-cache/mid_checkpoints/ 2>/dev/null || echo "No mid checkpoints"

# 4. Check SLURM job history
sacct -j 76356 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,TimeLimit

# 5. Check full error log for the actual failure point
grep -i "error\|failed\|killed" logs/nanochat_speedrun_76356.err | tail -50
```

## Expected Training Timeline

Based on your base training (7 hours for 21,400 iterations):

| Phase | Estimated Time | Status |
|-------|---------------|--------|
| Tokenizer | 5-10 min | ✅ Done |
| Base Training | 7 hours | ✅ Done |
| Base Evaluation | 5-10 min | ✅ Done |
| Midtraining | 2-3 hours | ❌ Failed/Incomplete |
| Mid Evaluation | 10-20 min | ❌ Never ran |
| SFT | 2-3 hours | ❌ Never ran |
| SFT Evaluation | 10-20 min | ❌ Never ran |
| **Total** | **~12-15 hours** | **Partial** |

Your job stopped after ~7 hours, suggesting a time limit issue.

## Recommended Next Steps

1. **Immediate**: Use the base model you have:
   ```bash
   python -m scripts.chat_cli --source=base -p "test"
   ```

2. **Short-term**: Check SLURM job settings and resubmit with longer time limit

3. **Long-term**: Consider checkpointing strategy to resume from failures

## Files to Review on HPC

```bash
# Full output log
less logs/nanochat_speedrun_76356.out

# Full error log
less logs/nanochat_speedrun_76356.err

# SLURM job info
scontrol show job 76356

# Check available checkpoints
find /scratch/ptolemy/users/ncj79/nanochat-cache -name "*.pt" -type f
```
