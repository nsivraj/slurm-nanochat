# ⚠️ IMPORTANT: Ptolemy HPC Internet Access Limitations

## Critical Information

**GPU compute nodes on Ptolemy do NOT have internet access!**

This is a common HPC security practice, but it means the nanochat workflow must be split into two phases:

## Two-Phase Workflow

### Phase 1: Data Download (on ptolemy-devel-1)

**Server:** `ptolemy-devel-1.arc.msstate.edu` (has internet)
**Duration:** ~30-60 minutes
**Data Size:** ~24GB

```bash
# SSH to login node
ssh [username]@ptolemy-login.arc.msstate.edu
# SSH to devel node
ssh [username]@ptolemy-devel-1.arc.msstate.edu

# Navigate to nanochat
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Download all required data
bash scripts/download_data.sh
```

This script downloads:

- 240 dataset shards (~24GB) for training
- Evaluation bundle (~162MB)
- Identity conversations (~2.3MB)
- Builds and trains the BPE tokenizer

### Phase 2: Training (on GPU compute nodes)

**Server:** GPU compute nodes (via SLURM, NO internet)
**Duration:** ~4 hours
**Resource:** 8x A100 GPUs

```bash
# Submit from any Ptolemy login node
cd /scratch/ptolemy/users/$USER/slurm-nanochat
sbatch scripts/speedrun.slurm
```

The SLURM job will:

1. **Verify all data is downloaded** (fails if not)
2. Run training pipeline (pretraining → midtraining → SFT)
3. Generate report with results
4. To get the report after running the training: scp [username]@ptolemy-login.arc.msstate.edu:/scratch/ptolemy/users/$USER/slurm-nanochat/report.md .
5. Example: scp ncj79@ptolemy-login.arc.msstate.edu:/scratch/ptolemy/users/ncj79/slurm-nanochat/report.md .

## What Happens If You Skip Phase 1?

If you submit the SLURM job without downloading data first:

```
❌ ERROR: Required data not found!

GPU compute nodes do NOT have internet access.
All data must be downloaded BEFORE submitting this job.

To download required data:
1. SSH to development server (which has internet):
   ssh [username]@ptolemy-devel-1.arc.msstate.edu
2. Navigate to nanochat directory:
   cd /scratch/ptolemy/users/$USER/slurm-nanochat
3. Run the data download script:
   bash scripts/download_data.sh
4. Wait for download to complete (~30-60 minutes for ~24GB data)
5. Re-submit this SLURM job:
   sbatch scripts/speedrun.slurm
```

The job will fail immediately (within seconds) with this error message.

## Why This Design?

### HPC Security Model

- Compute nodes are isolated from internet for security
- Only development/login nodes have internet access
- Prevents compromised jobs from exfiltrating data
- Standard practice across most HPC facilities

### Our Solution

1. **`scripts/download_data.sh`** - Run on devel node with internet

   - Downloads all required data
   - Builds tokenizer
   - Stores in scratch storage

2. **`scripts/speedrun.slurm`** - Run on GPU nodes without internet
   - Verifies data exists (fails fast if not)
   - Uses pre-downloaded data only
   - No network operations during training

## Quick Reference

| Task                 | Server                 | Internet? | Duration  |
| -------------------- | ---------------------- | --------- | --------- |
| Setup environment    | ptolemy-devel-1        | ✅ Yes    | 15-30 min |
| Download data        | ptolemy-devel-1        | ✅ Yes    | 30-60 min |
| Submit SLURM job     | Any login node         | N/A       | < 1 min   |
| Training (via SLURM) | GPU compute nodes      | ❌ No     | ~4 hours  |
| Chat with model      | GPU node (interactive) | ❌ No     | N/A       |

## Files Created for This Workflow

- **`scripts/download_data.sh`** ⭐ NEW

  - Handles all internet-dependent operations
  - Must be run on ptolemy-devel-1
  - Downloads ~24GB of data
  - Verifies downloads complete successfully

- **`scripts/speedrun.slurm`** (modified)
  - Removed all internet operations
  - Added data verification checks
  - Fails fast with helpful error if data missing
  - Safe to run on isolated GPU nodes

## Common Mistakes to Avoid

❌ **Don't:** Try to download data from within SLURM job
✅ **Do:** Download data on ptolemy-devel-1 first

❌ **Don't:** Assume the original `speedrun.sh` will work
✅ **Do:** Use the modified `speedrun.slurm` for Ptolemy

❌ **Don't:** Skip the data download step to save time
✅ **Do:** Run `download_data.sh` even though it takes ~1 hour

❌ **Don't:** Submit job from ptolemy-devel-1
✅ **Do:** Submit from login node after data is downloaded

## Verification

Before submitting the SLURM job, verify data is downloaded:

```bash
# Check dataset shards
ls -lh /scratch/ptolemy/users/$USER/nanochat-cache/data/*.bin | wc -l
# Should show 240 or more

# Check tokenizer
ls -lh /scratch/ptolemy/users/$USER/nanochat-cache/tokenizer/
# Should show tokenizer_2pow16.model

# Check eval bundle
ls -lh /scratch/ptolemy/users/$USER/nanochat-cache/eval_bundle/
# Should show evaluation files

# Check identity data
ls -lh /scratch/ptolemy/users/$USER/nanochat-cache/identity_conversations.jsonl
# Should exist (~2.3MB)
```

If all files exist, you're ready to submit the SLURM job!

## TL;DR

1. **First time only:** Run on ptolemy-devel-1

   ```bash
   bash scripts/download_data.sh  # ~1 hour, ~24GB
   ```

2. **Every time:** Submit SLURM job from any login node

   ```bash
   sbatch scripts/speedrun.slurm  # ~4 hours on GPUs
   ```

3. **The SLURM job will fail immediately if you skip step 1**

---

For detailed setup instructions, see [PTOLEMY_SETUP.md](PTOLEMY_SETUP.md)
