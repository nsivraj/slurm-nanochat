# Resume Training Failure - Missing MMLU Dataset

**Job ID**: 76383
**Script**: `scripts/resume_mid_sft.slurm`
**Failure Time**: 2025-11-01 21:55:04
**Phase**: Midtraining initialization

## Problem Summary

**Error**: `ConnectionError: Couldn't reach 'cais/mmlu' on the Hub (LocalEntryNotFoundError)`

**What happened**: The midtraining script (`scripts/mid_train.py`) tried to load the MMLU dataset from HuggingFace Hub, but:
1. GPU compute nodes have **no internet access**
2. The MMLU dataset was **not downloaded** during the initial data preparation
3. The script failed at line 100 when initializing the `MMLU` task

## Error Details

### Error Location

From `logs/nanochat_resume_mid_sft_76383.err` (lines 174-193):

```python
File "/scratch/ptolemy/users/ncj79/slurm-nanochat/scripts/mid_train.py", line 100, in <module>
    MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/scratch/ptolemy/users/ncj79/slurm-nanochat/tasks/mmlu.py", line 22, in __init__
    self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ConnectionError: Couldn't reach 'cais/mmlu' on the Hub (LocalEntryNotFoundError)
```

### What Worked

✅ **Environment setup** - All modules loaded correctly
✅ **Base model checkpoint verification** - Found at `/scratch/ptolemy/users/ncj79/nanochat-cache/base_checkpoints/d20`
✅ **Required data verification**:
- Identity conversations present
- Evaluation bundle present
✅ **WANDB_RUN validation** - Set to `my_training_run`
✅ **Model loading** - Base model loaded successfully (561M params, d20 config)
✅ **SmolTalk dataset** - Loaded from cache (already downloaded)

### What Failed

❌ **MMLU dataset** - Not in cache, tried to download from Hub but no internet access

## Root Cause Analysis

The `scripts/download_data.sh` script (used during initial setup) downloads:
1. ✅ Base training data (240 parquet shards)
2. ✅ Tokenizer training data
3. ✅ Identity conversations (for midtraining)
4. ✅ Evaluation bundle
5. ✅ SmolTalk dataset (cached)
6. ❌ **MMLU dataset** - NOT downloaded

The midtraining script requires MMLU but it was never pre-cached because:
- The original speedrun (job 76356) **failed before reaching midtraining**
- The download script doesn't pre-download MMLU
- MMLU is loaded on-demand during midtraining initialization

## Missing Dataset Details

**Dataset**: `cais/mmlu`
**Subset**: `auxiliary_train`
**Split**: `train`
**Purpose**: Provides 100K rows of multiple choice problems for midtraining
**Usage**: Part of the TaskMixture in `scripts/mid_train.py:100`

## Solution Options

### Option 1: Download MMLU on Development Node (RECOMMENDED)

Run this on `ptolemy-devel-1` (which has internet access):

```bash
# SSH to development server
ssh your_username@ptolemy-devel-1.arc.msstate.edu

# Navigate to nanochat
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Activate virtual environment
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate

# Set HuggingFace cache directory
export HF_HOME="/scratch/ptolemy/users/$USER/cache/huggingface"

# Download MMLU dataset
python3 << 'EOF'
from datasets import load_dataset
import os

# Set cache directory
os.environ['HF_HOME'] = f"/scratch/ptolemy/users/{os.environ['USER']}/cache/huggingface"

print("Downloading MMLU dataset...")
ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
print(f"✓ Downloaded {len(ds)} rows")
print(f"✓ Cached at: {os.environ['HF_HOME']}/datasets/cais___mmlu/")
EOF

# Verify download
ls -lh /scratch/ptolemy/users/$USER/cache/huggingface/datasets/cais___mmlu/

# Re-submit the resume job
WANDB_RUN=my_training_run sbatch --mail-user=your_email@msstate.edu scripts/resume_mid_sft.slurm
```

### Option 2: Update download_data.sh Script

Add MMLU download to the data preparation script:

```bash
# Edit scripts/download_data.sh
# Add before the "Download identity conversations" section:

echo "Downloading MMLU dataset for midtraining..."
python3 << 'EOF'
from datasets import load_dataset
print("Downloading MMLU auxiliary_train...")
ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
print(f"✓ Downloaded {len(ds)} rows")
EOF
```

Then run the updated script:
```bash
bash scripts/download_data.sh
```

### Option 3: Modify mid_train.py to Skip MMLU (NOT RECOMMENDED)

This would change the training data distribution and affect model quality.

## Recommended Next Steps

1. **Immediate**: Download MMLU on ptolemy-devel-1 using Option 1
2. **Verify**: Check that the dataset is cached:
   ```bash
   ls -lh /scratch/ptolemy/users/$USER/cache/huggingface/datasets/cais___mmlu/
   ```
3. **Resubmit**: Run the resume job again:
   ```bash
   WANDB_RUN=my_training_run sbatch --mail-user=your_email@msstate.edu scripts/resume_mid_sft.slurm
   ```
4. **Long-term**: Update `scripts/download_data.sh` to include MMLU for future runs

## Files to Check

```bash
# Check if MMLU is cached
ls -lh /scratch/ptolemy/users/$USER/cache/huggingface/datasets/cais___mmlu/

# Check what datasets are cached
find /scratch/ptolemy/users/$USER/cache/huggingface/datasets -maxdepth 2 -type d

# Review midtraining script data requirements
grep -n "load_dataset" scripts/mid_train.py tasks/*.py
```

## Expected Behavior After Fix

Once MMLU is downloaded, the midtraining should:
1. Load base model checkpoint (d20, step 21400) ✅ Already working
2. Initialize task mixture with:
   - GSM8K
   - **MMLU (auxiliary_train)** ← Will now work
   - SmolTalk
   - CustomJSON
   - SpellingBee
3. Run midtraining for ~2-3 hours
4. Save checkpoint to `mid_checkpoints/`
5. Continue to SFT phase

## Additional Notes

**Why didn't the original speedrun hit this?**
- The original speedrun (job 76356) was supposed to run all phases in one job
- It would have downloaded MMLU during midtraining (has ~30 min before timeout)
- But the job timed out before reaching midtraining
- Now we're trying to resume, and MMLU was never cached

**Python module version note:**
- Line 1 shows: `python/3.12.5` module not found (minor warning)
- This is non-critical - the venv already has Python 3.12.5
- The module load failure doesn't affect execution

**TF32 warnings:**
- Lines 19-34 show PyTorch TF32 deprecation warnings
- These are harmless warnings about future API changes
- Not related to the failure
