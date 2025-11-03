# Diagnosis: Dataset Download Script Issues (2025-11-01)

**Status:** ✅ RESOLVED

**Jobs affected:**
- Job 76386 - Resume midtraining failed with MMLU "all" config error
- Job 76387 - Resume midtraining failed with MMLU "all" config error (after first fix)

---

## Problem Summary

The `scripts/download_after_basetraining.sh` script had multiple issues that prevented proper dataset downloading for midtraining and SFT phases:

1. **Missing MMLU "all" config** - Validation dataset not downloaded
2. **Missing validation splits** - GSM8K and SmolTalk test splits not downloaded
3. **False "already cached" messages** - Script checked directory existence, not specific configs
4. **SmolTalk loading error** - `None` subset parameter causing "dataclass" error

---

## Issue 1: Missing MMLU "all" Config

### Error Message
```
ValueError: Couldn't find cache for cais/mmlu for config 'all'
Available configs in the cache: ['auxiliary_train']
```

**Location:** `scripts/mid_train.py:109`

### Root Cause

The midtraining script uses **two different MMLU configurations**:

**Training data** (line 100):
```python
MMLU(subset="auxiliary_train", split="train")  # 100K rows
```

**Validation data** (line 109):
```python
MMLU(subset="all", split="test", stop=5200)  # 14K rows, use 5.2K
```

The download script only downloaded `auxiliary_train`, not `all`.

### Solution

Added MMLU "all" config to download list:
```python
("MMLU all (midtraining validation)", "cais/mmlu", "all", "test"),
```

---

## Issue 2: Missing Validation Splits

### Root Cause Analysis

After analyzing `scripts/mid_train.py` and `scripts/chat_sft.py`, discovered that **validation datasets** were also needed:

**Midtraining validation** (mid_train.py:107-111):
```python
val_dataset = TaskMixture([
    SmolTalk(split="test"),  # ❌ Not downloaded
    MMLU(subset="all", split="test", stop=5200),  # ❌ Not downloaded
    GSM8K(subset="main", split="test", stop=420),  # ❌ Not downloaded
])
```

**SFT validation** (chat_sft.py:93):
```python
val_ds = SmolTalk(split="test")  # ❌ Not downloaded
```

### Solution

Extended download list to include all validation splits:
```python
datasets_to_download = [
    # Midtraining datasets (training)
    ("MMLU auxiliary_train (midtraining train)", "cais/mmlu", "auxiliary_train", "train"),
    ("GSM8K (midtraining train)", "openai/gsm8k", "main", "train"),
    ("SmolTalk (midtraining train)", "HuggingFaceTB/smol-smoltalk", None, "train"),

    # Midtraining datasets (validation)
    ("MMLU all (midtraining validation)", "cais/mmlu", "all", "test"),
    ("GSM8K (midtraining validation)", "openai/gsm8k", "main", "test"),
    ("SmolTalk (midtraining validation)", "HuggingFaceTB/smol-smoltalk", None, "test"),

    # SFT datasets (training)
    ("ARC-Easy (SFT train)", "allenai/ai2_arc", "ARC-Easy", "train"),
    ("ARC-Challenge (SFT train)", "allenai/ai2_arc", "ARC-Challenge", "train"),
    ("GSM8K (SFT train)", "openai/gsm8k", "main", "train"),
    ("SmolTalk (SFT train)", "HuggingFaceTB/smol-smoltalk", None, "train"),

    # SFT datasets (validation)
    ("SmolTalk (SFT validation)", "HuggingFaceTB/smol-smoltalk", None, "test"),
]
```

---

## Issue 3: False "Already Cached" Messages

### Problem

Script reported datasets as "already cached" when only the base directory existed, not the specific config:

```
✓ Already cached (skipping download)
  Cached at: /scratch/ptolemy/users/ncj79/cache/huggingface/datasets/cais___mmlu/
```

But when training ran, it failed because the `all` config wasn't in that directory.

### Root Cause

Original code only checked directory existence:
```python
cache_path = os.path.join(hf_home, "datasets", dataset_id.replace('/', '___'))
if os.path.exists(cache_path):
    print(f"  ✓ Already cached (skipping download)")
    download_count += 1
    continue
```

HuggingFace stores different configs in **subdirectories** within the dataset folder, so checking the base directory doesn't verify a specific config exists.

### Solution

Actually **load the dataset** to verify the specific config is cached:
```python
# Try to load the dataset (will use cache if available, download if not)
try:
    if subset is not None and subset != "None":
        ds = load_dataset(dataset_id, subset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)

    print(f"  ✓ Successfully loaded {name}")
    print(f"    Rows: {len(ds):,}")
    download_count += 1
except Exception as e:
    print(f"  ❌ Failed to load/download {name}: {e}")
    error_count += 1
```

This way:
- If config is cached → loads from cache (fast)
- If config is missing → downloads it
- Message accurately reflects what happened

---

## Issue 4: SmolTalk "Dataclass" Error

### Error Message
```
❌ Failed to load/download SmolTalk (midtraining train): must be called with a dataclass type or instance
```

### Root Cause

The SmolTalk dataset doesn't use a config/subset parameter. Looking at `tasks/smoltalk.py:16`:
```python
self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
# Note: NO subset/config parameter!
```

But the download script was trying to pass `None` as a positional argument:
```python
# Original broken code
if subset:
    ds = load_dataset(dataset_id, subset, split=split)
else:
    ds = load_dataset(dataset_id, split=split)
```

In Python embedded in bash heredocs, `None` can become the string `"None"`, and even checking `if subset:` wasn't reliable.

### Solution (Thanks to Zen AI Analysis)

Use **kwargs pattern** with explicit parameter names:
```python
# Build kwargs dictionary
load_args = {
    "path": dataset_id,
    "split": split
}

# Only add 'name' parameter if subset is meaningful (not None or "None")
if subset is not None and subset != "None":
    load_args["name"] = subset

# Unpack kwargs into function call
ds = load_dataset(**load_args)
```

**Why this works:**
1. Uses explicit parameter names (`path`, `name`, `split`) instead of positional args
2. Only adds `name` parameter when we have a real subset
3. Handles both Python `None` and string `"None"` correctly
4. More robust and idiomatic Python

---

## Other Fixes

### Missing ARC-Challenge

SFT uses both ARC-Easy and ARC-Challenge (chat_sft.py:85-86):
```python
ARC(subset="ARC-Easy", split="train"),      # 2.3K rows
ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
```

Added to download list:
```python
("ARC-Challenge (SFT train)", "allenai/ai2_arc", "ARC-Challenge", "train"),
```

### Missing NANOCHAT_BASE_DIR

The word list download was failing because `NANOCHAT_BASE_DIR` wasn't set:
```bash
Warning: Failed to create the file /words_alpha.txt: Permission denied
```

Fixed by setting it at the top of the script:
```bash
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
mkdir -p $NANOCHAT_BASE_DIR
```

---

## Final Dataset List

The complete list of datasets now downloaded:

### Midtraining (Training)
1. MMLU (auxiliary_train, train) - 99,842 rows
2. GSM8K (main, train) - 7,473 rows
3. SmolTalk (train) - 460,000 rows

### Midtraining (Validation)
4. MMLU (all, test) - 14,042 rows
5. GSM8K (main, test) - 1,319 rows
6. SmolTalk (test) - 24,000 rows

### SFT (Training)
7. ARC-Easy (ARC-Easy, train) - 2,251 rows
8. ARC-Challenge (ARC-Challenge, train) - 1,119 rows
9. GSM8K (main, train) - Already counted (#2)
10. SmolTalk (train) - Already counted (#3)

### SFT (Validation)
11. SmolTalk (test) - Already counted (#6)

### Evaluation
12. HumanEval (test) - 164 rows for code generation evaluation

### Other
13. English word list (words_alpha.txt) - ~370K words for SpellingBee task

**Total unique datasets:** 9 datasets + 1 word list

---

## Testing Results

After all fixes (including HumanEval), the download script successfully loads all datasets:

```
[1/12] ✓ Successfully loaded MMLU auxiliary_train (midtraining train) - 99,842 rows
[2/12] ✓ Successfully loaded GSM8K (midtraining train) - 7,473 rows
[3/12] ✓ Successfully loaded SmolTalk (midtraining train) - 460,000 rows
[4/12] ✓ Successfully loaded MMLU all (midtraining validation) - 14,042 rows
[5/12] ✓ Successfully loaded GSM8K (midtraining validation) - 1,319 rows
[6/12] ✓ Successfully loaded SmolTalk (midtraining validation) - 24,000 rows
[7/12] ✓ Successfully loaded ARC-Easy (SFT train) - 2,251 rows
[8/12] ✓ Successfully loaded ARC-Challenge (SFT train) - 1,119 rows
[9/12] Skipping GSM8K (SFT train) (already processed)
[10/12] Skipping SmolTalk (SFT train) (already processed)
[11/12] Skipping SmolTalk (SFT validation) (already processed)
[12/12] ✓ Successfully loaded HumanEval (evaluation) - 164 rows

✅ All 9 unique datasets downloaded/verified successfully!
   (3 duplicate entries skipped)
```

---

## Lessons Learned

1. **Validation datasets are critical** - Don't assume only training data is needed
2. **HuggingFace configs are separate** - Each config must be downloaded individually
3. **Verify by loading, not directory checking** - Directory existence ≠ config existence
4. **Use kwargs for optional parameters** - More robust than positional args in bash heredocs
5. **Analyze source code thoroughly** - Read the actual training scripts to find all requirements
6. **Test splits matter** - Both train and test splits needed for validation
7. **Check evaluation tasks too** - Datasets used in evaluation (like HumanEval) must also be pre-downloaded
8. **Iterative fixing works** - Missing datasets can be identified one by one and added to the download script

---

## Next Steps

Users should:

1. **Run the updated download script:**
   ```bash
   bash scripts/download_after_basetraining.sh
   ```

2. **Verify all datasets downloaded:**
   - Should see 8 unique datasets successfully loaded
   - 3 duplicates skipped (GSM8K, SmolTalk appear in multiple phases)
   - English word list verified

3. **Submit training job:**
   ```bash
   WANDB_RUN=my_training_run sbatch scripts/resume_mid_sft.slurm
   ```

The job should now complete successfully through midtraining and SFT phases!

---

## Update: HumanEval Dataset Issue (2025-11-03)

### Problem

After fixing the midtraining datasets, Job 76389 failed with:
```
ConnectionError: Couldn't reach 'openai/openai_humaneval' on the Hub (LocalEntryNotFoundError)
```

**Location:** `tasks/humaneval.py:51`

### Root Cause

The HumanEval dataset is used for evaluation but was **not included** in the download script. Looking at `tasks/humaneval.py:51`:
```python
self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)
```

This dataset is used during midtraining evaluation to test code generation capabilities.

### Solution

Added HumanEval to the download list in `scripts/download_after_basetraining.sh`:
```python
# Evaluation datasets
("HumanEval (evaluation)", "openai/openai_humaneval", None, "test"),
```

### Result

After running the updated download script:
- ✅ HumanEval dataset successfully cached
- ✅ Midtraining evaluation completed without errors
- ✅ SFT phase ran successfully
- ✅ Chat CLI working: `python -m scripts.chat_cli -p "Hello"`

### Process Used

User successfully completed training by:
1. Running updated `scripts/download_after_basetraining.sh` on ptolemy-devel-1
2. Commenting out midtraining step in `scripts/resume_mid_sft.slurm` (already done)
3. Re-running evaluation step (now had HumanEval dataset)
4. SFT phase completed successfully
5. Model chat functionality confirmed working

---

## Related Files

- **Download script:** `scripts/download_after_basetraining.sh`
- **Resume script:** `scripts/resume_mid_sft.slurm`
- **Midtraining script:** `scripts/mid_train.py`
- **SFT script:** `scripts/chat_sft.py`
- **HumanEval task:** `tasks/humaneval.py`
- **Previous diagnosis:** `experiments/DIAGNOSIS_RESUME_FAILURE.md`
- **Troubleshooting:** `docs/how-to/troubleshoot-common-issues.md`
