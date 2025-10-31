# Quick Re-Run Guide

## TL;DR - Run These Commands

### Step 1: Download Data (on ptolemy-devel-1, ~60 minutes)
```bash
ssh <username>@ptolemy-devel-1.arc.msstate.edu
cd /scratch/ptolemy/users/$USER/slurm-nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
bash scripts/download_data.sh
```

**Wait for:**
```
✅ Data Download COMPLETE
✓ SmolTalk dataset cached
✓ Tiktoken tokenizers cached
```

### Step 2: Submit Job (any node, ~12 hours)
```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
sbatch scripts/speedrun.slurm
```

### Step 3: Monitor
```bash
# Check status
squeue -u $USER

# Watch progress
tail -f logs/nanochat_speedrun_*.out
```

---

## What Was Fixed?

1. ✅ **Pre-download GPT-2/GPT-4 tokenizers** → tok_eval won't fail
2. ✅ **Pre-download SmolTalk dataset** → midtraining won't fail
3. ✅ **Increased time limit to 14 hours** → full pipeline will complete
4. ✅ **Made tok_eval gracefully handle missing tokenizers** → no crashes

---

## Expected Timeline

| Time | Stage | Status |
|------|-------|--------|
| 0:00 | Job starts | Verification |
| 0:05 | Tokenizer eval | ✅ |
| 0:10 | Base training starts | Progress... |
| 7:00 | Base training done | ✅ CORE: 0.20 |
| 7:10 | Midtraining starts | Progress... |
| 9:30 | Midtraining done | ✅ Chat format learned |
| 9:40 | SFT starts | Progress... |
| 11:30 | SFT done | ✅ Task performance improved |
| 11:35 | Report generated | ✅ DONE! |

**Total:** ~10-12 hours (12 hour limit is max allowed by QOS)

---

## You'll Get Email Notifications:

- **BEGIN:** Training started
- **END:** Training completed successfully
- **FAIL:** Training failed (check error log)

---

## When Training Completes

View the report:
```bash
cat /scratch/ptolemy/users/$USER/slurm-nanochat/report.md
```

Chat with your model:
```bash
srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
python -m scripts.chat_cli
```

---

## Files Modified
- `scripts/download_data.sh` → Added SmolTalk & tokenizers
- `scripts/tok_eval.py` → Handle offline mode
- `scripts/speedrun.slurm` → 12 hour time limit (QOS maximum)

See `TRAINING_FIXES_APPLIED.md` for details.
