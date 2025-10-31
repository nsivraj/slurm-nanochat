# Troubleshooting Guide for nanochat on Ptolemy HPC

This document covers common issues encountered during setup and training.

## Table of Contents
1. [Python Version Issues](#python-version-issues)
2. [Missing Dependencies](#missing-dependencies)
3. [UV Command Not Found](#uv-command-not-found)
4. [Data Download Failures](#data-download-failures)
5. [SLURM Job Failures](#slurm-job-failures)
6. [Storage/Quota Issues](#storagequota-issues)

---

## Python Version Issues

### Error: `Package 'nanochat' requires a different Python: 3.9.14 not in '>=3.10'`

**Problem:** nanochat requires Python 3.10 or higher, but system default is Python 3.9

**Solution:**
```bash
# Load Python 3.12 module
module load python/3.12.5

# Recreate virtual environment with correct Python
rm -rf /scratch/ptolemy/users/$USER/nanochat-venv
python3 -m venv /scratch/ptolemy/users/$USER/nanochat-venv

# Activate and verify
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
python --version  # Should show 3.12.5

# Reinstall dependencies
pip install uv
pip install -e '.[gpu]'
```

**Prevention:** Always run `bash scripts/setup_environment.sh` which loads Python 3.12 automatically

---

## Missing Dependencies

### Error: `ModuleNotFoundError: No module named 'requests'`

**Problem:** Python packages not installed in virtual environment

**Solution:**
```bash
# Activate virtual environment
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate

# Install ALL dependencies
pip install -e '.[gpu]'

# Verify critical packages
python -c "import requests; import torch; import tqdm; print('All packages OK')"
```

**What `pip install -e '.[gpu]'` does:**
- Installs nanochat in editable mode
- Installs ALL dependencies from `pyproject.toml`
- Includes GPU extras (PyTorch with CUDA)
- Required packages: requests, tqdm, numpy, torch, etc.

### Error: Missing packages during `download_data.sh`

**Problem:** Script runs before dependencies are installed

**Solution:**
The `download_data.sh` script now checks for required packages and will fail early with instructions:

```bash
❌ ERROR: Missing required Python packages: requests tqdm torch

Please install dependencies first:
  pip install uv
  pip install -e '.[gpu]'

Then re-run this script
```

Follow the instructions shown in the error message.

---

## UV Command Not Found

### Error: `uv: command not found` when running download script

**Problem:** UV not installed or not in PATH

**Solution 1: Install UV properly**
```bash
# Activate virtual environment first
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate

# Install uv
pip install uv

# Verify it's available
which uv  # Should show: /scratch/ptolemy/users/$USER/nanochat-venv/bin/uv
uv --version
```

**Solution 2: Use alternative commands**
If UV continues to be problematic, you can install dependencies differently:

```bash
# Instead of: uv sync --extra gpu
# Use:
pip install -e '.[gpu]'
```

**Note:** The download script uses `uv run maturin develop` to build the Rust tokenizer. If UV is installed via pip in the venv, this should work.

---

## Data Download Failures

### Error: Verification says "Dataset shards: 0/240" but data exists

**Problem:** Script verification checks for wrong file extension

**Symptom:**
```
❌ Dataset shards: 0/240 (insufficient)
```

But you can see files in `/scratch/ptolemy/users/$USER/nanochat-cache/base_data/`

**Cause:** nanochat downloads `.parquet` files, not `.bin` files. Older versions of the verification script checked for `.bin` files.

**Solution:**
```bash
# Verify the data actually exists (check for .parquet files)
ls -1 /scratch/ptolemy/users/$USER/nanochat-cache/base_data/*.parquet | wc -l
# Should show 240

# If you see 240 parquet files, your data is complete!
# The scripts have been updated to check for .parquet files
```

**Fixed in:** Updated `scripts/download_data.sh` and `scripts/speedrun.slurm` to check for `.parquet` files instead of `.bin` files.

### Error: Internet connection test fails

**Problem:** Running on compute node (no internet) instead of devel node

**Solution:**
```bash
# Check which node you're on
hostname

# If you see "compute" or GPU node name, you're on the wrong server
# SSH to a devel server instead:
ssh [username]@ptolemy-devel-1.arc.msstate.edu
# OR
ssh [username]@ptolemy-devel-2.arc.msstate.edu

# Then run download script
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh
```

### Download is very slow

**Normal:** Downloading 240 shards (~24GB) takes 30-60 minutes

**Monitor progress:**
```bash
# In another terminal, watch data directory grow
watch -n 10 'du -sh /scratch/ptolemy/users/$USER/nanochat-cache/data/'

# Count downloaded shards
watch -n 10 'ls -1 /scratch/ptolemy/users/$USER/nanochat-cache/data/*.bin | wc -l'
```

**Expected:** Download rate depends on network speed. ~400-800 MB per minute is normal.

### Rust compilation fails

**Error:** Building rustbpe tokenizer fails

**Solution:**
```bash
# Make sure Rust/Cargo is installed
source $CARGO_HOME/env

# Check Cargo version
cargo --version

# If cargo not found, install Rust:
export CARGO_HOME="/scratch/ptolemy/users/$USER/.cargo"
export RUSTUP_HOME="/scratch/ptolemy/users/$USER/.rustup"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
source "$CARGO_HOME/env"

# Try building manually
cd /scratch/ptolemy/users/$USER/slurm-nanochat
pip install maturin
maturin develop --release --manifest-path rustbpe/Cargo.toml
```

---

## SLURM Job Failures

### Job fails immediately: "Required data not found"

**Problem:** Data not downloaded before submitting job

**Error message:**
```
❌ ERROR: Required data not found!
GPU compute nodes do NOT have internet access.
All data must be downloaded BEFORE submitting this job.
```

**Solution:**
```bash
# SSH to devel node
ssh [username]@ptolemy-devel-1.arc.msstate.edu

# Download data
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh

# Wait for completion (30-60 min)

# Then submit job
sbatch scripts/speedrun.slurm
```

### Job fails: Email not configured

**Error:**
```
❌ ERROR: Email address not provided!
```

**Solution:**
```bash
# Create .env.local file
cd /scratch/ptolemy/users/$USER/slurm-nanochat
cp .env.local.example .env.local
nano .env.local

# Change: email=your_email@msstate.edu
# Save and exit

# Re-submit job
sbatch scripts/speedrun.slurm
```

### Job pending for long time

**Check job status:**
```bash
squeue -u $USER

# Check partition availability
sinfo -p gpu-a100
```

**Common reasons:**
- All GPUs are in use (wait for resources)
- Job requesting too many resources
- Account limits reached

### Out of Memory (OOM) errors during training

**Solution:** Reduce batch size in SLURM script

```bash
# Edit scripts/speedrun.slurm
nano scripts/speedrun.slurm

# Find the training commands and add --device_batch_size parameter:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --device_batch_size=16 --run=$WANDB_RUN

# Default is 32, try 16, 8, 4, or 2
```

---

## Storage/Quota Issues

### Disk quota exceeded in home directory

**Problem:** Something is writing to home instead of scratch

**Check home usage:**
```bash
# Check home directory size
du -sh ~

# Should be < 100 MB
# If larger, find what's taking space:
du -h --max-depth=1 ~ | sort -h
```

**Solution:** Verify all environment variables point to scratch

```bash
# Check environment variables
echo $NANOCHAT_BASE_DIR  # Should be /scratch/ptolemy/users/$USER/nanochat-cache
echo $HF_HOME            # Should be /scratch/ptolemy/users/$USER/cache/huggingface
echo $TORCH_HOME         # Should be /scratch/ptolemy/users/$USER/cache/torch
echo $UV_CACHE_DIR       # Should be /scratch/ptolemy/users/$USER/cache/uv
echo $PIP_CACHE_DIR      # Should be /scratch/ptolemy/users/$USER/cache/pip

# If any point to home (~), re-run setup script:
bash scripts/setup_environment.sh
```

### Out of space in scratch

**Check scratch usage:**
```bash
# Your usage
du -sh /scratch/ptolemy/users/$USER/

# Breakdown by directory
du -h --max-depth=1 /scratch/ptolemy/users/$USER/ | sort -h
```

**Expected usage:**
- nanochat-cache: ~30-35 GB (dataset + models)
- nanochat-venv: ~2-3 GB
- cache directories: ~3-8 GB
- Total: ~35-50 GB

**Clean up if needed:**
```bash
# After assignment is graded, you can remove:

# Dataset (can re-download if needed)
rm -rf /scratch/ptolemy/users/$USER/nanochat-cache/data/

# Caches (can be regenerated)
rm -rf /scratch/ptolemy/users/$USER/cache/
```

---

## General Debugging Tips

### Enable verbose output

```bash
# For bash scripts, add -x flag
bash -x scripts/download_data.sh

# For Python scripts, add verbose flag if available
python -m nanochat.dataset -n 240 -v
```

### Check job logs

```bash
# View output log
cat logs/nanochat_speedrun_JOBID.out

# View error log
cat logs/nanochat_speedrun_JOBID.err

# Watch in real-time
tail -f logs/nanochat_speedrun_JOBID.out
```

### Verify module loading

```bash
# List loaded modules
module list

# Should see:
# - python/3.12.5
# - cuda-toolkit
# - spack-managed-x86-64_v3/v1.0
```

### Test GPU access

```bash
# Run GPU test script
python scripts/test_gpu.py

# Or quick check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check the main documentation:**
   - `PTOLEMY_SETUP.md` - Setup instructions
   - `IMPORTANT_PTOLEMY_NOTES.md` - Internet restrictions
   - `SCRATCH_STORAGE_VERIFICATION.md` - Storage configuration

2. **Check SLURM logs:**
   - Look in `logs/nanochat_speedrun_*.out` and `*.err`

3. **Verify the checklist:**
   ```bash
   # Python 3.12 loaded?
   python --version

   # Virtual environment activated?
   which python  # Should point to /scratch/.../nanochat-venv/bin/python

   # Dependencies installed?
   pip list | grep -E "torch|requests|tqdm"

   # Data downloaded? (check for .parquet files in base_data directory)
   ls -1 /scratch/ptolemy/users/$USER/nanochat-cache/base_data/*.parquet | wc -l  # Should be 240+

   # Tokenizer trained?
   ls -lh /scratch/ptolemy/users/$USER/nanochat-cache/tokenizer/tokenizer.pkl
   ```

4. **Contact Ptolemy HPC support:**
   - For cluster issues, resource limits, module problems
   - https://www.hpc.msstate.edu/

5. **Contact instructor/TA:**
   - For assignment-specific questions

---

## Quick Reference: Complete Setup Workflow

If you need to start fresh:

```bash
# 1. SSH to devel node
ssh [username]@ptolemy-devel-2.arc.msstate.edu

# 2. Navigate to project
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# 3. Configure email
cp .env.local.example .env.local
nano .env.local  # Set your email

# 4. Run setup (loads Python 3.12, creates venv)
bash scripts/setup_environment.sh

# 5. Install dependencies
pip install uv
pip install -e '.[gpu]'

# 6. Download data (~30-60 min)
bash scripts/download_data.sh

# 7. Submit job (from any node - logs directory created automatically)
sbatch scripts/speedrun.slurm

# 8. Monitor
squeue -u $USER
tail -f logs/nanochat_speedrun_*.out
```

This workflow should complete without errors if followed exactly.
