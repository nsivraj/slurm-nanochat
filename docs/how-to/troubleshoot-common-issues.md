# How-To: Troubleshoot Common Issues

This guide covers common problems across all platforms with solutions.

---

## Quick Diagnosis

**Which platform are you using?**

- [Local CPU Issues](#local-cpu-issues)
- [Ptolemy HPC Issues](#ptolemy-hpc-issues)
- [Production GPU Issues](#production-gpu-issues)
- [Cross-Platform Issues](#cross-platform-issues)

---

## Local CPU Issues

### 1. Rust edition2024 Error

**Error message:**
```
feature `edition2024` is required
The package requires the Cargo feature called `edition2024`
```

**Cause:** Rust version too old (need Oct 2025+ nightly build)

**Solution:**
```bash
rustup update
rustc --version  # Should be 1.93.0-nightly or newer
bash scripts/local_cpu_train.sh
```

**This is the #1 most common issue!**

### 2. `uv: command not found`

**Error message:**
```
scripts/local_cpu_train.sh: line 69: uv: command not found
```

**Cause:** UV installed to `~/.local/bin` but not in PATH

**Solution:**
```bash
export PATH="$HOME/.local/bin:$PATH"
bash scripts/local_cpu_train.sh
```

**Permanent fix** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Out of Memory

**Symptom:** System becomes unresponsive or script crashes during training

**Solution:** Edit `scripts/local_cpu_train.sh`:
```bash
# Reduce model size
--depth=2              # Instead of 4
--max_seq_len=512      # Instead of 1024

# Reduce batch size
--device_batch_size=1
--total_batch_size=512  # Instead of 1024
```

### 4. Training Too Slow

**Symptom:** Taking more than 4-5 hours

**Solution:** Reduce iterations in `scripts/local_cpu_train.sh`:
```bash
--num_iterations=20    # For base (instead of 50)
--num_iterations=50    # For mid/SFT (instead of 100)
```

### 5. Python Version Too Old

**Error:**
```
ERROR: Python 3.10 or higher required
Current Python version: 3.9
```

**Solution:**
```bash
# On macOS:
brew install python@3.10

# On Ubuntu/Debian:
sudo apt install python3.10

# Or use pyenv:
pyenv install 3.10.19
pyenv local 3.10.19
```

### 6. Download Failures

**Symptoms:** Failed to download dataset or eval_bundle

**Solution:**
```bash
# Check internet
ping -c 3 google.com

# Script is idempotent - just retry
bash scripts/local_cpu_train.sh
```

---

## Ptolemy HPC Issues

### 1. WANDB_RUN Not Set

**Error message:**
```
ERROR: WANDB_RUN environment variable is not set!
```

**Cause:** Job submitted without WANDB_RUN

**Solution:**
```bash
# ✅ CORRECT
WANDB_RUN=my_run sbatch scripts/speedrun.slurm

# ❌ WRONG
sbatch scripts/speedrun.slurm
```

**Why it matters:** Without WANDB_RUN, midtraining and SFT are SKIPPED!

### 2. WANDB_RUN Set to "dummy"

**Error message:**
```
ERROR: WANDB_RUN is set to 'dummy' which will skip midtraining and SFT!
```

**Solution:**
```bash
# Use any non-"dummy" name
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm
WANDB_RUN=test_run_1 sbatch scripts/speedrun.slurm
```

**Note:** You do NOT need a wandb account!

### 3. Data Not Downloaded

**Error message:**
```
Required data not found!
You must download data on a development node first.
```

**Cause:** Ran job without downloading data

**Solution:**
```bash
# SSH to devel node (has internet)
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# Download all data
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh

# Wait 30-60 minutes for ~24GB download

# Then submit job from any node
WANDB_RUN=my_run sbatch scripts/speedrun.slurm
```

### 4. Python Version Too Old

**Error:**
```
Package 'nanochat' requires a different Python: 3.9.14 not in '>=3.10'
```

**Cause:** Using system Python instead of module Python

**Solution:**
```bash
# Load Python 3.12 module
module load python/3.12.5

# Recreate venv
rm -rf /scratch/ptolemy/users/$USER/nanochat-venv
python3 -m venv /scratch/ptolemy/users/$USER/nanochat-venv

# Activate and reinstall
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
pip install uv
pip install -e '.[gpu]'
```

**Prevention:** Always run `bash scripts/setup_environment.sh`

### 5. Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'requests'
```

**Cause:** Dependencies not installed

**Solution:**
```bash
# Activate venv
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate

# Install ALL dependencies
pip install -e '.[gpu]'

# Verify
python -c "import requests; import torch; import tqdm; print('OK')"
```

### 6. Chat Model Not Found

**Error:**
```
FileNotFoundError: chatsft_checkpoints
```

**Cause:** SFT training didn't run (likely WANDB_RUN issue)

**Solution 1 - Use base model:**
```bash
python -m scripts.chat_cli -i mid
```

**Solution 2 - Re-run training correctly:**
```bash
WANDB_RUN=my_new_run sbatch scripts/speedrun.slurm
```

### 7. Wrong Node for Downloads

**Symptom:** Internet connection fails during data download

**Cause:** On compute node (no internet) instead of devel node

**Solution:**
```bash
# Check current node
hostname

# If on compute node, exit and SSH to devel
ssh <username>@ptolemy-devel-1.arc.msstate.edu

# Download data
cd /scratch/ptolemy/users/$USER/slurm-nanochat
bash scripts/download_data.sh
```

### 8. Storage Quota Exceeded

**Error:**
```
OSError: [Errno 122] Disk quota exceeded
```

**Cause:** Using home directory instead of scratch

**Solution:**
```bash
# Everything should be in /scratch
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Check quota
quota -s

# Clean up if needed
rm -rf ~/.cache/nanochat  # Move to scratch instead
```

### 9. Job Times Out at 12 Hours

**Symptom:** Job reaches time limit before completion

**Cause:** 12-hour QOS limit for class-cse8990

**What to check:**
```bash
# Check job log
tail -100 logs/nanochat_speedrun_*.out

# Base training should complete in ~7 hours
# If not, check for slow progress
```

**If base training is slow:**
- May be queue contention
- Check GPU utilization in logs
- Contact HPC support if abnormal

### 10. Out of Memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:** Reduce batch size in `scripts/speedrun.slurm`:
```bash
# Edit the training commands
--device_batch_size=16  # Instead of 32
```

---

## Production GPU Issues

### 1. SSH Disconnects Kill Training

**Problem:** Training stops when SSH connection drops

**Solution:** Use screen or tmux
```bash
# Start training in screen
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Detach: Ctrl+a d
# Reattach: screen -r speedrun
```

### 2. Download Very Slow

**Symptom:** Data download taking hours

**Check network:**
```bash
# Test download speed
curl -o /dev/null http://speedtest.tele2.net/100MB.zip

# Monitor download progress
watch -n 10 'du -sh ~/.cache/nanochat/base_data'
```

**Solution:** May just need to wait - 24GB takes 30-60 min on fast connection

### 3. GPU Not Detected

**Error:**
```
RuntimeError: No CUDA GPUs are available
```

**Check:**
```bash
nvidia-smi  # Should show 8 GPUs

# If not, instance may not have GPUs
# Contact provider
```

### 4. Forgot to Download Results

**Problem:** Terminated instance without saving results

**Prevention:**
```bash
# Before terminating, download:
scp ubuntu@<ip>:~/nanochat/report.md .
scp -r ubuntu@<ip>:~/.cache/nanochat/models/ ./models/
```

### 5. High Costs

**Problem:** Instance running longer than expected

**Monitor:**
```bash
# Check training progress
tail -f speedrun.log | grep "step.*loss"

# Check time elapsed
ps aux | grep base_train
```

**Terminate if stuck:**
```bash
pkill -f base_train.py
# Fix issue and restart
```

---

## Cross-Platform Issues

### 1. Model Quality is Poor

**Symptom:** Very low benchmark scores or nonsensical outputs

**Check report.md:**
```bash
cat report.md | grep "CORE"
```

**Expected scores:**
- Local CPU: CORE ~0.15 (intentionally poor - tiny model)
- Ptolemy/GPU: CORE ~0.20-0.22

**If much lower:**
- Check logs for errors
- Training may have failed silently
- Verify all phases completed

### 2. Rust Build Failures

**Error:**
```
Failed to build rustbpe
```

**Solution:**
```bash
# Update Rust
rustup update

# Check version
rustc --version  # Need 1.93.0-nightly (Oct 2025+)

# Try manual build
cargo build --manifest-path rustbpe/Cargo.toml --release
```

### 3. Tokenizer Issues

**Error:**
```
FileNotFoundError: tokenizer.pkl
```

**Cause:** Tokenizer training failed or looking in wrong location

**Solution:**
```bash
# Check if tokenizer exists
ls ~/.cache/nanochat/tokenizer/tokenizer.pkl          # Local
ls /scratch/ptolemy/users/$USER/nanochat-cache/tokenizer/  # Ptolemy

# If missing, re-run tokenizer training
python -m scripts.tok_train
```

### 4. Report Not Generated

**Symptom:** No `report.md` file after training

**Cause:** Report generation phase failed

**Solution:**
```bash
# Check logs for errors during report generation
tail -100 speedrun.log | grep -i error

# Try generating manually (if all training completed)
# This requires all checkpoints to exist
```

### 5. Wrong Device Detected

**Error:**
```
Expected CUDA but got CPU
```

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True (on GPU systems)
# Should print: False (on CPU systems)
```

**Fix:**
- Local CPU: This is expected, ignore
- Ptolemy/GPU: Check CUDA installation, contact HPC support
- Production: Check instance has GPUs

---

## Debugging Workflow

### Step 1: Identify the Platform
- Local CPU
- Ptolemy HPC
- Production GPU

### Step 2: Identify the Phase
- Setup / Environment
- Data Download
- Tokenizer Training
- Base Pretraining
- Midtraining
- SFT
- Inference / Chat

### Step 3: Check Logs

**Local CPU:**
```bash
# Terminal output shows errors directly
```

**Ptolemy:**
```bash
# Output log
tail -100 logs/nanochat_speedrun_*.out

# Error log
tail -100 logs/nanochat_speedrun_*.err
```

**Production GPU:**
```bash
# Screen log
tail -100 speedrun.log
```

### Step 4: Match Error to Solutions Above

Search this document for the error message or symptom.

### Step 5: Apply Fix and Retry

Most issues can be fixed and training restarted.

---

## Prevention Checklist

### Before Starting (Any Platform)

- [ ] Python 3.10+ installed
- [ ] Rust updated (`rustup update`)
- [ ] Sufficient disk space (~2GB local, ~30GB HPC/cloud)
- [ ] Internet connection working

### Before Submitting (Ptolemy HPC)

- [ ] On devel node for data download
- [ ] All data downloaded and verified
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Email configured in `.env.local`
- [ ] WANDB_RUN will be set (remember!)

### Before Starting (Production GPU)

- [ ] Using screen or tmux
- [ ] Monitoring costs
- [ ] Plan to download results before terminating

---

## Getting More Help

**Check platform-specific docs:**
- [Setup Environment](setup-environment.md)
- [Run a Training Job](run-a-training-job.md)

**Review tutorials:**
- [Local CPU Quickstart](../tutorials/01-local-cpu-quickstart.md)
- [HPC First Job](../tutorials/02-hpc-first-job.md)

**Examine session logs:**
- Check `experiments/` folder for historical issues and fixes

**For Ptolemy-specific help:**
- [HPC Environment Details](../explanation/hpc-environment.md)

---

## Summary of Most Common Issues

| Platform | #1 Issue | Quick Fix |
|----------|----------|-----------|
| **Local CPU** | Rust edition2024 | `rustup update` |
| **Ptolemy HPC** | WANDB_RUN not set | `WANDB_RUN=X sbatch ...` |
| **Production GPU** | SSH disconnects | Use `screen` |

**Remember:** 90% of issues are solved by:
1. Updating Rust (local CPU)
2. Setting WANDB_RUN (Ptolemy)
3. Using screen (production GPU)
