# Local CPU Training - Troubleshooting Guide

This document covers common issues when running local CPU training.

## Quick Fixes

### Before Running the Script

1. **Update Rust** (Most important!):
   ```bash
   rustup update
   rustc --version  # Should be 1.93.0-nightly or newer
   ```

2. **Ensure PATH is correct**:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

3. **Verify Python version**:
   ```bash
   python --version  # Should be 3.10+
   ```

## Common Errors

### Error 1: `uv: command not found`

**Full error**:
```
scripts/local_cpu_train.sh: line 69: uv: command not found
```

**Cause**: The `uv` package manager was installed to `~/.local/bin` but isn't in your PATH.

**Solution**:
```bash
# The script now auto-fixes this, but if it still fails:
export PATH="$HOME/.local/bin:$PATH"
bash scripts/local_cpu_train.sh
```

**Permanent fix** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

### Error 2: `edition2024` Feature Required

**Full error**:
```
error: failed to parse manifest at `.../rustbpe/Cargo.toml`

Caused by:
  feature `edition2024` is required

  The package requires the Cargo feature called `edition2024`, but that feature is not stabilized in this version of Cargo (1.82.0-nightly (ba8b39413 2024-08-16)).
```

**Cause**: Your Rust/Cargo version is too old. Edition2024 requires Rust 1.93.0-nightly (October 2025) or newer.

**Solution**:
```bash
# Update Rust to latest nightly
rustup update

# Verify the update worked
rustc --version   # Should show: rustc 1.93.0-nightly (d5419f1e9 2025-10-30) or newer
cargo --version   # Should show: cargo 1.93.0-nightly (6c1b61003 2025-10-28) or newer

# Now run the training script again
bash scripts/local_cpu_train.sh
```

**Why this happens**: The nanochat repository was updated to use Rust edition2024, which is a new feature only available in recent nightly builds.

---

### Error 3: Virtual Environment Not Found

**Full error**:
```
scripts/local_cpu_train.sh: line 80: .venv/bin/activate: No such file or directory
```

**Cause**: The virtual environment wasn't created, usually because `uv` command failed.

**Solution**:
```bash
# Fix uv PATH issue first
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment manually
uv venv

# Install dependencies
uv sync --extra cpu

# Now run the training script again
bash scripts/local_cpu_train.sh
```

---

### Error 4: `maturin` Build Failed

**Full error**:
```
× Failed to build `nanochat @ file://...`
├─▶ The build backend returned an error
╰─▶ Call to `maturin.build_editable` failed (exit status: 1)
```

**Cause**: Usually related to Rust edition2024 issue or missing dependencies.

**Solution**:
```bash
# 1. Update Rust first
rustup update

# 2. Verify Rust version
rustc --version  # Must be 1.93.0-nightly or newer

# 3. Try building manually to see detailed error
cargo build --manifest-path rustbpe/Cargo.toml --release

# 4. If that works, run training script again
bash scripts/local_cpu_train.sh
```

---

### Error 5: Out of Memory

**Symptoms**: Script crashes or system becomes unresponsive during training.

**Solution**: Reduce model size and sequence length:

Edit `scripts/local_cpu_train.sh`:
```bash
# Find the base_train section and change:
python -m scripts.base_train \
    --depth=2 \              # Reduced from 4
    --max_seq_len=512 \      # Reduced from 1024
    --device_batch_size=1 \
    --total_batch_size=512 \ # Reduced from 1024
    ...
```

---

### Error 6: Training Too Slow

**Symptoms**: Training is taking more than 4-5 hours.

**Solution**: Reduce the number of training iterations:

Edit `scripts/local_cpu_train.sh`:
```bash
# Base training
--num_iterations=20      # Reduced from 50

# Midtraining
--num_iterations=50      # Reduced from 100

# Supervised finetuning
--num_iterations=50      # Reduced from 100
```

---

### Error 7: Download Failures

**Symptoms**: 
```
Failed to download dataset
Failed to download eval_bundle
```

**Solution**:
```bash
# Check internet connection
ping -c 3 google.com

# The script is idempotent - just run it again
# It will skip already-downloaded files
bash scripts/local_cpu_train.sh
```

---

### Error 8: Python Version Too Old

**Error**:
```
ERROR: Python 3.10 or higher required
Current Python version: 3.9
```

**Solution**:
```bash
# Install Python 3.10+ using your system package manager
# On macOS:
brew install python@3.10

# On Ubuntu/Debian:
sudo apt install python3.10

# Or use pyenv:
pyenv install 3.10.19
pyenv local 3.10.19
```

---

## Step-by-Step Debugging

If the script fails, follow these steps:

### 1. Check Prerequisites
```bash
# Python version
python --version  # Need 3.10+

# Rust version (MOST IMPORTANT)
rustc --version   # Need 1.93.0-nightly (Oct 2025) or newer
cargo --version   # Need 1.93.0-nightly or newer

# Disk space
df -h ~  # Need ~2GB free

# Internet
ping -c 3 google.com
```

### 2. Update Rust (If Needed)
```bash
rustup update
rustc --version  # Verify it's 1.93.0+ nightly
```

### 3. Fix PATH for uv
```bash
export PATH="$HOME/.local/bin:$PATH"
which uv  # Should show: /Users/your_name/.local/bin/uv
```

### 4. Test rustbpe Build
```bash
cargo build --manifest-path rustbpe/Cargo.toml --release
# If this fails, Rust version is likely the issue
```

### 5. Run Training Script
```bash
bash scripts/local_cpu_train.sh
```

---

## Getting Help

If you're still stuck:

1. **Read detailed docs**: `cat LOCAL_CPU_TRAINING.md`
2. **Check error messages carefully**: They usually indicate the problem
3. **Compare with working setup**: Your Rust should be 1.93.0+ nightly
4. **Review script**: `cat scripts/local_cpu_train.sh`
5. **Check main docs**: `cat README.md`

---

## Summary of Required Versions

| Software | Minimum Version | Check Command |
|----------|----------------|---------------|
| Python | 3.10+ | `python --version` |
| Rust | 1.93.0-nightly (Oct 2025+) | `rustc --version` |
| Cargo | 1.93.0-nightly | `cargo --version` |
| uv | Latest | `uv --version` |
| Disk Space | 2GB free | `df -h ~` |

---

## Quick Reference: Most Common Fix

**90% of issues are fixed by updating Rust:**

```bash
rustup update
rustc --version  # Verify 1.93.0-nightly or newer
bash scripts/local_cpu_train.sh
```

If that doesn't work, check the other errors above.
