# Scratch Storage Verification

## Summary

All large data files, models, and caches are configured to write to `/scratch` volume, NOT the home directory. This is critical because home directories on Ptolemy HPC have limited disk quotas.

## Storage Layout

### All Data Stored in /scratch

```
/scratch/ptolemy/users/$USER/
├── nanochat-venv/              # Python virtual environment (~2-3 GB)
├── nanochat-cache/             # Main nanochat data (~30+ GB)
│   ├── data/                   # Dataset shards (~24 GB)
│   ├── models/                 # Trained model checkpoints (~2-5 GB)
│   ├── eval_bundle/            # Evaluation data (~162 MB)
│   ├── tokenizer/              # Trained tokenizer (~100 MB)
│   └── report/                 # Training reports (<1 MB)
├── cache/                      # Various caches
│   ├── huggingface/            # HF models & datasets (~1-5 GB)
│   ├── torch/                  # PyTorch cache (~500 MB)
│   ├── uv/                     # UV package manager cache (~1-2 GB)
│   └── pip/                    # Pip cache (~500 MB)
├── .cargo/                     # Rust/Cargo installation (~500 MB)
└── .rustup/                    # Rust toolchain (~1 GB)

TOTAL: ~30-50 GB (varies based on training)
```

### Home Directory Usage

Only minimal configuration files in home directory:
- `~/.bashrc` (if modified)
- Shell history files
- SSH keys
- **NO LARGE DATA FILES**

## Environment Variables Set

All scripts (`setup_environment.sh`, `download_data.sh`, `speedrun.slurm`) set these environment variables:

| Variable | Value | Purpose | Size Impact |
|----------|-------|---------|-------------|
| `NANOCHAT_BASE_DIR` | `/scratch/ptolemy/users/$USER/nanochat-cache` | Main nanochat data | ~30 GB |
| `HF_HOME` | `/scratch/ptolemy/users/$USER/cache/huggingface` | HuggingFace cache | ~1-5 GB |
| `TORCH_HOME` | `/scratch/ptolemy/users/$USER/cache/torch` | PyTorch cache | ~500 MB |
| `CARGO_HOME` | `/scratch/ptolemy/users/$USER/.cargo` | Rust/Cargo | ~500 MB |
| `RUSTUP_HOME` | `/scratch/ptolemy/users/$USER/.rustup` | Rust toolchain | ~1 GB |
| `UV_CACHE_DIR` | `/scratch/ptolemy/users/$USER/cache/uv` | UV package cache | ~1-2 GB |
| `PIP_CACHE_DIR` | `/scratch/ptolemy/users/$USER/cache/pip` | Pip package cache | ~500 MB |
| `VENV_PATH` | `/scratch/ptolemy/users/$USER/nanochat-venv` | Virtual env | ~2-3 GB |

**Total scratch usage: ~35-50 GB** (well within typical scratch quotas)

## Verification Commands

Run these commands to verify all data is in scratch:

```bash
# Check total scratch usage
du -sh /scratch/ptolemy/users/$USER/

# Check nanochat cache size
du -sh /scratch/ptolemy/users/$USER/nanochat-cache/

# Check virtual environment size
du -sh /scratch/ptolemy/users/$USER/nanochat-venv/

# Check all cache directories
du -sh /scratch/ptolemy/users/$USER/cache/*

# Verify NO large files in home directory
du -sh ~
# Should be < 100 MB
```

## Comparison: Original vs Ptolemy Setup

### Original nanochat (speedrun.sh)
```bash
# Uses home directory - WILL FILL HOME QUOTA
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"  # ❌ HOME DIRECTORY
```

### Ptolemy Setup (our scripts)
```bash
# Uses scratch - NO HOME QUOTA ISSUES
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"  # ✅ SCRATCH
```

## Why This Matters

### Home Directory Quotas on HPC
- Typical home quota: 50-100 GB
- Often includes backups (costs 2x space)
- Slower I/O than scratch
- **nanochat needs ~35-50 GB** → would exhaust home quota

### Scratch Storage Advantages
- Large quotas (typically 1-10 TB)
- No backups (saves space)
- Faster I/O for training
- Designed for temporary large datasets
- Automatically cleaned after project completion

## Potential Issues Prevented

### ❌ If Using Home Directory
```
# Job would fail with:
Disk quota exceeded
Cannot write to ~/.cache/nanochat/data/shard_123.bin
Training failed
```

### ✅ With Scratch Storage
```
# All data writes to scratch:
Writing to /scratch/ptolemy/users/$USER/nanochat-cache/data/shard_123.bin
✓ Success - plenty of space in scratch
```

## Files Modified for Scratch Storage

1. **`scripts/setup_environment.sh`**
   - Sets all cache environment variables to `/scratch`
   - Creates venv in `/scratch`
   - Displays all paths for verification

2. **`scripts/download_data.sh`**
   - Downloads all data to `/scratch`
   - Verifies sufficient space before downloading
   - All Rust/Cargo operations in `/scratch`

3. **`scripts/speedrun.slurm`**
   - All training artifacts to `/scratch`
   - Model checkpoints in `/scratch`
   - Reports copied to working directory (small files OK)

## Best Practices Applied

✅ **All large data in scratch**
- Dataset shards
- Model checkpoints
- Training caches

✅ **All build artifacts in scratch**
- Python virtual environment
- Rust/Cargo installation
- Package manager caches

✅ **Only config files in home**
- `.env.local` (few bytes)
- Shell configurations
- Small scripts

✅ **Fast I/O for training**
- Scratch has better I/O performance
- Multiple nodes can access scratch
- No backup overhead slowing writes

## Monitoring Disk Usage

### Check Scratch Usage
```bash
# Overall scratch usage
quota -s /scratch/ptolemy

# Your usage
du -sh /scratch/ptolemy/users/$USER/

# Detailed breakdown
du -h --max-depth=1 /scratch/ptolemy/users/$USER/ | sort -h
```

### Check Home Usage (Should be minimal)
```bash
# Your home usage
quota -s ~

# Detailed breakdown
du -h --max-depth=1 ~ | sort -h
```

### Warning Signs
- Home directory > 10 GB → Investigate what's there
- Home directory approaching quota → Move data to scratch
- Scratch directory > 100 GB → Normal for nanochat with multiple models

## Cleanup After Training

If you want to free up space after assignment:

```bash
# Remove training data (keep models)
rm -rf /scratch/ptolemy/users/$USER/nanochat-cache/data/

# Remove caches (can be re-downloaded)
rm -rf /scratch/ptolemy/users/$USER/cache/

# Remove everything (complete cleanup)
rm -rf /scratch/ptolemy/users/$USER/nanochat-cache/
rm -rf /scratch/ptolemy/users/$USER/nanochat-venv/
rm -rf /scratch/ptolemy/users/$USER/cache/
rm -rf /scratch/ptolemy/users/$USER/.cargo/
rm -rf /scratch/ptolemy/users/$USER/.rustup/
```

**Note:** For the assignment, keep the trained model and report until after grading!

## Verification Checklist

Before submitting job, verify scratch storage is configured:

- [ ] `NANOCHAT_BASE_DIR` points to `/scratch` (not `~/.cache`)
- [ ] `VENV_PATH` is in `/scratch` (not `.venv` in working directory)
- [ ] `CARGO_HOME` and `RUSTUP_HOME` are in `/scratch`
- [ ] `UV_CACHE_DIR` and `PIP_CACHE_DIR` are in `/scratch`
- [ ] Home directory is < 10 GB
- [ ] Sufficient scratch space available (check `quota -s /scratch/ptolemy`)

## Summary

✅ **All scripts verified to use /scratch for large data**
✅ **No home directory quota issues**
✅ **Optimal I/O performance for training**
✅ **Easy cleanup after assignment**

---

Last verified: 2025-10-30
