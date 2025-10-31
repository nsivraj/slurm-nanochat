# Ptolemy HPC Setup Complete ✓

The nanochat codebase has been successfully configured to run on Ptolemy HPC cluster.

## What Was Created

### SLURM Configuration Files

1. **`scripts/speedrun.slurm`**
   - Main SLURM job script for 8xA100 GPU training
   - Configures resources: 1 node, 8 GPUs, 256GB RAM, 8 hours
   - Runs complete nanochat pipeline (pretraining → midtraining → SFT → report)
   - **Verifies all data is downloaded before starting** (fails with helpful instructions if not)
   - Email notifications for job status
   - Automatic error handling and progress reporting
   - **No internet downloads** (all data must be pre-downloaded)

2. **`scripts/setup_environment.sh`**
   - Environment setup script for Ptolemy
   - Creates virtual environment in `/scratch/ptolemy/users/$USER/nanochat-venv`
   - Sets up CUDA modules and Python paths
   - Configures cache directories for HuggingFace, PyTorch, Rust/Cargo
   - Tests CUDA availability

3. **`scripts/download_data.sh`** ⭐ NEW - CRITICAL
   - **Must be run on ptolemy-devel-1 BEFORE submitting SLURM job**
   - Downloads all required data (~24GB, 30-60 minutes)
   - GPU compute nodes have NO internet access
   - Downloads dataset shards, eval bundle, identity conversations
   - Builds rustbpe tokenizer
   - Trains BPE tokenizer on 2B characters
   - Verifies all required data is present
   - Prevents job failures due to missing data

4. **`scripts/test_gpu.py`**
   - GPU test suite for validating setup
   - Tests PyTorch installation and CUDA availability
   - Verifies GPU operations with matrix multiplication benchmark
   - Checks multi-GPU configuration
   - Validates required dependencies

### Configuration Files

5. **`.env.local.example`**
   - Template for user configuration
   - Email notification setup
   - Optional Weights & Biases integration

### Documentation

6. **`PTOLEMY_SETUP.md`**
   - Comprehensive setup and usage guide
   - Step-by-step instructions for environment setup
   - Resource allocation details
   - Expected workflow and timeline
   - Output locations and file structure
   - Troubleshooting guide
   - Instructions for larger model training (d26)

7. **`ASSIGNMENT_README.md`**
   - Assignment-specific guide
   - Links code files to assignment requirements
   - Maps Transformer components to implementation
   - Code exploration tips and examples
   - Expected results and metrics
   - Assignment timeline recommendations

8. **`SETUP_COMPLETE.md`** (this file)
   - Summary of all changes
   - Next steps for running on Ptolemy

## Configuration Highlights

### Resource Allocation
- **Partition:** `gpu-a100` (Ptolemy's A100 GPU partition)
- **Account:** `class-cse8990` (your class allocation)
- **GPUs:** 8x A100 (80GB each)
- **Memory:** 256GB
- **Time Limit:** 8 hours (sufficient for ~4 hour speedrun)

### Storage Locations
All training artifacts stored in scratch for performance (NOT home directory):
```
/scratch/ptolemy/users/$USER/
├── nanochat-venv/          # Python virtual environment (~2-3 GB)
├── nanochat-cache/         # Training artifacts and models (~30 GB)
│   ├── data/               # Dataset shards (~24GB)
│   ├── models/             # Model checkpoints (~2-5 GB)
│   ├── eval_bundle/        # Evaluation data (~162 MB)
│   ├── report/             # Training reports (<1 MB)
│   └── tokenizer/          # Trained tokenizer (~100 MB)
├── cache/                  # Various caches (~3-8 GB total)
│   ├── huggingface/        # HF model cache (~1-5 GB)
│   ├── torch/              # PyTorch cache (~500 MB)
│   ├── uv/                 # UV package cache (~1-2 GB)
│   └── pip/                # Pip cache (~500 MB)
├── .cargo/                 # Rust/Cargo (~500 MB)
└── .rustup/                # Rust toolchain (~1 GB)

Total: ~35-50 GB in scratch (home directory stays < 100 MB)
```

**Why scratch?** Home directories have limited quotas (~50-100 GB). Scratch has larger quotas and better I/O performance for training.

### Key Environment Variables
All point to `/scratch` to avoid filling home directory quota:
```bash
NANOCHAT_BASE_DIR=/scratch/ptolemy/users/$USER/nanochat-cache
HF_HOME=/scratch/ptolemy/users/$USER/cache/huggingface
TORCH_HOME=/scratch/ptolemy/users/$USER/cache/torch
CARGO_HOME=/scratch/ptolemy/users/$USER/.cargo
RUSTUP_HOME=/scratch/ptolemy/users/$USER/.rustup
UV_CACHE_DIR=/scratch/ptolemy/users/$USER/cache/uv
PIP_CACHE_DIR=/scratch/ptolemy/users/$USER/cache/pip
OMP_NUM_THREADS=1
```

See [SCRATCH_STORAGE_VERIFICATION.md](SCRATCH_STORAGE_VERIFICATION.md) for detailed storage breakdown.

## Next Steps

### For First-Time Setup (on ptolemy-devel-1)

**IMPORTANT:** Steps 1-6 must be done on `ptolemy-devel-1` (which has internet access)

1. **SSH to Ptolemy development server:**
   ```bash
   ssh [your_username]@ptolemy-devel-1.arc.msstate.edu
   ```

2. **Navigate to nanochat directory:**
   ```bash
   cd /scratch/ptolemy/users/$USER/slurm-nanochat
   ```

3. **Configure email notifications:**
   ```bash
   cp .env.local.example .env.local
   nano .env.local
   # Change: email=your_email@msstate.edu
   ```

4. **Run setup script:**
   ```bash
   bash scripts/setup_environment.sh
   ```

5. **Install dependencies:**
   ```bash
   pip install uv
   uv sync --extra gpu
   ```

6. **⭐ CRITICAL: Download all training data:**
   ```bash
   bash scripts/download_data.sh
   ```

   This takes ~30-60 minutes and downloads ~24GB of data. **This step is REQUIRED before submitting the SLURM job because GPU compute nodes do NOT have internet access.**

7. **Test GPU setup (optional):**
   ```bash
   python scripts/test_gpu.py
   ```

### To Submit Training Job (any Ptolemy node)

1. **Create logs directory:**
   ```bash
   cd /scratch/ptolemy/users/$USER/slurm-nanochat
   mkdir -p logs
   ```

2. **Submit SLURM job:**
   ```bash
   sbatch scripts/speedrun.slurm
   ```

   Or with email:
   ```bash
   sbatch --mail-user=your_email@msstate.edu scripts/speedrun.slurm
   ```

   **The job will immediately verify all data is downloaded. If not, it will fail with instructions.**

3. **Monitor job:**
   ```bash
   # Check status
   squeue -u $USER

   # Watch output (replace JOBID)
   tail -f logs/nanochat_speedrun_JOBID.out
   ```

### After Training Completes

1. **Review results:**
   ```bash
   cat report.md
   ```

2. **Chat with your model:**
   ```bash
   # Request interactive session
   srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash

   # Activate environment
   source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
   export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"

   # Chat
   python -m scripts.chat_cli -p "Why is the sky blue?"
   ```

## Expected Timeline

- **Setup:** 15-30 minutes (one-time, on ptolemy-devel-1)
- **Data download:** 30-60 minutes (one-time, on ptolemy-devel-1) ⭐ REQUIRED
- **Job submission:** < 1 minute (any Ptolemy node)
- **Training:** ~4 hours (unattended, on GPU compute nodes)
- **Total:** ~5-6 hours from start to chatting with your model

## Files Modified vs Original nanochat

### New Files (Ptolemy-specific)
- `scripts/setup_environment.sh` ← NEW
- `scripts/download_data.sh` ← NEW ⭐ CRITICAL (must run before SLURM job)
- `scripts/speedrun.slurm` ← NEW (modified to check for downloaded data)
- `scripts/test_gpu.py` ← NEW
- `.env.local.example` ← NEW
- `PTOLEMY_SETUP.md` ← NEW
- `ASSIGNMENT_README.md` ← NEW
- `SETUP_COMPLETE.md` ← NEW
- `IMPORTANT_PTOLEMY_NOTES.md` ← NEW (internet access limitations)
- `SCRATCH_STORAGE_VERIFICATION.md` ← NEW (storage configuration)

### Unchanged Files (Original nanochat)
- All files in `nanochat/` directory
- All files in `scripts/` (except new ones above)
- All files in `tasks/`
- All files in `rustbpe/`
- `speedrun.sh` (original bash version, not used on SLURM)
- `pyproject.toml`
- `README.md`
- All other original files

## Differences from Medical-Topic-Classification Setup

### Similarities
- Uses same SLURM partition (`gpu-a100`)
- Uses same account (`class-cse8990`)
- Similar virtual environment setup pattern
- Email notification system
- Scratch-based storage for performance
- Module loading (CUDA toolkit)

### Differences
- **More GPUs:** 8x A100 instead of 1-2 (nanochat benefits from multi-GPU)
- **More Memory:** 256GB vs 32GB (larger model training)
- **Longer Runtime:** 8 hours vs 2 hours (complete pipeline)
- **Different Dependencies:** Uses `uv` package manager + Rust/Cargo
- **Distributed Training:** Uses `torchrun` for multi-GPU coordination
- **Two-Phase Workflow:** Data download (devel node) → Training (GPU nodes) ⭐
- **No Internet on GPU Nodes:** All data pre-downloaded on ptolemy-devel-1
- **Data Verification:** SLURM job checks for required data before starting
- **Rust Compilation:** Builds custom BPE tokenizer from Rust source (on devel node)

## Support Resources

- **Detailed Setup:** See `PTOLEMY_SETUP.md`
- **Assignment Guide:** See `ASSIGNMENT_README.md`
- **Original Docs:** See `README.md`
- **Ptolemy HPC:** https://www.hpc.msstate.edu/computing/ptolemy/
- **nanochat Repo:** https://github.com/karpathy/nanochat

## Notes

- This setup is optimized for the assignment's "Analytical Deconstruction" step
- All scripts are tested against the reference medical-topic-classification setup
- Email notifications will alert you when training starts, ends, or fails
- The speedrun configuration trains a 561M parameter model in ~4 hours
- Total cost on Ptolemy is covered by class allocation
- Training runs unattended - perfect for overnight or weekend runs

## Verification Checklist

Before submitting your first job, verify (all on ptolemy-devel-1):

- [ ] Cloned/copied nanochat to `/scratch/ptolemy/users/$USER/slurm-nanochat`
- [ ] Created `.env.local` with your email
- [ ] Ran `scripts/setup_environment.sh` on ptolemy-devel-1
- [ ] Installed `uv`: `pip install uv`
- [ ] Installed dependencies: `uv sync --extra gpu`
- [ ] **⭐ Downloaded all training data: `bash scripts/download_data.sh`** (CRITICAL!)
- [ ] Verified data download completed successfully (check script output)
- [ ] Created `logs/` directory: `mkdir -p logs`
- [ ] Ready to submit: `sbatch scripts/speedrun.slurm`

**If you skip the data download step, the SLURM job will fail immediately with instructions.**

Good luck with your nanochat training and assignment!
