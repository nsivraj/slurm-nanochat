# How-To: Setup Environment

This guide covers environment setup for all three training platforms: Local CPU, Ptolemy HPC, and Production GPU.

---

## Choose Your Platform

**Local CPU** - Train on your laptop/desktop
→ [Jump to Local CPU Setup](#local-cpu-setup)

**Ptolemy HPC** - Train on MSU's cluster
→ [Jump to Ptolemy HPC Setup](#ptolemy-hpc-setup)

**Production GPU** - Train on cloud GPUs
→ [Jump to Production GPU Setup](#production-gpu-setup)

---

## Local CPU Setup

### Prerequisites
- Python 3.10+ installed
- Rust/Cargo (latest nightly recommended)
- ~2GB free disk space
- Internet connection
- 8GB+ RAM recommended

### Step-by-Step Setup

#### 1. Verify Python Version

```bash
python --version  # Should be 3.10 or higher
```

If you need to install/update Python, visit [python.org](https://www.python.org/).

#### 2. Install/Update Rust

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update to latest nightly
rustup update
rustc --version  # Should be 1.93.0-nightly or newer (Oct 2025+)
```

**Important:** nanochat requires Rust edition2024, only available in recent nightly builds.

#### 3. Clone Repository

```bash
git clone <repo-url>
cd slurm-nanochat
```

#### 4. Run Training Script

The script handles all setup automatically:

```bash
bash scripts/local_cpu_train.sh
```

This will:
- Create Python virtual environment (`.venv/`)
- Install UV package manager
- Install all dependencies
- Download training data (~400MB)
- Train tokenizer and model
- Generate report

### What Gets Installed

**Python Packages:**
- `torch` - PyTorch (CPU version)
- `numpy`, `tqdm` - Utilities
- `requests` - Data downloading
- `huggingface_hub` - Dataset access

**System Tools:**
- `uv` - Fast Python package manager
- Rust toolchain - For BPE tokenizer

### File Locations

```
~/.cache/nanochat/          # All training artifacts
├── base_data/              # Training data (4 shards)
├── tokenizer/              # Trained tokenizer
├── models/                 # Model checkpoints
├── eval_bundle/            # Evaluation data
└── report/                 # Training logs

./                          # Repository
├── .venv/                  # Python virtual environment
├── report.md               # Final report
└── scripts/                # Training scripts
```

### Troubleshooting

**`uv: command not found`**
```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Rust edition2024 error**
```bash
rustup update
```

**Out of memory**
Edit `scripts/local_cpu_train.sh`:
```bash
--depth=2              # Smaller model
--max_seq_len=512      # Shorter sequences
```

---

## Ptolemy HPC Setup

### Prerequisites
- Ptolemy HPC account
- `class-cse8990` allocation
- SSH access configured
- Email address for notifications

### Step-by-Step Setup

#### 1. SSH to Development Node

**CRITICAL:** Use a devel node that has internet access!

```bash
ssh <username>@ptolemy-devel-1.arc.msstate.edu
```

Why devel node? GPU compute nodes have **no internet access**.

#### 2. Navigate to Scratch Space

**IMPORTANT:** Use `/scratch/ptolemy/users/$USER/` for all storage!

```bash
cd /scratch/ptolemy/users/$USER
```

Home directory has tiny quotas - don't use it for data or models!

#### 3. Clone Repository

```bash
git clone <repo-url> slurm-nanochat
cd slurm-nanochat
```

#### 4. Configure Email Notifications

```bash
cp .env.local.example .env.local
nano .env.local
```

Set your email:
```bash
EMAIL=your_email@msstate.edu
```

Save (Ctrl+X, Y, Enter).

#### 5. Run Setup Script

```bash
bash scripts/setup_environment.sh
```

This will:
- Load Python 3.12.5 module
- Create venv in `/scratch/ptolemy/users/$USER/nanochat-venv`
- Install UV package manager
- Install all dependencies

#### 6. Download Training Data

**MUST** be done on devel node (has internet):

```bash
# Still on ptolemy-devel-1
bash scripts/download_data.sh
```

**Time:** 30-60 minutes for ~24GB

This downloads:
- 240 dataset shards (~24GB)
- Evaluation bundle (~162MB)
- Identity conversations (~2.3MB)
- GPT-2/GPT-4 tokenizers
- SmolTalk dataset

**Go get coffee - this takes a while!**

### What Gets Installed

**Python Modules:**
- Python 3.12.5 (loaded via module system)
- PyTorch with CUDA support
- All nanochat dependencies

**Storage Layout:**
```
/scratch/ptolemy/users/$USER/
├── slurm-nanochat/              # Repository
│   ├── .env.local               # Email config
│   └── logs/                    # Job logs
│
├── nanochat-venv/               # Virtual environment
│
└── nanochat-cache/              # All training data
    ├── base_data/               # 240 shards (~24GB)
    ├── tokenizer/               # Tokenizer
    ├── models/                  # Checkpoints
    ├── eval_bundle/             # Eval data
    └── report/                  # Reports
```

### Critical Notes

**WANDB_RUN Requirement:**
- You **MUST** set `WANDB_RUN` when submitting jobs
- Any non-"dummy" name works (e.g., `my_run`, `test_1`)
- No wandb account needed
- Without this, midtraining and SFT are **SKIPPED**

**Internet Access:**
- Development nodes (devel-1, devel-2): ✅ Has internet
- GPU compute nodes: ❌ No internet
- **Always download data on devel node first**

**Storage:**
- Home directory: ~5GB quota (tiny!)
- Scratch: ~5TB quota (use this!)
- Put everything in `/scratch/ptolemy/users/$USER/`

### Troubleshooting

**Python version too old**
```bash
# Setup script handles this automatically
# It loads module python/3.12.5
```

**Missing dependencies**
```bash
# SSH to devel node
ssh <username>@ptolemy-devel-1.arc.msstate.edu
cd /scratch/ptolemy/users/$USER/slurm-nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
pip install -e '.[gpu]'
```

**Data download fails**
- Check internet connection on devel node
- Verify you're on devel-1 or devel-2 (not compute node)
- Check disk space: `df -h /scratch/ptolemy/users/$USER`

---

## Production GPU Setup

### Prerequisites
- Cloud GPU provider account (Lambda, RunPod, etc.)
- 8xH100 GPU instance
- SSH access
- Budget (~$100 for full run)

### Step-by-Step Setup

#### 1. Rent GPU Instance

Recommended configuration:
- **GPUs:** 8x H100 80GB
- **RAM:** 256GB+
- **Storage:** 100GB+
- **Provider:** Lambda Labs, RunPod, etc.

#### 2. SSH into Instance

```bash
ssh ubuntu@<instance-ip>
```

#### 3. Clone Repository

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

#### 4. Run Training Script

Use screen for long-running process:

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

Detach with `Ctrl-a d` and let it run (~4 hours).

### What Gets Installed

The `speedrun.sh` script handles everything:
- Python virtual environment
- PyTorch with CUDA
- All dependencies
- Dataset download (~24GB)
- Tokenizer training
- Full model training

### File Locations

```
~/.cache/nanochat/          # All training artifacts
├── base_data/              # 240 shards (~24GB)
├── tokenizer/              # Tokenizer
├── models/                 # Checkpoints
├── eval_bundle/            # Eval data
└── report/                 # Reports

./                          # Repository
├── .venv/                  # Virtual environment
├── speedrun.log            # Screen log
└── report.md               # Final report
```

### Monitoring

```bash
# Reattach to screen session
screen -r speedrun

# View log file
tail -f speedrun.log

# Check GPU usage
nvidia-smi
```

### Cost Management

**Estimated Costs:**
- 8xH100 @ ~$3/GPU/hour = ~$24/hour
- Training time: ~4 hours
- **Total:** ~$96

**Cost Saving Tips:**
- Terminate instance immediately after training
- Use preemptible/spot instances if available
- Monitor progress to catch errors early

---

## Platform Comparison

| Aspect | Local CPU | Ptolemy HPC | Production GPU |
|--------|-----------|-------------|----------------|
| **Setup Time** | 10 min | 1-2 hours | 30 min |
| **Data Size** | 400MB | 24GB | 24GB |
| **Training Time** | 1-3 hours | ~12 hours | ~4 hours |
| **Cost** | Free | Free | ~$100 |
| **Complexity** | Easy | Medium | Medium |
| **Internet** | Required | Devel only | Required |
| **Model Quality** | Poor | Production | Production |

---

## Next Steps

After setup is complete:

**Local CPU**
→ [Local CPU Quickstart](../tutorials/01-local-cpu-quickstart.md)

**Ptolemy HPC**
→ [HPC First Job](../tutorials/02-hpc-first-job.md)

**Production GPU**
→ See main [README.md](../../README.md)

---

## Common Issues Across All Platforms

### Rust edition2024 Error
**Solution:** Update Rust to latest nightly
```bash
rustup update
rustc --version  # Should be 1.93.0-nightly+
```

### Out of Memory
**Solution:** Reduce model size or batch size
```bash
# Edit training script
--device_batch_size=16  # Reduce from 32
--depth=12              # Smaller model
```

### Download Failures
**Solution:** Check internet and retry
- Verify internet connection
- Check disk space
- Scripts are idempotent (safe to retry)

### Python Version Issues
**Solution:**
- Local CPU: Install Python 3.10+
- Ptolemy: Use devel node (has Python 3.12.5)
- Production: Usually pre-installed

---

For platform-specific troubleshooting:
→ [Troubleshoot Common Issues](troubleshoot-common-issues.md)
