# nanochat Setup Guide for Ptolemy HPC

This guide explains how to set up and run nanochat on the Ptolemy HPC cluster at MSU.

## Prerequisites

- Access to Ptolemy HPC cluster
- Account with `class-cse8990` allocation
- SSH access configured

## Quick Start

### 1. Clone the Repository (if not already done)

```bash
# On Ptolemy
cd /scratch/ptolemy/users/$USER
git clone git@github.com:nsivraj/slurm-nanochat.git
cd nanochat
```

### 2. Configure Email Notifications

Create `.env.local` file:

```bash
cp .env.local.example .env.local
# Edit the file
nano .env.local
```

Change `email=your_email@msstate.edu` to your actual email.

### 3. Set Up Environment (on ptolemy-devel-1 or ptolemy-devel-2)

**IMPORTANT:** Run these steps on a devel server (which has internet access):

```bash
# SSH to development server (either devel-1 or devel-2)
ssh [your_username]@ptolemy-devel-1.arc.msstate.edu
# OR
ssh [your_username]@ptolemy-devel-2.arc.msstate.edu

# Navigate to nanochat directory
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Run setup script (loads Python 3.12, creates venv in /scratch)
bash scripts/setup_environment.sh

# Install uv package manager
pip install uv

# Install nanochat and ALL dependencies (REQUIRED)
pip install -e '.[gpu]'
```

**Important notes:**
- nanochat requires Python 3.10+. The setup script loads Python 3.12.5 automatically
- `pip install -e '.[gpu]'` installs nanochat in editable mode with all GPU dependencies
- This includes: torch, requests, tqdm, numpy, and all other required packages

### 4. Download Training Data (on ptolemy-devel-1 or ptolemy-devel-2)

**CRITICAL:** GPU compute nodes do NOT have internet access. All data must be downloaded BEFORE submitting the SLURM job.

```bash
# Still on ptolemy-devel-1
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Run data download script (~30-60 minutes for ~24GB data)
bash scripts/download_data.sh
```

This script will:

- Download 240 dataset shards (~24GB)
- Build and train the BPE tokenizer
- Download evaluation bundle (~162MB)
- Download identity conversations (~2.3MB)
- Verify all required data is present

### 5. Submit SLURM Job

From any Ptolemy login node:

```bash
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Create logs directory
mkdir -p logs

# Submit the job
sbatch scripts/speedrun.slurm
```

Or specify email directly:

```bash
sbatch --mail-user=your_email@msstate.edu scripts/speedrun.slurm
```

**Note:** If you forgot to download data, the job will fail immediately with instructions on how to download it.

### 6. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch the output log (replace JOBID with your actual job ID)
tail -f logs/nanochat_speedrun_JOBID.out

# Check for errors
tail -f logs/nanochat_speedrun_JOBID.err
```

## Resource Allocation

The speedrun configuration requests:

- **Partition:** `gpu-a100`
- **Nodes:** 1
- **GPUs:** 8 x A100 (80GB each)
- **CPUs:** 16
- **Memory:** 256GB
- **Time Limit:** 8 hours (actual runtime ~4-5 hours)

## Expected Workflow

### Data Download Phase (on ptolemy-devel-1, ~30-60 min)

Run `scripts/download_data.sh` which:

- Downloads 240 dataset shards (~24GB)
- Builds rustbpe tokenizer
- Trains BPE tokenizer on 2B characters
- Downloads evaluation bundle (~162MB)
- Downloads identity conversations (~2.3MB)

### Training Phase (on GPU compute nodes, ~4 hours)

The SLURM script (`speedrun.slurm`) runs:

1. **Tokenizer Evaluation** (~5 min)

   - Evaluates compression ratio on test data

2. **Base Model Pretraining** (~2-3 hours)

   - Trains d20 model (561M parameters)
   - Trains on ~11.2B tokens
   - Evaluates CORE score

3. **Midtraining** (~30-45 min)

   - Teaches conversation format
   - Uses pre-downloaded identity conversations
   - Adds special tokens and tool use
   - Evaluates chat capabilities

4. **Supervised Finetuning** (~30-45 min)

   - Domain adaptation
   - Improves task performance
   - Final evaluation

5. **Report Generation** (~5 min)
   - Creates `report.md` with all metrics
   - Includes model performance benchmarks

## Output Locations

All outputs are stored in `/scratch/ptolemy/users/$USER/nanochat-cache/`:

```
nanochat-cache/
├── data/              # Dataset shards
├── models/            # Trained model checkpoints
├── eval_bundle/       # Evaluation data
├── report/            # Training reports
└── tokenizer/         # Trained tokenizer
```

The final `report.md` is also copied to the working directory.

## After Training Completes

### View Results

```bash
# Read the report
cat report.md

# Check model performance metrics at the bottom of the report
```

### Chat with Your Model (Interactive Session)

Request an interactive GPU session:

```bash
srun --account=class-cse8990 --partition=gpu-a100 --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
```

Then activate environment and chat:

```bash
cd /scratch/ptolemy/users/$USER/nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"

# Chat via CLI
python -m scripts.chat_cli -p "Why is the sky blue?"

# Or interactive chat
python -m scripts.chat_cli
```

### Serve Web UI (requires port forwarding)

From interactive session:

```bash
python -m scripts.chat_web
```

Then set up SSH port forwarding from your local machine:

```bash
# On your local machine
ssh -L 8000:compute-node-name:8000 [username]@ptolemy.arc.msstate.edu
```

Visit `http://localhost:8000` in your browser.

## Troubleshooting

### Out of Memory (OOM) Errors

If you run out of GPU memory, reduce batch size in the SLURM script:

```bash
# Edit scripts/speedrun.slurm
# Change the training commands to include --device_batch_size
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --device_batch_size=16 --run=$WANDB_RUN
```

### Dependency Installation Issues

Always install dependencies on `ptolemy-devel-1` server:

```bash
ssh [username]@ptolemy-devel-1.arc.msstate.edu
cd /scratch/ptolemy/users/$USER/nanochat
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate
uv sync --extra gpu
```

### Job Fails Early

Check error log:

```bash
cat logs/nanochat_speedrun_JOBID.err
```

Common issues:

- Email not configured in `.env.local`
- Virtual environment not created
- Dependencies not installed
- Insufficient disk space in `/scratch`

### Data Not Downloaded Error

If the SLURM job fails with "Required data not found":

```bash
# SSH to devel node (which has internet)
ssh [username]@ptolemy-devel-1.arc.msstate.edu

# Navigate to nanochat
cd /scratch/ptolemy/users/$USER/slurm-nanochat

# Run data download script
bash scripts/download_data.sh

# Wait for completion, then re-submit job
sbatch scripts/speedrun.slurm
```

### Dataset Download Slow

The download takes ~30-60 minutes for ~24GB. This is normal. The script shows progress as it downloads each shard.

## Training Larger Models

To train the d26 model (~$300, 12 hours):

Edit `scripts/speedrun.slurm` to modify these lines:

```bash
# Download more data shards (450 instead of 240)
python -m nanochat.dataset -n 450 &

# Increase depth, reduce batch size to avoid OOM
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16 --run=$WANDB_RUN

# Use same batch size for midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16 --run=$WANDB_RUN
```

Also increase time limit:

```bash
#SBATCH --time=16:00:00
```

## File Structure

```
slurm-nanochat/
├── scripts/
│   ├── setup_environment.sh    # Environment setup for Ptolemy
│   ├── speedrun.slurm          # SLURM job script
│   ├── base_train.py           # Base model training
│   ├── chat_cli.py             # Chat via command line
│   ├── chat_web.py             # Chat via web UI
│   └── ...                     # Other training scripts
├── nanochat/                   # Core nanochat library
├── .env.local.example          # Example configuration
├── PTOLEMY_SETUP.md           # This file
└── README.md                   # Original nanochat README
```

## Support

- **nanochat Documentation:** See main [README.md](README.md)
- **Ptolemy HPC Help:** [https://www.hpc.msstate.edu/computing/ptolemy/](https://www.hpc.msstate.edu/computing/ptolemy/)
- **SLURM Documentation:** [https://slurm.schedmd.com/](https://slurm.schedmd.com/)

## Assignment Note

This setup is designed to fulfill the "Analytical Deconstruction" requirement of Assignment 3:

> Clone and run the nanochat repository on your available compute resources.
> This exercise will prepare you for the next assignment, where you will design and train your own LLM.

After training completes, you'll have:

- A working ChatGPT-style model trained from scratch
- Performance metrics and evaluation results
- Hands-on experience with the complete LLM pipeline
- Understanding of Transformer implementation details
