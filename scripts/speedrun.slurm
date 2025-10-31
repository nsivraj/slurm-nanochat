#!/bin/bash
#SBATCH --job-name=nanochat-speedrun
#SBATCH --account=class-cse8990
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/nanochat_speedrun_%j.out
#SBATCH --error=logs/nanochat_speedrun_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# SLURM job script for nanochat speedrun on Ptolemy HPC
# This is the "Best ChatGPT clone that $100 can buy"
# Designed to run in ~4 hours on 8XH100 node (or 8XA100 a bit slower)
#
# Usage: sbatch --mail-user=your_email@msstate.edu scripts/speedrun.slurm
# Or set EMAIL environment variable: EMAIL=your_email@msstate.edu sbatch scripts/speedrun.slurm

# Get email from .env.local if it exists, otherwise use environment variable
if [ -f ".env.local" ]; then
    ENV_EMAIL=$(grep '^email=' .env.local | cut -d'=' -f2)
    MAIL_USER=${EMAIL:-$ENV_EMAIL}
else
    MAIL_USER=${EMAIL:-your_email@msstate.edu}
fi

# Fail if email is not set properly
if [ "$MAIL_USER" = "your_email@msstate.edu" ] || [ -z "$MAIL_USER" ]; then
    echo "❌ ERROR: Email address not provided!"
    echo "Usage:"
    echo "  1. Set email in .env.local file (email=your_email@msstate.edu)"
    echo "  OR"
    echo "  2. sbatch --mail-user=your_email@msstate.edu scripts/speedrun.slurm"
    echo "  OR"
    echo "  3. EMAIL=your_email@msstate.edu sbatch scripts/speedrun.slurm"
    exit 1
fi

echo "Email notifications will be sent to: $MAIL_USER"

echo "=================================================="
echo "nanochat Speedrun Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load required CUDA modules (for GPU support)
module purge
module load cuda-toolkit
module load spack-managed-x86-64_v3/v1.0

# Show loaded modules
echo "Loaded modules:"
module list

# Set environment variables
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="/scratch/ptolemy/users/$USER/cache/huggingface"
export TORCH_HOME="/scratch/ptolemy/users/$USER/cache/torch"
export CARGO_HOME="/scratch/ptolemy/users/$USER/.cargo"
export RUSTUP_HOME="/scratch/ptolemy/users/$USER/.rustup"
export UV_CACHE_DIR="/scratch/ptolemy/users/$USER/cache/uv"
export PIP_CACHE_DIR="/scratch/ptolemy/users/$USER/cache/pip"

# Create necessary directories
mkdir -p logs
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $CARGO_HOME
mkdir -p $RUSTUP_HOME
mkdir -p $UV_CACHE_DIR
mkdir -p $PIP_CACHE_DIR

# Activate virtual environment
VENV_PATH="/scratch/ptolemy/users/$USER/nanochat-venv"
source $VENV_PATH/bin/activate

# Show environment info
echo -e "\nEnvironment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"

# GPU verification
echo -e "\nGPU Verification:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Test PyTorch CUDA
echo -e "\nPyTorch CUDA Test:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# ============================================================================
# CRITICAL: Verify all required data is downloaded
# GPU compute nodes do NOT have internet access!
# Data must be downloaded on ptolemy-devel-1 BEFORE submitting this job
# ============================================================================

echo -e "\n=================================================="
echo "Verifying Required Data (GPU nodes have no internet!)"
echo "=================================================="

DATA_CHECK_FAILED=false

# Check dataset shards
DATA_DIR="$NANOCHAT_BASE_DIR/data"
if [ -d "$DATA_DIR" ]; then
    SHARD_COUNT=$(ls -1 $DATA_DIR/*.bin 2>/dev/null | wc -l)
    if [ $SHARD_COUNT -ge 240 ]; then
        echo "✓ Dataset shards: $SHARD_COUNT/240"
    else
        echo "❌ Dataset shards: $SHARD_COUNT/240 (need at least 240)"
        DATA_CHECK_FAILED=true
    fi
else
    echo "❌ Dataset directory not found: $DATA_DIR"
    DATA_CHECK_FAILED=true
fi

# Check tokenizer
TOKENIZER_PATH="$NANOCHAT_BASE_DIR/tokenizer/tokenizer_2pow16.model"
if [ -f "$TOKENIZER_PATH" ]; then
    echo "✓ Tokenizer trained"
else
    echo "❌ Tokenizer not found: $TOKENIZER_PATH"
    DATA_CHECK_FAILED=true
fi

# Check eval bundle
EVAL_BUNDLE_PATH="$NANOCHAT_BASE_DIR/eval_bundle"
if [ -d "$EVAL_BUNDLE_PATH" ]; then
    echo "✓ Evaluation bundle present"
else
    echo "❌ Evaluation bundle not found: $EVAL_BUNDLE_PATH"
    DATA_CHECK_FAILED=true
fi

# Check identity conversations
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    echo "✓ Identity conversations present"
else
    echo "❌ Identity conversations not found: $IDENTITY_FILE"
    DATA_CHECK_FAILED=true
fi

# Fail if any required data is missing
if [ "$DATA_CHECK_FAILED" = true ]; then
    echo -e "\n=================================================="
    echo "❌ ERROR: Required data not found!"
    echo "=================================================="
    echo ""
    echo "GPU compute nodes do NOT have internet access."
    echo "All data must be downloaded BEFORE submitting this job."
    echo ""
    echo "To download required data:"
    echo ""
    echo "1. SSH to development server (which has internet):"
    echo "   ssh [username]@ptolemy-devel-1.arc.msstate.edu"
    echo ""
    echo "2. Navigate to nanochat directory:"
    echo "   cd /scratch/ptolemy/users/\$USER/slurm-nanochat"
    echo ""
    echo "3. Run the data download script:"
    echo "   bash scripts/download_data.sh"
    echo ""
    echo "4. Wait for download to complete (~30-60 minutes for ~24GB data)"
    echo ""
    echo "5. Re-submit this SLURM job:"
    echo "   sbatch scripts/speedrun.slurm"
    echo ""
    echo "=================================================="
    exit 1
fi

echo -e "\n✅ All required data verified!"
echo "=================================================="

# Setup wandb (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy  # Skip wandb logging
fi

# -----------------------------------------------------------------------------
# Start the speedrun workflow

echo -e "\n=================================================="
echo "Starting nanochat speedrun workflow..."
echo "=================================================="

# Clear and initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Evaluate tokenizer (already trained via download_data.sh)

echo -e "\n[1/5] Evaluating tokenizer..."
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Pretrain the d20 model (561M parameters)
echo -e "\n[2/5] Pretraining base model (this will take ~2-3 hours)..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN

# Evaluate base model
echo "Evaluating base model..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining

echo -e "\n[3/5] Midtraining (teaching conversation format)..."

# Identity conversations already downloaded via download_data.sh
# Run midtraining and eval
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning

echo -e "\n[4/5] Supervised finetuning..."

# Train SFT and eval
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate final report

echo -e "\n[5/5] Generating final report..."
python -m nanochat.report generate

# Copy report to working directory for easy access
if [ -f "$NANOCHAT_BASE_DIR/report/report.md" ]; then
    cp $NANOCHAT_BASE_DIR/report/report.md ./report.md
    echo "✓ Report saved to ./report.md"
fi

# Check if training was successful
if [ $? -eq 0 ]; then
    echo -e "\n✅ nanochat speedrun completed successfully!"

    echo -e "\nModel artifacts location:"
    echo "$NANOCHAT_BASE_DIR"

    echo -e "\nNext steps:"
    echo "1. Review the report: cat report.md"
    echo "2. Chat with your model via CLI:"
    echo "   python -m scripts.chat_cli -p 'Why is the sky blue?'"
    echo "3. Or serve via web UI (requires port forwarding):"
    echo "   python -m scripts.chat_web"

else
    echo -e "\n❌ Training failed!"
    exit 1
fi

echo -e "\n=================================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds (~$(($SECONDS / 3600)) hours)"
echo "=================================================="
