#!/bin/bash
#SBATCH --job-name=nanochat-resume-mid-sft
#SBATCH --account=class-cse8990
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/nanochat_resume_mid_sft_%j.out
#SBATCH --error=logs/nanochat_resume_mid_sft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# SLURM job script to resume nanochat training from midtraining phase
# This script assumes base model training is already complete
# It will run: midtraining + SFT (no base training)
#
# Usage: sbatch --mail-user=your_email@msstate.edu scripts/resume_mid_sft.slurm
# Or set EMAIL environment variable: EMAIL=your_email@msstate.edu sbatch scripts/resume_mid_sft.slurm

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
    echo "  2. sbatch --mail-user=your_email@msstate.edu scripts/resume_mid_sft.slurm"
    echo "  OR"
    echo "  3. EMAIL=your_email@msstate.edu sbatch scripts/resume_mid_sft.slurm"
    exit 1
fi

echo "Email notifications will be sent to: $MAIL_USER"

echo "=================================================="
echo "nanochat Resume Training: Midtraining + SFT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load required modules (Python 3.12, CUDA)
module purge
module load python/3.12.5
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

# --- WandB Offline Configuration ---
# GPU nodes have no internet, so we run WandB in offline mode.
# Logs will be saved locally and can be synced later from a devel node.
export WANDB_MODE=offline
export WANDB_DIR="$NANOCHAT_BASE_DIR/wandb"

# Create necessary directories
mkdir -p logs
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $WANDB_DIR
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
# CRITICAL: Verify base model checkpoint exists
# This script assumes base training is already complete
# ============================================================================

echo -e "\n=================================================="
echo "Verifying Base Model Checkpoint"
echo "=================================================="

BASE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
if [ -d "$BASE_CHECKPOINT_DIR" ]; then
    echo "✓ Base model checkpoint found: $BASE_CHECKPOINT_DIR"
    # List checkpoint files
    echo -e "\nCheckpoint contents:"
    ls -lh $BASE_CHECKPOINT_DIR
else
    echo "❌ ERROR: Base model checkpoint not found!"
    echo "Expected location: $BASE_CHECKPOINT_DIR"
    echo ""
    echo "This script requires a completed base model training."
    echo "Please run base training first using:"
    echo "  sbatch scripts/speedrun.slurm"
    echo ""
    echo "Or run just base training with:"
    echo "  torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=my_training_run"
    echo ""
    exit 1
fi

# ============================================================================
# CRITICAL: Verify required data is present
# ============================================================================

echo -e "\n=================================================="
echo "Verifying Required Data"
echo "=================================================="

DATA_CHECK_FAILED=false

# Check identity conversations (needed for midtraining)
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    echo "✓ Identity conversations present"
else
    echo "❌ Identity conversations not found: $IDENTITY_FILE"
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

# Fail if any required data is missing
if [ "$DATA_CHECK_FAILED" = true ]; then
    echo -e "\n=================================================="
    echo "❌ ERROR: Required data not found!"
    echo "=================================================="
    echo ""
    echo "Please download required data on ptolemy-devel-1:"
    echo "  bash scripts/download_data.sh"
    echo ""
    exit 1
fi

echo -e "\n✅ All required data verified!"

# ============================================================================
# CRITICAL: Validate WANDB_RUN variable
# ============================================================================

echo -e "\n=================================================="
echo "Validating WANDB_RUN Configuration"
echo "=================================================="

# Check if WANDB_RUN is set
if [ -z "$WANDB_RUN" ]; then
    echo "❌ ERROR: WANDB_RUN environment variable is not set!"
    echo ""
    echo "To fix this, set WANDB_RUN when submitting the job:"
    echo "  WANDB_RUN=my_training_run sbatch scripts/resume_mid_sft.slurm"
    echo ""
    echo "Note: You don't need wandb configured. Any non-'dummy' name will work."
    echo ""
    exit 1
fi

# Warn if WANDB_RUN is set to 'dummy'
if [ "$WANDB_RUN" = "dummy" ]; then
    echo "❌ ERROR: WANDB_RUN is set to 'dummy'!"
    echo ""
    echo "Please set WANDB_RUN to any other name:"
    echo "  WANDB_RUN=my_training_run sbatch scripts/resume_mid_sft.slurm"
    echo ""
    exit 1
fi

echo "✓ WANDB_RUN is set to: $WANDB_RUN"
echo "=================================================="

# -----------------------------------------------------------------------------
# Start the resume workflow (midtraining + SFT)

echo -e "\n=================================================="
echo "Resuming nanochat training from midtraining phase..."
echo "=================================================="

# -----------------------------------------------------------------------------
# Midtraining

echo -e "\n[1/3] Midtraining (teaching conversation format)..."
echo "This will take approximately 2-3 hours"

# Run midtraining and eval
# torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
# if [ $? -ne 0 ]; then
#     echo "❌ Midtraining failed!"
#     exit 1
# fi

echo "Evaluating midtraining checkpoint..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
if [ $? -ne 0 ]; then
    echo "❌ Midtraining evaluation failed!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Supervised Finetuning

echo -e "\n[2/3] Supervised finetuning..."
echo "This will take approximately 2-3 hours"

# Train SFT and eval
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
if [ $? -ne 0 ]; then
    echo "❌ SFT training failed!"
    exit 1
fi

echo "Evaluating SFT checkpoint..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
if [ $? -ne 0 ]; then
    echo "❌ SFT evaluation failed!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Generate final report

echo -e "\n[3/3] Generating final report..."
python -m nanochat.report generate

# Copy report to working directory for easy access
if [ -f "$NANOCHAT_BASE_DIR/report/report.md" ]; then
    cp $NANOCHAT_BASE_DIR/report/report.md ./report.md
    echo "✓ Report saved to ./report.md"
fi

# Check if training was successful
if [ $? -eq 0 ]; then
    echo -e "\n✅ Midtraining and SFT completed successfully!"

    echo -e "\nModel artifacts location:"
    echo "$NANOCHAT_BASE_DIR"

    echo -e "\nCheckpoints created:"
    echo "  - Midtraining: $NANOCHAT_BASE_DIR/mid_checkpoints/"
    echo "  - SFT: $NANOCHAT_BASE_DIR/chatsft_checkpoints/"

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
