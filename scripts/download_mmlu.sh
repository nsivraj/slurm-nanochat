#!/bin/bash
# Download MMLU dataset for midtraining
# Must be run on ptolemy-devel-1 (has internet access)
#
# Usage: bash scripts/download_mmlu.sh

set -e  # Exit on error

echo "=================================================="
echo "Downloading MMLU Dataset for Midtraining"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/mid_train.py" ]; then
    echo "❌ ERROR: Must be run from nanochat root directory"
    echo "Current directory: $(pwd)"
    echo ""
    echo "Please cd to the nanochat directory first:"
    echo "  cd /scratch/ptolemy/users/\$USER/slurm-nanochat"
    echo "  bash scripts/download_mmlu.sh"
    exit 1
fi

# Set cache directory
export HF_HOME="/scratch/ptolemy/users/$USER/cache/huggingface"
mkdir -p $HF_HOME

echo "Cache directory: $HF_HOME"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  WARNING: Virtual environment not activated"
    echo "Attempting to activate: /scratch/ptolemy/users/$USER/nanochat-venv"

    VENV_PATH="/scratch/ptolemy/users/$USER/nanochat-venv"
    if [ -d "$VENV_PATH" ]; then
        source $VENV_PATH/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "❌ ERROR: Virtual environment not found at $VENV_PATH"
        echo "Please create the virtual environment first or activate it manually"
        exit 1
    fi
    echo ""
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Download MMLU dataset
echo "Downloading MMLU dataset (cais/mmlu)..."
echo "This may take a few minutes..."
echo ""

python3 << 'EOF'
import os
from datasets import load_dataset

# Ensure cache directory is set
hf_home = f"/scratch/ptolemy/users/{os.environ['USER']}/cache/huggingface"
os.environ['HF_HOME'] = hf_home
print(f"Using HuggingFace cache: {hf_home}")
print()

# Download MMLU auxiliary_train split (used in midtraining)
print("Downloading MMLU auxiliary_train split...")
try:
    ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    print(f"✓ Successfully downloaded MMLU auxiliary_train")
    print(f"  Rows: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Cached at: {hf_home}/datasets/cais___mmlu/auxiliary_train/")
except Exception as e:
    print(f"❌ Failed to download MMLU: {e}")
    exit(1)

print()
print("=" * 60)
print("✅ MMLU dataset download complete!")
print("=" * 60)
EOF

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Verifying cached dataset..."

    MMLU_CACHE="$HF_HOME/datasets/cais___mmlu"
    if [ -d "$MMLU_CACHE" ]; then
        echo "✓ MMLU dataset cached successfully"
        echo ""
        echo "Cache location:"
        ls -lh $MMLU_CACHE
        echo ""
        echo "=================================================="
        echo "Next Steps:"
        echo "=================================================="
        echo ""
        echo "1. Resubmit the resume training job:"
        echo "   WANDB_RUN=my_training_run sbatch --mail-user=your_email@msstate.edu scripts/resume_mid_sft.slurm"
        echo ""
        echo "2. Monitor the job:"
        echo "   squeue -u \$USER"
        echo ""
        echo "3. Check logs after submission:"
        echo "   tail -f logs/nanochat_resume_mid_sft_*.out"
        echo ""
    else
        echo "⚠️  WARNING: MMLU cache directory not found"
        echo "Expected: $MMLU_CACHE"
        echo "Download may have failed"
        exit 1
    fi
else
    echo ""
    echo "❌ MMLU download failed!"
    echo "Please check the error messages above"
    exit 1
fi
