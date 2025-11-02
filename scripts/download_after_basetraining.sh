#!/bin/bash
# Download datasets required for midtraining and SFT phases
# Must be run on ptolemy-devel-1 (has internet access)
#
# Usage: bash scripts/download_after_basetraining.sh

set -e  # Exit on error

echo "=================================================="
echo "Downloading Datasets for Midtraining & SFT"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/mid_train.py" ]; then
    echo "❌ ERROR: Must be run from nanochat root directory"
    echo "Current directory: $(pwd)"
    echo ""
    echo "Please cd to the nanochat directory first:"
    echo "  cd /scratch/ptolemy/users/\$USER/slurm-nanochat"
    echo "  bash scripts/download_after_basetraining.sh"
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

# Download all datasets required for midtraining and SFT
echo "Downloading datasets required for midtraining and SFT..."
echo "This may take 5-15 minutes..."
echo ""

python3 << 'EOF'
import os
from datasets import load_dataset

# Ensure cache directory is set
hf_home = f"/scratch/ptolemy/users/{os.environ['USER']}/cache/huggingface"
os.environ['HF_HOME'] = hf_home
print(f"Using HuggingFace cache: {hf_home}")
print()

datasets_to_download = [
    # Midtraining datasets
    ("MMLU (midtraining)", "cais/mmlu", "auxiliary_train", "train"),
    ("GSM8K (midtraining)", "openai/gsm8k", "main", "train"),
    ("SmolTalk (midtraining)", "HuggingFaceTB/smol-smoltalk", None, "train"),

    # SFT datasets
    ("ARC (SFT)", "allenai/ai2_arc", "ARC-Easy", "train"),
    ("GSM8K (SFT)", "openai/gsm8k", "main", "train"),
    ("SmolTalk (SFT)", "HuggingFaceTB/smol-smoltalk", None, "train"),
]

download_count = 0
error_count = 0

for name, dataset_id, subset, split in datasets_to_download:
    print(f"[{download_count + 1}/{len(datasets_to_download)}] Downloading {name}...")
    print(f"  Dataset: {dataset_id}")
    if subset:
        print(f"  Subset: {subset}")
    print(f"  Split: {split}")

    try:
        if subset:
            ds = load_dataset(dataset_id, subset, split=split)
        else:
            ds = load_dataset(dataset_id, split=split)

        print(f"  ✓ Successfully downloaded {name}")
        print(f"    Rows: {len(ds):,}")
        print(f"    Cached at: {hf_home}/datasets/{dataset_id.replace('/', '___')}/")
        download_count += 1
    except Exception as e:
        print(f"  ❌ Failed to download {name}: {e}")
        error_count += 1

    print()

print("=" * 60)
if error_count == 0:
    print(f"✅ All {download_count} datasets downloaded successfully!")
else:
    print(f"⚠️  Downloaded {download_count} datasets, {error_count} failed")
    print("Some datasets may already be cached or may not be critical")
print("=" * 60)
EOF

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Verifying cached datasets..."
    echo ""

    # Check critical datasets
    CRITICAL_DATASETS=(
        "$HF_HOME/datasets/cais___mmlu"
        "$HF_HOME/datasets/openai___gsm8k"
        "$HF_HOME/datasets/HuggingFaceTB___smol-smoltalk"
    )

    ALL_FOUND=true
    for dataset in "${CRITICAL_DATASETS[@]}"; do
        if [ -d "$dataset" ]; then
            echo "✓ Found: $dataset"
        else
            echo "⚠️  Missing: $dataset"
            ALL_FOUND=false
        fi
    done

    echo ""
    if [ "$ALL_FOUND" = true ]; then
        echo "=================================================="
        echo "✅ All critical datasets verified!"
        echo "=================================================="
        echo ""
        echo "Next Steps:"
        echo ""
        echo "1. To resume training from midtraining + SFT:"
        echo "   WANDB_RUN=my_resume_run sbatch scripts/resume_mid_sft.slurm"
        echo ""
        echo "2. To run full training from scratch:"
        echo "   WANDB_RUN=my_full_run sbatch scripts/speedrun.slurm"
        echo ""
        echo "3. Monitor the job:"
        echo "   squeue -u \$USER"
        echo ""
        echo "4. Check logs after submission:"
        echo "   tail -f logs/nanochat_*_*.out"
        echo ""
    else
        echo "=================================================="
        echo "⚠️  WARNING: Some datasets missing"
        echo "=================================================="
        echo ""
        echo "Some critical datasets were not found in cache."
        echo "Training may still work if datasets are already cached"
        echo "or if they download during training (requires internet)."
        echo ""
        echo "If training fails with 'LocalEntryNotFoundError',"
        echo "try running this script again."
        echo ""
    fi
else
    echo ""
    echo "❌ Dataset download failed!"
    echo "Please check the error messages above"
    exit 1
fi
