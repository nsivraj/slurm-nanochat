#!/bin/bash
# Data Download Script for nanochat on Ptolemy HPC
# MUST be run on ptolemy-devel-1 (which has internet access)
# GPU compute nodes do NOT have internet access

echo "=================================================="
echo "nanochat Data Download for Ptolemy HPC"
echo "=================================================="

# Check if we're on a devel node
HOSTNAME=$(hostname)
if [[ ! "$HOSTNAME" =~ devel ]]; then
    echo "⚠️  WARNING: You appear to be on a compute node: $HOSTNAME"
    echo "   Compute nodes do NOT have internet access!"
    echo ""
    echo "Please run this script on ptolemy-devel-1:"
    echo "  ssh [username]@ptolemy-devel-1.arc.msstate.edu"
    echo "  cd /scratch/ptolemy/users/\$USER/slurm-nanochat"
    echo "  bash scripts/download_data.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set environment variables
export NANOCHAT_BASE_DIR="/scratch/ptolemy/users/$USER/nanochat-cache"
export HF_HOME="/scratch/ptolemy/users/$USER/cache/huggingface"
export TORCH_HOME="/scratch/ptolemy/users/$USER/cache/torch"
export CARGO_HOME="/scratch/ptolemy/users/$USER/.cargo"
export RUSTUP_HOME="/scratch/ptolemy/users/$USER/.rustup"
export UV_CACHE_DIR="/scratch/ptolemy/users/$USER/cache/uv"
export PIP_CACHE_DIR="/scratch/ptolemy/users/$USER/cache/pip"
export OMP_NUM_THREADS=1

# Create necessary directories
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $CARGO_HOME
mkdir -p $RUSTUP_HOME
mkdir -p $UV_CACHE_DIR
mkdir -p $PIP_CACHE_DIR

echo -e "\nNANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"

# Activate virtual environment
VENV_PATH="/scratch/ptolemy/users/$USER/nanochat-venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "\n❌ ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please run setup_environment.sh first:"
    echo "  bash scripts/setup_environment.sh"
    exit 1
fi

source $VENV_PATH/bin/activate

echo -e "\nPython: $(which python)"
echo "Python version: $(python --version)"

# Check Python version (need 3.10+)
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "\n❌ ERROR: Python $REQUIRED_VERSION or higher required"
    echo "Current Python version: $PYTHON_VERSION"
    echo ""
    echo "Please load Python 3.12 module:"
    echo "  module load python/3.12.5"
    echo "  Then recreate virtual environment and retry"
    exit 1
fi

# Check for required Python packages
echo -e "\nChecking required Python packages..."
MISSING_PACKAGES=()

python -c "import requests" 2>/dev/null || MISSING_PACKAGES+=("requests")
python -c "import tqdm" 2>/dev/null || MISSING_PACKAGES+=("tqdm")
python -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "\n❌ ERROR: Missing required Python packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Please install dependencies first:"
    echo "  pip install uv"
    echo "  pip install -e '.[gpu]'"
    echo ""
    echo "Then re-run this script"
    exit 1
fi

echo "✓ All required packages found"

# Test internet connectivity
echo -e "\n=================================================="
echo "Testing Internet Connectivity"
echo "=================================================="

if curl -s --connect-timeout 5 https://www.google.com > /dev/null 2>&1; then
    echo "✓ Internet connection verified"
else
    echo "❌ ERROR: No internet connection detected!"
    echo "This script requires internet access to download data."
    echo "Please ensure you are on ptolemy-devel-1 server."
    exit 1
fi

# Check if Rust/Cargo is installed
echo -e "\n=================================================="
echo "Checking Rust/Cargo Installation"
echo "=================================================="

if [ ! -f "$CARGO_HOME/bin/cargo" ]; then
    echo "Installing Rust and Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
else
    echo "✓ Rust/Cargo already installed"
fi

source "$CARGO_HOME/env"
echo "Cargo version: $(cargo --version)"

# Build the rustbpe tokenizer
echo -e "\n=================================================="
echo "Building rustbpe Tokenizer"
echo "=================================================="

if [ -d "rustbpe" ]; then
    echo "Building rustbpe tokenizer (this may take a few minutes)..."
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

    if [ $? -eq 0 ]; then
        echo "✓ rustbpe tokenizer built successfully"
    else
        echo "❌ ERROR: Failed to build rustbpe tokenizer"
        exit 1
    fi
else
    echo "❌ ERROR: rustbpe directory not found"
    echo "Please ensure you are in the nanochat directory"
    exit 1
fi

# Download dataset shards
echo -e "\n=================================================="
echo "Downloading Dataset Shards"
echo "=================================================="

echo "This will download ~24GB of training data."
echo "Download location: $NANOCHAT_BASE_DIR/data/"
echo ""

# Check if data already exists
DATA_DIR="$NANOCHAT_BASE_DIR/data"
if [ -d "$DATA_DIR" ]; then
    EXISTING_SHARDS=$(ls -1 $DATA_DIR/*.bin 2>/dev/null | wc -l)
    echo "Found $EXISTING_SHARDS existing data shards"

    if [ $EXISTING_SHARDS -ge 240 ]; then
        echo "✓ Sufficient data shards already downloaded"
        read -p "Re-download anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping dataset download"
            SKIP_DATASET=true
        fi
    fi
fi

if [ "$SKIP_DATASET" != "true" ]; then
    echo "Downloading 8 shards for tokenizer training (~800MB)..."
    python -m nanochat.dataset -n 8

    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to download initial dataset shards"
        exit 1
    fi

    echo -e "\n✓ Initial shards downloaded"
    echo ""
    echo "Downloading remaining 240 shards for pretraining (~24GB)..."
    echo "This will take 10-30 minutes depending on connection speed..."

    python -m nanochat.dataset -n 240

    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to download full dataset"
        exit 1
    fi

    echo "✓ All dataset shards downloaded successfully"
fi

# Train the tokenizer (requires data to be downloaded)
echo -e "\n=================================================="
echo "Training Tokenizer"
echo "=================================================="

TOKENIZER_PATH="$NANOCHAT_BASE_DIR/tokenizer"
if [ -f "$TOKENIZER_PATH/tokenizer_2pow16.model" ]; then
    echo "✓ Tokenizer already trained"
    read -p "Re-train anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping tokenizer training"
        SKIP_TOKENIZER=true
    fi
fi

if [ "$SKIP_TOKENIZER" != "true" ]; then
    echo "Training tokenizer on ~2B characters (this may take 10-20 minutes)..."
    python -m scripts.tok_train --max_chars=2000000000

    if [ $? -eq 0 ]; then
        echo "✓ Tokenizer trained successfully"
    else
        echo "❌ ERROR: Failed to train tokenizer"
        exit 1
    fi
fi

# Download evaluation bundle
echo -e "\n=================================================="
echo "Downloading Evaluation Bundle"
echo "=================================================="

EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
EVAL_BUNDLE_PATH="$NANOCHAT_BASE_DIR/eval_bundle"

if [ -d "$EVAL_BUNDLE_PATH" ]; then
    echo "✓ Evaluation bundle already downloaded"
else
    echo "Downloading eval_bundle (~162MB)..."
    cd $NANOCHAT_BASE_DIR
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL

    if [ $? -eq 0 ]; then
        echo "Extracting..."
        unzip -q eval_bundle.zip
        rm eval_bundle.zip
        echo "✓ Evaluation bundle downloaded and extracted"
    else
        echo "❌ ERROR: Failed to download evaluation bundle"
        exit 1
    fi
    cd - > /dev/null
fi

# Download synthetic identity conversations
echo -e "\n=================================================="
echo "Downloading Synthetic Identity Data"
echo "=================================================="

IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_FILE" ]; then
    echo "✓ Identity conversations already downloaded"
else
    echo "Downloading identity_conversations.jsonl (~2.3MB)..."
    curl -L -o $IDENTITY_FILE https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    if [ $? -eq 0 ]; then
        echo "✓ Identity conversations downloaded"
    else
        echo "❌ ERROR: Failed to download identity conversations"
        exit 1
    fi
fi

# Verify all required data is present
echo -e "\n=================================================="
echo "Verifying Downloaded Data"
echo "=================================================="

VERIFICATION_FAILED=false

# Check dataset shards (nanochat downloads to base_data directory as .parquet files)
SHARD_COUNT=$(ls -1 $NANOCHAT_BASE_DIR/base_data/*.parquet 2>/dev/null | wc -l)
if [ $SHARD_COUNT -ge 240 ]; then
    echo "✓ Dataset shards: $SHARD_COUNT/240"
else
    echo "❌ Dataset shards: $SHARD_COUNT/240 (insufficient)"
    VERIFICATION_FAILED=true
fi

# Check tokenizer (saved as tokenizer.pkl, not tokenizer_2pow16.model)
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "✓ Tokenizer trained"
else
    echo "❌ Tokenizer not found: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
    VERIFICATION_FAILED=true
fi

# Check eval bundle
if [ -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "✓ Evaluation bundle present"
else
    echo "❌ Evaluation bundle not found"
    VERIFICATION_FAILED=true
fi

# Check identity data
if [ -f "$IDENTITY_FILE" ]; then
    echo "✓ Identity conversations present"
else
    echo "❌ Identity conversations not found"
    VERIFICATION_FAILED=true
fi

# Final summary
echo -e "\n=================================================="
if [ "$VERIFICATION_FAILED" = true ]; then
    echo "❌ Data Download INCOMPLETE"
    echo "=================================================="
    echo ""
    echo "Some required data is missing. Please review errors above."
    echo "You can re-run this script to retry failed downloads."
    exit 1
else
    echo "✅ Data Download COMPLETE"
    echo "=================================================="
    echo ""
    echo "All required data has been downloaded and verified."
    echo ""
    echo "Storage usage:"
    du -sh $NANOCHAT_BASE_DIR
    echo ""
    echo "Next steps:"
    echo "1. Review the setup: cat PTOLEMY_SETUP.md"
    echo "2. Submit SLURM job: sbatch scripts/speedrun.slurm"
    echo "3. Monitor progress: squeue -u \$USER"
    echo ""
    echo "The GPU compute nodes will use the data from:"
    echo "$NANOCHAT_BASE_DIR"
fi
