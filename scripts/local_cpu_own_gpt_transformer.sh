#!/bin/bash

# Quick Iteration Script for Learning GPT Transformer Implementation
# This script runs a single training iteration to test your normans_gpt.py implementation
#
# Usage: bash scripts/local_cpu_own_gpt_transformer.sh
#
# Expected runtime: 10-30 seconds per iteration
# This is for rapid iteration while implementing nanochat/normans_gpt.py

echo "=================================================="
echo "GPT Transformer Learning - Quick Iteration"
echo "=================================================="
echo ""
echo "This script runs ONE training step to test your"
echo "normans_gpt.py implementation."
echo ""
echo "First-time setup (automatic):"
echo "  - Downloads training data if needed (~400MB)"
echo "  - Trains tokenizer if needed (~10-15 min)"
echo ""
echo "Learning workflow:"
echo "  1. Run this script"
echo "  2. See what NotImplementedError you hit"
echo "  3. Implement that method in nanochat/normans_gpt.py"
echo "  4. Run this script again"
echo "  5. Repeat until training works!"
echo ""

# -----------------------------------------------------------------------------
# Environment Setup

echo ""
echo "=================================================="
echo "Setting up environment..."
echo "=================================================="

# Set environment variables
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python Virtual Environment Setup with uv

echo ""
echo "Setting up Python virtual environment..."

# Install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session (uv installs to ~/.local/bin)
    export PATH="$HOME/.local/bin:$PATH"

    # Also check if it was installed elsewhere
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
else
    echo "✓ uv already installed"
fi

# Ensure uv is in PATH (in case it was already installed)
if [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment (if it doesn't exist)
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    uv venv
else
    echo "✓ Virtual environment already exists"
fi

# Install dependencies for CPU
echo "Installing Python dependencies (CPU version)..."
uv sync --extra cpu

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# -----------------------------------------------------------------------------
# Rust and Cargo Setup

echo ""
echo "=================================================="
echo "Setting up Rust and Cargo..."
echo "=================================================="

# Install Rust/Cargo if not already installed
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust and Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    source "$HOME/.cargo/env"
else
    echo "✓ Rust/Cargo already installed"
    source "$HOME/.cargo/env"
fi

echo "Cargo version: $(cargo --version)"
echo "Rust version: $(rustc --version)"

# Build the rustbpe tokenizer (only if not already built)
if [ ! -f ".venv/lib/python3.*/site-packages/rustbpe.*.so" ]; then
    echo ""
    echo "Building rustbpe tokenizer (this may take a few minutes)..."
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

    if [ $? -eq 0 ]; then
        echo "✓ rustbpe tokenizer built successfully"
    else
        echo "❌ ERROR: Failed to build rustbpe tokenizer"
        exit 1
    fi
else
    echo "✓ rustbpe tokenizer already built"
fi

# -----------------------------------------------------------------------------
# Data Setup

echo ""
echo "=================================================="
echo "Checking for required data..."
echo "=================================================="

# Check if training data exists (needed for tokenizer training)
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/*.parquet 2>/dev/null)" ]; then
    echo "⚠️  Training data not found"
    echo ""
    echo "Downloading 4 data shards (~400MB)..."
    echo "This will take a few minutes depending on your connection."
    echo ""

    python -m nanochat.dataset -n 4

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Training data downloaded successfully"
    else
        echo ""
        echo "❌ ERROR: Failed to download training data"
        exit 1
    fi
else
    echo "✓ Training data found"
fi

# Check if tokenizer exists
TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
if [ ! -f "$TOKENIZER_FILE" ]; then
    echo ""
    echo "⚠️  Tokenizer not found"
    echo ""
    echo "Training tokenizer on 1B characters..."
    echo "This will take approximately 10-15 minutes."
    echo ""
    echo "What's happening:"
    echo "  1. Loading training data from $DATA_DIR"
    echo "  2. Training BPE tokenizer (vocab size: 65536)"
    echo "  3. Saving to $TOKENIZER_FILE"
    echo ""

    python -m scripts.tok_train --max_chars=1000000000

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Tokenizer trained successfully"
        echo ""
        echo "Evaluating tokenizer compression ratio..."
        python -m scripts.tok_eval
    else
        echo ""
        echo "❌ ERROR: Failed to train tokenizer"
        exit 1
    fi
else
    echo "✓ Tokenizer found"
fi

# Download eval bundle if needed for CORE metric (skip if we're not evaluating)
EVAL_BUNDLE_DIR="$NANOCHAT_BASE_DIR/eval_bundle"
if [ ! -d "$EVAL_BUNDLE_DIR" ]; then
    echo ""
    echo "Note: CORE evaluation disabled (eval bundle not present)"
    echo "This is fine for quick iteration testing."
else
    echo "✓ Evaluation bundle found"
fi

# -----------------------------------------------------------------------------
# wandb Setup (disabled for local iteration)

export WANDB_RUN=dummy  # Skip wandb logging for quick iteration

# -----------------------------------------------------------------------------
# Run Single Training Iteration

echo ""
echo "=================================================="
echo "Running ONE Training Iteration"
echo "=================================================="
echo ""
echo "Command:"
echo "  python -m scripts.base_train \\"
echo "    --depth=4 \\"
echo "    --max_seq_len=512 \\"
echo "    --device_batch_size=1 \\"
echo "    --eval_tokens=512 \\"
echo "    --core_metric_every=-1 \\"
echo "    --total_batch_size=512 \\"
echo "    --num_iterations=1"
echo ""
echo "Watch for [DEBUG] output and NotImplementedError!"
echo ""
echo "=================================================="
echo ""

# Run the training
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --eval_tokens=512 \
    --core_metric_every=-1 \
    --total_batch_size=512 \
    --num_iterations=1

TRAIN_EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Report Results

echo ""
echo "=================================================="
echo "Iteration Complete"
echo "=================================================="
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! Training iteration completed without errors!"
    echo ""
    echo "This means your implementation got past this step."
    echo "Keep going - implement more methods!"
    echo ""
    echo "Next steps:"
    echo "  1. Check if loss decreased (look for 'loss' in output above)"
    echo "  2. Run again to test the next iteration"
    echo "  3. Once several iterations work, try more steps:"
    echo "     python -m scripts.base_train --depth=4 --max_seq_len=512 \\"
    echo "       --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 \\"
    echo "       --total_batch_size=512 --num_iterations=10"
else
    echo "❌ Training failed (exit code: $TRAIN_EXIT_CODE)"
    echo ""
    echo "Look for the error message above."
    echo ""
    echo "If you see 'NotImplementedError':"
    echo "  1. Read the error message - it tells you what to implement"
    echo "  2. Look at the [DEBUG] output to understand the context"
    echo "  3. Open nanochat/normans_gpt.py"
    echo "  4. Implement the method mentioned in the error"
    echo "  5. Run this script again"
    echo ""
    echo "If you see a different error:"
    echo "  1. Read the error message and traceback"
    echo "  2. Check for shape mismatches, device issues, etc."
    echo "  3. Compare your implementation with nanochat/gpt.py"
    echo "  4. Add debug prints to understand what's happening"
    echo "  5. Fix the issue and run again"
    echo ""
    echo "For help, see:"
    echo "  - docs/how-to/03-how-to-write-your-own-gpt-transformer.md"
    echo "  - experiments/NORMANS_GPT_LEARNING_STATUS.md"
fi

echo ""
echo "=================================================="
echo ""

exit $TRAIN_EXIT_CODE
