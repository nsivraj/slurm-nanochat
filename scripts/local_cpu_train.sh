#!/bin/bash

# Local CPU Training Script for nanochat
# This script runs the complete nanochat training pipeline on a CPU with minimal data
# Designed for learning and understanding the code, not for production models
#
# Usage: bash scripts/local_cpu_train.sh
#
# Expected runtime: 1-3 hours on a modern CPU
# Expected disk usage: ~2GB (data + model artifacts)
# Expected RAM usage: ~8GB minimum (16GB recommended)

echo "=================================================="
echo "nanochat Local CPU Training"
echo "=================================================="
echo ""
echo "This script will train a tiny nanochat model on your CPU."
echo "Training will take approximately 1-3 hours."
echo ""
echo "What you'll get:"
echo "  - Complete training pipeline experience"
echo "  - Understanding of all training phases"
echo "  - A working (but tiny and not very smart) chatbot"
echo ""
echo "What you won't get:"
echo "  - A production-quality model"
echo "  - Good benchmark scores"
echo "  - Fast inference"
echo ""
echo "This is for LEARNING, not for production!"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

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

# Build the rustbpe tokenizer
echo ""
echo "Building rustbpe tokenizer (this may take a few minutes)..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

if [ $? -eq 0 ]; then
    echo "✓ rustbpe tokenizer built successfully"
else
    echo "❌ ERROR: Failed to build rustbpe tokenizer"
    exit 1
fi

# -----------------------------------------------------------------------------
# wandb Setup (disabled by default for local training)

if [ -z "$WANDB_RUN" ]; then
    export WANDB_RUN=dummy  # Skip wandb logging
fi

# -----------------------------------------------------------------------------
# Initialize Report

echo ""
echo "=================================================="
echo "Initializing training report..."
echo "=================================================="

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Phase 1: Tokenizer Training

echo ""
echo "=================================================="
echo "[1/5] Tokenizer Training and Evaluation"
echo "=================================================="
echo ""
echo "Downloading minimal training data (4 shards, ~400MB)..."

# Download 4 data shards (~400MB) for tokenizer training
# Each shard is ~250M characters, so 4 shards = ~1B characters
python -m nanochat.dataset -n 4

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to download dataset"
    exit 1
fi

echo ""
echo "✓ Data downloaded successfully"
echo ""
echo "Training tokenizer on ~1B characters..."
echo "This will take approximately 10-15 minutes..."

# Train tokenizer on 1B characters (vs 2B in production)
python -m scripts.tok_train --max_chars=1000000000

if [ $? -eq 0 ]; then
    echo "✓ Tokenizer trained successfully"
else
    echo "❌ ERROR: Failed to train tokenizer"
    exit 1
fi

echo ""
echo "Evaluating tokenizer compression ratio..."
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Phase 2: Base Model Pretraining

echo ""
echo "=================================================="
echo "[2/5] Base Model Pretraining"
echo "=================================================="
echo ""
echo "Training tiny d4 model (4 layers, ~8M parameters)..."
echo "This will take approximately 30-60 minutes..."
echo ""
echo "Parameters:"
echo "  - Depth: 4 layers (vs 20 for production)"
echo "  - Context: 1024 tokens (vs 2048 for production)"
echo "  - Batch size: 1 sequence per step"
echo "  - Iterations: 50 (vs ~5000 for production)"
echo ""

# Download eval bundle for CORE metric
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "Downloading evaluation bundle (~162MB)..."
    cd $NANOCHAT_BASE_DIR
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    echo "✓ Evaluation bundle downloaded"
    cd - > /dev/null
else
    echo "✓ Evaluation bundle already present"
fi

# Train base model
# Note: No torchrun needed for single CPU
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50 \
    --run=$WANDB_RUN

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Base model training failed"
    exit 1
fi

echo ""
echo "Evaluating base model..."

# Evaluate base model on validation data
python -m scripts.base_loss \
    --device_batch_size=1 \
    --split_tokens=4096

# Evaluate base model on CORE benchmark
python -m scripts.base_eval \
    --max-per-task=16

echo "✓ Base model training completed"

# -----------------------------------------------------------------------------
# Phase 3: Midtraining

echo ""
echo "=================================================="
echo "[3/5] Midtraining (Conversation Format)"
echo "=================================================="
echo ""
echo "Teaching model conversation format and special tokens..."
echo "This will take approximately 15-30 minutes..."
echo ""

# Download identity conversations if not present
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ]; then
    echo "Downloading synthetic identity conversations (~2.3MB)..."
    curl -L -o $IDENTITY_FILE https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    echo "✓ Identity conversations downloaded"
else
    echo "✓ Identity conversations already present"
fi

# Run midtraining
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100 \
    --run=$WANDB_RUN

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Midtraining failed"
    exit 1
fi

echo ""
echo "Evaluating midtrained model..."

# Evaluate midtraining results
# Note: Results will be poor on this tiny model, but we're just exercising the code paths
python -m scripts.chat_eval \
    --source=mid \
    --max-new-tokens=128 \
    --max-problems=20

echo "✓ Midtraining completed"

# -----------------------------------------------------------------------------
# Phase 4: Supervised Finetuning

echo ""
echo "=================================================="
echo "[4/5] Supervised Finetuning (SFT)"
echo "=================================================="
echo ""
echo "Fine-tuning on instruction-following data..."
echo "This will take approximately 15-30 minutes..."
echo ""

# Run supervised finetuning
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16 \
    --run=$WANDB_RUN

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Supervised finetuning failed"
    exit 1
fi

echo ""
echo "Evaluating finetuned model..."

# Evaluate SFT results
python -m scripts.chat_eval \
    --source=sft \
    --max-new-tokens=128 \
    --max-problems=20

echo "✓ Supervised finetuning completed"

# -----------------------------------------------------------------------------
# Phase 5: Report Generation

echo ""
echo "=================================================="
echo "[5/5] Generating Training Report"
echo "=================================================="

python -m nanochat.report generate

# Copy report to current directory for easy access
if [ -f "$NANOCHAT_BASE_DIR/report/report.md" ]; then
    cp $NANOCHAT_BASE_DIR/report/report.md ./report.md
    echo "✓ Report saved to ./report.md"
fi

# -----------------------------------------------------------------------------
# Training Complete

echo ""
echo "=================================================="
echo "✅ Local CPU Training Complete!"
echo "=================================================="
echo ""
echo "Training Summary:"
echo "  - Model: 4-layer Transformer (~8M parameters)"
echo "  - Training data: ~1B characters (4 shards)"
echo "  - Training time: Check report for exact timing"
echo ""
echo "What was trained:"
echo "  ✓ Tokenizer (BPE, 65536 vocab)"
echo "  ✓ Base model (pretraining)"
echo "  ✓ Midtraining (conversation format)"
echo "  ✓ Supervised finetuning"
echo ""
echo "Model artifacts saved to:"
echo "  $NANOCHAT_BASE_DIR"
echo ""
echo "Training report:"
echo "  ./report.md"
echo ""
echo "Next steps:"
echo ""
echo "1. Review the training report:"
echo "   cat report.md"
echo ""
echo "2. Chat with your model via CLI:"
echo "   source .venv/bin/activate"
echo "   python -m scripts.chat_cli -p 'Why is the sky blue?'"
echo ""
echo "3. Or use interactive mode:"
echo "   python -m scripts.chat_cli"
echo ""
echo "4. Or start the web interface:"
echo "   python -m scripts.chat_web"
echo "   # Then open http://localhost:8000 in your browser"
echo ""
echo "5. Analyze the code:"
echo "   - Read scripts/base_train.py to understand pretraining"
echo "   - Read scripts/mid_train.py to understand midtraining"
echo "   - Read scripts/chat_sft.py to understand finetuning"
echo "   - Read nanochat/gpt.py to see the Transformer architecture"
echo ""
echo "6. Read the documentation:"
echo "   cat LOCAL_CPU_TRAINING.md"
echo ""
echo "Important notes:"
echo "  - This is a TINY model for learning, not production"
echo "  - Expect poor performance and many mistakes"
echo "  - The goal is understanding the training pipeline"
echo "  - For production models, use GPU training (speedrun.sh)"
echo ""
echo "=================================================="
echo "Happy learning! 🚀"
echo "=================================================="
