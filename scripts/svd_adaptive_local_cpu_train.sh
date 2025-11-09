#!/bin/bash

# SVD Adaptive Training Script for Local CPU
# This script runs the adaptive SVD training with smart defaults for Phase 1 validation
#
# Usage:
#   bash scripts/svd_adaptive_local_cpu_train.sh
#   bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=3000
#   bash scripts/svd_adaptive_local_cpu_train.sh --depth=6 --num_iterations=2000 --wandb_run=my-run-id
#
# All parameters can be specified in any order and any combination.
# Unspecified parameters will use the defaults below.

# ============================================================================
# Default Configuration (Phase 1 Extended Validation)
# ============================================================================

# Model configuration
DEPTH=4                          # Tiny model for fast iteration
MAX_SEQ_LEN=512                  # Shorter context for CPU training
DEVICE_BATCH_SIZE=1              # CPU can only handle small batches
TOTAL_BATCH_SIZE=512             # Accumulate gradients to simulate larger batch

# Training configuration
NUM_ITERATIONS=2000              # 2000 steps for first mode switch observation (use 3000 for more data)
EVAL_EVERY=200                   # Evaluate every 200 steps
EVAL_TOKENS=512                  # Quick eval on CPU
CORE_METRIC_EVERY=-1             # Skip CORE metric (too slow on CPU)
SAMPLE_EVERY=200                 # Generate samples every 200 steps

# SVD specific configuration
SVD_INTERVAL=20                  # Run SVD analysis every 20 steps (default in training script)

# Device configuration
DEVICE_TYPE=cpu                  # Use CPU (MPS doesn't support SVD operations)

# WandB configuration
WANDB_RUN=""                     # Empty = auto-generate, or specify custom run ID

# Output configuration
LOG_FILE="svd_training_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

echo "=================================================="
echo "SVD Adaptive Training - Parameter Setup"
echo "=================================================="
echo ""

# Parse all command-line arguments
for arg in "$@"; do
    case $arg in
        --depth=*)
            DEPTH="${arg#*=}"
            shift
            ;;
        --max_seq_len=*)
            MAX_SEQ_LEN="${arg#*=}"
            shift
            ;;
        --device_batch_size=*)
            DEVICE_BATCH_SIZE="${arg#*=}"
            shift
            ;;
        --total_batch_size=*)
            TOTAL_BATCH_SIZE="${arg#*=}"
            shift
            ;;
        --num_iterations=*)
            NUM_ITERATIONS="${arg#*=}"
            shift
            ;;
        --eval_every=*)
            EVAL_EVERY="${arg#*=}"
            shift
            ;;
        --eval_tokens=*)
            EVAL_TOKENS="${arg#*=}"
            shift
            ;;
        --core_metric_every=*)
            CORE_METRIC_EVERY="${arg#*=}"
            shift
            ;;
        --sample_every=*)
            SAMPLE_EVERY="${arg#*=}"
            shift
            ;;
        --svd_interval=*)
            SVD_INTERVAL="${arg#*=}"
            shift
            ;;
        --device_type=*)
            DEVICE_TYPE="${arg#*=}"
            shift
            ;;
        --wandb_run=*)
            WANDB_RUN="${arg#*=}"
            shift
            ;;
        --log_file=*)
            LOG_FILE="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: bash scripts/svd_adaptive_local_cpu_train.sh [OPTIONS]"
            echo ""
            echo "Options (all optional, can be specified in any order):"
            echo "  --depth=N                  Model depth (default: 4)"
            echo "  --max_seq_len=N            Max sequence length (default: 512)"
            echo "  --device_batch_size=N      Batch size per device (default: 1)"
            echo "  --total_batch_size=N       Total effective batch size (default: 512)"
            echo "  --num_iterations=N         Training steps (default: 2000)"
            echo "  --eval_every=N             Eval interval (default: 200)"
            echo "  --eval_tokens=N            Tokens per eval (default: 512)"
            echo "  --core_metric_every=N      CORE metric interval (default: -1, disabled)"
            echo "  --sample_every=N           Sample generation interval (default: 200)"
            echo "  --svd_interval=N           SVD analysis interval (default: 20)"
            echo "  --device_type=TYPE         Device type: cpu/cuda/mps (default: cpu)"
            echo "  --wandb_run=ID             WandB run ID (default: auto-generated)"
            echo "  --log_file=PATH            Output log file (default: svd_training_TIMESTAMP.log)"
            echo ""
            echo "Examples:"
            echo "  # Use all defaults (2000 steps)"
            echo "  bash scripts/svd_adaptive_local_cpu_train.sh"
            echo ""
            echo "  # Run for 3000 steps to see more mode switches"
            echo "  bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=3000"
            echo ""
            echo "  # Larger model with custom WandB run"
            echo "  bash scripts/svd_adaptive_local_cpu_train.sh --depth=6 --num_iterations=2000 --wandb_run=phase1-extended"
            echo ""
            echo "  # More frequent SVD analysis"
            echo "  bash scripts/svd_adaptive_local_cpu_train.sh --svd_interval=10 --num_iterations=2000"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown parameter: $arg"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# ============================================================================
# Display Configuration
# ============================================================================

echo "Training Configuration:"
echo "----------------------"
echo "Model:"
echo "  depth:              $DEPTH"
echo "  max_seq_len:        $MAX_SEQ_LEN"
echo ""
echo "Training:"
echo "  num_iterations:     $NUM_ITERATIONS"
echo "  device_batch_size:  $DEVICE_BATCH_SIZE"
echo "  total_batch_size:   $TOTAL_BATCH_SIZE"
echo "  eval_every:         $EVAL_EVERY"
echo "  eval_tokens:        $EVAL_TOKENS"
echo "  core_metric_every:  $CORE_METRIC_EVERY"
echo "  sample_every:       $SAMPLE_EVERY"
echo ""
echo "SVD Adaptive:"
echo "  svd_interval:       $SVD_INTERVAL (SVD analysis every N steps)"
echo ""
echo "Device:"
echo "  device_type:        $DEVICE_TYPE"
echo ""
echo "Logging:"
echo "  wandb_run:          ${WANDB_RUN:-[auto-generated]}"
echo "  log_file:           $LOG_FILE"
echo ""

# ============================================================================
# Estimated Runtime
# ============================================================================

# Rough estimate: ~2.4 seconds per step on modern CPU for depth=4
ESTIMATED_MINUTES=$((NUM_ITERATIONS * 24 / 10 / 60))
if [ $ESTIMATED_MINUTES -lt 1 ]; then
    ESTIMATED_TIME="< 1 minute"
elif [ $ESTIMATED_MINUTES -lt 60 ]; then
    ESTIMATED_TIME="~$ESTIMATED_MINUTES minutes"
else
    ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))
    ESTIMATED_TIME="~$ESTIMATED_HOURS hours"
fi

echo "Estimated runtime: $ESTIMATED_TIME"
echo ""
echo "Expected behavior:"
echo "  Steps 0-200:   Initial training, no mode switches"
echo "  Steps 200-400: Patterns forming, metrics approaching thresholds"
echo "  Steps 400-800: Expected FIRST MODE SWITCH to low-rank"
echo "  Steps 800+:    Low-rank mode active, r decreasing adaptively"
echo ""

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "=================================================="
echo "Pre-flight Checks"
echo "=================================================="
echo ""

# Check if Python virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Python virtual environment not detected"
    echo "   Attempting to activate .venv..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "❌ ERROR: .venv not found. Please run setup first:"
        echo "   uv venv && uv sync --extra cpu"
        exit 1
    fi
else
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
fi

# Check if adaptive SVD training script exists
if [ ! -f "scripts/base_train_adaptive_svd.py" ]; then
    echo "❌ ERROR: scripts/base_train_adaptive_svd.py not found"
    echo "   Make sure you're in the nanochat project directory"
    exit 1
fi
echo "✓ Adaptive SVD training script found"

# Check if adaptive_svd module exists
if ! python -c "import nanochat.adaptive_svd" 2>/dev/null; then
    echo "❌ ERROR: nanochat.adaptive_svd module not found"
    echo "   Make sure the adaptive SVD implementation is in place"
    exit 1
fi
echo "✓ Adaptive SVD module available"

# Check if model checkpoint exists (needed for adaptive SVD training)
CHECKPOINT_DIR="$HOME/.cache/nanochat/models"
CHECKPOINT_FILE="$CHECKPOINT_DIR/d${DEPTH}_base.pt"
if [ ! -f "$CHECKPOINT_FILE" ] && [ $NUM_ITERATIONS -lt 100 ]; then
    echo "⚠️  Warning: No base model checkpoint found at $CHECKPOINT_FILE"
    echo "   The training will initialize from scratch"
fi

echo ""

# ============================================================================
# Training Confirmation
# ============================================================================

echo "=================================================="
echo "Ready to Start Training"
echo "=================================================="
echo ""

read -p "Press Enter to start training or Ctrl+C to cancel..."
echo ""

# ============================================================================
# Run Training
# ============================================================================

echo "=================================================="
echo "Starting SVD Adaptive Training"
echo "=================================================="
echo ""
echo "Training output will be saved to: $LOG_FILE"
echo "Monitor in real-time: tail -f $LOG_FILE"
echo ""
echo "To monitor SVD metrics in parallel, open a new terminal and run:"
echo "  bash scripts/svd_monitor.sh --wandb_run=<run_id>"
echo ""
echo "Starting training in 3 seconds..."
sleep 3
echo ""

# Build the command with all parameters
CMD="python -m scripts.base_train_adaptive_svd"
CMD="$CMD --depth=$DEPTH"
CMD="$CMD --max_seq_len=$MAX_SEQ_LEN"
CMD="$CMD --device_batch_size=$DEVICE_BATCH_SIZE"
CMD="$CMD --total_batch_size=$TOTAL_BATCH_SIZE"
CMD="$CMD --num_iterations=$NUM_ITERATIONS"
CMD="$CMD --eval_every=$EVAL_EVERY"
CMD="$CMD --eval_tokens=$EVAL_TOKENS"
CMD="$CMD --core_metric_every=$CORE_METRIC_EVERY"
CMD="$CMD --sample_every=$SAMPLE_EVERY"
CMD="$CMD --device_type=$DEVICE_TYPE"

# Add wandb run if specified
if [ -n "$WANDB_RUN" ]; then
    CMD="$CMD --run=$WANDB_RUN"
fi

# Display command
echo "Executing command:"
echo "$CMD"
echo ""
echo "=================================================="
echo ""

# Run training and tee output to log file
$CMD 2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Training Complete
# ============================================================================

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training Completed Successfully"
else
    echo "❌ Training Failed (exit code: $EXIT_CODE)"
fi
echo "=================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training Summary:"
    echo "  Steps completed:    $NUM_ITERATIONS"
    echo "  Model depth:        $DEPTH"
    echo "  Log file:           $LOG_FILE"
    echo ""
    echo "Next Steps:"
    echo ""
    echo "1. Review training log:"
    echo "   cat $LOG_FILE"
    echo ""
    echo "2. Search for mode switches:"
    echo "   grep 'SWITCHED TO' $LOG_FILE"
    echo ""
    echo "3. Check WandB dashboard for metrics:"
    echo "   https://wandb.ai/"
    echo "   Look for: svd/block0.attn.c_q/*"
    echo ""
    echo "4. Analyze SVD metrics:"
    echo "   grep 'SVD Analysis' $LOG_FILE"
    echo ""
    echo "5. View final metrics summary:"
    echo "   tail -100 $LOG_FILE"
    echo ""
    echo "Documentation:"
    echo "  experiments/BASE_TRAIN_SVD_ENHANCEMENT_STATUS.md"
    echo "  experiments/SVD_MONITORING_GUIDE.md"
    echo ""
else
    echo "Training failed. Check the log for errors:"
    echo "  tail -50 $LOG_FILE"
    echo ""
fi

exit $EXIT_CODE
