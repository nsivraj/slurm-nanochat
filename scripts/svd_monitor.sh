#!/bin/bash

# SVD Metrics Monitoring Script
# This script monitors adaptive SVD training metrics in real-time
# and provides alerts when mode switches are imminent
#
# Usage:
#   bash scripts/svd_monitor.sh --wandb_run=abc123def456
#   bash scripts/svd_monitor.sh --wandb_run=abc123def456 --refresh_interval=5
#   bash scripts/svd_monitor.sh --wandb_run=abc123def456 --warn_margin=0.08 --principal_danger=0.35
#
# All parameters can be specified in any order and any combination.

# ============================================================================
# Default Configuration
# ============================================================================

# WandB configuration
WANDB_RUN=""                     # REQUIRED: WandB run ID to monitor
WANDB_PROJECT="nanochat"         # WandB project name
WANDB_ENTITY=""                  # WandB entity (username/team), empty = use default

# Monitoring configuration
REFRESH_INTERVAL=10              # Check for new metrics every N seconds
FOLLOW=true                      # Follow mode (continuous monitoring)

# Alert thresholds (match training defaults)
WARN_MARGIN=0.05                 # Alert when within this distance of threshold
PRINCIPAL_DANGER=0.4             # Clobbering threshold (alert if principal_alignment > this)
MINOR_SAFE=0.6                   # Safety threshold (good if minor_alignment > this)
ANGLE_STABLE=0.1                 # Stable subspace threshold (good if subspace_angle < this)
RECONSTRUCTION_SAFE=0.01         # Safe reconstruction threshold (good if error < this)

# Output configuration
QUIET=false                      # Suppress non-alert output
SUMMARY_ONLY=false               # Only show summary at end

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

echo "=================================================="
echo "SVD Metrics Monitor - Setup"
echo "=================================================="
echo ""

# Parse all command-line arguments
for arg in "$@"; do
    case $arg in
        --wandb_run=*)
            WANDB_RUN="${arg#*=}"
            shift
            ;;
        --project=*)
            WANDB_PROJECT="${arg#*=}"
            shift
            ;;
        --entity=*)
            WANDB_ENTITY="${arg#*=}"
            shift
            ;;
        --refresh_interval=*)
            REFRESH_INTERVAL="${arg#*=}"
            shift
            ;;
        --warn_margin=*)
            WARN_MARGIN="${arg#*=}"
            shift
            ;;
        --principal_danger=*)
            PRINCIPAL_DANGER="${arg#*=}"
            shift
            ;;
        --minor_safe=*)
            MINOR_SAFE="${arg#*=}"
            shift
            ;;
        --angle_stable=*)
            ANGLE_STABLE="${arg#*=}"
            shift
            ;;
        --reconstruction_safe=*)
            RECONSTRUCTION_SAFE="${arg#*=}"
            shift
            ;;
        --no-follow)
            FOLLOW=false
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --summary-only)
            SUMMARY_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash scripts/svd_monitor.sh --wandb_run=RUN_ID [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --wandb_run=ID             WandB run ID to monitor (REQUIRED)"
            echo ""
            echo "Optional WandB settings:"
            echo "  --project=NAME             WandB project name (default: nanochat)"
            echo "  --entity=NAME              WandB entity/username (default: use WandB default)"
            echo ""
            echo "Monitoring settings:"
            echo "  --refresh_interval=N       Seconds between checks (default: 10)"
            echo "  --no-follow                Single check instead of continuous monitoring"
            echo "  --quiet                    Suppress non-alert output"
            echo "  --summary-only             Only show summary at end"
            echo ""
            echo "Alert threshold settings:"
            echo "  --warn_margin=X            Alert margin from thresholds (default: 0.05)"
            echo "  --principal_danger=X       Clobbering threshold (default: 0.4)"
            echo "  --minor_safe=X             Safety threshold (default: 0.6)"
            echo "  --angle_stable=X           Stable subspace threshold (default: 0.1)"
            echo "  --reconstruction_safe=X    Reconstruction error threshold (default: 0.01)"
            echo ""
            echo "Examples:"
            echo "  # Basic monitoring (get run ID from WandB dashboard)"
            echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456"
            echo ""
            echo "  # Faster refresh rate"
            echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456 --refresh_interval=5"
            echo ""
            echo "  # More sensitive alerts"
            echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456 --warn_margin=0.08"
            echo ""
            echo "  # Custom thresholds (earlier clobbering detection)"
            echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456 --principal_danger=0.35"
            echo ""
            echo "  # Single check (no continuous monitoring)"
            echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456 --no-follow"
            echo ""
            echo "  # Quiet mode (only show alerts)"
            echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456 --quiet"
            echo ""
            echo "Finding your WandB run ID:"
            echo "  1. Go to https://wandb.ai/"
            echo "  2. Open your project (e.g., 'nanochat')"
            echo "  3. Click on the training run"
            echo "  4. Copy the run ID from the URL:"
            echo "     https://wandb.ai/<entity>/<project>/runs/<RUN_ID>"
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
# Validate Configuration
# ============================================================================

# Check if wandb_run is specified
if [ -z "$WANDB_RUN" ]; then
    echo "❌ ERROR: --wandb_run parameter is required"
    echo ""
    echo "Usage: bash scripts/svd_monitor.sh --wandb_run=<run_id>"
    echo ""
    echo "To find your run ID:"
    echo "  1. Go to https://wandb.ai/"
    echo "  2. Open your project"
    echo "  3. Click on the training run"
    echo "  4. Copy the run ID from the URL"
    echo ""
    echo "Example:"
    echo "  bash scripts/svd_monitor.sh --wandb_run=abc123def456"
    echo ""
    echo "Use --help for more options"
    exit 1
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "Monitoring Configuration:"
echo "------------------------"
echo "WandB:"
echo "  run_id:                $WANDB_RUN"
echo "  project:               $WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then
    echo "  entity:                $WANDB_ENTITY"
fi
echo ""
echo "Monitoring:"
echo "  refresh_interval:      ${REFRESH_INTERVAL}s"
echo "  follow_mode:           $FOLLOW"
echo "  quiet:                 $QUIET"
echo "  summary_only:          $SUMMARY_ONLY"
echo ""
echo "Alert Thresholds:"
echo "  warn_margin:           ±$WARN_MARGIN"
echo "  principal_danger:      >$PRINCIPAL_DANGER (clobbering risk)"
echo "  minor_safe:            >$MINOR_SAFE (safe gradients)"
echo "  angle_stable:          <$ANGLE_STABLE (stable subspace)"
echo "  reconstruction_safe:   <$RECONSTRUCTION_SAFE (safe compression)"
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

# Check if monitoring script exists
if [ ! -f "scripts/monitor_svd_metrics.py" ]; then
    echo "❌ ERROR: scripts/monitor_svd_metrics.py not found"
    echo "   Make sure you're in the nanochat project directory"
    exit 1
fi
echo "✓ Monitoring script found"

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "❌ ERROR: wandb package not installed"
    echo "   Install with: pip install wandb"
    exit 1
fi
echo "✓ WandB package available"

# Check WandB login status
if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo "⚠️  Warning: Not logged into WandB"
    echo "   If monitoring fails, run: wandb login"
else
    echo "✓ WandB authentication verified"
fi

echo ""

# ============================================================================
# Start Monitoring
# ============================================================================

echo "=================================================="
echo "Starting SVD Metrics Monitoring"
echo "=================================================="
echo ""
echo "Monitoring WandB run: $WANDB_RUN"
echo "Project: $WANDB_PROJECT"
echo ""
echo "What to expect:"
echo "  🟢 Steps 0-200:   Quiet (no alerts)"
echo "  🟡 Steps 200-400: INFO alerts (trends detected)"
echo "  🟡 Steps 400-600: WARNING alerts (approaching thresholds)"
echo "  🔴 Steps 600-800: CRITICAL alerts (switch imminent)"
echo "  🟢 Steps 800+:    Mode switches completed"
echo ""
echo "Press Ctrl+C to stop monitoring and show summary"
echo ""
echo "=================================================="
echo ""

# Build the monitoring command
CMD="python scripts/monitor_svd_metrics.py"
CMD="$CMD --wandb-run $WANDB_RUN"
CMD="$CMD --project $WANDB_PROJECT"

# Add optional entity
if [ -n "$WANDB_ENTITY" ]; then
    CMD="$CMD --entity $WANDB_ENTITY"
fi

# Add monitoring settings
if [ "$FOLLOW" = true ]; then
    CMD="$CMD --refresh-interval $REFRESH_INTERVAL"
else
    CMD="$CMD --no-follow"
fi

# Add threshold settings
CMD="$CMD --warn-margin $WARN_MARGIN"
CMD="$CMD --principal-danger $PRINCIPAL_DANGER"
CMD="$CMD --minor-safe $MINOR_SAFE"
CMD="$CMD --angle-stable $ANGLE_STABLE"
CMD="$CMD --reconstruction-safe $RECONSTRUCTION_SAFE"

# Add output settings
if [ "$QUIET" = true ]; then
    CMD="$CMD --quiet"
fi

if [ "$SUMMARY_ONLY" = true ]; then
    CMD="$CMD --summary-only"
fi

# Display command (for debugging)
if [ "$QUIET" = false ]; then
    echo "Executing:"
    echo "$CMD"
    echo ""
    echo "=================================================="
    echo ""
fi

# Run the monitoring script
$CMD

# Capture exit code
EXIT_CODE=$?

# ============================================================================
# Monitoring Complete
# ============================================================================

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Monitoring Completed"
else
    echo "⚠️  Monitoring Stopped (exit code: $EXIT_CODE)"
fi
echo "=================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Summary displayed above."
    echo ""
    echo "To restart monitoring:"
    echo "  bash scripts/svd_monitor.sh --wandb_run=$WANDB_RUN"
    echo ""
elif [ $EXIT_CODE -eq 130 ]; then
    # Exit code 130 = Ctrl+C (user interrupt)
    echo "Monitoring interrupted by user (Ctrl+C)"
    echo ""
else
    echo "Monitoring failed. Possible causes:"
    echo "  - Invalid WandB run ID"
    echo "  - Network connection issue"
    echo "  - WandB authentication expired (run: wandb login)"
    echo "  - Run not found in project '$WANDB_PROJECT'"
    echo ""
    echo "Check the error message above for details."
    echo ""
fi

exit $EXIT_CODE
