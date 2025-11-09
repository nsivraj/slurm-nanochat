# SVD Metrics Monitoring Guide

This guide explains how to use the `scripts/monitor_svd_metrics.py` script to track adaptive SVD training in real-time and receive alerts when mode switches are imminent.

---

## Overview

The monitoring script provides:

1. **Real-time alerts** when metrics approach switch thresholds
2. **Trend analysis** to predict when switches will occur
3. **Visual indicators** of training progress toward low-rank mode
4. **Summary statistics** across all monitored layers

### ⚠️ Prerequisites

**WandB Authentication Required**: This script requires WandB to be configured with authentication:
```bash
wandb login
```

**Alternative for local CPU training**: If WandB authentication is not available:
- Run training with default settings (dummy mode, no `--wandb_run` parameter)
- All SVD metrics are still logged to the training output
- Analyze results from log files after training completes
- See `experiments/SVD_ENHANCEMENT_PHASE1_RUN_INSTRUCTIONS.md` for log analysis

**For Ptolemy cluster**: Use `WANDB_MODE=offline` to save metrics locally without internet connection

### Alert Levels

- 🚨 **CRITICAL**: Immediate action needed (switch happening now or very soon)
- ⚠️ **WARNING**: Approaching threshold (within 0.05 of trigger)
- ℹ️ **INFO**: Useful trends or observations

---

## Installation

Make the script executable:

```bash
chmod +x scripts/monitor_svd_metrics.py
```

Required package for WandB monitoring:

```bash
pip install wandb
```

---

## Usage

### Method 1: Monitor WandB Run (Recommended)

**Basic usage:**

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run <run_id> \
  --project nanochat
```

**With custom refresh interval:**

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run abc123def456 \
  --project nanochat \
  --refresh-interval 5  # Check every 5 seconds
```

**With custom thresholds:**

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run abc123def456 \
  --warn-margin 0.08 \           # Alert when within 0.08 of threshold
  --principal-danger 0.35 \      # Lower clobbering threshold
  --minor-safe 0.65              # Higher safety threshold
```

**Finding your WandB run ID:**

1. Go to your WandB dashboard: https://wandb.ai/
2. Navigate to your project (e.g., "nanochat")
3. Click on the training run
4. The run ID is in the URL: `https://wandb.ai/<entity>/<project>/runs/<run_id>`

### Method 2: Monitor Log File

**Follow log file in real-time:**

```bash
python scripts/monitor_svd_metrics.py \
  --log-file training.log \
  --follow
```

**Parse existing log file:**

```bash
python scripts/monitor_svd_metrics.py \
  --log-file training.log
```

*Note*: Log file parsing is currently a stub. You'll need to adapt the parser to your log format.

---

## What the Monitor Tracks

### Decision Thresholds

The monitor checks against the same thresholds used in the training code:

**Immediate Clobbering Trigger:**
- `principal_alignment > 0.4` → Switch to low-rank NOW

**Optimal Conditions Trigger (all must be true):**
- `subspace_angle < 0.1` (stable subspace)
- `minor_alignment > 0.6` (safe gradients)
- `reconstruction_error < 0.01` (safe compression)

### Example Output

```
================================================================================
Step 450 | block0.attn.c_q
--------------------------------------------------------------------------------
⚠️  WARNING: Approaching optimal conditions: ✓ Safe reconstruction,
           subspace_angle=0.12 (within 0.02 of 0.1),
           minor_alignment=0.58 (within 0.02 of 0.6)
ℹ️  INFO: Subspace stabilizing: 0.121 → ~15 steps to threshold

Current metrics:
  principal_alignment: 0.3200 (danger at >0.4)
  minor_alignment:     0.5800 (safe at >0.6)
  subspace_angle:      0.1210 (stable at <0.1)
  reconstruction_error:0.0045 (safe at <0.01)
  mode:                FULL
  r:                   246
================================================================================
```

### Trend Analysis

The monitor tracks recent history (last 5-20 steps) to detect:

1. **Stabilizing subspace**: Consistent decrease in `subspace_angle`
2. **Safer gradients**: Consistent increase in `minor_alignment`
3. **Time to threshold**: Estimates steps remaining until switch

Example trend alert:

```
ℹ️  INFO: Gradients becoming safer: 0.580 → ~12 steps to threshold
```

---

## Typical Workflow

### Phase 2 Training with Monitoring

**Terminal 1 - Start training:**

```bash
python -m scripts.base_train_adaptive_svd \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --total_batch_size=512 \
  --num_iterations=2000 \
  --device_type=cuda \
  2>&1 | tee training.log
```

**Terminal 2 - Monitor in real-time:**

```bash
# Wait ~30 seconds for WandB run to initialize
# Then get the run ID from WandB dashboard

python scripts/monitor_svd_metrics.py \
  --wandb-run <your_run_id> \
  --project nanochat \
  --refresh-interval 10
```

**What to expect:**

```
Steps 0-200:   No alerts (early training, unstable)

Steps 200-400: INFO alerts about trends
               "Subspace stabilizing: 0.180 → ~80 steps to threshold"

Steps 400-600: WARNING alerts
               "Approaching optimal conditions: 2/3 criteria met"

Steps 600-800: CRITICAL alert
               "OPTIMAL CONDITIONS MET! All criteria satisfied → SWITCH IMMINENT"

Step 650:      Mode switch observed!
               "[block0.attn.c_q] SWITCHED TO LOW-RANK (r=220)"
```

---

## Summary Report

Press `Ctrl+C` to stop monitoring and show summary:

```
================================================================================
SVD MONITORING SUMMARY
================================================================================

block0.attn.c_q
  Latest step: 800
  Mode: LOW-RANK
  Rank: 195
  Metrics:
    principal_alignment: 0.3500
    minor_alignment:     0.6800
    subspace_angle:      0.0650
    reconstruction_error:0.0032
  Distance to switch:
    Clobbering threshold: +0.0500 🟢 SAFE
    Optimal conditions: 3/3 met
      Stable subspace:  ✓ (angle=0.0650 vs <0.1)
      Safe gradients:   ✓ (minor=0.6800 vs >0.6)
      Safe reconstruction: ✓ (error=0.0032 vs <0.01)
================================================================================
```

---

## Customizing Thresholds

If you want more/fewer alerts, adjust the warning margin:

**More sensitive (more alerts):**

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run <run_id> \
  --warn-margin 0.10  # Alert when within 0.10 of threshold
```

**Less sensitive (fewer alerts):**

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run <run_id> \
  --warn-margin 0.02  # Only alert when very close
```

**Custom decision thresholds:**

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run <run_id> \
  --principal-danger 0.35 \  # Trigger clobbering protection earlier
  --minor-safe 0.65 \        # Require higher gradient safety
  --angle-stable 0.08        # Require more stable subspace
```

---

## Integration with Phase 2/3

### Phase 2 (All Attention Layers)

Monitor all layers simultaneously:

```bash
python scripts/monitor_svd_metrics.py \
  --wandb-run <run_id> \
  --project nanochat
```

You'll see alerts for:
- `block0.attn.c_q`
- `block0.attn.c_k`
- `block0.attn.c_v`
- `block0.attn.c_proj`
- `block1.attn.c_q`
- ... (all layers)

Different layers may switch at different times depending on their learning dynamics.

### Phase 3 (Production Scale)

On Ptolemy cluster, monitor from your local machine:

```bash
# Make sure you're logged into WandB
wandb login

# Monitor the remote run
python scripts/monitor_svd_metrics.py \
  --wandb-run <ptolemy_run_id> \
  --entity <your_wandb_entity> \
  --project nanochat \
  --refresh-interval 30  # Longer interval for large-scale runs
```

---

## Troubleshooting

### "wandb package not installed"

```bash
pip install wandb
```

### "WandB connection error"

The monitor will automatically retry. Check your internet connection and WandB status.

### "No metrics collected yet"

Wait for training to reach the first SVD analysis step (typically step 20 with default `svd_interval=20`).

### Log file parsing not working

The log file parser is currently a stub. To implement:

1. Edit `scripts/monitor_svd_metrics.py`
2. Find the `monitor_log_file()` function
3. Add parsing logic for your log format
4. Look for lines containing SVD metrics and extract them

---

## Expected Timeline for Phase 2

Based on projections, here's when you should expect alerts:

| Step Range | Expected Activity |
|------------|------------------|
| 0-200 | Quiet (no alerts) |
| 200-400 | INFO alerts (trends detected) |
| 400-600 | WARNING alerts (approaching thresholds) |
| 600-800 | CRITICAL alerts (switch imminent/occurring) |
| 800+ | Mode switches completed, monitoring continues |

---

## Advanced: Custom Monitoring Logic

You can import the `SVDMonitor` class in your own scripts:

```python
from scripts.monitor_svd_metrics import SVDMonitor, SwitchThresholds, SVDMetrics

# Create monitor
thresholds = SwitchThresholds(warn_margin=0.05)
monitor = SVDMonitor(thresholds)

# Add metrics during training
metrics = SVDMetrics(
    step=100,
    layer_name="block0.attn.c_q",
    mode=0.0,
    r=246,
    principal_alignment=0.32,
    minor_alignment=0.48,
    subspace_angle=0.15,
    reconstruction_error=0.008
)
monitor.add_metrics(metrics)

# Get summary
monitor.print_summary()
```

---

## Next Steps

1. ✅ Script created: `scripts/monitor_svd_metrics.py`
2. ⏳ Test with Phase 2 training (depth=4, 2000 steps)
3. ⏳ Validate predictions against actual switch behavior
4. ⏳ Use for Phase 3 production training on Ptolemy

**Ready to use for Phase 2!** 🚀
