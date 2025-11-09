# Phase 1: Running Adaptive SVD Training (Proof-of-Concept)

**Date**: November 7, 2025
**Status**: Ready to run
**Purpose**: Validate that adaptive SVD mechanics work without crashing

---

## Quick Start (Copy-Paste Command)

```bash
# Navigate to project directory
cd /Users/norman.jarvis/forge/work/code/coderockit/msu-phd/slurm-nanochat

# Activate virtual environment (if needed)
source .venv/bin/activate

# Run Phase 1 training (CPU mode - required for MPS compatibility)
python -m scripts.base_train_adaptive_svd \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --eval_tokens=512 \
  --core_metric_every=-1 \
  --total_batch_size=512 \
  --num_iterations=100 \
  --device_type=cpu
```

**Expected duration**: 5-10 minutes on CPU

---

## What This Command Does

### Model Configuration
- `--depth=4`: Small model with 4 transformer layers (~37M parameters)
- `--max_seq_len=512`: Context length of 512 tokens
- `--device_batch_size=1`: Batch size of 1 (minimal memory usage)

### Training Configuration
- `--num_iterations=100`: Run for 100 training steps
- `--total_batch_size=512`: Total batch size of 512 tokens
- This results in gradient accumulation steps of 512/512 = 1 (no accumulation)

### Evaluation Configuration
- `--eval_tokens=512`: Minimal evaluation (just for validation)
- `--core_metric_every=-1`: Disable CORE metric (too slow for proof-of-concept)

### Device Configuration
- `--device_type=cpu`: **IMPORTANT** - Use CPU instead of MPS/GPU
  - **Why CPU?** Apple MPS backend doesn't support all SVD operations
  - The error message: `NotImplementedError: The operator 'aten::_linalg_svd.U' is not currently implemented for the MPS device`
  - For Phase 1, CPU is sufficient (slow but functional)
  - For Phase 2/3, we'll run on CUDA (Ptolemy cluster)

### Adaptive SVD Configuration (Hard-coded in script)
- `adaptive_svd = True`: Enable adaptive SVD
- `adaptive_svd_target_layer = 0`: Only apply to layer 0
- `svd_interval = 20`: Run SVD analysis every 20 steps

---

## Expected Output

### 1. Initialization
```
nanochat banner (ASCII art)
Autodetected device type: cpu
Vocab size: 65,536
num_layers: 4
model_dim: 256
num_heads: 2
Number of parameters: 36,700,160
```

### 2. Training Progress
```
step 00001/00100 (1.00%) | loss: 11.003613 | lrm: 1.00 | dt: 42.27ms | ...
step 00002/00100 (2.00%) | loss: 10.800964 | lrm: 1.00 | dt: 15.34ms | ...
...
```

### 3. SVD Analysis (every 20 steps)
```
=== SVD Analysis at step 20 ===
  [block0.attn.c_q] Initial SVD computed
=== SVD Analysis complete ===
```

**First time (step 20)**: Computes initial SVD for comparison baseline

**Subsequent times (steps 40, 60, 80, 100)**: Analyzes gradients and makes decisions:
- Example: `[block0.attn.c_q] SWITCHED TO LOW-RANK (r=246) - Reason: gradient_clobbering_risk`
- Or: `[block0.attn.c_q] Updated r: 246 -> 230`

### 4. Final Statistics
```
Peak memory usage: XXX MiB
Total training time: X.XX m
Minimum validation bpb: X.XXXX
```

---

## Success Criteria Checklist

After the run completes, verify:

- [ ] **Training completes without errors** (exit code 0)
- [ ] **SVD analysis runs at steps 20, 40, 60, 80, 100**
- [ ] **Initial SVD computed at step 20**
- [ ] **Metrics computed** (principal_alignment, subspace_angle, etc.)
- [ ] **At least one mode switch occurs** (full → lowrank or vice versa)
  - Look for `SWITCHED TO LOW-RANK` or `SWITCHED TO FULL MATRIX` messages
- [ ] **Final validation loss is reasonable** (should decrease from initial ~11.0)
- [ ] **Model checkpoint is saved** (in `base_checkpoints/d4/`)

---

## Monitoring the Run

### Option 1: Watch in Real-Time
```bash
# Run with output visible
python -m scripts.base_train_adaptive_svd \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --eval_tokens=512 \
  --core_metric_every=-1 \
  --total_batch_size=512 \
  --num_iterations=100 \
  --device_type=cpu
```

### Option 2: Run in Background with Logging
```bash
# Run in background and log to file
python -m scripts.base_train_adaptive_svd \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --eval_tokens=512 \
  --core_metric_every=-1 \
  --total_batch_size=512 \
  --num_iterations=100 \
  --device_type=cpu \
  > phase1_training.log 2>&1 &

# Get the process ID
echo $!

# Monitor progress
tail -f phase1_training.log

# Check for SVD analysis
grep "SVD Analysis" phase1_training.log

# Check for mode switches
grep "SWITCHED" phase1_training.log
```

### Option 3: Use tmux/screen
```bash
# Start tmux session
tmux new -s phase1

# Run training
python -m scripts.base_train_adaptive_svd \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --eval_tokens=512 \
  --core_metric_every=-1 \
  --total_batch_size=512 \
  --num_iterations=100 \
  --device_type=cpu

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t phase1
```

---

## Troubleshooting

### Issue: MPS Device Error
```
NotImplementedError: The operator 'aten::_linalg_svd.U' is not currently implemented for the MPS device
```

**Solution**: Add `--device_type=cpu` to force CPU usage

### Issue: Out of Memory
```
RuntimeError: [enforce fail at ...] DefaultCPUAllocator
```

**Solution**: Reduce batch size or sequence length:
```bash
python -m scripts.base_train_adaptive_svd \
  --depth=4 \
  --max_seq_len=256 \      # Reduced from 512
  --device_batch_size=1 \
  --total_batch_size=256 \ # Reduced from 512
  ...
```

### Issue: Import Error
```
ModuleNotFoundError: No module named 'nanochat.adaptive_svd'
```

**Solution**: Make sure you're in the project root directory and virtual environment is activated

### Issue: Slow Training
**Expected**: CPU training is ~10-20x slower than GPU. This is normal for Phase 1.
- Step time on CPU: ~40-50ms per step
- Total time for 100 steps: ~5-10 minutes

**If you need faster**:
- Wait for Phase 2/3 to run on CUDA (Ptolemy)
- Or accept the slow speed for validation purposes

---

## What to Look For

### Good Signs ✅
1. **SVD analysis completes without errors**
2. **Metrics are logged**: principal_alignment, minor_alignment, subspace_angle
3. **At least one mode switch** (full→lowrank due to clobbering detection)
4. **Training loss decreases steadily** (from ~11.0 to ~7.0-8.0)
5. **No crashes or exceptions**

### Warning Signs ⚠️
1. **No mode switches at all** - May indicate:
   - Thresholds too conservative
   - Need more training steps
   - Or genuinely no clobbering (which is fine for validation)

2. **Constant mode switching** (every cycle) - May indicate:
   - Thresholds too aggressive
   - Need to adjust decision logic
   - Oscillating behavior (not ideal but not fatal)

3. **SVD errors** - Contact if you see:
   - NaN values in metrics
   - SVD convergence failures
   - Dimension mismatch errors

---

## After Successful Completion

### 1. Check the Output Checkpoint
```bash
ls -lh base_checkpoints/d4/

# Should see:
# checkpoint.pt
# config.json
```

### 2. Extract Key Metrics from Log
```bash
# Get all SVD analysis summaries
grep -A 10 "SVD Analysis at step" phase1_training.log

# Get mode switches
grep "SWITCHED" phase1_training.log

# Get final statistics
tail -20 phase1_training.log
```

### 3. Test Model Generation (Optional)
```python
# In Python REPL
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine
import torch

# Load checkpoint
checkpoint = torch.load('base_checkpoints/d4/checkpoint.pt', map_location='cpu')

# Recreate model
config_kwargs = checkpoint['metadata']['model_config']
config = GPTConfig(**config_kwargs)
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate text
tokenizer = get_tokenizer()
engine = Engine(model, tokenizer)
prompt = "The capital of France is"
tokens = tokenizer(prompt, prepend="<|bos|>")
sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=20, temperature=0)
print(tokenizer.decode(sample[0]))
```

### 4. Document Results
Create a summary in `experiments/PHASE1_RESULTS.md`:
- Did training complete successfully? ✅/❌
- Number of mode switches observed:
- Example metrics from step 60/80:
  - principal_alignment:
  - minor_alignment:
  - subspace_angle:
- Final training loss:
- Any issues encountered:
- Ready for Phase 2? ✅/❌

---

## Next Steps After Phase 1

Once Phase 1 succeeds:

### Phase 2 Preparation
1. Update `adaptive_svd_target_layer` to `-1` (all layers)
2. Update target layers to include all attention matrices
3. Run full training (~2000-3000 steps)
4. Compare with baseline

### Phase 3 Preparation
1. Scale to depth=20 (production model)
2. Deploy on Ptolemy cluster (8xA100 GPUs)
3. Full base training run (21,400 steps, 7-8 hours)
4. Comprehensive analysis and comparison

---

## Quick Reference: File Locations

**Training Script**: `scripts/base_train_adaptive_svd.py`
**SVD Metrics**: `nanochat/adaptive_svd.py`
**Model Definition**: `nanochat/gpt.py` (AdaptiveSVDLinear class)
**Tests**: `tests/test_adaptive_svd.py`
**Documentation**:
- Main experiment doc: `experiments/BASE_TRAIN_SVD_ENHANCEMENT.md`
- Status tracking: `experiments/BASE_TRAIN_SVD_ENHANCEMENT_STATUS.md`
- Architecture guide: `experiments/SVD_ENHANCEMENT_ARCHITECTURE_MODIFICATIONS.md`
- **This file**: `experiments/PHASE1_RUN_INSTRUCTIONS.md`

---

## Contact/Support

If you encounter issues:
1. Check the error message carefully
2. Review the troubleshooting section above
3. Check `experiments/BASE_TRAIN_SVD_ENHANCEMENT_STATUS.md` for known issues
4. Document the error and context for discussion

**Good luck with Phase 1! 🚀**

---

## NEW: Simplified Shell Script Workflow

Two new shell scripts have been created for easier execution and monitoring:

### Scripts Available

1. **`scripts/svd_adaptive_local_cpu_train.sh`** - Training script with smart defaults
2. **`scripts/svd_monitor.sh`** - Real-time monitoring with alerts

### ⚠️ WandB Monitoring Note

**Issue**: The monitoring script requires WandB online mode with authentication (`wandb login`).

**Current approach**: For local CPU training, we run without WandB (dummy mode):
- All SVD metrics are logged to the console/file
- No WandB dashboard needed
- Simpler execution without authentication dependencies
- Full analysis possible from log files

**For future Ptolemy runs**: Use `WANDB_MODE=offline` to save metrics locally without internet connection.

---

## Simplified Workflow (No WandB - Recommended for Local CPU)

### Single Terminal: Start Training

```bash
cd /Users/norman.jarvis/forge/work/code/coderockit/msu-phd/slurm-nanochat

# Quick start with defaults (2000 steps)
bash scripts/svd_adaptive_local_cpu_train.sh

# OR: Extended run (3000 steps for more observation)
bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=3000
```

**The script will:**
- Check your environment
- Display configuration
- Show estimated runtime (~80 minutes for 2000 steps)
- Ask for confirmation before starting
- Log all output to screen AND file

**All SVD metrics logged in real-time:**
- SVD analysis every 20 steps
- Mode switches when they occur
- Metrics: principal_alignment, minor_alignment, subspace_angle, reconstruction_error
- No WandB needed - everything in the log file

**After training completes**, analyze the log file for results.

---

## Alternative: Two-Terminal Workflow with WandB (If Configured)

If you have WandB authenticated and want real-time monitoring:

### Terminal 1: Start Training

```bash
cd /Users/norman.jarvis/forge/work/code/coderockit/msu-phd/slurm-nanochat

# Enable WandB with custom run name
bash scripts/svd_adaptive_local_cpu_train.sh \
  --num_iterations=2000 \
  --wandb_run=phase1-extended 2>&1 | tee svd_training_run.log
```

### Terminal 2: Monitor Metrics

After training starts and you have the WandB run ID:

```bash
cd /Users/norman.jarvis/forge/work/code/coderockit/msu-phd/slurm-nanochat

# Replace abc123def456 with your actual run ID
bash scripts/svd_monitor.sh --wandb_run=abc123def456 2>&1 | tee svd_monitoring_run.log
```

**Note**: This requires `wandb login` to be completed first.

---

## What You'll See

### Training Terminal (Terminal 1)

```
==================================================
SVD Adaptive Training - Parameter Setup
==================================================

Training Configuration:
----------------------
Model:
  depth:              4
  max_seq_len:        512

Training:
  num_iterations:     2000
  device_batch_size:  1
  total_batch_size:   512

SVD Adaptive:
  svd_interval:       20 (SVD analysis every N steps)

Estimated runtime: ~80 minutes

...

Step 20/2000 | loss: 10.543 | ...
=== SVD Analysis at step 20 ===
  [block0.attn.c_q] Initial SVD computed
=== SVD Analysis complete ===

...

Step 640/2000 | loss: 7.123 | ...
=== SVD Analysis at step 640 ===
  [block0.attn.c_q] SWITCHED TO LOW-RANK (r=220) - Reason: optimal_lowrank_conditions
=== SVD Analysis complete ===
```

### Monitoring Terminal (Terminal 2)

```
==================================================
Starting SVD Metrics Monitoring
==================================================

Monitoring WandB run: abc123def456
Project: nanochat

What to expect:
  🟢 Steps 0-200:   Quiet (no alerts)
  🟡 Steps 200-400: INFO alerts (trends detected)
  🟡 Steps 400-600: WARNING alerts (approaching thresholds)
  🔴 Steps 600-800: CRITICAL alerts (switch imminent)
  🟢 Steps 800+:    Mode switches completed

Press Ctrl+C to stop monitoring and show summary

==================================================

⚠️  Step 480 | block0.attn.c_q
    WARNING: Approaching optimal conditions
    - subspace_angle: 0.12 (within 0.02 of 0.1)
    - minor_alignment: 0.58 (within 0.02 of 0.6)

🚨 Step 640 | block0.attn.c_q
    CRITICAL: OPTIMAL CONDITIONS MET! → SWITCH IMMINENT
    All 3 criteria satisfied
```

---

## Customization Examples

All parameters can be specified in any order, any combination.

### Different Run Lengths

```bash
# Quick test (100 steps, ~4 minutes)
bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=100

# Standard Phase 1 (2000 steps, ~80 minutes)
bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=2000

# Extended Phase 1 (3000 steps, ~120 minutes)
bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=3000

# Very long (5000 steps, ~200 minutes)
bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=5000
```

### More Frequent SVD Analysis

```bash
# Default: SVD every 20 steps
bash scripts/svd_adaptive_local_cpu_train.sh

# More frequent: SVD every 10 steps (finer-grained detection)
bash scripts/svd_adaptive_local_cpu_train.sh --svd_interval=10

# Less frequent: SVD every 50 steps (faster, less overhead)
bash scripts/svd_adaptive_local_cpu_train.sh --svd_interval=50
```

### Larger Model

```bash
# Default: depth=4 (~37M params)
bash scripts/svd_adaptive_local_cpu_train.sh

# Larger: depth=6 (~83M params, slower)
bash scripts/svd_adaptive_local_cpu_train.sh --depth=6 --num_iterations=2000
```

### Custom WandB Run ID

```bash
# Auto-generated run ID
bash scripts/svd_adaptive_local_cpu_train.sh

# Custom run ID for easy identification
bash scripts/svd_adaptive_local_cpu_train.sh --wandb_run=phase1-extended-2000steps

# Then monitor with:
bash scripts/svd_monitor.sh --wandb_run=phase1-extended-2000steps
```

### Multiple Parameters Combined

```bash
# Any parameters, any order
bash scripts/svd_adaptive_local_cpu_train.sh \
  --depth=6 \
  --num_iterations=3000 \
  --svd_interval=15 \
  --wandb_run=depth6-3ksteps \
  --eval_every=300
```

---

## Monitoring Customization

### Faster Refresh

```bash
# Default: Check every 10 seconds
bash scripts/svd_monitor.sh --wandb_run=abc123def456

# Faster: Check every 5 seconds
bash scripts/svd_monitor.sh --wandb_run=abc123def456 --refresh_interval=5

# Slower: Check every 30 seconds (less network traffic)
bash scripts/svd_monitor.sh --wandb_run=abc123def456 --refresh_interval=30
```

### More Sensitive Alerts

```bash
# Default: Alert when within 0.05 of threshold
bash scripts/svd_monitor.sh --wandb_run=abc123def456

# More sensitive: Alert when within 0.08
bash scripts/svd_monitor.sh --wandb_run=abc123def456 --warn_margin=0.08

# Less sensitive: Alert when within 0.02 (fewer alerts)
bash scripts/svd_monitor.sh --wandb_run=abc123def456 --warn_margin=0.02
```

### Custom Decision Thresholds

```bash
# Earlier clobbering detection (lower threshold)
bash scripts/svd_monitor.sh \
  --wandb_run=abc123def456 \
  --principal_danger=0.35

# More conservative switching (higher safety requirements)
bash scripts/svd_monitor.sh \
  --wandb_run=abc123def456 \
  --minor_safe=0.65 \
  --angle_stable=0.08
```

### Quiet Mode

```bash
# Show everything (default)
bash scripts/svd_monitor.sh --wandb_run=abc123def456

# Only show alerts (quiet mode)
bash scripts/svd_monitor.sh --wandb_run=abc123def456 --quiet
```

---

## Finding Your WandB Run ID

### Method 1: From Console Output

Look for this line in the training terminal:

```
wandb: ⭐️ View run at https://wandb.ai/yourname/nanochat/runs/abc123def456
                                                                 ^^^^^^^^^^^^
                                                                 This is your run ID
```

### Method 2: From WandB Dashboard

1. Go to https://wandb.ai/
2. Click on your project (e.g., "nanochat")
3. Click on the active run
4. Copy the run ID from the URL:
   ```
   https://wandb.ai/<entity>/<project>/runs/<RUN_ID>
   ```

---

## Expected Timeline (depth=4, 2000 steps)

| Step Range | Duration | What's Happening | Alerts |
|------------|----------|------------------|--------|
| 0-200 | ~8 min | Initial learning, unstable subspace | None |
| 200-400 | ~8 min | Patterns forming, metrics improving | INFO |
| 400-600 | ~8 min | Subspace stabilizing | WARNING |
| 600-800 | ~8 min | **FIRST SWITCH EXPECTED** | CRITICAL |
| 800-2000 | ~48 min | Low-rank mode, r decreasing | INFO |

**Total**: ~80 minutes on modern CPU (MacBook Pro M1/M2)

---

## Shell Script Help

Both scripts have built-in help:

```bash
# Show all training options
bash scripts/svd_adaptive_local_cpu_train.sh --help

# Show all monitoring options
bash scripts/svd_monitor.sh --help
```

---

## Quick Reference Card

**Start training with defaults:**
```bash
bash scripts/svd_adaptive_local_cpu_train.sh
```

**Monitor (replace with your run ID):**
```bash
bash scripts/svd_monitor.sh --wandb_run=<YOUR_RUN_ID>
```

**Training with 3000 steps:**
```bash
bash scripts/svd_adaptive_local_cpu_train.sh --num_iterations=3000
```

**Monitoring with faster refresh:**
```bash
bash scripts/svd_monitor.sh --wandb_run=<ID> --refresh_interval=5
```

---

**Ready to start the extended Phase 1 validation!** 🚀
