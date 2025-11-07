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
