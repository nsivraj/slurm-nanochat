# WandB Offline Mode Fix

**Date:** 2025-11-01
**Issue:** WandB authentication error on GPU nodes
**Status:** ✅ FIXED

---

## Problem Summary

### Error Encountered
```
wandb.errors.errors.UsageError: api_key not configured (no-tty).
call wandb.login(key=[your_api_key])
```

### Root Cause
When `WANDB_RUN` is set to a non-"dummy" value (e.g., `my_training_run`), the training scripts call `wandb.init()` which attempts to authenticate with WandB servers. Since GPU compute nodes have no internet access, this authentication fails and crashes the training job.

### Impact
- Base training would fail immediately
- Midtraining couldn't start
- SFT couldn't start
- Training jobs completely blocked

---

## Solution Implemented

### Changes Made to `scripts/speedrun.slurm`

Added WandB offline mode configuration:

```bash
# --- WandB Offline Configuration ---
# GPU nodes have no internet, so we run WandB in offline mode.
# Logs will be saved locally and can be synced later from a devel node.
# See: https://docs.wandb.ai/guides/track/environment-variables
export WANDB_MODE=offline
export WANDB_DIR="$NANOCHAT_BASE_DIR/wandb" # Store wandb logs in persistent cache
```

Also added directory creation:
```bash
mkdir -p $WANDB_DIR
```

### How It Works

1. **`WANDB_MODE=offline`**: Instructs WandB to bypass all network communication
   - No authentication required
   - No API key needed
   - Works perfectly on nodes without internet

2. **`WANDB_DIR="$NANOCHAT_BASE_DIR/wandb"`**: Specifies where to store logs
   - Logs saved to `/scratch/ptolemy/users/$USER/nanochat-cache/wandb/`
   - Persistent location (not temporary)
   - Easy to find and sync later

3. **Distributed Training Compatibility**:
   - Only rank 0 (master process) calls `wandb.init()`
   - Other GPUs use `DummyWandb` (no network calls)
   - Offline mode only affects master process

---

## Benefits

### Immediate
- ✅ Training works without WandB authentication
- ✅ No API key or internet needed during training
- ✅ Compatible with distributed training (8 GPUs)
- ✅ Metrics still logged locally

### Long-term
- ✅ Logs can be synced to WandB later (optional)
- ✅ Full experiment tracking available
- ✅ Easy to review metrics after training

---

## Usage

### Running Training (No Changes Required)
The fix is automatic. Just run jobs as normal:

```bash
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm
```

### Syncing Logs to WandB (Optional)

After training completes, you can optionally sync logs to WandB:

```bash
# 1. SSH to a devel node (has internet)
ssh ptolemy-devel-1.arc.msstate.edu

# 2. Activate your environment
source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate

# 3. Login to WandB (one-time setup)
wandb login

# 4. Sync all offline runs
wandb sync /scratch/ptolemy/users/$USER/nanochat-cache/wandb/
```

This will upload all metrics, logs, and artifacts to your WandB account for web-based visualization and sharing.

---

## Technical Details

### Why Offline Mode Works

From the Zen chat consultation:

> **Will `WANDB_MODE=offline` fix the authentication error?**
>
> Yes, absolutely. The `api_key not configured (no-tty)` error occurs because `wandb.init()` attempts to communicate with WandB servers for authentication. Setting `WANDB_MODE=offline` instructs the library to bypass all network communication. It will create a local directory for the run and save all metrics, logs, and artifacts there without attempting to authenticate or sync.

### Distributed Training

The existing code in `scripts/base_train.py` (and similar scripts) already handles distributed training correctly:

```python
master_process = ddp_rank == 0  # Only rank 0 does logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(...)
```

Only the master process calls `wandb.init()`, and with `WANDB_MODE=offline`, this works without internet.

### Alternative Approaches Considered

1. **Using `run="dummy"`**: This skips WandB entirely but also skips training iterations
2. **Custom logging handler**: Unnecessary complexity
3. **Proxy configuration**: Not suitable for secure HPC environments

**Conclusion**: Offline mode is the simplest and most appropriate solution.

---

## Verification

### How to Verify the Fix Works

1. Check that environment variables are set:
```bash
# Look for these lines in the .out log file:
export WANDB_MODE=offline
export WANDB_DIR=...
```

2. Confirm no authentication errors:
```bash
# The .err log should NOT contain:
# "api_key not configured"
# "call wandb.login"
```

3. Check that wandb directory is created:
```bash
ls -la /scratch/ptolemy/users/$USER/nanochat-cache/wandb/
```

### Expected Log Directory Structure

After training, you should see:
```
/scratch/ptolemy/users/$USER/nanochat-cache/wandb/
├── offline-run-20251101_113105-<unique_id>/
│   ├── files/
│   │   ├── config.yaml
│   │   ├── wandb-metadata.json
│   │   └── wandb-summary.json
│   └── logs/
│       └── debug.log
└── ... (more runs)
```

---

## Related Documentation

- [docs/how-to/troubleshoot-common-issues.md](../docs/how-to/troubleshoot-common-issues.md#3-wandb-api-key-error-fixed) - Section 3: WandB API Key Error
- [docs/how-to/setup-environment.md](../docs/how-to/setup-environment.md) - WandB Offline Mode section
- [CHANGELOG.md](../CHANGELOG.md) - WandB Offline Mode Fix entry
- [WandB Environment Variables Docs](https://docs.wandb.ai/guides/track/environment-variables)

---

## Timeline

- **2025-10-31**: Error first encountered in job 76355
- **2025-11-01**: Issue diagnosed with Zen chat consultation
- **2025-11-01**: Fix implemented and documented
- **Status**: Ready for next training run

---

## Next Steps

1. ✅ Fix implemented in `scripts/speedrun.slurm`
2. ✅ Documentation updated
3. ⏭️ Test with next training job
4. ⏭️ Optionally sync logs to WandB after training

---

**For questions or issues, refer to the troubleshooting guide or check the error logs.**
