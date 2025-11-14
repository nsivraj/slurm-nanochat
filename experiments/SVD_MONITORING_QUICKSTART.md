# SVD Monitoring Quick Start

## ⚠️ Prerequisite

**Requires WandB authentication**: Run `wandb login` first.

**Alternative**: For local CPU training without WandB, skip monitoring and analyze log files after training completes.

---

## One-Command Monitoring

```bash
# Get your WandB run ID from: https://wandb.ai/your_project/runs
python scripts/monitor_svd_metrics.py --wandb-run <run_id> --project nanochat
```

## What You'll See

### 🟢 Safe (Steps 0-400)
```
No alerts - metrics far from thresholds
```

### 🟡 Approaching (Steps 400-600)
```
⚠️  WARNING: Approaching optimal conditions
ℹ️  INFO: Subspace stabilizing: 0.121 → ~15 steps to threshold
```

### 🔴 Imminent (Steps 600-800)
```
🚨 CRITICAL: OPTIMAL CONDITIONS MET! → SWITCH IMMINENT
```

## Key Metrics

| Metric | Switch When | Current Phase 1 |
|--------|-------------|-----------------|
| `principal_alignment` | > 0.4 | ~0.2-0.3 |
| `minor_alignment` | > 0.6 | ~0.4-0.5 |
| `subspace_angle` | < 0.1 | ~0.2-0.5 |
| `reconstruction_error` | < 0.01 | ~0.003-0.01 |

## Expected Timeline (depth=4, 2000 steps)

```
Steps 0-200:   🟢 No alerts
Steps 200-400: 🟢 INFO only (trends)
Steps 400-600: 🟡 WARNINGs appear
Steps 600-800: 🔴 CRITICAL - switches occur
Steps 800+:    🟢 Low-rank mode active
```

## Stop Monitoring

Press `Ctrl+C` to see summary:

```
================================================================================
SVD MONITORING SUMMARY
================================================================================

block0.attn.c_q
  Latest step: 800
  Mode: LOW-RANK
  Rank: 195
  Distance to switch:
    Optimal conditions: 3/3 met ✓
```

## Full Documentation

See `experiments/SVD_MONITORING_GUIDE.md` for:
- Detailed usage examples
- Custom threshold configuration
- Log file monitoring
- Integration with Phase 2/3
- Troubleshooting

---

**Ready to use with Phase 2!** 🚀
