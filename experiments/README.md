# Training Experiments and Session Logs

This directory contains historical records of training runs, session summaries, and project status snapshots.

---

## Purpose

These files document:
- **Specific training runs** and their results
- **Session summaries** of work completed
- **Project status** at various points in time
- **Issues discovered** and fixes applied

This is separate from the main documentation (in `docs/`), which describes **how to use the system**.

---

## Files in This Directory

### Training Run History

- **`PTOLEMY_SESSION_STATUS.md`** - Comprehensive Ptolemy HPC training history
  - All training runs (Oct 30-31, 2025)
  - WANDB_RUN issue discovery and fix
  - Expected results for next run

- **`2025-11-01-summary.md`** - November 1st session summary
  - WANDB_RUN fix details
  - Documentation updates applied

- **`SESSION_STATUS.md`** - Original session status (historical)

### Local CPU Training

- **`SESSION_SUMMARY_LOCAL_CPU.md`** - Local CPU training session summary
  - Scripts created
  - Documentation written
  - Features implemented

### Project Status Snapshots

- **`PROJECT_STATUS.md`** - Project status from Oct 30, 2025
  - Environment setup completion
  - Data download status
  - Training job submission
  - Assignment context

- **`SUMMARY_OF_CHANGES.md`** - Summary of local CPU training setup
  - Files created
  - Features added
  - Non-invasive design

---

## How to Use These Files

### For Resuming Work

If you need to remember what was done in a previous session:
1. Check the most recent summary file
2. Review `PTOLEMY_SESSION_STATUS.md` for training history
3. Check `PROJECT_STATUS.md` for overall progress

### For Understanding Issues

If you encounter a problem:
1. Check if it's documented in these session logs
2. Look for fixes that were applied
3. See if the issue recurred

### For Academic Record-Keeping

These files provide a complete audit trail of:
- Training attempts and results
- Issues encountered and resolved
- Time invested and progress made

---

## Organization

Files are organized by:
- **Platform**: Ptolemy HPC vs Local CPU
- **Date**: Most recent at top of this README
- **Type**: Session summary vs status snapshot vs training log

---

## Adding New Files

When adding new experiment logs:

1. **Use consistent naming**:
   - Session summaries: `YYYY-MM-DD-summary.md`
   - Training runs: `training-run-JOBID.md`
   - Status snapshots: `status-YYYY-MM-DD.md`

2. **Include key information**:
   - Date and time
   - What was attempted
   - Results (success/failure)
   - Issues encountered
   - Fixes applied

3. **Update this README** with a brief description

---

## Difference from `docs/`

| `experiments/` | `docs/` |
|----------------|---------|
| Historical records | Evergreen guides |
| Specific runs | General how-to |
| What happened | How to do it |
| Time-specific | Timeless |
| Session logs | Documentation |

---

## Retention Policy

These files should be kept for:
- ✅ Academic record-keeping
- ✅ Understanding project evolution
- ✅ Troubleshooting similar issues
- ✅ Assignment submission documentation

Consider archiving (but not deleting) after:
- Project completion
- Course ends
- Assignment submitted

---

**Last Updated:** 2025-11-01

For current documentation, see [`docs/`](../docs/index.md)
