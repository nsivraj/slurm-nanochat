# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [2025-11-01] - Dataset Requirements & Resume Training

### Added - Critical Dataset Support
- **NEW REQUIREMENT**: Additional datasets required for midtraining and SFT
  - `scripts/download_after_basetraining.sh` - Downloads all midtraining/SFT datasets
  - MMLU (auxiliary_train) - For midtraining
  - GSM8K (main) - For midtraining and SFT
  - SmolTalk (default) - For midtraining and SFT (uses cached version)
  - ARC (ARC-Easy) - For SFT
  - English word list (words_alpha.txt) - For SpellingBee task in midtraining
- `scripts/resume_mid_sft.slurm` - Resume training from midtraining phase
  - Skips base training (saves ~7 hours)
  - Verifies base model checkpoint exists
  - Runs midtraining + SFT only (~4-6 hours)

### Added - Diagnosis & Documentation
- `experiments/DIAGNOSIS_TRAINING_INCOMPLETE.md` - Analysis of first training failure
- `experiments/DIAGNOSIS_RESUME_FAILURE.md` - Analysis of MMLU dataset error
- Updated all documentation with dataset requirements
- Added resume training workflow to all guides
- Updated troubleshooting guide with dataset errors

### Fixed - Dataset Download Issues
- Fixed `download_after_basetraining.sh` to check cache before downloading
- Handles already-cached datasets gracefully (no false errors)
- Skips duplicate dataset entries automatically
- Added English word list download for SpellingBee task
- Improved error messages and verification

### Changed - Documentation
- `docs/how-to/setup-environment.md` - Added dataset download step
- `docs/how-to/run-a-training-job.md` - Added resume training section
- `docs/how-to/troubleshoot-common-issues.md` - Added dataset error solutions
- `docs/tutorials/02-hpc-first-job.md` - Updated with dataset requirements
- `docs/explanation/hpc-environment.md` - Added resume workflow

### WandB Offline Mode Fix (2025-11-01)
- **CRITICAL**: Fixed WandB authentication error on GPU nodes
  - Added `WANDB_MODE=offline` to `scripts/speedrun.slurm`
  - Added `WANDB_DIR` to store logs in persistent location
  - GPU nodes can now run training without internet/authentication
  - Metrics are logged locally and can be synced later (optional)

### Documentation Reorganization (2025-11-01)
- Restructured all documentation following Diátaxis framework
- Created `docs/` directory with tutorials, how-to, explanation, and reference sections
- Moved session logs to `experiments/` directory
- Created comprehensive navigation via `docs/index.md`
- Consolidated redundant documentation across platforms
- Updated troubleshooting guide with WandB offline mode information
- Updated setup guide with WandB configuration details

---

## [2025-11-01] - Critical WANDB_RUN Fix

### Fixed
- **CRITICAL**: Fixed WANDB_RUN issue causing midtraining and SFT to be skipped
  - Added mandatory validation to `scripts/speedrun.slurm`
  - Job now fails immediately if `WANDB_RUN` not set or equals "dummy"
  - Clear error messages guide users to correct usage

### Changed
- Updated `PTOLEMY_SETUP.md` with WANDB_RUN requirements
- Updated `QUICK_RERUN_GUIDE.md` with correct submission commands
- Enhanced chat documentation to explain model selection (`-i mid` vs `-i sft`)

### Added
- `PTOLEMY_SESSION_STATUS.md` - Comprehensive training history and status
- `SESSION_2025_11_01_SUMMARY.md` - Detailed summary of Nov 1 fixes
- `WANDB_RUN_FIX_SUMMARY.md` - Technical analysis of the issue

---

## [2025-10-31] - Second Training Run

### Changed
- Increased SLURM time limit to 12 hours (from 4 hours)
- Pre-downloaded GPT-2 and GPT-4 tokenizers for offline evaluation
- Pre-downloaded SmolTalk dataset for midtraining

### Fixed
- Tokenizer evaluation now works offline
- `tok_eval.py` updated to gracefully handle missing wandb

### Completed
- ✅ Base model training (CORE: 0.2118)
- ✅ Tokenizer evaluation (using cached tokenizers)
- ⚠️ Midtraining ran in dummy mode (WANDB_RUN issue discovered)
- ⚠️ SFT ran in dummy mode (WANDB_RUN issue discovered)

---

## [2025-10-30] - Initial Ptolemy Setup and First Training Run

### Added
- `scripts/setup_environment.sh` - Automated environment setup
- `scripts/download_data.sh` - Data download script for devel node
- `scripts/speedrun.slurm` - SLURM job script for 8xA100
- `scripts/test_gpu.py` - GPU configuration test
- `.env.local.example` - Email notification template
- `PTOLEMY_SETUP.md` - Complete Ptolemy setup guide
- `ASSIGNMENT_README.md` - Assignment-specific documentation
- `TROUBLESHOOTING.md` - Common issues and solutions
- `PROJECT_STATUS.md` - Project status tracking

### Fixed
- Python version issue (system Python 3.9 too old, using 3.12.5)
- Dependency installation (switched from `uv sync` to `pip install -e '.[gpu]'`)
- UV PATH issue (installed in venv)
- Data path verification (updated for `.parquet` files in `base_data/`)
- Tokenizer path (updated for `tokenizer.pkl`)

### Completed
- ✅ Environment setup on Ptolemy
- ✅ Downloaded all training data (~24GB)
- ✅ First training run (Job 76322)
  - ✅ Base model training completed
  - ❌ Tokenizer eval failed (no internet on compute node)
  - ❌ Midtraining failed (couldn't download SmolTalk)

---

## [2025-10-XX] - Local CPU Training Support

### Added
- `scripts/local_cpu_train.sh` - Complete CPU training pipeline
- `local_cpu_docs/` directory with comprehensive documentation:
  - `START_HERE_LOCAL_CPU.md` - Entry point
  - `LOCAL_CPU_QUICKSTART.md` - Quick reference
  - `LOCAL_CPU_TRAINING.md` - Detailed training guide
  - `LOCAL_CPU_ANALYSIS_GUIDE.md` - Code analysis guide
  - `LOCAL_CPU_README.md` - Overview
  - `TRAINING_COMPARISON.md` - CPU vs GPU vs HPC comparison
  - `TROUBLESHOOTING_LOCAL_CPU.md` - CPU-specific issues
  - `SESSION_SUMMARY_LOCAL_CPU.md` - Session summary

### Features
- Train tiny models (4 layers, ~8M params) on any CPU
- Complete pipeline: tokenizer → base → mid → SFT → report
- Training time: 1-3 hours
- Minimal data download: 4 shards (~400MB)
- Non-invasive: doesn't modify existing GPU scripts

---

## [Initial] - Fork from nanochat

### Added
- Complete nanochat codebase from [karpathy/nanochat](https://github.com/karpathy/nanochat)
- Original `README.md`
- Core training scripts (`scripts/`)
- Model implementation (`nanochat/`)
- Task implementations (`tasks/`)
- Test suite (`tests/`)

---

## Summary of Changes by Category

### Environment Support
- ✅ **Ptolemy HPC** - SLURM-based training on 8xA100
- ✅ **Local CPU** - Training on any laptop/desktop
- ✅ **Production GPU** - Original cloud GPU workflow (unchanged)

### Critical Fixes
1. **WANDB_RUN issue** - Ensures midtraining and SFT actually run
2. **Offline training** - Pre-download all internet-required data
3. **Python version** - Use Python 3.12.5 on Ptolemy
4. **Dependency management** - Correct installation procedure

### Documentation Improvements
- Reorganized into `docs/` with clear structure
- Session logs moved to `experiments/`
- Comprehensive guides for each platform
- Clear troubleshooting documentation
- Assignment-specific guidance

### New Features
- Email notifications for SLURM jobs
- Validation checks before training starts
- CPU training support for learning
- Detailed environment comparison

---

## Migration Guide

### From Old Structure to New

**Old location** → **New location**

Top-level docs:
- `START_HERE.md` → `docs/index.md`
- `DOCUMENTATION_INDEX.md` → `docs/index.md`
- `TROUBLESHOOTING.md` → `docs/how-to/troubleshoot-common-issues.md`
- `QUICK_RESUME.md` → `docs/how-to/resume-a-job.md` (to be created)

Platform-specific:
- `local_cpu_docs/LOCAL_CPU_QUICKSTART.md` → `docs/tutorials/01-local-cpu-quickstart.md`
- `ptolemy_slurm_docs/RESUME_HERE.md` → `docs/tutorials/02-hpc-first-job.md`
- `local_cpu_docs/TRAINING_COMPARISON.md` → `docs/explanation/training-environments.md`

Session logs:
- `PROJECT_STATUS.md` → `experiments/PROJECT_STATUS.md`
- `SUMMARY_OF_CHANGES.md` → `experiments/SUMMARY_OF_CHANGES.md`
- All `SESSION_*.md` → `experiments/`

---

**For current documentation, see [`docs/index.md`](docs/index.md)**
