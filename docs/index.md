# nanochat Documentation

**Complete LLM Training Pipeline - From Tokenization to ChatGPT**

Last Updated: 2025-11-01

---

## 🚀 New Here? Start With These

### Choose Your Training Environment

**Running locally on CPU?**
👉 Start here: [Local CPU Quickstart](tutorials/01-local-cpu-quickstart.md)

**Running on Ptolemy HPC cluster?**
👉 Start here: [HPC First Job](tutorials/02-hpc-first-job.md)

**Running on production GPU (Lambda/RunPod)?**
👉 See the main [README.md](../README.md)

**Not sure which to use?**
👉 Read: [Training Environment Comparison](explanation/training-environments.md)

---

## 📚 Documentation Structure

This documentation follows the [Diátaxis framework](https://diataxis.fr/), organizing content by user need:

### [Tutorials](tutorials/) - Learning-Oriented
Step-by-step guides for getting started:
- [Local CPU Quickstart](tutorials/01-local-cpu-quickstart.md) - Train a tiny model in 1-3 hours
- [HPC First Job](tutorials/02-hpc-first-job.md) - Submit your first Ptolemy training job

### [How-To Guides](how-to/) - Problem-Oriented
Practical guides for specific tasks:
- [Setup Environment](how-to/setup-environment.md) - Complete setup for all platforms
- [Run a Training Job](how-to/run-a-training-job.md) - Submit and monitor training
- [Troubleshoot Common Issues](how-to/troubleshoot-common-issues.md) - Fix common problems
- [Analyze Results](how-to/analyze-results.md) - Understand your training outputs

### [Explanation](explanation/) - Understanding-Oriented
High-level concepts and design decisions:
- [Training Environments](explanation/training-environments.md) - Compare Local CPU vs GPU vs HPC
- [HPC Environment Details](explanation/hpc-environment.md) - Ptolemy-specific constraints
- [Project Architecture](explanation/project-architecture.md) - Understanding the codebase

### [Reference](reference/) - Information-Oriented
Technical specifications and API details:
- [Configuration Parameters](reference/configuration.md) - All training parameters
- [SLURM Scripts](reference/slurm-scripts.md) - HPC job submission reference
- [Data Pipeline](reference/data-pipeline.md) - Dataset and tokenizer details

---

## 🎓 For MSU CSE8990 Students

### Assignment: Transformer Architecture Analysis

**Objective:** Train and analyze a complete ChatGPT-style model

**What You'll Build:**
- 560M parameter Transformer model
- Trained on 11.2B tokens
- Complete pipeline: tokenization → pretraining → midtraining → SFT

### Recommended Learning Path

#### Week 1: Learn & Setup
1. **Understand the options**: Read [Training Environments](explanation/training-environments.md)
2. **Start small**: Try [Local CPU Quickstart](tutorials/01-local-cpu-quickstart.md) (1-3 hours)
3. **Set up HPC**: Follow [HPC First Job](tutorials/02-hpc-first-job.md)

#### Week 2: Train & Study
1. **Submit job**: Use [Run a Training Job](how-to/run-a-training-job.md)
2. **Study code**: While training runs (~12 hours)
   - `nanochat/gpt.py` - Transformer architecture
   - `scripts/base_train.py` - Training loop
   - `nanochat/tokenizer.py` - BPE tokenization
3. **Monitor progress**: Check logs and email notifications

#### Week 3: Analyze
1. **Review results**: Read generated `report.md`
2. **Chat with model**: Test your trained ChatGPT
3. **Deep dive**: Use [Analyze Results](how-to/analyze-results.md)
4. **Compare**: Local CPU vs HPC results

#### Week 4: Report
1. Write analysis covering:
   - Transformer architecture implementation
   - Training pipeline stages
   - Performance metrics
   - Lessons learned

### Key Files to Study

**Core Architecture** (`nanochat/gpt.py`):
- Lines 51-130: `CausalSelfAttention` - Multi-head attention
- Lines 132-150: `MLP` - Feed-forward network
- Lines 152-200: `TransformerBlock` - Complete transformer block
- Lines 202-300: `GPT` - Full model

**Training** (`scripts/base_train.py`):
- Training loop and optimization
- Distributed training setup
- Evaluation integration

**Tokenization** (`nanochat/tokenizer.py`):
- BPE implementation
- Vocabulary construction

---

## ⚡ Quick Commands

### Local CPU Training
```bash
# One command - complete pipeline
bash scripts/local_cpu_train.sh

# Chat with model after training
source .venv/bin/activate
python -m scripts.chat_cli
```

### Ptolemy HPC Training
```bash
# CRITICAL: Always set WANDB_RUN
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm

# Monitor progress
squeue -u $USER
tail -f logs/nanochat_speedrun_*.out

# Chat with model (after training)
python -m scripts.chat_cli
```

### Production GPU Training
```bash
# Standard cloud GPU workflow
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Serve web interface
python -m scripts.chat_web
```

---

## 🎯 Common Tasks

| I want to... | Go to... |
|-------------|----------|
| Get started quickly (CPU) | [Local CPU Quickstart](tutorials/01-local-cpu-quickstart.md) |
| Get started on HPC | [HPC First Job](tutorials/02-hpc-first-job.md) |
| Understand differences | [Training Environments](explanation/training-environments.md) |
| Set up my environment | [Setup Environment](how-to/setup-environment.md) |
| Submit a training job | [Run a Training Job](how-to/run-a-training-job.md) |
| Fix an error | [Troubleshoot Common Issues](how-to/troubleshoot-common-issues.md) |
| Understand results | [Analyze Results](how-to/analyze-results.md) |
| Learn about Ptolemy | [HPC Environment Details](explanation/hpc-environment.md) |

---

## ⚠️ Critical Information

### For Ptolemy HPC Users

**MUST SET WANDB_RUN!**
```bash
# ✅ CORRECT
WANDB_RUN=my_run sbatch scripts/speedrun.slurm

# ❌ WRONG - Will skip midtraining and SFT!
sbatch scripts/speedrun.slurm
```

**Other Important Notes:**
- No internet on GPU compute nodes (pre-download all data)
- 12-hour time limit (QOS maximum)
- Use `/scratch/ptolemy/users/$USER/` for storage
- Data must be downloaded on devel node first

See: [HPC Environment Details](explanation/hpc-environment.md)

### For Local CPU Users

**Start Small!**
- Tiny model (1M params) = 5 minutes ✅
- Small model (5M params) = 30 minutes ✅
- Medium model (15M params) = 2-3 hours ⚠️
- Large model (50M params) = 8+ hours ⚠️

See: [Training Environments](explanation/training-environments.md)

---

## 📊 Platform Comparison

| Feature | Local CPU | Production GPU | Ptolemy HPC |
|---------|-----------|----------------|-------------|
| **Model Size** | 1M-50M params | 561M params | 561M params |
| **Training Time** | 5 min - 12 hours | ~4 hours | ~12 hours |
| **Cost** | Free | ~$100 | Free |
| **Setup Time** | 10 minutes | 1-2 hours | 1-2 hours |
| **Quality** | Experimental | Production | Production |
| **Best For** | Learning | Quick results | Class assignment |

Full comparison: [Training Environments](explanation/training-environments.md)

---

## 🆘 Getting Help

### By Issue Type

**Setup Problems**
→ [Setup Environment](how-to/setup-environment.md)

**Training Errors**
→ [Troubleshoot Common Issues](how-to/troubleshoot-common-issues.md)

**Understanding Results**
→ [Analyze Results](how-to/analyze-results.md)

**Ptolemy-Specific Issues**
→ [HPC Environment Details](explanation/hpc-environment.md)

**Not Sure Where to Look?**
→ [Troubleshoot Common Issues](how-to/troubleshoot-common-issues.md) - Start here

---

## 🔄 What's New (2025-11-01)

### Critical Fixes Applied
- ✅ **Fixed WANDB_RUN issue** - Midtraining and SFT were being skipped
- ✅ **Added validation** to `speedrun.slurm` - Job fails fast with clear errors
- ✅ **Updated documentation** - All guides reflect latest fixes
- ✅ **Organized docs** into logical structure (Diátaxis framework)

### Documentation Reorganization
- All platform-specific docs consolidated
- Clear separation: tutorials, how-to, explanation, reference
- Session logs moved to `experiments/` folder
- Better navigation and discoverability

See: [CHANGELOG.md](../CHANGELOG.md) for complete history

---

## 📁 Project Overview

### What is nanochat?

nanochat is a complete implementation of a ChatGPT-style language model in a single, clean, minimal codebase. It includes:

- ✅ **Full training pipeline**: Tokenization → Pretraining → Midtraining → SFT → RL
- ✅ **Production-ready**: Web UI, CLI chat, comprehensive evaluations
- ✅ **Educational focus**: Clean, hackable, well-documented code
- ✅ **Scalable**: 1M params (CPU) to 560M params (GPU cluster)

### This Fork

This is a fork adapted for **MSU CSE8990** with:
- ✅ Ptolemy HPC cluster support (SLURM)
- ✅ Local CPU training for development
- ✅ Comprehensive documentation
- ✅ All fixes and optimizations applied

Original project: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)

---

## 🎉 Ready to Begin!

Choose your path:

### 🖥️ I have access to Ptolemy HPC
→ [HPC First Job](tutorials/02-hpc-first-job.md)

### 💻 I want to use my laptop/desktop
→ [Local CPU Quickstart](tutorials/01-local-cpu-quickstart.md)

### 📚 I want to learn about options first
→ [Training Environments](explanation/training-environments.md)

---

**Questions?** Browse the documentation sections above or check the [troubleshooting guide](how-to/troubleshoot-common-issues.md).

**Good luck with your training! 🚀**
