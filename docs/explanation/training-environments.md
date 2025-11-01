# Training Environment Comparison

This document provides a comprehensive comparison of the three ways to train nanochat: **Local CPU**, **Production GPU**, and **Ptolemy HPC**.

---

## Quick Comparison

| Aspect | Local CPU | Production GPU | Ptolemy HPC |
|--------|-----------|----------------|-------------|
| **Purpose** | Learning & understanding | Production model | Class assignment |
| **Hardware** | Your laptop/desktop | 8xH100 GPUs (cloud) | 8xA100 GPUs (SLURM) |
| **Cost** | Free | ~$100 | Free (class allocation) |
| **Time** | 1-3 hours | ~4 hours | ~12 hours (A100 slower) |
| **Model Quality** | Poor (kindergartener) | Good (outperforms GPT-2) | Good (outperforms GPT-2) |
| **Script** | `scripts/local_cpu_train.sh` | `speedrun.sh` | `scripts/speedrun.slurm` |
| **Data Setup** | Automatic | Automatic | Manual (devel node) |
| **Internet** | Required | Required | Devel node only |

---

## 1. Hardware & Environment

### Local CPU
- **Device**: CPU (any modern processor) or MPS (Mac M1/M2/M3)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: ~2GB
- **Setup**: Run script, everything automatic
- **Portability**: Run anywhere with Python

### Production GPU
- **Device**: 8xH100 80GB GPUs
- **Provider**: Lambda Labs, RunPod, etc.
- **Cost**: ~$3/GPU/hour × 8 × 4 hours = $96
- **Setup**: Cloud VM, run script
- **Portability**: Any cloud provider with H100s

### Ptolemy HPC
- **Device**: 8xA100 40GB GPUs
- **Provider**: Mississippi State University ARC
- **Cost**: Free (class allocation)
- **Setup**: SLURM job submission
- **Constraints**: No internet on compute nodes
- **Queue time**: Variable

---

## 2. Data Requirements

### Local CPU
- **Shards**: 4 (~400MB)
- **Tokenizer training**: 1B characters
- **Pretraining data**: ~1B tokens
- **Download time**: ~5-10 minutes
- **Download location**: Automatic during script

### Production GPU
- **Shards**: 240 (~24GB)
- **Tokenizer training**: 2B characters
- **Pretraining data**: ~11B tokens
- **Download time**: ~30-60 minutes
- **Download location**: Automatic during script

### Ptolemy HPC
- **Shards**: 240 (~24GB)
- **Tokenizer training**: 2B characters
- **Pretraining data**: ~11B tokens
- **Download time**: ~30-60 minutes
- **Download location**: **MUST run on devel node first**
  - SSH to ptolemy-devel-1 (has internet)
  - Run `bash scripts/download_data.sh`
  - Data cached in `/scratch/ptolemy/users/$USER/`
  - Compute nodes have NO internet access

---

## 3. Model Architecture

### Local CPU
```bash
--depth=4                 # 4 layers
--max_seq_len=1024       # 1024 token context
--device_batch_size=1    # 1 sequence at a time
--total_batch_size=1024  # 1024 tokens per step
# Result: ~8M parameters
```

### Production GPU
```bash
--depth=20                # 20 layers
--max_seq_len=2048       # 2048 token context
--device_batch_size=32   # 32 sequences per GPU
--total_batch_size=524288  # 524K tokens per step
# Result: ~561M parameters
```

### Ptolemy HPC
```bash
--depth=20                # 20 layers
--max_seq_len=2048       # 2048 token context
--device_batch_size=32   # 32 sequences per GPU
--total_batch_size=524288  # 524K tokens per step
# Result: ~561M parameters
```

---

## 4. Training Iterations

### Local CPU
- **Base pretraining**: 50 iterations
- **Midtraining**: 100 iterations
- **Supervised finetuning**: 100 iterations
- **Total optimization steps**: 250

### Production GPU
- **Base pretraining**: ~5000 iterations (calculated)
- **Midtraining**: ~1000 iterations
- **Supervised finetuning**: ~500 iterations
- **Total optimization steps**: ~6500

### Ptolemy HPC
- **Base pretraining**: ~5000 iterations (calculated)
- **Midtraining**: ~1000 iterations
- **Supervised finetuning**: ~500 iterations
- **Total optimization steps**: ~6500

---

## 5. Training Time Breakdown

### Local CPU (1-3 hours total)
| Phase | Time |
|-------|------|
| Setup & data download | 10-15 min |
| Tokenizer training | 10-15 min |
| Base pretraining | 30-60 min |
| Midtraining | 15-30 min |
| Supervised finetuning | 15-30 min |
| Evaluation & report | 5-10 min |

### Production GPU (4 hours total)
| Phase | Time |
|-------|------|
| Setup & data download | 30-60 min |
| Tokenizer training | 15-20 min |
| Base pretraining | 2-2.5 hours |
| Midtraining | 30-45 min |
| Supervised finetuning | 20-30 min |
| Evaluation & report | 10-15 min |

### Ptolemy HPC (12 hours total)
| Phase | Time |
|-------|------|
| Data download (devel node) | 30-60 min |
| Queue wait | Variable |
| Tokenizer evaluation | 5-10 min |
| Base pretraining | 8-9 hours |
| Midtraining | 1-2 hours |
| Supervised finetuning | 45-90 min |
| Evaluation & report | 15-20 min |

*Note: A100s are slower than H100s, hence longer time*

---

## 6. Expected Performance

### Local CPU
```
| Metric          | BASE     | MID      | SFT      |
|-----------------|----------|----------|----------|
| CORE            | 0.15     | -        | -        |
| ARC-Challenge   | -        | 0.20     | 0.21     |
| ARC-Easy        | -        | 0.25     | 0.28     |
| GSM8K           | -        | 0.01     | 0.02     |
| HumanEval       | -        | 0.00     | 0.02     |
| MMLU            | -        | 0.24     | 0.25     |
| ChatCORE        | -        | 0.05     | 0.07     |
```
**Quality**: Like a kindergartener - cute but not useful

### Production GPU
```
| Metric          | BASE     | MID      | SFT      |
|-----------------|----------|----------|----------|
| CORE            | 0.22     | -        | -        |
| ARC-Challenge   | -        | 0.29     | 0.28     |
| ARC-Easy        | -        | 0.36     | 0.39     |
| GSM8K           | -        | 0.03     | 0.05     |
| HumanEval       | -        | 0.07     | 0.09     |
| MMLU            | -        | 0.31     | 0.32     |
| ChatCORE        | -        | 0.07     | 0.09     |
```
**Quality**: Slightly outperforms GPT-2 (2019)

### Ptolemy HPC
```
| Metric          | BASE     | MID      | SFT      |
|-----------------|----------|----------|----------|
| CORE            | 0.22     | -        | -        |
| ARC-Challenge   | -        | 0.29     | 0.28     |
| ARC-Easy        | -        | 0.36     | 0.39     |
| GSM8K           | -        | 0.03     | 0.05     |
| HumanEval       | -        | 0.07     | 0.09     |
| MMLU            | -        | 0.31     | 0.32     |
| ChatCORE        | -        | 0.07     | 0.09     |
```
**Quality**: Same as production GPU (same config)

---

## 7. Complete Workflows

### Local CPU Workflow
```bash
# 1. Clone repo (if needed)
git clone <repo_url>
cd slurm-nanochat

# 2. Run training (one command!)
bash scripts/local_cpu_train.sh

# 3. Wait 1-3 hours

# 4. Review results
cat report.md

# 5. Chat with model
source .venv/bin/activate
python -m scripts.chat_cli
```

**Pros:**
- ✅ Simple one-command setup
- ✅ Run anywhere
- ✅ Fast to complete
- ✅ Great for learning
- ✅ No cost

**Cons:**
- ❌ Poor model quality
- ❌ Not useful for production
- ❌ Slow per iteration

### Production GPU Workflow
```bash
# 1. Rent 8xH100 node (Lambda, RunPod, etc.)

# 2. SSH into node

# 3. Clone repo
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 4. Run speedrun script
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 5. Wait ~4 hours (detach with Ctrl+a d)

# 6. Review results
cat report.md

# 7. Serve web interface
python -m scripts.chat_web
# Access via public IP: http://<node-ip>:8000
```

**Pros:**
- ✅ Good model quality
- ✅ Relatively fast
- ✅ Production-ready
- ✅ Standard cloud workflow

**Cons:**
- ❌ Costs ~$100
- ❌ Requires cloud account
- ❌ Need to manage VM

### Ptolemy HPC Workflow
```bash
# 1. SSH to development node (has internet!)
ssh username@ptolemy-devel-1.arc.msstate.edu

# 2. Navigate to scratch space
cd /scratch/ptolemy/users/$USER

# 3. Clone repo
git clone <repo_url> slurm-nanochat
cd slurm-nanochat

# 4. Setup environment (one time)
bash scripts/setup_environment.sh

# 5. Download data (MUST do on devel node!)
bash scripts/download_data.sh
# Wait 30-60 minutes for ~24GB download

# 6. Submit SLURM job (CRITICAL: Set WANDB_RUN!)
WANDB_RUN=my_training_run sbatch scripts/speedrun.slurm

# 7. Monitor job
squeue -u $USER
tail -f logs/nanochat_speedrun_<jobid>.out

# 8. Wait ~12 hours (queue + training)

# 9. Review results
cat report.md
```

**Pros:**
- ✅ Free (class allocation)
- ✅ Good model quality
- ✅ Learn HPC workflow
- ✅ Professional environment

**Cons:**
- ❌ Complex setup
- ❌ No internet on compute nodes
- ❌ Queue wait times
- ❌ SLURM learning curve
- ❌ Longer training (A100 vs H100)

---

## 8. Use Cases

### When to Use Local CPU
- ✅ Learning the codebase
- ✅ Understanding training phases
- ✅ Testing code changes quickly
- ✅ Debugging training scripts
- ✅ No GPU access available
- ✅ First time with the repo
- ✅ Analyzing the training pipeline

### When to Use Production GPU
- ✅ Need a usable model quickly
- ✅ Have budget (~$100)
- ✅ Want GPT-2 grade performance
- ✅ Production deployment
- ✅ Research experiments
- ✅ Comfortable with cloud VMs

### When to Use Ptolemy HPC
- ✅ Class assignment (required!)
- ✅ Learning HPC workflow
- ✅ No budget for cloud GPUs
- ✅ MSU student with access
- ✅ Practice with SLURM
- ✅ Need good model for free

---

## 9. Storage Locations

### Local CPU
```
~/.cache/nanochat/                 # All artifacts
├── base_data/                     # 4 shards (~400MB)
├── tokenizer/                     # Tokenizer
├── models/                        # Checkpoints
├── eval_bundle/                   # Eval data
└── report/                        # Reports

./                                 # Repo root
├── .venv/                         # Python venv
├── report.md                      # Final report
└── speedrun.log (optional)        # If using screen
```

### Production GPU
```
~/.cache/nanochat/                 # All artifacts
├── base_data/                     # 240 shards (~24GB)
├── tokenizer/                     # Tokenizer
├── models/                        # Checkpoints
├── eval_bundle/                   # Eval data
└── report/                        # Reports

./                                 # Repo root
├── .venv/                         # Python venv
├── report.md                      # Final report
└── speedrun.log                   # Screen log
```

### Ptolemy HPC
```
/scratch/ptolemy/users/$USER/nanochat-cache/
├── base_data/                     # 240 shards (~24GB)
├── tokenizer/                     # Tokenizer
├── models/                        # Checkpoints
├── eval_bundle/                   # Eval data
└── report/                        # Reports

/scratch/ptolemy/users/$USER/slurm-nanochat/
├── nanochat-venv/                 # Python venv
├── logs/                          # SLURM logs
├── report.md                      # Final report
└── .env.local                     # Email config
```

---

## Recommendation by Goal

### Goal: Learn the codebase
→ **Use Local CPU** (`scripts/local_cpu_train.sh`)
- Fast iteration
- Complete pipeline
- No cost
- See: `docs/tutorials/local-cpu-quickstart.md`

### Goal: Get a working chatbot quickly
→ **Use Production GPU** (`speedrun.sh` on Lambda/RunPod)
- 4 hours to completion
- Good model quality
- Simple cloud workflow
- See main `README.md`

### Goal: Complete class assignment
→ **Use Ptolemy HPC** (`scripts/speedrun.slurm`)
- Free for students
- Learn HPC workflow
- Required for class
- See: `docs/how-to/setup-environment.md`

### Goal: Experiment with modifications
→ **Use Local CPU first**, then GPU
- Test on CPU (fast, cheap)
- Validate on GPU (quality check)
- Iterate quickly

### Goal: Research / scaling laws
→ **Use Production GPU**
- Flexible configuration
- Quick turnaround
- Multiple experiments
- Cost scales with usage

---

## Summary

| Setup | Best For | Time | Cost | Quality |
|-------|----------|------|------|---------|
| **Local CPU** | Learning | 1-3h | Free | Poor |
| **Production GPU** | Production | 4h | $100 | Good |
| **Ptolemy HPC** | Class | 12h | Free | Good |

**Recommended path:**
1. Start with **Local CPU** to understand the pipeline
2. Read the generated `report.md` and study the code
3. If needed, scale up to **Production GPU** or **Ptolemy HPC**

---

## Next Steps

- Local CPU training: See `docs/tutorials/local-cpu-quickstart.md`
- Production GPU: See main `README.md`
- Ptolemy HPC: See `docs/how-to/setup-environment.md`
- Code analysis: See `docs/how-to/analyze-results.md`
