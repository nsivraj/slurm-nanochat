# Local CPU Training for nanochat - Complete Guide

## 🎯 Purpose

This directory contains everything you need to run nanochat training **locally on your CPU** with minimal data. This setup is specifically designed for **learning and understanding the code**, not for training production-quality models.

## 📚 Documentation Index

We've created a complete set of documentation for local CPU training:

1. **[LOCAL_CPU_QUICKSTART.md](LOCAL_CPU_QUICKSTART.md)** - Start here! One command to run everything
2. **[LOCAL_CPU_TRAINING.md](LOCAL_CPU_TRAINING.md)** - Comprehensive guide with all details
3. **[LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)** - Deep dive into the codebase
4. **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** - Compare CPU vs GPU vs HPC training

## 🚀 Quick Start (TL;DR)

```bash
# From the repo root, run this ONE command:
bash scripts/local_cpu_train.sh
```

That's it! Wait 1-3 hours and you'll have a complete (tiny) trained chatbot.

## 📖 What You'll Learn

After running local CPU training, you'll understand:

✅ **Tokenizer Training** - How BPE tokenization works
✅ **Base Model Pretraining** - The Transformer architecture and training loop
✅ **Midtraining** - Teaching conversation format and special tokens
✅ **Supervised Finetuning** - Instruction following and helpfulness
✅ **Evaluation** - Benchmarking (CORE, ARC, MMLU, GSM8K, HumanEval)
✅ **Inference** - KV caching and text generation
✅ **Complete Pipeline** - From raw data to chatbot

## 🎓 Learning Path

### Step 1: Run the Training (1-3 hours)
```bash
bash scripts/local_cpu_train.sh
```

Watch it progress through all 5 phases:
1. Tokenizer training
2. Base model pretraining
3. Midtraining
4. Supervised finetuning
5. Report generation

### Step 2: Review Results
```bash
# Read the training report
cat report.md

# Check what was created
ls ~/.cache/nanochat/
```

### Step 3: Chat with Your Model
```bash
# Activate environment
source .venv/bin/activate

# Try a prompt
python -m scripts.chat_cli -p "Why is the sky blue?"

# Or use interactive mode
python -m scripts.chat_cli

# Or start web interface
python -m scripts.chat_web
# Then open: http://localhost:8000
```

### Step 4: Analyze the Code

Follow **[LOCAL_CPU_ANALYSIS_GUIDE.md](LOCAL_CPU_ANALYSIS_GUIDE.md)** to:
- Understand each training script
- Trace the Transformer architecture
- Study the training loop
- Learn about optimizers, data loaders, and evaluation

### Step 5: Compare with Production

Read **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** to understand:
- How local CPU differs from GPU training
- Why your model is tiny and not very smart
- When to use each training method
- How to scale up to production

## 📊 What to Expect

### Your Model Will:
- ✅ Complete training successfully (all phases)
- ✅ Generate grammatically correct text
- ✅ Respond to questions (often incorrectly)
- ✅ Show basic conversational ability
- ✅ Demonstrate you understand the pipeline

### Your Model Will NOT:
- ❌ Be very smart (it's tiny!)
- ❌ Score well on benchmarks (expected)
- ❌ Compete with ChatGPT/GPT-4
- ❌ Be useful for production

**This is normal and expected!** You're training an ~8M parameter model on minimal data. The purpose is **learning**, not production.

## 🔧 What Was Created

### New Training Script
- **`scripts/local_cpu_train.sh`** - Complete CPU training pipeline
  - Downloads only 4 data shards (~400MB vs 24GB)
  - Trains tiny 4-layer model (~8M params vs 561M)
  - Runs 50 base iterations (vs ~5000)
  - Completes in 1-3 hours
  - No modifications to existing GPU/SLURM scripts

### New Documentation
- **`LOCAL_CPU_QUICKSTART.md`** - Quick reference guide
- **`LOCAL_CPU_TRAINING.md`** - Comprehensive training guide
- **`LOCAL_CPU_ANALYSIS_GUIDE.md`** - Code analysis guide
- **`TRAINING_COMPARISON.md`** - CPU vs GPU vs HPC comparison
- **`LOCAL_CPU_README.md`** - This file (overview)

### Preserved Existing Files
✅ No changes to `speedrun.sh` (production GPU)
✅ No changes to `scripts/speedrun.slurm` (Ptolemy HPC)
✅ No changes to `PTOLEMY_SETUP.md` (HPC docs)
✅ No changes to any training scripts
✅ Complete separation of concerns

## 🎯 Key Differences: CPU vs GPU Training

| Aspect | Local CPU | Production GPU | Ptolemy HPC |
|--------|-----------|----------------|-------------|
| **Purpose** | Learning | Production | Class work |
| **Time** | 1-3 hours | 4 hours | 12 hours |
| **Cost** | Free | ~$100 | Free |
| **Quality** | Poor | Good | Good |
| **Data** | 400MB | 24GB | 24GB |
| **Model** | 4 layers, 8M params | 20 layers, 561M params | 20 layers, 561M params |
| **Setup** | One command | One command | Multi-step |
| **Script** | `scripts/local_cpu_train.sh` | `speedrun.sh` | `scripts/speedrun.slurm` |

See **[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)** for complete details.

## 🗂️ Files Created

```
slurm-nanochat/
├── scripts/
│   └── local_cpu_train.sh              # NEW: CPU training script
├── LOCAL_CPU_README.md                 # NEW: This file
├── LOCAL_CPU_QUICKSTART.md             # NEW: Quick start guide
├── LOCAL_CPU_TRAINING.md               # NEW: Full training guide
├── LOCAL_CPU_ANALYSIS_GUIDE.md         # NEW: Code analysis
├── TRAINING_COMPARISON.md              # NEW: Compare methods
│
# Existing files unchanged:
├── speedrun.sh                         # UNCHANGED: GPU training
├── scripts/speedrun.slurm              # UNCHANGED: HPC training
├── PTOLEMY_SETUP.md                    # UNCHANGED: HPC docs
├── README.md                           # UNCHANGED: Main docs
└── ... (all other files unchanged)
```

## ⚙️ Technical Details

### Training Configuration

#### Phase 1: Tokenizer
- Data: 4 shards (~1B characters)
- Vocab size: 65,536 (2^16)
- Algorithm: BPE (Byte Pair Encoding)
- Output: `~/.cache/nanochat/tokenizer/tokenizer.pkl`

#### Phase 2: Base Pretraining
- Model: 4 layers, 12 heads, 768 hidden dim
- Parameters: ~8 million
- Context: 1024 tokens
- Iterations: 50
- Batch size: 1024 tokens per step
- Output: `~/.cache/nanochat/models/base_model.pt`

#### Phase 3: Midtraining
- Input: Base model + conversational data
- Dataset: SmolTalk + identity conversations
- Iterations: 100
- Purpose: Learn conversation format
- Output: `~/.cache/nanochat/models/mid_model.pt`

#### Phase 4: Supervised Finetuning
- Input: Midtrained model + instruction data
- Iterations: 100
- Purpose: Improve instruction following
- Output: `~/.cache/nanochat/models/sft_model.pt`

#### Phase 5: Report Generation
- Compiles all metrics
- Creates markdown report
- Output: `./report.md`

### System Requirements

- **CPU**: Any modern processor (multi-core recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for data and models
- **OS**: Linux, macOS, or WSL on Windows
- **Python**: 3.10 or higher
- **Internet**: Required for initial downloads

### What Gets Downloaded

1. **Training data**: 4 shards (~400MB compressed)
2. **Evaluation bundle**: CORE benchmark data (~162MB)
3. **Identity conversations**: Personality data (~2.3MB)
4. **SmolTalk dataset**: Conversational data (~500MB)
5. **Python packages**: PyTorch, etc. (varies)
6. **Rust/Cargo**: For tokenizer (varies)

## 🐛 Troubleshooting

### Out of Memory
```bash
# Edit scripts/local_cpu_train.sh
# Reduce depth: --depth=2 (instead of 4)
# Reduce context: --max_seq_len=512 (instead of 1024)
```

### Too Slow
```bash
# Edit scripts/local_cpu_train.sh
# Reduce iterations: --num_iterations=20 (instead of 50)
```

### Download Fails
- Check internet connection
- Retry the script (it's idempotent)
- Downloads resume automatically

### Script Errors
- Check Python version: `python --version` (need 3.10+)
- Check disk space: `df -h ~`
- Read error messages carefully

## 📞 Getting Help

1. **Check documentation**: Read the guides mentioned above
2. **Review error messages**: They usually indicate the problem
3. **Check troubleshooting**: See sections in each guide
4. **Compare scripts**: `diff scripts/local_cpu_train.sh dev/runcpu.sh`
5. **Read main docs**: `cat README.md`

## 🎓 After Local Training

Once you've completed local CPU training and understand the pipeline:

### Option 1: Experiment Locally
- Modify hyperparameters in the script
- Try different model depths
- Adjust iteration counts
- Create custom identity conversations
- Add your own evaluation tasks

### Option 2: Scale to GPU
- Run production training: `bash speedrun.sh` (cloud)
- Or HPC training: `sbatch scripts/speedrun.slurm` (Ptolemy)
- Apply what you learned to larger models
- See actual good performance

### Option 3: Deep Code Analysis
- Follow **LOCAL_CPU_ANALYSIS_GUIDE.md**
- Trace every training step
- Understand the Transformer architecture
- Study the optimization process
- Learn distributed training concepts

## 🎯 Success Criteria

You'll know you're successful when you can:

✅ Run the full training pipeline locally
✅ Understand what each phase does
✅ Explain the training report metrics
✅ Chat with your trained model
✅ Trace code execution through training scripts
✅ Understand why your model is small and weak
✅ Know how to scale to production

## 📚 Additional Resources

### In This Repo
- Main README: `cat README.md`
- GPU training: `cat speedrun.sh`
- HPC training: `cat PTOLEMY_SETUP.md`
- Original CPU demo: `cat dev/runcpu.sh`

### External
- [nanochat repo](https://github.com/karpathy/nanochat)
- [Andrej Karpathy's introduction](https://github.com/karpathy/nanochat/discussions/1)
- [Transformer paper](https://arxiv.org/abs/1706.03762)
- [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 🎉 Conclusion

You now have everything you need to:
1. ✅ Run local CPU training
2. ✅ Understand the complete pipeline
3. ✅ Analyze the codebase
4. ✅ Compare different training methods
5. ✅ Scale to production when ready

**Start with**: `bash scripts/local_cpu_train.sh`

**Happy learning!** 🚀

---

## Quick Links

- **[Quick Start](LOCAL_CPU_QUICKSTART.md)** - One command to run everything
- **[Full Guide](LOCAL_CPU_TRAINING.md)** - Complete documentation
- **[Code Analysis](LOCAL_CPU_ANALYSIS_GUIDE.md)** - Understand the code
- **[Comparison](TRAINING_COMPARISON.md)** - CPU vs GPU vs HPC

**Questions?** Check the guides above or read `README.md`.
