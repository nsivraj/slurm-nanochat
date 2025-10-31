# Local CPU Training Guide for nanochat

This guide explains how to run the complete nanochat training pipeline locally on a CPU with minimal data. This setup is designed for learning and understanding the code, not for training production-quality models.

## Overview

This local CPU setup allows you to:
- Train with only **4-8 data shards** (~400-800MB instead of 24GB)
- Use a **tiny 4-layer model** (vs 20-layer production model)
- Complete training in **1-3 hours** on a modern CPU
- Experience **all training phases**: tokenizer training, base pretraining, midtraining, supervised finetuning, and report generation
- Understand the complete training workflow from start to finish

## Prerequisites

- Python 3.10 or higher
- Rust/Cargo (installed automatically by the script)
- ~2GB of disk space for data and model artifacts
- ~8GB RAM minimum (16GB recommended)
- Internet connection for downloading data and dependencies

## Quick Start

### Step 1: Run the Local CPU Training Script

From the repository root, execute:

```bash
bash scripts/local_cpu_train.sh
```

This single script will:
1. Set up the Python virtual environment with `uv`
2. Install Rust and build the tokenizer
3. Download minimal training data (4 shards, ~400MB)
4. Train the tokenizer on 1B characters
5. Pretrain a 4-layer base model (50 iterations)
6. Perform midtraining (100 iterations)
7. Run supervised finetuning (100 iterations)
8. Generate a comprehensive report

### Step 2: Monitor Progress

The script will display progress for each phase:
```
==================================================
[1/5] Training Tokenizer...
==================================================
Training tokenizer on ~1B characters...

==================================================
[2/5] Base Model Pretraining...
==================================================
Training tiny d4 model (4 layers, ~8M parameters)...
```

Expect the full run to take **1-3 hours** depending on your CPU.

### Step 3: Review Results

After completion, check:
1. **Training report**: `cat report.md`
2. **Model artifacts**: `ls ~/.cache/nanochat/`
3. **Training logs**: Review terminal output

## Training Phases Explained

### Phase 1: Tokenizer Training (~10-15 minutes)
- Downloads 4 data shards (~400MB compressed text)
- Trains BPE tokenizer on 1B characters
- Evaluates compression ratio vs GPT-2/GPT-4 tokenizers
- Output: `~/.cache/nanochat/tokenizer/tokenizer.pkl`

### Phase 2: Base Model Pretraining (~30-60 minutes)
- Trains a 4-layer Transformer (depth=4)
- Uses 1024 token context length (vs 2048 production)
- Runs 50 optimization steps
- Evaluates on CORE benchmark (subset)
- Output: `~/.cache/nanochat/models/base_model.pt`

### Phase 3: Midtraining (~15-30 minutes)
- Downloads SmolTalk conversational dataset
- Downloads synthetic identity conversations
- Teaches model conversation format and special tokens
- Runs 100 optimization steps
- Evaluates on chat benchmarks (ARC, MMLU, HumanEval, etc.)
- Output: `~/.cache/nanochat/models/mid_model.pt`

### Phase 4: Supervised Finetuning (~15-30 minutes)
- Fine-tunes on high-quality instruction-following data
- Runs 100 optimization steps
- Re-evaluates on all benchmarks
- Output: `~/.cache/nanochat/models/sft_model.pt`

### Phase 5: Report Generation (~1 minute)
- Compiles all metrics and evaluations
- Generates markdown report with tables
- Output: `./report.md`

## Understanding the Training Parameters

The local CPU script uses significantly reduced parameters compared to production:

| Parameter | Production (GPU) | Local CPU | Purpose |
|-----------|-----------------|-----------|---------|
| depth | 20 layers | 4 layers | Model size |
| max_seq_len | 2048 tokens | 1024 tokens | Context window |
| device_batch_size | 32 | 1 | Memory usage |
| total_batch_size | 524288 | 1024 | Training efficiency |
| num_iterations | ~5000 | 50 (base) | Training steps |
| data shards | 240 (~24GB) | 4 (~400MB) | Training data |
| training time | ~4 hours (8xH100) | ~1-3 hours (CPU) | Wall clock time |

## Chatting with Your Model

After training completes, you can interact with your model:

### Option 1: Command Line Interface

```bash
# Activate the virtual environment
source .venv/bin/activate

# Single prompt
python -m scripts.chat_cli -p "Why is the sky blue?"

# Interactive mode
python -m scripts.chat_cli
```

### Option 2: Web Interface

```bash
# Activate the virtual environment
source .venv/bin/activate

# Start web server
python -m scripts.chat_web
```

Then open your browser to `http://localhost:8000`

## Expected Results

Your tiny 4-layer CPU-trained model will have:
- **~8 million parameters** (vs 561M for production d20 model)
- **Poor performance** on benchmarks (this is expected!)
- **Basic conversational ability** (like a kindergartener)
- **Frequent mistakes and hallucinations**

**This is normal!** The goal is to understand the training pipeline, not to create a production model.

Example report metrics you might see:

| Metric | BASE | MID | SFT |
|--------|------|-----|-----|
| CORE | 0.15 | - | - |
| ARC-Challenge | - | 0.20 | 0.21 |
| ARC-Easy | - | 0.25 | 0.28 |
| GSM8K | - | 0.01 | 0.02 |
| HumanEval | - | 0.00 | 0.02 |
| MMLU | - | 0.24 | 0.25 |

## Troubleshooting

### Issue: Out of Memory

**Solution**: Edit `scripts/local_cpu_train.sh` and reduce parameters further:
```bash
# In base_train section, change:
--depth=4          # Try 3 or 2
--max_seq_len=1024 # Try 512
```

### Issue: Training is Too Slow

**Solution**: Reduce iterations in `scripts/local_cpu_train.sh`:
```bash
# In base_train section:
--num_iterations=50  # Try 20 or 10

# In mid_train section:
--num_iterations=100  # Try 50

# In chat_sft section:
--num_iterations=100  # Try 50
```

### Issue: Download Failures

**Solution**: Check internet connection and retry. The script is idempotent and will skip already-downloaded data.

### Issue: Rust/Cargo Installation Fails

**Solution**: Manually install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

## Analyzing the Code

After running the training, you can examine:

### Key Training Scripts
- `scripts/tok_train.py` - Tokenizer training logic
- `scripts/base_train.py` - Base model pretraining
- `scripts/mid_train.py` - Midtraining implementation
- `scripts/chat_sft.py` - Supervised finetuning

### Core Model Files
- `nanochat/gpt.py` - Transformer model architecture
- `nanochat/dataloader.py` - Data loading and tokenization
- `nanochat/engine.py` - Inference engine with KV cache
- `nanochat/tokenizer.py` - BPE tokenizer wrapper

### Evaluation Scripts
- `scripts/tok_eval.py` - Tokenizer compression evaluation
- `scripts/base_eval.py` - CORE benchmark evaluation
- `scripts/chat_eval.py` - Chat model benchmarks
- `scripts/base_loss.py` - Bits per byte calculation

### Data and Reports
- `nanochat/dataset.py` - Dataset downloading and reading
- `nanochat/report.py` - Report generation utilities

## Comparing with Production Training

To see the differences between local CPU and production GPU training:

```bash
# Compare parameters
diff scripts/local_cpu_train.sh speedrun.sh

# Read production setup
cat PTOLEMY_SETUP.md

# Review SLURM configuration
cat scripts/speedrun.slurm
```

## Next Steps

1. **Experiment with hyperparameters**: Try different model depths, sequence lengths, or iteration counts
2. **Add more data**: Download more shards to improve model quality
3. **Custom identity**: Create your own identity conversations (see `dev/gen_synthetic_data.py`)
4. **Read the code**: Use the trained model as context while studying the implementation
5. **Scale up**: Once you understand the pipeline, run on GPU with `bash speedrun.sh`

## Additional Resources

- **Main README**: `cat README.md` - Full project documentation
- **CPU demo script**: `cat dev/runcpu.sh` - Original minimal CPU example
- **Production speedrun**: `cat speedrun.sh` - Full GPU training script
- **Ptolemy HPC setup**: `cat PTOLEMY_SETUP.md` - HPC cluster instructions

## File Outputs

After training, you'll have:

```
~/.cache/nanochat/
├── base_data/              # Training data shards (*.parquet)
├── tokenizer/              # Trained tokenizer
│   └── tokenizer.pkl
├── eval_bundle/            # CORE evaluation data
├── identity_conversations.jsonl  # Personality data
├── models/                 # Model checkpoints
│   ├── base_model.pt
│   ├── mid_model.pt
│   └── sft_model.pt
└── report/                 # Training metrics and logs
    └── report.md

./report.md                 # Final training report (copy)
```

## License

Same as nanochat: MIT

## Questions?

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Examine the training scripts in `scripts/`
4. Compare with `dev/runcpu.sh` for reference
