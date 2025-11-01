# Tutorial: Local CPU Quickstart

**Goal:** Train your first nanochat model on CPU in 1-3 hours

**What you'll learn:**
- How to run the complete training pipeline
- What each training phase does
- How to chat with your trained model
- How the pieces fit together

**Prerequisites:**
- Python 3.10+ installed
- Rust/Cargo installed (latest nightly)
- ~2GB free disk space
- Internet connection

---

## Step 1: One-Command Training

From the repository root, run:

```bash
bash scripts/local_cpu_train.sh
```

That's it! The script handles everything automatically.

### What's Happening?

The script will:
1. ✅ Install Python dependencies via `uv`
2. ✅ Download ~400MB of training data (4 shards)
3. ✅ Train a BPE tokenizer (~10-15 min)
4. ✅ Pretrain a 4-layer model (~30-60 min)
5. ✅ Run midtraining (~15-30 min)
6. ✅ Run supervised finetuning (~15-30 min)
7. ✅ Generate a training report

**Total time: 1-3 hours** (depending on your CPU)

---

## Step 2: Understanding the Phases

While training runs, here's what each phase does:

### Phase 1: Tokenizer Training
- **Input**: Raw text (1B characters from FineWeb)
- **Output**: BPE tokenizer with 65,536 vocabulary
- **Purpose**: Learn efficient text → number encoding
- **Time**: ~10-15 minutes

### Phase 2: Base Model Pretraining
- **Input**: Tokenized training data
- **Output**: Base language model
- **Purpose**: Learn general language patterns
- **Time**: ~30-60 minutes
- **Evaluation**: CORE benchmark score

### Phase 3: Midtraining
- **Input**: Base model + conversation data
- **Output**: Model that understands chat format
- **Purpose**: Teach special tokens, tool use, turn-taking
- **Time**: ~15-30 minutes
- **Evaluation**: ARC, MMLU, HumanEval, GSM8K

### Phase 4: Supervised Finetuning (SFT)
- **Input**: Midtrained model + instruction data
- **Output**: Final chat model
- **Purpose**: Improve helpfulness and instruction-following
- **Time**: ~15-30 minutes
- **Evaluation**: Re-run all chat benchmarks

### Phase 5: Report Generation
- **Input**: All training metrics
- **Output**: `report.md` file
- **Purpose**: Document complete training results
- **Time**: ~1 minute

---

## Step 3: Review Your Results

Once training completes, check the report:

```bash
cat report.md
```

You'll see something like:

```markdown
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

**Note:** These scores are intentionally low - this is a tiny model for learning!

---

## Step 4: Chat with Your Model

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Try Single Questions

```bash
python -m scripts.chat_cli -p "Tell me a joke"
python -m scripts.chat_cli -p "What is 2+2?"
python -m scripts.chat_cli -p "Explain what a transformer is"
```

### Interactive Mode

```bash
python -m scripts.chat_cli
```

Then type your questions. Type `exit` to quit.

### Web Interface

```bash
python -m scripts.chat_web
```

Then open http://localhost:8000 in your browser.

---

## Step 5: Understand What You Built

### Model Architecture

Your tiny model:
- **4 transformer layers** (vs 20 for production)
- **~8M parameters** (vs 560M for production)
- **1024 token context** (vs 2048 for production)
- **65,536 vocabulary size** (same as production)

### Training Data

- **4 shards** (~400MB) vs 240 shards (~24GB) for production
- **~1B tokens** vs ~11B tokens for production
- **FineWeb dataset** (same as production)

### Why Is It Not Very Smart?

This is intentional! The tiny model:
- ❌ Makes lots of mistakes
- ❌ Hallucinates frequently
- ❌ Has limited knowledge
- ✅ But demonstrates the complete pipeline!

**Purpose**: Learning how it works, not creating a useful model

---

## Common Issues

### `uv: command not found`

The script adds `uv` to PATH automatically. If this persists:

```bash
export PATH="$HOME/.local/bin:$PATH"
bash scripts/local_cpu_train.sh
```

### Rust `edition2024` Error

Update Rust:

```bash
rustup update
rustc --version  # Should be 1.93.0-nightly or newer
```

### Out of Memory

Edit `scripts/local_cpu_train.sh` to reduce model size:

```bash
--depth=2              # Smaller model
--max_seq_len=512      # Shorter sequences
```

### Too Slow

Reduce iterations in `scripts/local_cpu_train.sh`:

```bash
--num_iterations=20    # Fewer training steps
```

---

## Next Steps

### 1. Study the Code

While the training is fresh in your mind:

- `nanochat/gpt.py` - Transformer architecture
- `scripts/base_train.py` - Training loop
- `nanochat/tokenizer.py` - BPE tokenization
- `nanochat/dataloader.py` - Data loading

### 2. Experiment

Try modifying hyperparameters in `scripts/local_cpu_train.sh`:

```bash
# Deeper model
--depth=6

# Larger model dimension
--width=512

# Different learning rate
--matrix_lr=0.02
```

### 3. Compare

See how local CPU compares to GPU:

[Training Environment Comparison](../explanation/training-environments.md)

### 4. Scale Up

When ready, try:
- **Ptolemy HPC**: [HPC First Job](02-hpc-first-job.md)
- **Production GPU**: See main [README.md](../../README.md)

---

## What You Learned

✅ How to run the complete training pipeline
✅ What each training phase does
✅ How to evaluate and chat with your model
✅ The difference between tiny and production models
✅ How to troubleshoot common issues

---

## File Locations

All training artifacts are saved to `~/.cache/nanochat/`:

```
~/.cache/nanochat/
├── base_data/              # Training data (4 shards)
├── tokenizer/              # Trained BPE tokenizer
├── models/                 # Model checkpoints
│   ├── base/              # Base model
│   ├── mid/               # Midtrained model
│   └── sft/               # SFT model (final)
├── eval_bundle/            # Evaluation datasets
└── report/                 # Training logs

./report.md                 # Final report (copied to repo)
./.venv/                    # Python virtual environment
```

---

**Congratulations!** You've trained your first ChatGPT-style model from scratch!

Next tutorial: [HPC First Job](02-hpc-first-job.md) - Scale up to production quality
