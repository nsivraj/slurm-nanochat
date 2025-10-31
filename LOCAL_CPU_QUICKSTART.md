# Local CPU Training - Quick Start Guide

This is a condensed guide to get you started quickly with local CPU training.

## One-Command Setup

```bash
bash scripts/local_cpu_train.sh
```

That's it! The script handles everything automatically.

## What Happens

The script will:
1. ✅ Install dependencies (Python packages, Rust, etc.)
2. ✅ Download ~400MB of training data (4 shards)
3. ✅ Train a tokenizer (~10-15 min)
4. ✅ Pretrain a 4-layer model (~30-60 min)
5. ✅ Run midtraining (~15-30 min)
6. ✅ Run supervised finetuning (~15-30 min)
7. ✅ Generate a training report (~1 min)

**Total time: 1-3 hours**

## After Training

### View the Report

```bash
cat report.md
```

### Chat with Your Model

```bash
# Activate the virtual environment
source .venv/bin/activate

# Single question
python -m scripts.chat_cli -p "Tell me a joke"

# Interactive mode
python -m scripts.chat_cli
```

### Start Web Interface

```bash
source .venv/bin/activate
python -m scripts.chat_web
```

Then open: http://localhost:8000

## Expected Results

Your model will:
- ✅ Complete all training phases successfully
- ✅ Generate coherent (but often wrong) responses
- ✅ Demonstrate basic conversational ability
- ❌ NOT be very smart (it's tiny!)
- ❌ NOT score well on benchmarks (expected)
- ❌ Make lots of mistakes and hallucinate

**This is normal!** The purpose is to learn the pipeline, not create a production model.

## Key Differences from Production

| Aspect | Production (GPU) | Local CPU |
|--------|------------------|-----------|
| Model size | 561M params (20 layers) | ~8M params (4 layers) |
| Training data | 24GB (240 shards) | 400MB (4 shards) |
| Training time | ~4 hours on 8xH100 | 1-3 hours on CPU |
| Hardware cost | ~$100 | Free (your CPU) |
| Result quality | Outperforms GPT-2 | Like a kindergartener |

## Understanding the Training Phases

### 1. Tokenizer Training
- **Input**: Raw text data (4 shards, ~1B characters)
- **Output**: BPE tokenizer with 65,536 vocabulary
- **Purpose**: Convert text to tokens efficiently

### 2. Base Model Pretraining
- **Input**: Tokenized training data
- **Output**: Base language model (understands text patterns)
- **Purpose**: Learn general language understanding
- **Evaluation**: CORE benchmark

### 3. Midtraining
- **Input**: Base model + conversational data
- **Output**: Model that understands conversation format
- **Purpose**: Teach special tokens, tool use, multi-turn chat
- **Evaluation**: ARC, MMLU, HumanEval, GSM8K

### 4. Supervised Finetuning (SFT)
- **Input**: Midtrained model + instruction data
- **Output**: Final chat model
- **Purpose**: Improve instruction-following and helpfulness
- **Evaluation**: Re-run all chat benchmarks

### 5. Report Generation
- **Input**: All training metrics
- **Output**: Markdown report with tables
- **Purpose**: Document training results

## Analyzing the Code

After training, study these key files:

### Training Scripts (in `scripts/`)
```bash
# Tokenizer
scripts/tok_train.py      # Train the tokenizer
scripts/tok_eval.py       # Evaluate compression

# Base model
scripts/base_train.py     # Pretraining loop
scripts/base_eval.py      # CORE evaluation
scripts/base_loss.py      # Bits per byte

# Chat model
scripts/mid_train.py      # Midtraining
scripts/chat_sft.py       # Supervised finetuning
scripts/chat_eval.py      # Chat benchmarks

# Inference
scripts/chat_cli.py       # CLI chat interface
scripts/chat_web.py       # Web UI
```

### Core Library (in `nanochat/`)
```bash
nanochat/gpt.py           # Transformer architecture
nanochat/tokenizer.py     # BPE tokenizer wrapper
nanochat/dataloader.py    # Data loading
nanochat/engine.py        # Inference engine (KV cache)
nanochat/adamw.py         # Optimizer (embeddings)
nanochat/muon.py          # Optimizer (matrices)
nanochat/checkpoint_manager.py  # Save/load models
nanochat/report.py        # Report generation
```

### Tasks (in `tasks/`)
```bash
tasks/arc.py              # Science questions
tasks/gsm8k.py            # Math problems
tasks/humaneval.py        # Coding tasks
tasks/mmlu.py             # General knowledge
tasks/smoltalk.py         # Conversational data
```

## Troubleshooting

### `uv: command not found`
The script automatically adds `uv` to PATH. If you still see this error:
```bash
export PATH="$HOME/.local/bin:$PATH"
bash scripts/local_cpu_train.sh
```

### Rust `edition2024` Error
Update Rust to the latest version:
```bash
rustup update
rustc --version  # Should be 1.93.0-nightly or newer
bash scripts/local_cpu_train.sh
```

### Out of Memory
Reduce model size or sequence length in the script:
```bash
# Edit scripts/local_cpu_train.sh
--depth=2              # Smaller model
--max_seq_len=512      # Shorter sequences
```

### Too Slow
Reduce iterations:
```bash
# Edit scripts/local_cpu_train.sh
--num_iterations=20    # Fewer steps
```

### Download Fails
Check internet connection and retry. The script is idempotent.

## Next Steps

1. **Read the full guide**: `cat LOCAL_CPU_TRAINING.md`
2. **Experiment**: Modify hyperparameters in the script
3. **Compare**: See differences with `diff scripts/local_cpu_train.sh speedrun.sh`
4. **Study code**: Read the training scripts while the model is fresh in your mind
5. **Scale up**: Once comfortable, try GPU training with `bash speedrun.sh`

## File Locations

```
~/.cache/nanochat/          # All training artifacts
├── base_data/              # Training data (*.parquet)
├── tokenizer/              # Trained tokenizer
├── models/                 # Model checkpoints
├── eval_bundle/            # Evaluation data
└── report/                 # Training logs

./report.md                 # Final report (copied here)
./.venv/                    # Python virtual environment
```

## Questions?

See `LOCAL_CPU_TRAINING.md` for detailed explanations, or review the main `README.md`.
