# Project Architecture: nanochat Codebase

This document explains the architecture and organization of the nanochat codebase.

---

## Overview

nanochat is a complete, end-to-end LLM training pipeline in a single, clean, hackable codebase. It implements everything needed to train a ChatGPT-style model from scratch.

**Design Philosophy:**
- Minimal dependencies
- Clean, readable code
- No complex abstractions
- Single cohesive codebase
- Maximally forkable

---

## Repository Structure

```
slurm-nanochat/
├── nanochat/                   # Core library
│   ├── gpt.py                  # Transformer model
│   ├── tokenizer.py            # BPE tokenizer wrapper
│   ├── dataloader.py           # Data loading
│   ├── engine.py               # Inference engine
│   ├── adamw.py                # AdamW optimizer
│   ├── muon.py                 # Muon optimizer
│   ├── checkpoint_manager.py   # Save/load models
│   ├── core_eval.py            # CORE benchmark
│   ├── loss_eval.py            # Bits per byte
│   ├── dataset.py              # Data downloading
│   ├── execution.py            # Python code execution
│   ├── report.py               # Report generation
│   ├── configurator.py         # Better than argparse
│   ├── common.py               # Utilities
│   └── ui.html                 # Web UI
│
├── scripts/                    # Training & eval scripts
│   ├── tok_train.py            # Train tokenizer
│   ├── tok_eval.py             # Eval tokenizer
│   ├── base_train.py           # Base pretraining
│   ├── base_eval.py            # Eval base model
│   ├── base_loss.py            # Base model sampling
│   ├── mid_train.py            # Midtraining
│   ├── chat_sft.py             # Supervised finetuning
│   ├── chat_eval.py            # Eval chat model
│   ├── chat_rl.py              # Reinforcement learning
│   ├── chat_cli.py             # CLI chat interface
│   ├── chat_web.py             # Web chat interface
│   ├── speedrun.slurm          # SLURM job script (HPC)
│   ├── local_cpu_train.sh      # Local CPU training
│   ├── setup_environment.sh    # Ptolemy setup
│   └── download_data.sh        # Data download (HPC)
│
├── tasks/                      # Evaluation tasks
│   ├── arc.py                  # Science questions
│   ├── gsm8k.py                # Math problems
│   ├── humaneval.py            # Coding tasks
│   ├── mmlu.py                 # General knowledge
│   ├── smoltalk.py             # Conversation data
│   ├── spellingbee.py          # Spelling/counting
│   ├── customjson.py           # Custom tasks
│   └── common.py               # Task utilities
│
├── rustbpe/                    # Rust BPE tokenizer
│   ├── Cargo.toml              # Rust config
│   ├── README.md               # Why Rust?
│   └── src/
│       └── lib.rs              # Tokenizer implementation
│
├── tests/                      # Test suite
│   └── test_rustbpe.py         # Tokenizer tests
│
├── docs/                       # Documentation
│   ├── index.md                # Main docs entry
│   ├── tutorials/              # Learning-oriented
│   ├── how-to/                 # Problem-oriented
│   ├── explanation/            # Understanding-oriented
│   └── reference/              # Information-oriented
│
├── experiments/                # Session logs & status
│
├── dev/                        # Development tools
│   ├── gen_synthetic_data.py  # Generate identity data
│   ├── repackage_data_reference.py  # Data processing
│   └── runcpu.sh               # Original CPU script
│
├── speedrun.sh                 # Production GPU training
├── run1000.sh                  # $1000 tier model
├── pyproject.toml              # Python dependencies
├── uv.lock                     # Dependency lock file
├── README.md                   # Main README
└── CHANGELOG.md                # Project history
```

---

## Core Components

### 1. Transformer Model (`nanochat/gpt.py`)

The heart of nanochat - implements the GPT architecture.

**Key Classes:**

**`CausalSelfAttention`** (lines 51-130)
- Multi-head self-attention
- Causal masking for autoregressive generation
- Rotary positional embeddings (RoPE)
- QK normalization
- Multi-Query Attention (MQA) support

**`MLP`** (lines 132-150)
- Feed-forward network
- ReLU² activation (not GELU)
- Up-projection → activation → down-projection

**`TransformerBlock`** (lines 152-200)
- Pre-normalization (RMSNorm)
- Attention + MLP with residual connections
- Standard transformer block architecture

**`GPT`** (lines 202-300)
- Token embeddings (`wte`)
- Positional embeddings (`wpe`) or RoPE
- Stack of transformer blocks
- Final layer norm
- Language modeling head

**Model Scaling:**
```python
depth = 20                          # Number of layers
model_dim = depth * 64              # 1,280 dimensions
num_heads = (model_dim + 127) // 128  # 10 heads
# Result: 561M parameters
```

### 2. Tokenizer (`nanochat/tokenizer.py`, `rustbpe/`)

**Rust Implementation:**
- BPE (Byte Pair Encoding)
- Trained on ~2B characters
- Vocabulary: 65,536 tokens (2^16)
- Fast training and encoding

**Python Wrapper:**
- `Tokenizer` class wraps Rust implementation
- Compatible with GPT-2/GPT-4 format
- Special tokens: `<|endoftext|>`, `<|tool|>`, etc.

**Why Rust?**
- 10-100x faster than pure Python
- Memory efficient
- Training tokenizer is computationally intensive

### 3. Data Loading (`nanochat/dataloader.py`)

**`DistributedDataLoader`:**
- Sharded data loading
- Distributed across GPUs
- Memory-mapped `.parquet` files
- Automatic batching
- Sequence packing

**Features:**
- Efficient I/O
- Deterministic shuffling
- Multi-GPU support via DDP
- Handles large datasets (24GB+)

### 4. Training Infrastructure

**Optimizers:**

**`nanochat/muon.py`:**
- Muon optimizer for matrix parameters
- Momentum-based
- Better for transformer weights

**`nanochat/adamw.py`:**
- AdamW for embeddings
- Distributed implementation
- Handles large embedding tables

**Checkpoint Management (`nanochat/checkpoint_manager.py`):**
- Save/load model states
- Optimizer state preservation
- Distributed checkpoint handling

### 5. Inference Engine (`nanochat/engine.py`)

**`TextEngine`:**
- KV cache for efficient generation
- Temperature, top-p, top-k sampling
- Streaming generation
- Tool calling support

**Features:**
- Fast autoregressive generation
- Memory-efficient caching
- Multiple sampling strategies

---

## Training Pipeline

### Phase 1: Tokenizer Training

**Script:** `scripts/tok_train.py`

**Process:**
1. Download text data
2. Train BPE tokenizer (Rust)
3. Save to `tokenizer.pkl`
4. Evaluate compression ratio

**Output:**
- Tokenizer with 65,536 vocab
- Compression metrics vs GPT-2/GPT-4

### Phase 2: Base Pretraining

**Script:** `scripts/base_train.py`

**Process:**
1. Initialize model from scratch
2. Train on raw text data
3. Optimize for next-token prediction
4. Evaluate CORE benchmark

**Key Parameters:**
- `--depth=20` (model size)
- `--num_iterations` (training steps)
- `--device_batch_size` (per-GPU batch)
- `--total_batch_size` (effective batch)

**Output:**
- Base language model
- CORE score (~0.20-0.22)

### Phase 3: Midtraining

**Script:** `scripts/mid_train.py`

**Process:**
1. Load base model
2. Train on conversational data
3. Teach special tokens and format
4. Evaluate chat benchmarks

**Datasets:**
- SmolTalk (HuggingFace)
- Identity conversations (synthetic)

**Output:**
- Conversation-aware model
- ARC, MMLU, HumanEval, GSM8K scores

### Phase 4: Supervised Finetuning (SFT)

**Script:** `scripts/chat_sft.py`

**Process:**
1. Load midtrained model
2. Train on instruction-following data
3. Improve helpfulness
4. Re-evaluate all benchmarks

**Output:**
- Final chat model
- Improved benchmark scores
- Production-ready chatbot

### Phase 5: Reinforcement Learning (Optional)

**Script:** `scripts/chat_rl.py`

**Process:**
- Policy gradient training
- Reward modeling
- Further alignment

**Note:** Not always used in basic training runs

---

## Evaluation Framework

### CORE Benchmark (`nanochat/core_eval.py`)

**Purpose:** Evaluate base model language understanding

**Method:**
- Perplexity on held-out data
- Compression efficiency
- General language capability

**Score:** 0-1 (higher is better)
- ~0.15: Tiny models (local CPU)
- ~0.20-0.22: Medium models (GPU)
- ~0.25+: Large models

### Task-Specific Evals (`tasks/`)

**ARC (Science Questions):**
- Multiple choice
- Challenge & Easy variants
- Tests reasoning

**MMLU (General Knowledge):**
- 57 subjects
- Multiple choice
- Broad knowledge test

**GSM8K (Math):**
- Grade-school math
- Chain-of-thought helpful
- Tests reasoning

**HumanEval (Coding):**
- Python coding tasks
- Functional correctness
- Tests code generation

**ChatCORE:**
- CORE benchmark for chat models
- Conversational capability

---

## Data Flow

### Training Data Flow

```
FineWeb Dataset (HuggingFace)
    ↓
Download as .parquet shards
    ↓
DistributedDataLoader
    ↓
Tokenize with BPE
    ↓
Batch & pack sequences
    ↓
Feed to GPT model
    ↓
Compute loss (cross-entropy)
    ↓
Backprop & optimize
    ↓
Save checkpoints
```

### Inference Data Flow

```
User prompt (text)
    ↓
Tokenize with BPE
    ↓
Feed to GPT model
    ↓
Generate tokens autoregressively
    ↓
Use KV cache for efficiency
    ↓
Sample with temperature/top-p
    ↓
Detokenize to text
    ↓
Return response
```

---

## Model Configurations

### Tiny (Local CPU)
```python
depth = 4
width = 256
max_seq_len = 1024
# ~8M parameters
# Training: ~1-3 hours on CPU
```

### Small (Production d20)
```python
depth = 20
width = 1280  # depth * 64
max_seq_len = 2048
# ~561M parameters
# Training: ~4 hours on 8xH100, ~12 hours on 8xA100
```

### Medium (d26)
```python
depth = 26
width = 1664
max_seq_len = 2048
# ~1.1B parameters
# Training: ~12 hours on 8xH100
```

### Large (d32)
```python
depth = 32
width = 2048
max_seq_len = 2048
# ~1.9B parameters
# Training: ~33 hours on 8xH100
```

---

## Platform Adaptations

### Local CPU (`scripts/local_cpu_train.sh`)
- Tiny model (depth=4)
- Minimal data (4 shards)
- CPU-only PyTorch
- Reduced iterations

### Ptolemy HPC (`scripts/speedrun.slurm`)
- Full model (depth=20)
- All data (240 shards)
- SLURM job submission
- Two-phase workflow (download → train)
- A100 GPUs

### Production GPU (`speedrun.sh`)
- Full model (depth=20)
- All data (240 shards)
- Direct execution
- H100 GPUs
- Cloud environment

---

## Key Design Decisions

### 1. Minimal Dependencies

Only essential packages:
- `torch` - Deep learning
- `numpy` - Numerics
- `tqdm` - Progress bars
- `requests` - Downloads
- `huggingface_hub` - Datasets

**No:**
- transformers library
- Complex frameworks
- Unnecessary abstractions

### 2. Single Codebase

Everything in one repo:
- Model implementation
- Training scripts
- Evaluation tasks
- Tokenizer
- Inference engine

**Benefit:** Easy to understand and modify

### 3. Rust for Tokenizer

BPE training is slow in Python:
- Rust implementation 10-100x faster
- Memory efficient
- Critical for large vocabularies

### 4. Distributed Training

Uses PyTorch DDP:
- Data parallel across 8 GPUs
- Gradient synchronization
- Scales efficiently

### 5. Modular Scripts

Separate scripts for each phase:
- `tok_train.py` - Tokenizer
- `base_train.py` - Base model
- `mid_train.py` - Midtraining
- `chat_sft.py` - SFT

**Benefit:** Can run phases independently

---

## Customization Points

### Model Architecture

Edit `nanochat/gpt.py`:
- Change attention mechanism
- Modify MLP structure
- Add new components

### Training Procedure

Edit training scripts:
- Adjust hyperparameters
- Change learning rate schedules
- Modify optimization

### Evaluation

Add new tasks in `tasks/`:
- Implement `Task` class
- Define evaluation logic
- Add to task mixture

### Data

Modify data sources:
- Change dataset in `scripts/`
- Add custom data loaders
- Mix multiple datasets

---

## Performance Characteristics

### Compute Requirements

| Model | Parameters | FLOPs | GPU Memory | Time (8xH100) |
|-------|------------|-------|------------|---------------|
| d4 (CPU) | 8M | 4e17 | N/A (CPU) | 1-3 hours |
| d20 | 561M | 4e19 | ~40GB | ~4 hours |
| d26 | 1.1B | 1e20 | ~60GB | ~12 hours |
| d32 | 1.9B | 2e20 | ~80GB | ~33 hours |

### Training Speed

**Throughput:**
- 8xH100: ~500K tokens/sec
- 8xA100: ~300K tokens/sec
- CPU: ~5K tokens/sec

**Bottlenecks:**
- I/O (data loading)
- GPU memory (model size)
- Communication (multi-GPU)

---

## Summary

nanochat is a complete, end-to-end LLM training pipeline that demonstrates:
- ✅ How to build transformers from scratch
- ✅ How to train on large-scale data
- ✅ How to evaluate language models
- ✅ How to create a chat interface
- ✅ How to optimize for different platforms

**Key Strength:** Minimal, hackable, educational codebase that produces real results.

---

For detailed code analysis:
→ Study `nanochat/gpt.py` for architecture
→ Study `scripts/base_train.py` for training loop
→ Study `nanochat/engine.py` for inference

For usage:
→ [Local CPU Quickstart](../tutorials/01-local-cpu-quickstart.md)
→ [HPC First Job](../tutorials/02-hpc-first-job.md)
