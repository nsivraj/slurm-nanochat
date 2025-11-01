# Local CPU Training - Code Analysis Guide

This guide helps you analyze and understand the nanochat codebase using your locally-trained model as context.

## Purpose

After running local CPU training, you have:
- ✅ A complete trained model (all phases)
- ✅ Fresh understanding of the training process
- ✅ Real training metrics and logs
- ✅ Working knowledge of all components

Use this to deeply understand the codebase!

## Step-by-Step Analysis

### Step 1: Understand the Training Flow

Start with the training script you just ran:

```bash
# Read the script that orchestrated everything
cat scripts/local_cpu_train.sh

# Compare with production GPU version
diff scripts/local_cpu_train.sh speedrun.sh

# Compare with minimal CPU demo
diff scripts/local_cpu_train.sh dev/runcpu.sh
```

**Key Questions:**
- What parameters were reduced for CPU training?
- Why these specific values?
- What would break if you changed them?

### Step 2: Analyze Tokenizer Training

```bash
# Read the tokenizer training script
cat scripts/tok_train.py

# Read the evaluation script
cat scripts/tok_eval.py

# Examine the core tokenizer implementation
cat nanochat/tokenizer.py

# Look at the Rust tokenizer (performance-critical)
cat rustbpe/src/lib.rs
```

**Key Concepts:**
- BPE (Byte Pair Encoding) algorithm
- Vocabulary size (65,536 = 2^16)
- Character-to-token compression ratio
- Special tokens handling

**Your Training:**
- Check your tokenizer's compression ratio in `report.md`
- Compare with GPT-2 and GPT-4 tokenizers
- Find your tokenizer at `~/.cache/nanochat/tokenizer/tokenizer.pkl`

### Step 3: Analyze Base Model Pretraining

```bash
# Read the pretraining script
cat scripts/base_train.py

# Examine the GPT model architecture
cat nanochat/gpt.py

# Look at the data loader
cat nanochat/dataloader.py

# Check the optimizer implementations
cat nanochat/adamw.py
cat nanochat/muon.py
```

**Key Concepts:**
- Transformer architecture (attention, MLP, LayerNorm)
- Training loop (forward pass, loss, backward pass, optimizer step)
- Distributed data loading and tokenization
- Optimizer selection (AdamW for embeddings, Muon for matrices)
- Gradient accumulation
- Learning rate schedules

**Your Training:**
- Review base model metrics in `report.md`
- Find CORE score (measures general capability)
- Check loss curve (bits per byte)
- Examine sample text generations

**Trace the Training Flow:**

1. **Model Initialization** (`nanochat/gpt.py`):
   ```python
   # GPTConfig defines architecture
   config = GPTConfig(
       vocab_size=65536,
       n_layer=4,        # Your depth
       n_head=12,        # Attention heads
       n_embd=768,       # Hidden dimension
   )
   model = GPT(config)
   ```

2. **Data Loading** (`nanochat/dataloader.py`):
   ```python
   # Loads parquet files, tokenizes, creates batches
   train_loader = tokenizing_distributed_data_loader(
       base_dir=base_dir,
       split="train",
       ...
   )
   ```

3. **Training Loop** (`scripts/base_train.py:~100-200`):
   - Get batch from data loader
   - Forward pass through model
   - Calculate loss (cross-entropy)
   - Backward pass (compute gradients)
   - Optimizer step (update weights)
   - Periodic evaluation and checkpointing

4. **Checkpointing** (`nanochat/checkpoint_manager.py`):
   - Saves model state, optimizer state, training step
   - Enables resuming from interruptions

### Step 4: Analyze Midtraining

```bash
# Read the midtraining script
cat scripts/mid_train.py

# Look at the SmolTalk dataset loader
cat tasks/smoltalk.py

# Check the custom JSON loader (for identity data)
cat tasks/customjson.py

# See the common task infrastructure
cat tasks/common.py
```

**Key Concepts:**
- Conversation format (special tokens like <|im_start|>, <|im_end|>)
- Multi-turn dialogue structure
- Tool use teaching
- Identity imprinting
- Task mixture (combining multiple datasets)

**Your Training:**
- Check midtraining metrics in `report.md`
- Compare performance before/after midtraining
- Look at the identity conversations: `cat ~/.cache/nanochat/identity_conversations.jsonl`

**Trace Conversation Format:**

Example conversation structure:
```
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
```

See how this is parsed in `scripts/mid_train.py` and `tasks/smoltalk.py`.

### Step 5: Analyze Supervised Finetuning

```bash
# Read the SFT script
cat scripts/chat_sft.py

# Compare with midtraining
diff scripts/mid_train.py scripts/chat_sft.py
```

**Key Concepts:**
- Instruction-following data format
- Per-sequence optimization (vs batched)
- Evaluation during training
- Final model selection

**Your Training:**
- Check SFT metrics in `report.md`
- Look for improvements on benchmarks
- Your final model: `~/.cache/nanochat/models/sft_model.pt`

### Step 6: Analyze Evaluation

```bash
# Base model evaluation (CORE)
cat scripts/base_eval.py
cat nanochat/core_eval.py

# Chat model evaluation
cat scripts/chat_eval.py

# Individual task implementations
cat tasks/arc.py          # Science questions
cat tasks/gsm8k.py        # Math problems
cat tasks/humaneval.py    # Coding
cat tasks/mmlu.py         # General knowledge

# Loss evaluation
cat scripts/base_loss.py
cat nanochat/loss_eval.py
```

**Key Concepts:**
- CORE benchmark (from DCLM paper)
- Multiple choice tasks
- Code generation tasks
- Math problem solving
- Perplexity and bits-per-byte metrics

**Your Training:**
- Compare your scores across BASE → MID → SFT
- Understand why scores are low (tiny model!)
- See which tasks improved most

### Step 7: Analyze Inference

```bash
# Read the inference engine
cat nanochat/engine.py

# CLI interface
cat scripts/chat_cli.py

# Web interface
cat scripts/chat_web.py

# See the HTML/CSS/JS frontend
cat nanochat/ui.html
```

**Key Concepts:**
- KV (Key-Value) cache for efficient generation
- Top-k and top-p sampling
- Temperature control
- Stop tokens
- Streaming generation

**Try It:**
```bash
source .venv/bin/activate

# Single prompt
python -m scripts.chat_cli -p "Explain transformers"

# Interactive
python -m scripts.chat_cli

# Adjust sampling (lower temp = more deterministic)
python -m scripts.chat_cli -p "Count to 10" --temperature 0.5

# Longer responses
python -m scripts.chat_cli -p "Write a story" --max-new-tokens 256
```

### Step 8: Analyze Report Generation

```bash
# Read the report utilities
cat nanochat/report.py

# Check your generated report
cat report.md

# See where metrics are collected
grep -r "report.append" scripts/
```

**Your Report Structure:**
1. System information
2. Tokenizer metrics
3. Base model metrics
4. Midtraining metrics
5. SFT metrics
6. Summary table

## Deep Dive: The Transformer Architecture

Open `nanochat/gpt.py` and trace through the model:

### 1. Token Embeddings
```python
# Line ~70-80 in gpt.py
self.wte = nn.Embedding(vocab_size, n_embd)  # Token embeddings
self.wpe = nn.Embedding(max_seq_len, n_embd)  # Position embeddings
```

Your input tokens become vectors in high-dimensional space.

### 2. Transformer Blocks
```python
# Line ~100-120 in gpt.py
self.blocks = nn.ModuleList([
    Block(n_embd, n_head, ...) for _ in range(n_layer)
])
```

Each block has:
- **Self-attention**: Tokens attend to previous tokens
- **MLP**: Feed-forward network for processing
- **LayerNorm**: Normalization for stability

### 3. Output Head
```python
# Line ~140-150 in gpt.py
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
```

Converts hidden states back to vocabulary logits.

### 4. Forward Pass
```python
# Line ~200-250 in gpt.py
def forward(self, idx):
    # Get embeddings
    tok_emb = self.wte(idx)
    pos_emb = self.wpe(positions)
    x = tok_emb + pos_emb

    # Pass through transformer blocks
    for block in self.blocks:
        x = block(x)

    # Get logits
    logits = self.lm_head(x)
    return logits
```

## Understanding Your Model's Behavior

### Why Is My Model Not Very Smart?

1. **Tiny size**: ~8M parameters vs 561M (production) vs 175B (GPT-3)
2. **Limited data**: 1B characters vs 54B (production) vs trillions (GPT-3)
3. **Short training**: 50 steps vs ~5000 (production)
4. **Small context**: 1024 tokens vs 2048+ (modern models)

### What Can It Do?

- ✅ Generate grammatically coherent text
- ✅ Follow basic conversation format
- ✅ Attempt to answer questions (often wrong)
- ✅ Show emergent patterns (rhyming, counting, basic facts)
- ❌ Complex reasoning
- ❌ Factual accuracy
- ❌ Math beyond simple addition
- ❌ Code generation

### Interesting Experiments

Try these prompts and observe behavior:

```bash
source .venv/bin/activate

# Test knowledge
python -m scripts.chat_cli -p "What is the capital of France?"

# Test math
python -m scripts.chat_cli -p "What is 15 + 27?"

# Test creativity
python -m scripts.chat_cli -p "Write a haiku about computers"

# Test reasoning
python -m scripts.chat_cli -p "If John is taller than Mary, and Mary is taller than Sue, who is tallest?"

# Test code
python -m scripts.chat_cli -p "Write a Python function to check if a number is prime"
```

## Parameter Sensitivity Analysis

Want to understand how parameters affect training? Try modifying `scripts/local_cpu_train.sh`:

### Experiment 1: Model Depth
```bash
# Current: --depth=4
# Try: --depth=2 (smaller, faster, worse)
# Try: --depth=6 (larger, slower, better - if memory allows)
```

### Experiment 2: Context Length
```bash
# Current: --max_seq_len=1024
# Try: --max_seq_len=512 (less memory, less context)
# Try: --max_seq_len=2048 (more memory, more context)
```

### Experiment 3: Training Iterations
```bash
# Current: --num_iterations=50 (base), 100 (mid/sft)
# Try: --num_iterations=100 (base) - longer pretraining
# Try: --num_iterations=20 (base) - faster but worse
```

### Experiment 4: Batch Size
```bash
# Current: --total_batch_size=1024
# Try: --total_batch_size=2048 - more stable gradients
# Try: --total_batch_size=512 - less stable but faster
```

After each change, compare the `report.md` metrics!

## Comparing CPU vs GPU Training

Create a comparison table:

| Aspect | Local CPU | Production GPU |
|--------|-----------|----------------|
| Script | `scripts/local_cpu_train.sh` | `speedrun.sh` / `scripts/speedrun.slurm` |
| Data download | 4 shards | 240 shards |
| Tokenizer chars | 1B | 2B |
| Model depth | 4 | 20 |
| Parameters | ~8M | ~561M |
| Context length | 1024 | 2048 |
| Base iterations | 50 | ~5000 |
| Training time | 1-3 hrs (CPU) | ~4 hrs (8xH100) |
| Cost | Free | ~$100 |

## Advanced Topics

### 1. Distributed Training
The production scripts use `torchrun` for multi-GPU training:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

Read `scripts/base_train.py` to see how it handles:
- DDP (Distributed Data Parallel)
- Gradient synchronization
- Multi-process communication

### 2. Custom Optimizers
nanochat uses two optimizers:
- **AdamW** for embeddings: `nanochat/adamw.py`
- **Muon** for weight matrices: `nanochat/muon.py`

Compare with standard PyTorch optimizers.

### 3. Mixed Precision Training
Look for `torch.cuda.amp` usage in training scripts (GPU only).

## Questions to Answer

As you read the code, try to answer:

1. **Tokenization**: How does BPE training work? Why vocab size 65536?
2. **Architecture**: Why use causal attention? What's the purpose of positional embeddings?
3. **Training**: Why separate optimizers for embeddings vs matrices?
4. **Evaluation**: How is CORE score calculated? What does it measure?
5. **Inference**: How does KV caching speed up generation?
6. **Conversation**: How do special tokens structure multi-turn dialogue?

## Resources

- **Main README**: `cat README.md`
- **Local training guide**: `cat LOCAL_CPU_TRAINING.md`
- **Quick start**: `cat LOCAL_CPU_QUICKSTART.md`
- **Production setup**: `cat PTOLEMY_SETUP.md`

## Next Steps

1. ✅ Understand each training phase
2. ✅ Trace code execution for one full training iteration
3. ✅ Experiment with hyperparameters
4. ✅ Implement a custom evaluation task
5. ✅ Create custom identity conversations
6. ✅ Try GPU training on a larger model

Happy analyzing! 🔍
