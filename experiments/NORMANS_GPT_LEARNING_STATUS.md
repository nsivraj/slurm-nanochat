# Norman's GPT Learning Effort - Status

**Last Updated**: 2025-11-03
**Approach**: Iterative implementation with training feedback
**Goal**: Deeply understand transformer architecture by implementing it step by step

---

## Overview

This document tracks progress on implementing `nanochat/normans_gpt.py` - a ground-up rewrite of the GPT transformer to learn by doing.

**Methodology**:
- Start with skeleton code containing only structure and debug statements
- Run training script to hit first `NotImplementedError`
- Implement the required method by understanding the reference
- Run again to see the next error
- Repeat until full model training succeeds

---

## Current Status

### Setup Complete ✅

- [x] Created `nanochat/normans_gpt.py` with skeleton structure
- [x] Added comprehensive debug statements throughout
- [x] Updated all imports to use `normans_gpt`:
  - `scripts/base_train.py:22`
  - `nanochat/checkpoint_manager.py:12`
  - `scripts/learn_via_debug/debug_gpt_components.py:21`
- [x] Verified training script runs and hits first error
- [x] Created documentation for the learning workflow

**First Error Encountered**:
```
NotImplementedError: TODO: Implement GPT.__init__ - create transformer dict with wte and h, create lm_head, setup rotary embeddings
```

### Implementation Progress

Track your progress as you implement each component:

#### Phase 1: Model Initialization
- [x] `GPTConfig` dataclass (already complete)
- [x] `GPT.__init__` - Model architecture setup
- [ ] `GPT._precompute_rotary_embeddings` - Rotary embedding cache
- [x] `Block.__init__` - Transformer block initialization
- [ ] `CausalSelfAttention.__init__` - Attention layer initialization
- [ ] `MLP.__init__` - Feed-forward network initialization

#### Phase 2: Weight Initialization
- [ ] `GPT.init_weights` - Initialize all weights
- [ ] `GPT._init_weights` - Initialize individual modules

#### Phase 3: Forward Pass
- [ ] `norm()` - RMS normalization function
- [ ] `apply_rotary_emb()` - Rotary embeddings function
- [ ] `CausalSelfAttention.forward` - Attention computation
- [ ] `MLP.forward` - Feed-forward computation
- [ ] `Block.forward` - Transformer block forward
- [ ] `GPT.forward` - Full model forward pass

#### Phase 4: Optimization
- [ ] `GPT.setup_optimizers` - Muon and AdamW optimizers
- [ ] `GPT.get_device` - Device detection helper
- [ ] `GPT.estimate_flops` - FLOPs estimation (optional)

#### Phase 5: Generation (Optional)
- [ ] `GPT.generate` - Autoregressive inference

---

## Training Commands

Quick iteration script (recommended):
```bash
bash scripts/local_cpu_own_gpt_transformer.sh
```

**First run**: Automatically downloads data (~400MB) and trains tokenizer (~10-15 min)
**Subsequent runs**: 10-30 seconds per iteration

Or run the command directly:
```bash
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --eval_tokens=512 \
    --core_metric_every=-1 \
    --total_batch_size=512 \
    --num_iterations=1
```

Full training pipeline (1-3 hours):
```bash
bash scripts/local_cpu_train.sh
```

---

## Learning Log

### Session 2025-11-03: Initial Setup

**What was accomplished**:
- Created skeleton implementation with all method signatures
- Added debug statements to show execution flow
- Updated all imports across codebase
- Verified setup works (hits expected error)
- Documented the iterative workflow

**Key insights**:
- Starting with `NotImplementedError` placeholders makes the path clear
- Debug statements will show exactly when each component is called
- Can iterate very quickly with `--num_iterations=1`

**Next session goals**:
- Implement `GPT.__init__`
- Understand transformer architecture composition
- Get to the first weight initialization

### Session 2025-11-03: Deep Dive into Embeddings and Fundamentals

**What was accomplished**:
- ✅ Implemented `GPT.__init__` successfully
- ✅ Fixed tokenizer path issue in iteration script
- ✅ Understood fundamental concepts of embeddings and tokenization

**Key Learnings - PyTorch Basics**:

**Q: What is `nn`?**
- `nn` = PyTorch's neural network module (`import torch.nn as nn`)
- Contains building blocks: `nn.Linear`, `nn.Embedding`, `nn.Module`, etc.
- All neural network layers come from this module

**Q: What is `nn.ModuleDict`?**
- Special dictionary that holds neural network modules
- Unlike regular Python dict, PyTorch properly tracks all modules inside
- Automatically registers parameters for training
- Handles device transfers (CPU/GPU) correctly
- Ensures modules are saved/loaded in checkpoints
- Example: `self.transformer = nn.ModuleDict({"wte": nn.Embedding(...), "h": nn.ModuleList(...)})`

**Key Learnings - Configuration**:

**Q: What is `vocab_size`?**
- Number of unique tokens the model knows (65,536 in our case = 2^16)
- Each token has a unique ID: 0 to 65,535
- Comes from the trained tokenizer
- Model must match: `nn.Embedding(vocab_size, n_embd)`

**Q: What is `n_embd`?**
- Embedding dimension = size of vector representing each token
- Each token becomes a vector of `n_embd` numbers
- Our small model: 256 dimensions
- Calculated as: `depth * 64` (e.g., 4 * 64 = 256)
- Larger models use 768, 1024, or even 12,288 dimensions

**Q: Where do these config values come from?**
- `vocab_size`: From tokenizer training (line 84 in base_train.py)
- `n_embd`: Calculated from `--depth` argument (line 89: `depth * 64`)
- `n_layer`: From `--depth` argument
- `n_head`: Calculated from model_dim (line 90)
- All values flow into `GPTConfig(**model_config_kwargs)` (line 108)

**Q: Why is `n_embd = depth * 64`? Where does 64 come from?**
- The `64` is called the **aspect ratio** (depth-to-width ratio)
- **Not mathematically derived** - it's an empirical design choice (hyperparameter)
- Aspect ratio controls the relationship between:
  - **Depth**: How many layers (vertical)
  - **Width**: How big each layer is (horizontal)
- **Why 64 specifically?**
  - ✅ Well-tested by research community
  - ✅ Good balance between depth and width
  - ✅ Works well in practice across many model sizes
  - ✅ Powers of 2 are efficient for hardware (64, 128, 256)
- **Not magic!** Aspect ratio 60 or 70 would probably work fine too
- Different model sizes use different aspect ratios:
  - Small models: 64 (our case: 4 layers × 64 = 256 dim)
  - Medium models: 64-96 (GPT-2: 12 layers × 64 = 768 dim)
  - Large models: 96-128 (GPT-3: 96 layers × 128 = 12,288 dim)
- **Trade-offs**:
  - Small aspect ratio (deep & narrow): More hierarchical, harder to train
  - Large aspect ratio (shallow & wide): Easier to train, less hierarchical
  - 64 is the sweet spot found through experimentation
- **Key insight**: Neural network architecture is part art, part science!
  - Many values are hyperparameters (chosen empirically)
  - Not derived from mathematical formulas
  - Found through experimentation and research

**Q: Why create n_layer Block instances? Why not just one?**
- **Each Block is one complete transformer layer** (attention + MLP)
- We stack `n_layer` blocks to create a **deep network**
- **Sequential processing**: Data flows through blocks one at a time
  ```
  Input → [Block 0] → [Block 1] → [Block 2] → [Block 3] → Output
  ```
- **Hierarchical learning**: Each layer learns different abstraction levels
  - Block 0: Basic patterns ("the" often precedes nouns)
  - Block 1: Grammar (subject-verb agreement)
  - Block 2: Semantics (word meanings, relationships)
  - Block 3: Complex patterns (complete understanding)
- **Why depth matters**: More layers = more powerful model
  - 1 layer: Very limited (can only learn simple patterns)
  - 4 layers: Basic understanding (our learning model)
  - 12 layers: Good understanding (GPT-2 small)
  - 96 layers: Exceptional understanding (GPT-3)
- **How it works in code**:
  ```python
  # Create 4 blocks
  "h": nn.ModuleList([Block(config, 0), Block(config, 1), Block(config, 2), Block(config, 3)])

  # Process data through all blocks sequentially
  for block in self.transformer.h:
      x = block(x, cos_sin, kv_cache)  # Each block refines the representation
  ```
- **All blocks have same architecture** but learn different things because they process data at different pipeline stages
- **Industry standard**: All transformers (GPT, BERT, LLaMA) use stacked layers
- **Think of it like**: Photo filters stacked - each transformation builds on the previous one
- **For learning**: 4 layers is perfect (fast, small, demonstrates concepts)

**Key Learnings - Tokenization**:

**Q: What does tokenizer training do?**
- Command: `python -m scripts.tok_train --max_chars=1000000000`
- Learns 65,536 tokens from 1 billion characters of text
- Uses BPE (Byte Pair Encoding) algorithm
- Takes ~10-15 minutes
- Saves to: `~/.cache/nanochat/tokenizer/tokenizer.pkl`
- Only needs to be done once!

**Q: How does BPE work?**
- Start with 256 base tokens (all bytes)
- Find most frequent pair of tokens in training data
- Merge that pair into a new token
- Repeat for 65,280 iterations (65,536 - 256 = 65,280 merges)
- Result: Common words = single tokens, rare words = multiple tokens
- Example: "The" might be token 258, "cat" might be token 1543
- Rare word "xylophone" might split: ["xyl", "oph", "one"]

**Q: Why BPE over other algorithms?**
- **Simple**: Easy to implement and understand
- **Effective**: Good compression, handles any text
- **Fast**: Training takes minutes
- **Proven**: Used by GPT-2/3/4, LLaMA, Mistral
- **Byte-level**: Works for any language/symbols
- **Alternatives**: WordPiece (BERT), Unigram (T5), but BPE is the standard for GPT-style models

**Key Learnings - Embeddings**:

**Q: How do tokens become vectors?**
- `nn.Embedding` creates a lookup table (matrix)
- Shape: `(vocab_size, n_embd)` = `(65536, 256)`
- Each row is one token's vector
- Token ID → Look up that row in the matrix
- Example: Token 42 → Get row 42 (256 numbers)
- It's literally just indexing: `embedding.weight[token_id]`

**Q: When is `nn.Embedding` initialized?**
- **Step 1**: Creating `nn.Embedding(65536, 256)` immediately fills with random values
- **Step 2**: `model.init_weights()` re-initializes with different random values (line 113 in base_train.py)
- Both use normal distribution: `N(mean=0, std=1.0)`
- After creation, embeddings are ready but random (meaningless)

**Q: When/how do embeddings learn?**
- Learning happens during training loop (lines 269-286 in base_train.py)
- **Every training iteration**:
  1. Forward pass: Use current embedding values
  2. Compute loss: How wrong are predictions?
  3. Backward pass: `loss.backward()` computes gradients for each embedding dimension
  4. Update: `optimizer.step()` adjusts embeddings based on gradients
- Only tokens that appear in the batch get updated
- Takes thousands of iterations to learn good embeddings
- Final result: Similar tokens have similar vectors

**Q: What do the 256 dimensions represent?**
- **Not predefined!** The model learns what's useful for prediction
- Individual dimensions usually don't have clear meanings
- **Distributed representations**: Meaning spreads across multiple dimensions
  - No single dimension = "is it an animal?"
  - Instead: Dimensions 5, 17, 42, 103 TOGETHER might capture "animal-ness"
- Emergent from optimization, not designed by humans
- Sometimes patterns emerge (e.g., gender, plurality) but not reliably
- The 256 numbers work together in complex ways to minimize loss
- Think: "Abstract learned patterns that help predict the next token"

**Q: What's stored after training?**
- The trained weight matrix in `model.transformer.wte.weight`
- Shape: `(65536, 256)`
- Each row is a learned 256-dimensional vector for one token
- Saved in checkpoint: `state_dict["transformer.wte.weight"]`
- These embeddings ARE the learned representation of language!

**Visual Summary - Complete Flow**:
```
1. Tokenizer Training (tok_train.py)
   Text → BPE Algorithm → 65,536 tokens → tokenizer.pkl

2. Model Creation (GPT.__init__)
   Config(vocab_size=65536, n_embd=256) → nn.Embedding(65536, 256) → Random initialization

3. Weight Initialization (init_weights)
   Apply init_weights() → Re-randomize all weights → Model ready for training

4. Training Loop (base_train.py)
   For each batch:
     - Forward: embedding[token_ids] → get vectors → process through model → predictions
     - Loss: compare predictions to actual next tokens
     - Backward: compute gradients for embeddings
     - Update: adjust embeddings to reduce loss

5. After Training
   Embeddings learned! Similar tokens have similar vectors in 256D space
```

**Insights**:
- Embeddings are learned parameters, just like all other weights
- The "meaning" is distributed across all 256 dimensions
- We can't easily explain WHY the model learns what it does
- Trust the optimization process to discover useful patterns
- This is the beauty and mystery of neural networks!
- **Architecture is part art, part science**: Many design choices (like aspect ratio 64) are empirical, not mathematically optimal
- Hyperparameters are chosen through experimentation, not pure theory

**Next session goals**:
- Implement `Block.__init__`
- Understand transformer block architecture
- Continue building up the model step by step

### Session 2025-11-03: Block Architecture and Data Flow

**What was accomplished**:
- ✅ Implemented `Block.__init__` successfully
- ✅ Understood matrix dimensions through transformer blocks
- ✅ Clarified what "512 tokens" means in the architecture
- ✅ Deep dive into max_seq_len and its impact on learning

**Key Learnings - Block Architecture**:

**Q: What does Block.__init__ create?**
```python
def __init__(self, config, layer_idx):
    super().__init__()

    # 1. Self-Attention: Looks at relationships between tokens
    self.attn = CausalSelfAttention(config, layer_idx)

    # 2. MLP: Processes each position independently
    self.mlp = MLP(config)
```

- **Two components per block**: Attention + MLP
- **Attention**: Gathers information from other positions in sequence
  - "Causal" = can only look at previous positions (left-to-right)
  - Example: Processing "sat" can look back at "The" and "cat"
  - Learns relationships between tokens
- **MLP**: Transforms each position independently
  - Expands to 4x size internally (more learning capacity)
  - Projects back to original size
  - Applies non-linear transformations
- **This pattern repeats**: With n_layer=4, we have 4 identical Block structures stacked

**Q: Why need BOTH attention and MLP?**
- **Attention** = Information gathering (communicate between positions)
- **MLP** = Information processing (transform what was gathered)
- Without attention: No communication between tokens (each processed in isolation)
- Without MLP: Only linear transformations (limited learning capacity)
- Together: Powerful combination that enables language understanding

**Key Learnings - Matrix Dimensions**:

**Q: What are the matrix dimensions through the transformer blocks?**

With our config (`depth=4`, so `n_embd=256`, `n_layer=4`, `max_seq_len=512`):

```
Input token IDs:     (batch_size, 512)
                            ↓
Token Embedding (wte):  (B, 512, 256)
                            ↓
Apply norm:             (B, 512, 256)
                            ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:                  (B, 512, 256)
  → norm → Attention:   (B, 512, 256)
  → Residual:           (B, 512, 256)
  → norm → MLP:         (B, 512, 256)
  → Residual:           (B, 512, 256)
Output:                 (B, 512, 256)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 1, 2, 3: Same dimensions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            ↓
Final norm:             (B, 512, 256)
                            ↓
lm_head projection:     (B, 512, 65536)
```

**Critical insight**: Dimensions stay **(B, 512, 256) through ALL blocks!**

**Why constant dimensions?**
- Enables residual connections: `x = x + attn(x)` requires same shape
- Each layer refines representation without changing size
- Only changes at final projection to vocabulary

**Inside Attention (dimensions change temporarily)**:
```
Input:              (B, 512, 256)
  → Project to Q:   (B, 512, 256)
  → Project to K:   (B, 512, 256)
  → Project to V:   (B, 512, 256)
  → Reshape heads:  (B, 512, n_head=2, head_dim=128)
  → Attention:      (B, 512, 2, 128)
  → Reshape back:   (B, 512, 256)
  → c_proj:         (B, 512, 256)
Output:             (B, 512, 256)
```

**Inside MLP (expands then contracts)**:
```
Input:              (B, 512, 256)
  → c_fc (expand):  (B, 512, 1024)  # 256 * 4
  → relu^2:         (B, 512, 1024)
  → c_proj:         (B, 512, 256)   # Back to n_embd
Output:             (B, 512, 256)
```

**Key Learnings - What Are "512 Tokens"?**:

**Q: What does "512 tokens" mean in the architecture?**
- The **512** comes from `--max_seq_len=512` parameter
- It means processing **sequences of 512 tokens at a time**
- NOT 512 different words - it's the sequence length

**Concrete example**:
```
Text: "The cat sat on the mat. The dog ran in the park."

After tokenization:
Token IDs: [150, 8901, 23456, 891, 150, 4567, 19, ...]
           ↑                                        ↑
           Token 0                              Token 13
           (14 tokens total)

But model expects 512 tokens, so dataloader combines sentences:
[sentence1_tokens] + [sentence2_tokens] + [sentence3_tokens] + ...
= [tok0, tok1, tok2, ..., tok511]  ← 512 token IDs total
```

**Shape breakdown**:
```python
# (batch_size, sequence_length, embedding_dim)
tensor.shape = (B, 512, 256)

# What each dimension means:
# B:    How many sequences in parallel (batch)
# 512:  How many tokens in each sequence (max_seq_len)
# 256:  Size of vector for each token (n_embd)
```

**Example with batch_size=2**:
```
Sequence 1: "The cat sat..." (512 tokens)
Sequence 2: "Dogs run fast..." (512 tokens)

Shape: (2, 512, 256)
       ↑  ↑    ↑
       │  │    └─ 256-dim vector per token
       │  └────── 512 tokens per sequence
       └───────── 2 sequences in this batch
```

**Key Learnings - Does max_seq_len Improve Learning?**:

**Q: Does increasing max_seq_len improve learning or embedding quality?**

**Short answer**:
- ✅ Improves **long-range pattern learning**
- ❌ Does NOT directly improve **embedding quality**
- ⚠️ Costs **quadratically more** compute and memory

**What improves with longer sequences**:

**1. Long-Range Dependencies** ✅
```
max_seq_len=512:
"The cat [400 tokens]. It meowed."
 ↑                        ↑
 Token 0              Token 450

Attention CAN connect "cat" and "meowed" ✅

max_seq_len=128:
Chunk 1: "The cat [100 tokens]"
Chunk 2: "[...] It meowed."

Attention CANNOT see "cat" from different chunk ❌
```

**2. Embedding Quality** ❌ Not directly affected
```
Token "cat" learns from ALL occurrences during training:
- Sequence 1: "The cat sat"
- Sequence 2: "A cat ran"
- Sequence 3: "Cats are animals"
... thousands more ...

Embeddings learn from LOCAL context (nearby tokens),
not from being in longer sequences.
```

**3. Trade-offs**:

| Aspect | max_seq_len=128 | max_seq_len=512 | max_seq_len=1024 |
|--------|----------------|----------------|------------------|
| Training Speed | ✅ Fast | Medium | ❌ Slow (4x) |
| Memory Usage | ✅ Low | Medium | ❌ High (4x) |
| Long-range Learning | ❌ Poor | ✅ Good | ✅✅ Excellent |
| Short-range Learning | ✅ Same | ✅ Same | ✅ Same |
| Embedding Quality | ✅ Same | ✅ Same | ✅ Same |
| Good For | Quick iteration | Balanced | Production |

**What ACTUALLY improves embeddings**:
1. ✅ **More training data** - More examples in different contexts
2. ✅ **More iterations** - More gradient updates to refine vectors
3. ✅ **Better architecture** - Deeper networks, better normalization
4. ⚠️ **Longer sequences** - Indirect benefit only (better model overall)

**Why 512 for learning?**
- **Fast enough**: 30 seconds per iteration (vs minutes for 1024)
- **Not too limited**: Can learn some long-range patterns
- **Fits in memory**: Works on CPU and small GPUs
- **Sweet spot**: Balance of speed and capability

**Code understanding example**:
```python
# max_seq_len=128 (SHORT)
def process_data(data):
    result = []
# [Chunk boundary - attention can't cross]
    result.append(item)
    return result

# Model CANNOT learn that "result.append" relates to "result = []"

# max_seq_len=512 (LONGER)
def process_data(data):
    result = []              # Token 10
    for item in data:
        # ... 50 lines ...
        result.append(item)  # Token 200
    return result            # Token 205

# Model CAN learn relationships within 512 tokens! ✅
```

**Bottom Line**:
- **Increasing max_seq_len**: Improves long-context understanding, NOT embedding quality directly
- **For learning**: Keep 512 for fast iteration
- **For production**: Use 1024+ where long context is critical
- **Attention complexity**: O(n²) - doubling sequence length = 4x compute!

**Insights**:
- The "512" in matrix dimensions means processing 512 tokens simultaneously
- Each token gets its own 256-dimensional vector that flows through all blocks
- Dimensions stay constant through blocks to enable residual connections
- Longer sequences help with long-range patterns, not basic embeddings
- Architecture choice (512 vs 1024) is about compute vs capability trade-off

**Next session goals**:
- Implement `CausalSelfAttention.__init__` (next needed component)
- Implement `MLP.__init__` (next needed component)
- Understand attention mechanism in detail
- Get model to fully initialize without errors

---

## Debugging Notes

### Debug Output Format

The skeleton includes print statements like:
```python
print0(f"[DEBUG] GPT.forward() called")
print0(f"[DEBUG]   idx.shape: {idx.shape}")
```

`print0()` only prints on rank 0 (to avoid duplicate output in distributed training).

### Common Patterns to Watch For

**Initialization order**:
1. GPT.__init__ calls Block.__init__ for each layer
2. Block.__init__ calls CausalSelfAttention.__init__ and MLP.__init__
3. After all modules created, GPT.init_weights() is called

**Forward pass order**:
1. Embed tokens → norm
2. For each block: norm → attention → residual, norm → MLP → residual
3. Final norm → lm_head → softcapping
4. Compute loss if training

**Optimizer setup**:
- Separates parameters into matrix_params, embedding_params, lm_head_params
- Uses Muon for matrices, AdamW for embeddings
- Scales learning rates by `sqrt(768/model_dim)`

---

## Questions to Answer

As you implement, document answers to these questions:

### Fundamental Questions (ANSWERED ✅)
- [x] **What is `nn`?** → PyTorch's neural network module (`torch.nn`)
- [x] **What is `nn.ModuleDict`?** → Smart dictionary that tracks neural network modules for PyTorch
- [x] **What is `vocab_size`?** → Number of unique tokens (65,536), from tokenizer
- [x] **What is `n_embd`?** → Embedding dimension (256), size of vector per token
- [x] **Why `n_embd = depth * 64`?** → 64 is the "aspect ratio" (depth-to-width), empirical design choice
- [x] **Why create n_layer Block instances?** → Stack layers for hierarchical learning, depth = power
- [x] **How are tokens converted to vectors?** → Lookup table (`nn.Embedding`), just indexing
- [x] **When do embeddings learn?** → Every training iteration via backpropagation
- [x] **What do the embedding dimensions represent?** → Abstract learned patterns (distributed representations)
- [x] **What does tokenizer training do?** → Learns 65,536 tokens using BPE algorithm
- [x] **Why BPE?** → Simple, effective, proven (used by GPT-2/3/4)
- [x] **What gets saved after training?** → Weight matrix in `transformer.wte.weight` (65536 x 256)

### Architecture Questions
- [x] **What do Block components do?** → Attention gathers info, MLP processes info
- [x] **Why both attention and MLP?** → Attention = communication, MLP = transformation
- [x] **Why do dimensions stay constant through blocks?** → Enables residual connections
- [x] **What does "512 tokens" mean?** → Sequence length (max_seq_len parameter)
- [x] **Does max_seq_len improve embeddings?** → No, improves long-range learning only
- [ ] Why untie embedding and lm_head weights?
- [ ] What's the advantage of Multi-Query Attention?
- [ ] Why use relu^2 instead of regular relu or GELU?
- [ ] What does QK normalization prevent?

### Implementation Questions
- [ ] Why are rotary embeddings registered as buffers not parameters?
- [ ] How does gradient accumulation simulate larger batches?
- [ ] Why normalize after embedding instead of before?
- [ ] What does softcapping prevent?

### Training Questions
- [ ] Why do embeddings need a different optimizer than matrices?
- [ ] How does the learning rate scaling formula work?
- [ ] What's the advantage of no bias in linear layers?
- [ ] Why use bfloat16 for embeddings and rotary cache?

---

## Issues Encountered

Document any bugs, errors, or confusions here:

### Issue 1: Tokenizer Path Mismatch
**Date**: 2025-11-03
**Problem**: Script kept re-training tokenizer even though it existed
**Error Message**: Script checked for `~/.cache/nanochat/tokenizer.pkl` but tokenizer was saved to `~/.cache/nanochat/tokenizer/tokenizer.pkl`
**Solution**: Updated script to check correct path: `TOKENIZER_FILE="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"`
**Learning**: Always verify actual file locations vs expected paths. The tokenizer saves to a subdirectory by default.

### Issue Template (for future use)
**Date**: YYYY-MM-DD
**Problem**: Brief description
**Error Message**: Copy full traceback
**Solution**: What fixed it
**Learning**: Key insight from solving it

---

## Milestones

Track major achievements:

- [x] **Setup Complete** (2025-11-03): Skeleton created, imports updated, verified working
- [x] **GPT.__init__ Implemented** (2025-11-03): Model can create transformer dict, embeddings, lm_head
- [x] **Deep Understanding** (2025-11-03): Understood embeddings, tokenization, BPE, distributed representations
- [x] **Block.__init__ Implemented** (2025-11-03): Transformer block creates attention and MLP layers
- [x] **Data Flow Understanding** (2025-11-03): Matrix dimensions, 512 tokens, max_seq_len impact
- [ ] **Model Initializes**: GPT model can be created without errors (need CausalSelfAttention, MLP)
- [ ] **Forward Pass Works**: Can run one forward pass
- [ ] **Backward Pass Works**: Gradients flow correctly
- [ ] **Training Step Completes**: One full train step runs
- [ ] **Loss Decreases**: Model learns over multiple steps
- [ ] **Full Pipeline**: Complete train → eval → save cycle
- [ ] **Generation Works**: Can generate text samples
- [ ] **Matches Reference**: Performance equals original `gpt.py`

---

## Performance Comparison

Once training works, compare against the reference implementation:

| Metric | normans_gpt.py | gpt.py (reference) | Notes |
|--------|----------------|---------------------|-------|
| Training loss (step 50) | TBD | TBD | Should match closely |
| Memory usage | TBD | TBD | Should be similar |
| Training speed | TBD | TBD | May be slightly slower |
| Final eval loss | TBD | TBD | Should match closely |
| CORE score | TBD | TBD | Should match closely |

---

## Code Quality Notes

Track refactorings or improvements made:

- Clean code structure matching reference
- Comprehensive debug statements added
- Clear TODO comments for guidance
- Type hints could be added (future enhancement)

---

## References

**Primary reference**: `nanochat/gpt.py` (the original working implementation)

**Key papers**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT architecture
- [LLaMA](https://arxiv.org/abs/2302.13971) - Rotary embeddings
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Simpler normalization
- [Muon Optimizer](https://arxiv.org/abs/2402.16788) - Matrix-specific optimization

**Documentation**:
- [How-To: Write Your Own GPT](../docs/how-to/03-how-to-write-your-own-gpt-transformer.md)
- [Local CPU Quickstart](../docs/tutorials/01-local-cpu-quickstart.md)
- [Project Architecture](../docs/explanation/project-architecture.md)

---

## Next Actions

1. **Implement CausalSelfAttention.__init__**
   - Create Q, K, V projection layers
   - Create output projection (c_proj)
   - Setup for multi-query attention

2. **Implement MLP.__init__**
   - Create expansion layer (c_fc)
   - Create contraction layer (c_proj)

3. **Run training again**
   - Should hit next NotImplementedError
   - Follow the debug output
   - See initialization complete

4. **Document learnings**
   - Update Questions to Answer section
   - Add any issues encountered
   - Record insights in Learning Log

---

## Success Criteria

This learning effort is successful when:

✅ You can explain every line of the transformer code
✅ You understand the execution flow from tokens to loss
✅ You can modify the architecture with confidence
✅ Your implementation trains successfully
✅ You can debug issues in transformer models
✅ You have deep intuition about what each component does

---

**Remember**: The goal is **understanding**, not speed. Take time to understand each piece before moving on.
