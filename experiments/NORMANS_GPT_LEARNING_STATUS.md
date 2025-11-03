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
- [ ] `GPTConfig` dataclass (already complete)
- [ ] `GPT.__init__` - Model architecture setup
- [ ] `GPT._precompute_rotary_embeddings` - Rotary embedding cache
- [ ] `Block.__init__` - Transformer block initialization
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

**Next session goals**:
- Implement `Block.__init__`
- Understand transformer block architecture
- Continue building up the model step by step

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
- [x] **How are tokens converted to vectors?** → Lookup table (`nn.Embedding`), just indexing
- [x] **When do embeddings learn?** → Every training iteration via backpropagation
- [x] **What do the embedding dimensions represent?** → Abstract learned patterns (distributed representations)
- [x] **What does tokenizer training do?** → Learns 65,536 tokens using BPE algorithm
- [x] **Why BPE?** → Simple, effective, proven (used by GPT-2/3/4)
- [x] **What gets saved after training?** → Weight matrix in `transformer.wte.weight` (65536 x 256)

### Architecture Questions
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
- [ ] **Model Initializes**: GPT model can be created without errors (need Block, MLP, Attention)
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

1. **Implement GPT.__init__**
   - Create transformer ModuleDict with wte and h
   - Create lm_head Linear layer
   - Setup rotary embedding buffers

2. **Run training again**
   - See what's needed next
   - Follow the debug output

3. **Document learnings**
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
