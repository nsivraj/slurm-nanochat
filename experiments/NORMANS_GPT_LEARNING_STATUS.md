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

### Issue Template
**Date**: YYYY-MM-DD
**Problem**: Brief description
**Error Message**: Copy full traceback
**Solution**: What fixed it
**Learning**: Key insight from solving it

---

## Milestones

Track major achievements:

- [x] **Setup Complete** (2025-11-03): Skeleton created, imports updated, verified working
- [ ] **Model Initializes**: GPT model can be created without errors
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
