# How-To: Write Your Own GPT Transformer

This guide shows you how to implement your own GPT transformer from scratch by iteratively building `nanochat/normans_gpt.py` while running the training loop to see what each component does.

---

## Overview

Instead of just reading the transformer code, you'll **write it yourself** step by step, running the training script after each implementation to understand what each component does through debug output.

This hands-on approach helps you:
- Understand the execution flow of transformer training
- See exactly when each component is called
- Debug issues as they arise
- Build deep intuition about transformer architecture

---

## Prerequisites

Before starting, make sure you have:
1. Completed the [Local CPU Quickstart](../tutorials/01-local-cpu-quickstart.md)
2. Verified your environment is working
3. Basic understanding of PyTorch and neural networks

---

## Setup: The Skeleton Implementation

The codebase already includes `nanochat/normans_gpt.py` - a skeleton implementation with:
- Complete structure mirroring `nanochat/gpt.py`
- `NotImplementedError` placeholders for every method
- Debug print statements showing execution flow
- TODO comments explaining what to implement

All imports have been updated to use `normans_gpt` instead of `gpt`:
- `scripts/base_train.py:22`
- `nanochat/checkpoint_manager.py:12`
- `scripts/learn_via_debug/debug_gpt_components.py:21`

---

## The Iterative Workflow

### Step 1: Run the Training Script

Start with a quick training iteration:

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

You'll see debug output and hit the first `NotImplementedError`.

### Step 2: Read the Error Message

The error tells you **exactly** what to implement next:

```
[DEBUG] ========================================
[DEBUG] Initializing GPT model
[DEBUG]   n_layer: 4
[DEBUG]   n_head: 2
[DEBUG]   n_kv_head: 2
[DEBUG]   n_embd: 256
[DEBUG]   vocab_size: 65536
[DEBUG]   sequence_len: 512
[DEBUG] ========================================
NotImplementedError: TODO: Implement GPT.__init__ - create transformer dict with wte and h, create lm_head, setup rotary embeddings
```

### Step 3: Open the Reference Implementation

Open `nanochat/gpt.py` (the original working code) side by side with `nanochat/normans_gpt.py`.

### Step 4: Implement the Method

In `nanochat/normans_gpt.py`, find the method mentioned in the error (e.g., `GPT.__init__`) and:

1. **Read the reference implementation** in `nanochat/gpt.py`
2. **Understand what it does** - don't just copy blindly
3. **Write it yourself** in `nanochat/normans_gpt.py`
4. **Remove the `raise NotImplementedError` line**

### Step 5: Run Again

Run the training script again:

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

You'll either:
- Hit the **next** `NotImplementedError` (good progress!)
- Get an error in your implementation (debug and fix it)

### Step 6: Repeat

Keep iterating: implement → run → debug → implement → run → debug...

---

## Implementation Order

The debug output will guide you through these components in order:

### Phase 1: Model Initialization

1. **GPTConfig dataclass** (already done - just data fields)
2. **GPT.__init__** - Set up model architecture
   - Create embedding layer (`wte`)
   - Create transformer blocks (`h`)
   - Create language modeling head (`lm_head`)
   - Setup rotary embeddings
3. **GPT._precompute_rotary_embeddings** - Create rotary embedding cache
4. **Block.__init__** - Initialize one transformer block
5. **CausalSelfAttention.__init__** - Initialize attention mechanism
6. **MLP.__init__** - Initialize feed-forward network

### Phase 2: Weight Initialization

7. **GPT.init_weights** - Initialize all model weights
8. **GPT._init_weights** - Initialize individual module weights

### Phase 3: Forward Pass

9. **norm()** function - RMS normalization
10. **apply_rotary_emb()** function - Rotary position embeddings
11. **CausalSelfAttention.forward** - Attention computation
12. **MLP.forward** - Feed-forward network
13. **Block.forward** - One transformer layer
14. **GPT.forward** - Full model forward pass

### Phase 4: Optimization

15. **GPT.setup_optimizers** - Create Muon and AdamW optimizers
16. **GPT.get_device** - Helper for device detection
17. **GPT.estimate_flops** - FLOPs estimation (optional)

### Phase 5: Generation (Optional)

18. **GPT.generate** - Autoregressive text generation
    - Only needed for inference, not training
    - Can skip this initially

---

## Tips for Success

### Understand, Don't Just Copy

The goal is **learning**, not completing quickly:
- Read the reference implementation carefully
- Understand what each line does
- Ask yourself "why is this needed?"
- Try to implement from memory after understanding

### Use the Debug Output

The debug statements show you:
- When each method is called
- What the input shapes are
- The flow of execution

Example output:
```
[DEBUG] Block.forward() called
[DEBUG]   Input shape: torch.Size([1, 512, 256])
[DEBUG] CausalSelfAttention.forward() called for layer 0
[DEBUG]   Input shape: torch.Size([1, 512, 256])
[DEBUG] norm() called with input shape: torch.Size([1, 512, 256])
```

### Test Incrementally

After implementing each method:
1. Run the training script
2. Verify the next debug message appears
3. Confirm shapes and values make sense

### Compare Outputs

As you implement methods, compare debug output with what you'd expect:
- Are tensor shapes correct?
- Are values reasonable (not NaN or inf)?
- Does the execution order make sense?

### Keep Notes

Document what you learn:
- Why is QK normalization important?
- What do rotary embeddings do?
- Why use relu^2 instead of relu?

---

## Alternative: Full Training Loop

Once you've implemented enough to get through initialization, you can run the full training script:

```bash
bash scripts/local_cpu_train.sh
```

This will:
- Train the tokenizer
- Train your implementation from scratch
- Show you how loss decreases (or doesn't!)
- Generate samples to see if the model learns

**Warning**: This takes 1-3 hours. Use the quick iteration approach first.

---

## Detailed Implementation Guide

### 1. GPT.__init__

**What it does**: Sets up the entire model architecture.

**Key components**:
```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(config.vocab_size, config.n_embd),
    "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
})
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

**Gotcha**: Rotary embeddings need to be registered as buffers, not parameters.

### 2. norm() Function

**What it does**: RMS normalization without learnable parameters.

**Implementation**:
```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

**Why**: Simpler than LayerNorm, often works just as well.

### 3. apply_rotary_emb() Function

**What it does**: Applies rotary position embeddings for relative positional encoding.

**Key insight**: Rotate pairs of dimensions using cos/sin.

**Gotcha**: Must split the last dimension in half.

### 4. CausalSelfAttention.forward

**What it does**: Multi-query attention with rotary embeddings and QK normalization.

**Key steps**:
1. Project to Q, K, V
2. Apply rotary embeddings to Q and K
3. Normalize Q and K
4. Compute attention with causal mask
5. Project back to residual stream

**Gotcha**: Handle different cases (training vs inference, with/without KV cache).

### 5. GPT.forward

**What it does**: Full forward pass through the model.

**Key steps**:
1. Embed tokens
2. Apply norm to embeddings
3. Pass through all blocks
4. Apply final norm
5. Project to vocabulary (with softcapping)
6. Compute loss if targets provided

**Gotcha**: Softcapping prevents logits from exploding: `softcap * tanh(logits / softcap)`.

### 6. GPT.setup_optimizers

**What it does**: Creates separate optimizers for different parameter groups.

**Key insight**:
- **Muon** optimizer for matrix parameters (transformer blocks)
- **AdamW** optimizer for embeddings and lm_head

**Why**: Different parameter types benefit from different optimization strategies.

---

## Debugging Common Issues

### "RuntimeError: Expected all tensors to be on the same device"

**Problem**: Model parts on different devices (CPU vs GPU vs MPS).

**Solution**: Make sure rotary embeddings are on the same device as model weights.

### "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

**Problem**: Shape mismatch in linear layers or attention.

**Solution**:
1. Print shapes at each step
2. Verify `head_dim = n_embd // n_head`
3. Check reshape operations preserve total elements

### "Loss is NaN"

**Problem**: Numerical instability.

**Solution**:
1. Check weight initialization (not too large)
2. Verify softcapping is applied
3. Check for division by zero in norm
4. Verify learning rates aren't too high

### "Model outputs don't change"

**Problem**: Gradients not flowing or weights not updating.

**Solution**:
1. Verify residual connections are implemented
2. Check optimizer is actually stepping
3. Make sure parameters are in optimizer param groups

---

## Verification Checklist

After implementing each component, verify:

- [ ] No `NotImplementedError` when running that code path
- [ ] Debug output shows expected shapes
- [ ] No NaN or Inf values
- [ ] Gradients are non-zero
- [ ] Loss decreases over iterations
- [ ] Model can be saved and loaded
- [ ] Generated samples improve over training

---

## Learning Outcomes

By the end of this exercise, you'll understand:

**Architecture**:
- How transformers compose embeddings, blocks, and output heads
- Why we use residual connections and normalization
- What rotary embeddings do for positional encoding

**Attention**:
- How multi-query attention reduces KV cache size
- Why we normalize queries and keys
- How causal masking prevents looking ahead

**Training**:
- Why different parameters need different optimizers
- How gradient accumulation increases batch size
- What softcapping does for stability

**Implementation Details**:
- Device management for heterogeneous models
- Efficient inference with KV caching
- Memory-efficient attention implementations

---

## Next Steps

After completing your implementation:

1. **Compare Performance**
   - Does your implementation match the original's loss curve?
   - Are gradients similar?
   - Is memory usage comparable?

2. **Experiment**
   - Try different activation functions
   - Modify the attention mechanism
   - Add new architectural features

3. **Read the Paper**
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
   - [LLaMA](https://arxiv.org/abs/2302.13971) (for rotary embeddings)

4. **Dive Deeper**
   - Study the optimizer implementations (Muon, AdamW)
   - Understand the dataloader and tokenization
   - Explore the evaluation tasks

---

## Reference Files

**Your implementation**: `nanochat/normans_gpt.py`

**Reference implementation**: `nanochat/gpt.py`

**Training scripts**:
- `scripts/base_train.py` - Main training loop
- `scripts/local_cpu_train.sh` - Full pipeline

**Debug output**: Terminal where you run training

---

## Getting Help

**If you're stuck**:
1. Check the debug output for clues
2. Compare your code with `nanochat/gpt.py`
3. Verify tensor shapes match expectations
4. Run with `--num_iterations=1` for fast iteration
5. Add more debug prints to understand flow

**Common questions**:
- "What order should I implement methods?" → Follow the execution order from debug output
- "Can I skip methods?" → No, implement each one as you hit it
- "How do I know it's correct?" → Loss should eventually decrease
- "Can I use the original gpt.py?" → Yes, but only as reference. Write it yourself!

---

**Related Documentation**:
- [Local CPU Quickstart](../tutorials/01-local-cpu-quickstart.md)
- [Troubleshooting Guide](./troubleshoot-common-issues.md)
- [Project Architecture](../explanation/project-architecture.md)
