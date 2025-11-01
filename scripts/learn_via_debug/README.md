# Learn via Debug Scripts

This folder contains instrumented debugging scripts designed to help you understand the nanochat GPT implementation by tracing execution during actual training.

## Purpose

These scripts are for **learning and understanding**, not production training. They:
- Add detailed logging to show what happens inside the model
- Trace tensor shapes, statistics, and transformations
- Explain architectural components as they execute
- Use the same local CPU training parameters as `scripts/local_cpu_train.sh`

## Prerequisites

Before running these scripts, you should:

1. Complete the local CPU quickstart: `bash scripts/local_cpu_train.sh`
2. Have training data downloaded (automatically done by the quickstart)
3. Have the tokenizer trained (automatically done by the quickstart)

## Available Scripts

### `debug_gpt_components.py`

Traces the execution of key GPT components during training:

**Components traced:**
1. **Token embeddings (wte)** - How discrete tokens become continuous vectors
2. **Positional embeddings (RoPE)** - How position information is encoded via rotation
3. **Transformer Block stack** - Detailed pre-norm residual architecture:
   - **Part 1**: Pre-norm (RMSNorm) + **CausalSelfAttention** + Residual
     - Input/Output shape: [batch, seq_len, d_model] → [batch, seq_len, d_model]
     - Q, K, V projections with Multi-Query Attention (MQA) support
     - RoPE application to Q and K (not V)
     - QK normalization for training stability
     - Scaled dot-product attention with causal masking
     - Softmax and weighted sum over values
     - Head concatenation and output projection
   - **Part 2**: Pre-norm (RMSNorm) + MLP + Residual
   - Shows each step: normalization, projections, activations, residual additions
4. **RMSNorm** - Normalization without learnable parameters (shown before each operation)
5. **Language modeling head** - Final projection to vocabulary
6. **Model scaling** - Parameter breakdown and 561M target configuration

**Usage:**

```bash
# Basic usage with defaults (matches local_cpu_train.sh parameters)
python -m scripts.learn_via_debug.debug_gpt_components

# With custom parameters (still CPU-safe)
python -m scripts.learn_via_debug.debug_gpt_components \
    --depth=4 \
    --max_seq_len=512 \
    --num_iterations=5

# To trace the 561M parameter target configuration (WARNING: requires more RAM)
python -m scripts.learn_via_debug.debug_gpt_components \
    --depth=20 \
    --num_iterations=1
```

**Default parameters** (matches `scripts/local_cpu_train.sh`):
- `depth=4` - 4 transformer layers (~8M parameters)
- `max_seq_len=1024` - 1024 token context
- `device_batch_size=1` - 1 sequence per batch
- `total_batch_size=1024` - Total batch size in tokens
- `num_iterations=3` - Just 3 training iterations for debugging
- `device_type=cpu` - CPU-only mode

**What you'll see:**

The script provides detailed output showing:
- Input token shapes and statistics
- Token embedding transformation
- RoPE (Rotary Position Embeddings) computation
- For each transformer block (detailed trace for first 3 and last):
  - **Pre-norm architecture explanation**
  - **Part 1: CausalSelfAttention path**
    - Step 1a: RMSNorm before attention
    - Step 1b: **Detailed attention breakdown**:
      - [1b.i] Q, K, V projections (shows MQA if n_kv_head < n_head)
      - [1b.ii] RoPE application (Q before/after, K after, explains relative position)
      - [1b.iii] QK normalization (separate RMSNorm for Q and K)
      - [1b.iv] Transpose for multi-head (B,T,H,D → B,H,T,D)
      - [1b.v] Scaled dot-product attention (shows causal mask visualization for small T)
      - [1b.vi] Concatenate heads and output projection
    - Step 1c: Residual connection (x + attention output)
  - **Part 2: MLP path**
    - Step 2a: RMSNorm before MLP
    - Step 2b.i: MLP fc projection (expand to 4x dimension)
    - Step 2b.ii: ReLU² activation
    - Step 2b.iii: MLP output projection (back to model dimension)
    - Step 2c: Residual connection (x + MLP output)
- Final RMSNorm before LM head
- Logit computation and softcapping
- Loss computation
- Gradient information

**Sample output:**

```
================================================================================
                        MODEL SCALING ANALYSIS
================================================================================

Configuration:
  Depth (n_layer): 4
  Model dimension (n_embd): 256
  Number of heads (n_head): 2
  Head dimension: 128
  Sequence length: 1024
  Vocabulary size: 50304

Parameter Breakdown:
  Token embeddings (wte): 12,877,824 (81.5%)
  LM head (unembedding): 12,877,824 (81.5%)
  Transformer blocks: 1,576,960 (10.0%)
    - Attention per layer: 263,168
    - MLP per layer: 131,072

Total Parameters: 15,805,440 (15.8M)

================================================================================
                              FORWARD PASS #0
================================================================================

[Input tokens (idx)]
  Shape: (1, 1024)
  Dtype: torch.int64
  Device: cpu
  Mean: 24891.648438
  ...
```

## Learning Path

1. **Start here**: Run `debug_gpt_components.py` with defaults
2. **Read the output**: Follow the colored sections showing each component
3. **Compare to code**: Open `nanochat/gpt.py` and match the traced operations
4. **Experiment**: Try different `depth` values to see how model size scales
5. **Deep dive**: Modify the script to trace additional components you're curious about

## Important Notes

⚠️ **CPU Only**: These scripts are designed for local CPU training environments
- Default parameters match `scripts/local_cpu_train.sh`
- Safe for laptops and development machines
- Not optimized for GPU (use regular training scripts for that)

⚠️ **Memory Usage**:
- Default depth=4: ~500MB-1GB RAM
- depth=20: ~2-4GB RAM (the 561M parameter target)
- If you run out of memory, reduce `depth` or `max_seq_len`

⚠️ **Requires Training Data**:
- Run `bash scripts/local_cpu_train.sh` first to download data
- Or run `python -m nanochat.dataset -n 4` to download 4 shards

## Understanding the Output

### Color Coding

- **Purple/Header**: Section titles
- **Cyan**: Component names
- **Green**: Tensor values and statistics
- **Blue**: Explanatory text about what's happening
- **Yellow**: Important notes and formulas
- **Red**: Iteration markers

### Key Sections to Watch

1. **Token Embeddings**: See how the embedding matrix is shaped
2. **RoPE**: Understand how position info is encoded without learned params
3. **Block 0, 1, 2**: First few blocks show Q, K, V in detail
4. **Final Block**: Last block shows the output before LM head
5. **Gradients**: See how gradients flow backward

## Next Steps

After understanding the basics:

1. Study the attention mechanism in detail (`nanochat/gpt.py:51-110`)
2. Understand the MLP architecture (`nanochat/gpt.py:113-123`)
3. Read about RoPE: https://arxiv.org/abs/2104.09864
4. Explore the optimizer setup (`nanochat/gpt.py:213-242`)
5. Scale up to GPU training on HPC

## Questions?

If something is unclear:
1. Check the inline comments in `nanochat/gpt.py`
2. Read `docs/explanation/architecture.md`
3. Compare the debug output to the actual code line-by-line

## Contributing

To add more debug scripts:
1. Follow the naming pattern: `debug_<component>.py`
2. Use the same color scheme for consistency
3. Match local CPU training parameters by default
4. Add detailed explanatory text
5. Update this README
