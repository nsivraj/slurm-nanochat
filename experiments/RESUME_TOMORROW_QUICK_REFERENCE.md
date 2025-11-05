# Quick Reference - Resume Tomorrow

**Date**: 2025-11-03
**Next Task**: Implement `CausalSelfAttention.__init__` and `MLP.__init__`

---

## Start Here

```bash
# Run this command
bash scripts/local_cpu_own_gpt_transformer.sh
```

**You'll see**:
```
[DEBUG] Initializing CausalSelfAttention for layer 0
NotImplementedError: TODO: Implement CausalSelfAttention.__init__
```

---

## Files to Open

1. **nanochat/normans_gpt.py** - Your implementation (lines 89-119)
2. **nanochat/gpt.py** - Reference (lines 69-81)

---

## What to Implement

### CausalSelfAttention.__init__ (lines 108-112)

**Reference**: `nanochat/gpt.py` lines 69-81

**What it needs**:
```python
def __init__(self, config, layer_idx):
    super().__init__()
    self.layer_idx = layer_idx

    # Store dimensions
    self.n_head = config.n_head           # 2 (query heads)
    self.n_kv_head = config.n_kv_head     # 2 (key/value heads)
    self.n_embd = config.n_embd           # 256
    self.head_dim = config.n_embd // config.n_head  # 128

    # Create projection layers
    self.c_q = nn.Linear(...)  # Query projection
    self.c_k = nn.Linear(...)  # Key projection
    self.c_v = nn.Linear(...)  # Value projection
    self.c_proj = nn.Linear(...) # Output projection
```

**Key points**:
- Multi-Query Attention: Q uses n_head, K/V use n_kv_head
- All projections have `bias=False`
- Input: n_embd, Output varies by projection

### MLP.__init__ (lines 137-140)

**Reference**: `nanochat/gpt.py` lines 119-123

**What it needs**:
```python
def __init__(self, config):
    super().__init__()

    # Expand by 4x
    self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)

    # Contract back
    self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=False)
```

**Key points**:
- Expands from 256 → 1024
- Then contracts 1024 → 256
- No bias in either layer

---

## After Implementation

Run the script again:
```bash
bash scripts/local_cpu_own_gpt_transformer.sh
```

**Next error will likely be**:
```
NotImplementedError: TODO: Implement GPT._precompute_rotary_embeddings
```

---

## Documentation to Update

After implementing, update:

1. **experiments/NORMANS_GPT_LEARNING_STATUS.md**
   - Mark CausalSelfAttention.__init__ as complete
   - Mark MLP.__init__ as complete
   - Add any new questions/learnings

2. **CHANGELOG.md** (optional)
   - Add progress note if significant

---

## Quick Config Reference

Current configuration:
- `n_layer`: 4 (number of transformer blocks)
- `n_head`: 2 (query heads)
- `n_kv_head`: 2 (key/value heads)
- `n_embd`: 256 (embedding dimension)
- `head_dim`: 128 (n_embd // n_head)
- `vocab_size`: 65536
- `max_seq_len`: 512

Matrix dimensions:
- Through blocks: **(B, 512, 256)**
- MLP expansion: **(B, 512, 1024)**

---

## Documentation Files

- **Learning Status**: `experiments/NORMANS_GPT_LEARNING_STATUS.md`
- **Session Summary**: `experiments/SESSION_2025_11_03_NORMANS_GPT_IMPLEMENTATION.md`
- **How-To Guide**: `docs/how-to/03-how-to-write-your-own-gpt-transformer.md`
- **CHANGELOG**: `CHANGELOG.md`

---

## Remember

✅ **Understand, don't just copy** - Read the reference, understand it, then write it yourself
✅ **Run after each change** - See the next error immediately
✅ **Document questions** - Add them to the learning status file
✅ **Check shapes** - Matrix dimensions should make sense

---

**Ready to continue! Good luck! 🚀**
