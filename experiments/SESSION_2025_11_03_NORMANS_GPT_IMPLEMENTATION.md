# Session Summary: Norman's GPT Implementation - Nov 3, 2025

## Session Overview

**Date**: November 3, 2025
**Focus**: Begin implementing own GPT transformer by iterative development
**Approach**: Run training → Hit error → Implement → Repeat
**Duration**: Initial implementation session

---

## What Was Accomplished

### 1. Implemented Components ✅

#### GPT.__init__ (Main Model Architecture)
- Created transformer ModuleDict with:
  - `wte`: Token embeddings (vocab_size=65536 → n_embd=256)
  - `h`: Stack of n_layer=4 transformer blocks
- Created `lm_head`: Final projection to vocabulary
- Setup rotary embedding buffers (cos/sin precomputed)
- **File**: `nanochat/normans_gpt.py` lines 185-234

#### Block.__init__ (Transformer Block)
- Created two main components:
  - `attn`: CausalSelfAttention layer (information gathering)
  - `mlp`: MLP layer (information processing)
- Added detailed comments explaining each component
- **File**: `nanochat/normans_gpt.py` lines 161-183

### 2. Deep Learning Q&A Sessions ✅

Documented comprehensive understanding of:

**PyTorch Fundamentals**:
- What is `nn` and `nn.ModuleDict`
- How embeddings work as lookup tables
- Parameter registration and tracking

**Tokenization**:
- BPE algorithm (Byte Pair Encoding)
- How tokenizer creates 65,536 tokens from text
- Why BPE over other algorithms

**Embeddings**:
- When embeddings are initialized (random)
- How embeddings learn (backpropagation every iteration)
- What the 256 dimensions represent (distributed representations)
- Embeddings are learned parameters, not predefined

**Architecture**:
- Why n_embd = depth × 64 (aspect ratio, empirical choice)
- Why create n_layer blocks (hierarchical learning, depth = power)
- Block structure: Attention (gather) + MLP (process)

**Data Flow**:
- Matrix dimensions: (B, 512, 256) constant through blocks
- What "512 tokens" means (sequence length, not vocab)
- Inside Attention: temporary reshape to heads
- Inside MLP: expand 4x, then contract

**Sequence Length**:
- max_seq_len=512 parameter controls context window
- Longer sequences → better long-range learning
- Longer sequences → quadratically more compute
- Does NOT directly improve embedding quality
- 512 is sweet spot for learning (fast + capable)

### 3. Bug Fixes ✅

**Tokenizer Path Issue**:
- **Problem**: Script kept retraining tokenizer
- **Cause**: Checked `tokenizer.pkl` but actual path is `tokenizer/tokenizer.pkl`
- **Fix**: Updated `scripts/local_cpu_own_gpt_transformer.sh` line 161
- **Impact**: Saves ~10-15 minutes on subsequent runs

### 4. Documentation Updates ✅

**Updated Files**:
- `experiments/NORMANS_GPT_LEARNING_STATUS.md`
  - Added 3 learning session sections
  - Documented 20+ Q&A entries
  - Updated implementation checklist
  - Added "Where to Resume" section
- `CHANGELOG.md`
  - Added implementation progress
  - Documented bug fixes
  - Listed all learnings
- Created this session summary

---

## Key Insights Gained

### Technical Understanding

1. **Embeddings are just lookups**: `nn.Embedding` is literally indexing into a weight matrix
2. **Dimensions stay constant**: (B, 512, 256) through all blocks enables residual connections
3. **Attention communicates, MLP transforms**: Both needed for powerful learning
4. **Aspect ratio is empirical**: The "64" in n_embd = depth × 64 is a design choice, not math
5. **Sequence length affects context**: max_seq_len controls long-range learning, not embedding quality

### Learning Process

1. **Iterative approach works**: Hit error → Understand → Implement → Repeat
2. **Debug statements are valuable**: Show execution flow and data shapes
3. **Reference comparison essential**: Having working code (gpt.py) alongside helps
4. **Questions drive learning**: Each "why?" leads to deeper understanding
5. **Documentation is crucial**: Recording insights prevents re-learning

---

## Current State

### Implementation Status

**Completed** ✅:
- GPT.__init__
- Block.__init__

**Next to Implement**:
1. CausalSelfAttention.__init__ (next error when running)
2. MLP.__init__
3. Then forward pass methods

**Expected Next Error**:
```
[DEBUG] Initializing CausalSelfAttention for layer 0
NotImplementedError: TODO: Implement CausalSelfAttention.__init__
```

### Files Modified

1. `nanochat/normans_gpt.py`
   - Implemented GPT.__init__ (lines 185-234)
   - Implemented Block.__init__ (lines 161-183)

2. `scripts/local_cpu_own_gpt_transformer.sh`
   - Fixed tokenizer path (line 161)

3. `experiments/NORMANS_GPT_LEARNING_STATUS.md`
   - Added 3 session sections
   - Updated all tracking sections

4. `CHANGELOG.md`
   - Documented today's progress

---

## How to Resume Tomorrow

### Quick Start

```bash
# 1. Navigate to project directory
cd /Users/norman.jarvis/forge/work/code/coderockit/msu-phd/slurm-nanochat

# 2. Run the iteration script
bash scripts/local_cpu_own_gpt_transformer.sh

# 3. You'll hit CausalSelfAttention.__init__ error
# 4. Open nanochat/normans_gpt.py
# 5. Reference nanochat/gpt.py lines 69-81
# 6. Implement the __init__ method
# 7. Run again
```

### What to Implement Next

**CausalSelfAttention.__init__** needs:
- Store config parameters (n_head, n_kv_head, n_embd, head_dim)
- Create Q projection: `self.c_q = nn.Linear(...)`
- Create K projection: `self.c_k = nn.Linear(...)`
- Create V projection: `self.c_v = nn.Linear(...)`
- Create output projection: `self.c_proj = nn.Linear(...)`
- Handle Multi-Query Attention (different Q vs K/V heads)

**MLP.__init__** needs:
- Create expansion layer: `self.c_fc = nn.Linear(n_embd, n_embd * 4)`
- Create contraction layer: `self.c_proj = nn.Linear(n_embd * 4, n_embd)`

### Reference Materials

- **Your implementation**: `nanochat/normans_gpt.py`
- **Working reference**: `nanochat/gpt.py`
- **Learning status**: `experiments/NORMANS_GPT_LEARNING_STATUS.md`
- **How-to guide**: `docs/how-to/03-how-to-write-your-own-gpt-transformer.md`

---

## Questions for Future Sessions

### Architecture Questions (To Answer Later)

- [ ] Why untie embedding and lm_head weights?
- [ ] What's the advantage of Multi-Query Attention?
- [ ] Why use relu^2 instead of regular relu or GELU?
- [ ] What does QK normalization prevent?

### Implementation Questions (To Answer Later)

- [ ] Why are rotary embeddings registered as buffers not parameters?
- [ ] How does gradient accumulation simulate larger batches?
- [ ] Why normalize after embedding instead of before?
- [ ] What does softcapping prevent?

### Training Questions (To Answer Later)

- [ ] Why do embeddings need a different optimizer than matrices?
- [ ] How does the learning rate scaling formula work?
- [ ] What's the advantage of no bias in linear layers?
- [ ] Why use bfloat16 for embeddings and rotary cache?

---

## Milestones Achieved

- [x] Setup complete (skeleton + imports)
- [x] GPT.__init__ implemented
- [x] Deep understanding of fundamentals (embeddings, tokenization, architecture)
- [x] Block.__init__ implemented
- [x] Data flow understanding (dimensions, sequences, context)
- [ ] Model initializes (need Attention + MLP next)
- [ ] Forward pass works
- [ ] Training step completes
- [ ] Loss decreases
- [ ] Full pipeline works

---

## Statistics

**Lines of Code Implemented**: ~50 lines
**Q&A Sessions Documented**: 3 major sessions
**Questions Answered**: 20+
**Bugs Fixed**: 1 (tokenizer path)
**Files Modified**: 4
**Learning Hours**: ~2-3 hours of deep learning

---

## Notes for Tomorrow

1. **Start with the resume section** in NORMANS_GPT_LEARNING_STATUS.md
2. **Run the script first** to see the exact error
3. **Read the reference code** before implementing
4. **Understand, don't copy** - the goal is learning
5. **Document questions** as they arise
6. **Update status** after each implementation

---

**Session completed successfully! Ready to resume tomorrow with CausalSelfAttention.__init__** 🎉
