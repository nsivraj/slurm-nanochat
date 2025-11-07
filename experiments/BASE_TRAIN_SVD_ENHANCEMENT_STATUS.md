# BASE_TRAIN_SVD_ENHANCEMENT - Project Status

**Date**: November 7, 2025
**Project**: Adaptive SVD-Based Training for nanochat Base Training
**Goal**: Detect and avoid clobbering learned weights during training using dynamic SVD analysis

---

## Session Summary

### What We Accomplished Today

#### 1. ✅ Analyzed Training Loop and Model Architecture
- **File**: `scripts/base_train.py` (lines 181-315) - Main training loop identified
- **File**: `nanochat/gpt.py` (lines 51-200) - Model architecture analyzed
- **Key findings**:
  - Base training: ~7-8 hours, 21,400 steps, 561M parameters (d20 model)
  - Training phases: Base → Midtraining → SFT
  - Main matrices to enhance: `c_q`, `c_k`, `c_v`, `c_proj` (attention), `c_fc`, `c_proj` (MLP)

#### 2. ✅ Documented Weight Initialization
- **File**: `experiments/BASE_TRAIN_SVD_ENHANCEMENT.md`
- Explained initialization for:
  - `nn.Linear`: `std ≈ 0.02795` (scaled Gaussian)
  - `nn.Embedding`: `std = 1.0` (35x larger than Linear!)
  - Special zero initializations: `lm_head`, `attn.c_proj`, `mlp.c_proj` (~30% of weights start at zero)
- Added concrete examples with sample values for d20 model

#### 3. ✅ Designed Custom Adaptive SVD Approach
- **File**: `experiments/BASE_TRAIN_SVD_ENHANCEMENT.md` (lines 359-696)
- **Key differences from standard MiLoRA**:
  - Wₚ is **re-computed** every SVD cycle (not frozen)
  - r **starts large and decreases** (opposite of typical LoRA)
  - Training switches between **full matrix** and **B/A** dynamically
  - Goal: **Detect clobbering before it happens**

#### 4. ✅ Implemented SVD Comparison Metrics
All metric functions designed and documented with full Python code:

**Metric 1: Subspace Angular Distance**
- `compute_subspace_angle(U1, U2, k)` - Detects rotation of principal components
- Threshold: `angle > 0.1 radians` (5.7°) = significant rotation

**Metric 2: Singular Value Concentration**
- `compute_singular_value_concentration(S, percentile)` - Tracks information spread
- Higher concentration = mature patterns, lower = learning new patterns

**Metric 3: Reconstruction Error**
- `compute_reconstruction_error(W, U, S, V, r)` - Safety check before compression
- Threshold: `error < 0.01` (1%) = safe to compress

**Metric 4: Gradient Alignment** ⭐ **CLOBBERING DETECTOR**
- `compute_gradient_alignment(grad_W, U, S, r)` - Measures gradient vs principal/minor alignment
- Threshold: `principal_align > 0.4` = DANGER, about to clobber!

#### 5. ✅ Designed Adaptive r Selection Strategy
- `determine_next_r(S, S_prev, r_current, r_min)` - Auto-adjusts rank
- Uses:
  - Spectral gaps (natural cutoff points)
  - Energy thresholds (99.9% capture)
  - Rate of change in bottom singular values

#### 6. ✅ Created Master Decision Logic
- `should_use_lowrank_training(...)` - Decides: Full matrix or B/A?
- Rules:
  - **Switch to B/A**: Gradients attacking principals OR optimal conditions
  - **Stay full matrix**: Reconstruction unsafe OR warmup period
  - **Continue current**: Neutral conditions

---

## Current State

### Files Created/Modified

1. **`experiments/BASE_TRAIN_SVD_ENHANCEMENT.md`** ✅ COMPLETE
   - Full documentation of initialization
   - Complete implementation plan with all functions
   - Ready for code implementation

2. **`experiments/BASE_TRAIN_SVD_ENHANCEMENT_STATUS.md`** ✅ THIS FILE
   - Status tracking document

### Files Ready to Create (Next Steps)

1. **`nanochat/adaptive_svd.py`** ⏳ PENDING
   - New module with all SVD comparison metrics
   - All functions from lines 398-684 of BASE_TRAIN_SVD_ENHANCEMENT.md

2. **`scripts/base_train_adaptive_svd.py`** ⏳ PENDING
   - Modified version of `scripts/base_train.py`
   - Integrates adaptive SVD logic into training loop

3. **`tests/test_adaptive_svd.py`** ⏳ PENDING
   - Unit tests for SVD metrics on toy matrices
   - Validation before full training run

---

## Architecture Design Summary

### Training Flow with Adaptive SVD

```
INITIALIZATION:
  1. Initialize model weights (standard initialization)
  2. Compute initial SVD: W₀ = U₀Σ₀V₀ᵀ
  3. Set initial r = rank(W) - 10 (start large, nearly full rank)
  4. Training mode = "full_matrix"

TRAINING LOOP (each step):
  ┌─────────────────────────────────────┐
  │ 1. Forward pass (using current W)   │
  │ 2. Compute loss                      │
  │ 3. Backward pass (get grad_W)       │
  └─────────────────────────────────────┘
           │
           ├─ Every N steps (e.g., N=100): ANALYZE & ADAPT
           │  ┌──────────────────────────────────────────────┐
           │  │ A. Compute current SVD: W = UΣVᵀ            │
           │  │ B. Compare with previous SVD (U_prev, etc.)  │
           │  │ C. Compute 4 metrics:                        │
           │  │    - Subspace angle                          │
           │  │    - Gradient alignment ⭐                   │
           │  │    - Reconstruction error                    │
           │  │    - Singular value concentration            │
           │  │ D. Decide: should_use_lowrank_training()     │
           │  │    → Returns: use_lowrank, r_next            │
           │  │ E. If mode change needed:                    │
           │  │    - Decompose: W = Wₚ + Wₘ                 │
           │  │    - Initialize: Bₘ = Uₘ√Σₘ, Aₘ = √ΣₘVₘᵀ   │
           │  │ F. Update r for next cycle                   │
           │  │ G. Store current SVD as "previous"           │
           │  └──────────────────────────────────────────────┘
           │
           ├─ UPDATE WEIGHTS:
           │  ┌──────────────────────────────────────────────┐
           │  │ IF mode == "full_matrix":                    │
           │  │     optimizer.step()  # Update all of W      │
           │  │                                              │
           │  │ IF mode == "lowrank":                        │
           │  │     Re-compute Wₚ from current W             │
           │  │     Only update Bₘ and Aₘ                   │
           │  │     W_effective = Wₚ + Bₘ @ Aₘ               │
           │  └──────────────────────────────────────────────┘
           │
           ↓
    Continue to next training step
```

### Key Differences from Standard MiLoRA

| Aspect                | Standard MiLoRA           | Our Adaptive Approach          |
| --------------------- | ------------------------- | ------------------------------ |
| Wₚ (principal matrix) | Frozen once at init       | ✨ **Re-computed every cycle** |
| r (rank)              | Fixed (e.g., 128)         | ✨ **Dynamic, starts large**   |
| When to compress      | At initialization only    | ✨ **Based on metrics**        |
| Training target       | Always B and A            | ✨ **Full W OR B/A**           |
| Use case              | Fine-tuning (mid/SFT)     | ✨ **Base training**           |
| Clobbering detection  | Not applicable            | ✨ **Core feature!**           |

---

## Key Research Insights

### The Clobbering Problem

**Current training** (~7 hours base training):
- 21,400 steps updating ALL 561M parameters
- Early steps learn broad patterns (principal components form)
- Later steps can overwrite early learning (catastrophic forgetting)
- No mechanism to detect when this is happening

**Our solution**:
- Monitor gradient alignment with principal components
- If `principal_align > 0.4`: **DANGER!** About to clobber
- Switch to low-rank training, updating only minor components
- Principal patterns preserved, new learning happens in orthogonal subspace

### Why This Is Novel

1. **Dynamic Wₚ re-computation**: Unlike MiLoRA which freezes Wₚ, we re-compute it to capture newly learned stable patterns
2. **Adaptive r**: Responds to training dynamics, not fixed in advance
3. **Clobbering detection**: Uses gradient alignment as early warning system
4. **Base training focus**: Most research uses LoRA for fine-tuning; we're enhancing base pretraining

---

## Next Steps (Prioritized)

### Immediate Next Steps (Today/Tomorrow)

1. **Create `nanochat/adaptive_svd.py`** ⏰ NEXT
   - Copy functions from BASE_TRAIN_SVD_ENHANCEMENT.md (lines 398-684)
   - Add proper imports, docstrings, type hints
   - Estimated time: 30-45 minutes

2. **Create `tests/test_adaptive_svd.py`** ⏰ AFTER STEP 1
   - Test each metric on toy 10x10 matrices
   - Verify decision logic with known scenarios
   - Estimated time: 30 minutes

3. **Test on toy examples** ⏰ VALIDATION
   - Create 2-3 training scenarios with small matrices
   - Verify metrics detect expected patterns
   - Estimated time: 20 minutes

### Medium-Term Steps (This Week)

4. **Modify `scripts/base_train.py`** ⏰ INTEGRATION
   - Create `scripts/base_train_adaptive_svd.py` (copy + modify)
   - Integrate adaptive SVD logic into training loop
   - Add SVD tracking state between steps
   - Add logging for all metrics
   - Estimated time: 2-3 hours

5. **Local CPU test run** ⏰ VALIDATION
   - Run with `depth=4` on local machine
   - Verify logic works end-to-end
   - Check that metrics are computed correctly
   - Estimated time: 1 hour setup + 2-3 hours training

### Long-Term Steps (Next Week)

6. **Full Ptolemy run** ⏰ DEPLOYMENT
   - Submit job with `depth=20` on 8xA100
   - Monitor metrics during training
   - Compare results with baseline
   - Estimated time: ~7-10 hours training + analysis

7. **Analysis and iteration** ⏰ RESEARCH
   - Analyze metric logs
   - Adjust thresholds if needed
   - Document findings
   - Write up results

---

## Open Questions / Decisions Needed

### Hyperparameter Choices

1. **How often to run SVD analysis?**
   - Option A: Every 100 steps (more responsive, more compute)
   - Option B: Every 500 steps (less overhead, might miss transitions)
   - **Recommendation**: Start with 100, adjust based on overhead

2. **Initial r value?**
   - Option A: r = rank(W) - 10 (very conservative, almost full rank)
   - Option B: r = rank(W) // 2 (moderate compression from start)
   - **Recommendation**: Start conservative (Option A)

3. **Which matrices to apply this to?**
   - Option A: ALL weight matrices (most comprehensive)
   - Option B: Only attention matrices (c_q, c_k, c_v, c_proj)
   - Option C: Only MLP matrices (c_fc, c_proj)
   - **Recommendation**: Start with Option B (attention), then expand

### Threshold Values

Current defaults (can tune based on experiments):
- Subspace angle threshold: `0.1 radians` (5.7°)
- Principal alignment danger: `> 0.4` (40% of gradient)
- Minor alignment safe: `> 0.6` (60% of gradient)
- Reconstruction error safe: `< 0.01` (1%)
- Warmup period: `10 steps`

**Action**: Run small-scale experiments to validate these are reasonable

---

## Code Locations Quick Reference

### Existing Files (Read-Only)
- **Training loop**: `scripts/base_train.py:181-315`
- **Model architecture**: `nanochat/gpt.py:51-200`
- **Weight initialization**: `nanochat/gpt.py:157-184`
- **MiLoRA reference**: `/Users/norman.jarvis/forge/work/code/coderockit/msu-phd/cse8990-nlp/20251006-04-assignment/milora-final-paper.md`

### New Files (To Create)
- **Metrics module**: `nanochat/adaptive_svd.py` (doesn't exist yet)
- **Modified training**: `scripts/base_train_adaptive_svd.py` (doesn't exist yet)
- **Tests**: `tests/test_adaptive_svd.py` (doesn't exist yet)

### Documentation
- **Main doc**: `experiments/BASE_TRAIN_SVD_ENHANCEMENT.md`
- **Status (this file)**: `experiments/BASE_TRAIN_SVD_ENHANCEMENT_STATUS.md`

---

## Resources and References

### Papers
- **MiLoRA**: Wang et al., NAACL 2025 - Harnessing Minor Singular Components
- **LoRA**: Hu et al., 2021 - Low-Rank Adaptation of Large Language Models
- **LASER**: Sharma et al., 2024 - Layer-Selective Rank Reduction

### Key Concepts
- **SVD**: `W = UΣVᵀ` - Splits matrix into principal (large σᵢ) and minor (small σᵢ) components
- **Principal components**: Top (g-r) components, capture essential patterns
- **Minor components**: Bottom r components, safe to modify
- **Clobbering**: When new training overwrites previously learned patterns
- **Gradient alignment**: Measuring which subspace gradients affect

---

## Estimated Timeline

**Today (Nov 7)**:
- ✅ Analysis and design (COMPLETE)
- ⏰ Status documentation (IN PROGRESS)

**Next session (Later today or tomorrow)**:
- Create `nanochat/adaptive_svd.py` (45 min)
- Create `tests/test_adaptive_svd.py` (30 min)
- Run toy validation tests (20 min)
- **Total: ~1.5-2 hours**

**This week**:
- Modify training script (2-3 hours)
- Local CPU test run (3-4 hours total)

**Next week**:
- Full Ptolemy run (7-10 hours training)
- Analysis and iteration (variable)

---

## Success Criteria

### Immediate Success (Toy Tests)
- ✅ Metrics compute without errors
- ✅ Decision logic returns expected results for known scenarios
- ✅ Subspace angle detects rotation correctly

### Medium-Term Success (CPU Run)
- ✅ Training completes without crashes
- ✅ Metrics logged correctly throughout training
- ✅ Mode switches happen (full → lowrank or vice versa)
- ✅ Final model can generate coherent text

### Long-Term Success (Ptolemy Run)
- ✅ Training time competitive with baseline (~7-8 hours)
- ✅ Final loss comparable or better than baseline
- ✅ Metrics show clobbering detection working
- ✅ Evidence that r decreases over training (compression happening)
- ✅ Benchmark scores (CORE metric) at least as good as baseline

---

## Notes for Resuming Work

When you come back to this project:

1. **Read this file first** to understand current state
2. **Check**: `experiments/BASE_TRAIN_SVD_ENHANCEMENT.md` - Contains all the implementation details
3. **Start with**: Creating `nanochat/adaptive_svd.py` using functions from lines 398-684 of the main doc
4. **Remember**: The goal is to detect gradient clobbering BEFORE it happens using the 4 metrics

### Key Implementation Detail to Remember

The core innovation is in the training loop modification:

```python
# Current training (simplified):
for step in range(num_iterations):
    loss = model(x, y)
    loss.backward()
    optimizer.step()  # Updates ALL weights

# Our adaptive approach:
for step in range(num_iterations):
    loss = model(x, y)
    loss.backward()

    if step % svd_interval == 0:
        # ANALYZE: Compare SVD, compute metrics, decide mode
        use_lowrank, r_next, diagnostics = should_use_lowrank_training(...)

        if use_lowrank != current_mode:
            # SWITCH MODES
            if use_lowrank:
                # Decompose W into Wₚ + Bₘ @ Aₘ
                # Freeze Wₚ, make Bₘ and Aₘ trainable
            else:
                # Merge back to full W
                # Make W trainable again

    # UPDATE based on current mode
    if current_mode == "lowrank":
        # Only update Bₘ and Aₘ
        optimizer_lowrank.step()
    else:
        # Update full W
        optimizer.step()
```

---

## Implementation Roadmap (3-Phase Approach)

### Phase 1: Minimal Proof-of-Concept ⏰ CURRENT PHASE

**Goal**: Validate that adaptive SVD mechanics work without crashing

**Scope**:
- Single layer: Only `c_q` in transformer block 0
- Small model: depth=4 (4 layers, ~10M parameters)
- Short run: 100 training steps
- Local execution: CPU or single GPU

**Success Criteria**:
1. ✅ Training completes without errors
2. ✅ SVD metrics are computed correctly every `svd_interval` steps
3. ✅ Mode switching occurs (at least one switch between full/lowrank)
4. ✅ Logged metrics make sense (angles, alignments, reconstruction errors)
5. ✅ Model can still generate text after training

**Deliverables**:
1. `nanochat/gpt.py` - Add `AdaptiveSVDLinear` class
2. `scripts/base_train_adaptive_svd.py` - Modified training script
3. Validation run output showing metrics and mode switches

**Timeline**: 1-2 sessions (3-4 hours)

---

### Phase 2: Full Attention Coverage, Small Scale ⏳ AFTER PHASE 1

**Goal**: Verify approach works for all attention layers in a complete training run

**Scope**:
- All attention layers: `c_q`, `c_k`, `c_v`, `c_proj` in all 4 blocks
- Small model: depth=4 (4 layers, ~10M parameters)
- Full training: Complete base training (~2000-3000 steps)
- Local or single GPU execution

**Success Criteria**:
1. ✅ All attention layers switch modes appropriately
2. ✅ Final validation loss comparable to baseline (within 5%)
3. ✅ Model generates coherent text
4. ✅ At least one instance of clobbering detection (principal_alignment > 0.4)
5. ✅ r decreases over training for at least some layers

**Deliverables**:
1. Updated `nanochat/gpt.py` - All attention layers use `AdaptiveSVDLinear`
2. Training logs with all metrics
3. Comparison with baseline depth=4 model
4. Analysis document showing clobbering detection in action

**Timeline**: 2-3 sessions (4-6 hours dev + 2-3 hours training)

**Blockers**: Must complete Phase 1 successfully first

---

### Phase 3: Production Deployment ⏳ AFTER PHASE 2

**Goal**: Run full-scale experiment on production model

**Scope**:
- All attention layers in all 20 blocks
- Production model: depth=20 (20 layers, 561M parameters)
- Full base training: ~21,400 steps, 7-8 hours on 8xA100
- Ptolemy cluster execution

**Success Criteria**:
1. ✅ Training completes successfully on Ptolemy
2. ✅ Training time competitive with baseline (~7-8 hours)
3. ✅ Final validation loss within 3% of baseline
4. ✅ CORE metric comparable or better than baseline
5. ✅ Evidence of successful clobbering prevention
6. ✅ Compression happening (r decreasing for most layers)

**Deliverables**:
1. Slurm script for Ptolemy execution
2. Full training logs with all SVD metrics
3. Comprehensive analysis comparing with baseline
4. Research write-up documenting findings
5. Plots showing:
   - r evolution over training
   - Principal alignment spikes (clobbering events)
   - Mode switching patterns
   - Training loss comparison

**Timeline**: 1-2 sessions setup (2-4 hours) + 7-8 hours training + 1 session analysis (2-3 hours)

**Blockers**: Must complete Phase 2 successfully first

**Optional Extensions**:
- Apply to MLP layers (`c_fc`, `c_proj`)
- Experiment with different `svd_interval` values (50, 100, 500)
- Try different threshold values for decision logic
- Compare multiple runs with different random seeds

---

## Current Session Progress (November 7, 2025)

### Completed Today ✅

1. **Created `nanochat/adaptive_svd.py`** - All 6 metric functions implemented
   - `compute_subspace_angle()` - Principal subspace rotation detection
   - `compute_singular_value_concentration()` - Information distribution tracking
   - `compute_reconstruction_error()` - Compression safety check
   - `compute_gradient_alignment()` - **CLOBBERING DETECTOR** ⭐
   - `determine_next_r()` - Adaptive rank selection
   - `should_use_lowrank_training()` - Master decision logic

2. **Created `tests/test_adaptive_svd.py`** - Comprehensive test suite
   - 22 unit tests, all passing ✅
   - Tests for each individual metric function
   - Integration tests for complete workflow
   - Clobbering detection validation

3. **Enhanced documentation with U vs V explanation**
   - Updated `nanochat/adaptive_svd.py` with detailed docstrings
   - Added comprehensive section to `experiments/BASE_TRAIN_SVD_ENHANCEMENT.md`
   - Explains why we track left singular vectors (output space patterns)

4. **Created `experiments/SVD_ENHANCEMENT_ARCHITECTURE_MODIFICATIONS.md`**
   - Complete `AdaptiveSVDLinear` class design
   - Training loop integration strategy
   - Optimizer handling approach
   - 3-phase implementation roadmap
   - Expected challenges and solutions

### Ready to Start: Phase 1 Implementation ⏰

**Next immediate steps**:
1. Add `AdaptiveSVDLinear` class to `nanochat/gpt.py`
2. Modify GPTConfig to support `adaptive_svd` flag
3. Create `scripts/base_train_adaptive_svd.py` with SVD analysis integration
4. Run Phase 1 validation test (depth=4, 100 steps)
5. Analyze results and verify all metrics are working

---

**Status**: Phase 1 implementation ready to begin
**Confidence**: High - All design and testing complete
**Blockers**: None
**Last Updated**: November 7, 2025 (Session 2)
