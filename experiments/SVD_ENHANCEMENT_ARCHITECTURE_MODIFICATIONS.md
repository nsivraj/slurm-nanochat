# Architecture Modifications for Adaptive SVD Training

**Date**: November 7, 2025
**Purpose**: Document the changes needed to integrate adaptive SVD training into nanochat

This document explains in detail what needs to be modified to enable dynamic switching between full-matrix and low-rank training modes during base training.

---

## Overview: What We're Building

We're modifying the training loop to support **two training modes** that can switch dynamically:

1. **Full Matrix Mode**: Train all weights W directly (current approach)
2. **Low-Rank Mode**: Train only low-rank decomposition Bₘ @ Aₘ while freezing Wₚ

The switch between modes is determined by SVD analysis every N steps (e.g., N=100).

---

## Key Architectural Challenges

### Challenge 1: Dual Training Modes

**Problem**: PyTorch optimizers expect a fixed set of parameters. We need to switch which parameters are trainable.

**Solution**: Maintain two sets of parameters for each target matrix:
- `W`: The full weight matrix (always exists, but may not be trainable)
- `B, A`: Low-rank factors (created on-demand when switching to low-rank mode)

### Challenge 2: Dynamic Parameter Creation

**Problem**: We can't add new parameters to a model after the optimizer is created.

**Solution**: Pre-allocate ALL possible parameters (W, B, A for each target layer) at initialization, but control which ones have `requires_grad=True`.

### Challenge 3: Maintaining W = Wₚ + Bₘ @ Aₘ

**Problem**: During low-rank training, we need to:
- Keep Wₚ (principal components) frozen
- Update only Bₘ and Aₘ (minor components)
- Reconstruct W for forward passes

**Solution**: Use a custom forward hook that computes `W_effective = Wₚ + B @ A` before each forward pass when in low-rank mode.

---

## Required Modifications

### Modification 1: Extend GPT Model Class

**File**: `nanochat/gpt.py`

**What to add**: A wrapper class for adaptive SVD-enabled layers

```python
class AdaptiveSVDLinear(nn.Module):
    """
    A Linear layer that can switch between full-matrix and low-rank training.

    Attributes:
        W: Full weight matrix (d_out, d_in) - always present
        B: Low-rank left factor (d_out, r) - created when needed
        A: Low-rank right factor (r, d_in) - created when needed
        W_principal: Frozen principal components (d_out, d_in)
        bias: Optional bias (same as nn.Linear)

        mode: "full" or "lowrank" - current training mode
        r: Current rank for low-rank decomposition

    SVD tracking state:
        U_prev, S_prev, V_prev: Previous SVD for comparison
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # Primary weight matrix (always present)
        self.W = nn.Parameter(torch.empty(out_features, in_features))

        # Low-rank factors (initialized as None, created on first switch)
        self.register_buffer('B', None)  # Will be (out_features, r)
        self.register_buffer('A', None)  # Will be (r, in_features)
        self.register_buffer('W_principal', None)  # Frozen Wₚ

        # Bias (standard)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Training mode state
        self.mode = "full"  # Start in full-matrix mode
        self.r = None  # Current rank (set when switching to low-rank)

        # SVD tracking (for comparison between steps)
        self.register_buffer('U_prev', None)
        self.register_buffer('S_prev', None)
        self.register_buffer('V_prev', None)

    def forward(self, x):
        """Forward pass - use W or (Wₚ + B @ A) depending on mode."""
        if self.mode == "full":
            # Standard linear layer
            return F.linear(x, self.W, self.bias)
        else:
            # Low-rank mode: W_effective = W_principal + B @ A
            W_effective = self.W_principal + self.B @ self.A
            return F.linear(x, W_effective, self.bias)

    def switch_to_lowrank(self, r):
        """Switch from full-matrix to low-rank training."""
        # 1. Compute SVD of current W
        U, S, Vh = torch.linalg.svd(self.W.data, full_matrices=False)
        V = Vh.T

        g = len(S)
        k = g - r  # Number of principal components

        # 2. Decompose: W = Wₚ + Wₘ
        # Wₚ = top-k components (frozen)
        U_p = U[:, :k]
        S_p = S[:k]
        V_p = V[:, :k]
        self.W_principal = U_p @ torch.diag(S_p) @ V_p.T
        self.W_principal.requires_grad = False

        # Wₘ = bottom-r components (trainable as B @ A)
        U_m = U[:, k:]
        S_m = S[k:]
        V_m = V[:, k:]

        # 3. Initialize B and A using MiLoRA approach
        # B = Uₘ √Σₘ,  A = √Σₘ Vₘᵀ
        sqrt_S_m = torch.sqrt(torch.diag(S_m))
        self.B = nn.Parameter(U_m @ sqrt_S_m)
        self.A = nn.Parameter(sqrt_S_m @ V_m.T)

        # 4. Freeze W, make B and A trainable
        self.W.requires_grad = False
        self.B.requires_grad = True
        self.A.requires_grad = True

        # 5. Update mode and rank
        self.mode = "lowrank"
        self.r = r

        # 6. Store SVD for next comparison
        self.U_prev = U.clone()
        self.S_prev = S.clone()
        self.V_prev = V.clone()

    def switch_to_full(self):
        """Switch from low-rank to full-matrix training."""
        # 1. Merge B @ A back into W
        if self.mode == "lowrank":
            self.W.data = self.W_principal + self.B @ self.A

        # 2. Make W trainable, freeze B and A
        self.W.requires_grad = True
        if self.B is not None:
            self.B.requires_grad = False
        if self.A is not None:
            self.A.requires_grad = False

        # 3. Update mode
        self.mode = "full"

    def update_svd_tracking(self):
        """Update stored SVD for comparison at next cycle."""
        with torch.no_grad():
            if self.mode == "full":
                W_current = self.W.data
            else:
                W_current = self.W_principal + self.B @ self.A

            U, S, Vh = torch.linalg.svd(W_current, full_matrices=False)
            V = Vh.T

            self.U_prev = U.clone()
            self.S_prev = S.clone()
            self.V_prev = V.clone()
```

**Integration into GPT class**:

```python
# In nanochat/gpt.py, modify the CausalSelfAttention and MLP classes:

class CausalSelfAttention(nn.Module):
    def __init__(self, config, adaptive_svd=False):
        super().__init__()
        # ...existing code...

        if adaptive_svd:
            # Use adaptive SVD layers
            self.c_q = AdaptiveSVDLinear(config.n_embd, config.n_embd, bias=False)
            self.c_k = AdaptiveSVDLinear(config.n_embd, config.n_embd, bias=False)
            self.c_v = AdaptiveSVDLinear(config.n_embd, config.n_embd, bias=False)
            self.c_proj = AdaptiveSVDLinear(config.n_embd, config.n_embd, bias=False)
        else:
            # Standard layers (current approach)
            self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
            # ...etc...
```

---

### Modification 2: Training Loop Integration

**File**: `scripts/base_train_adaptive_svd.py` (new file, modified from `base_train.py`)

**Key changes**:

#### A. Add SVD Analysis Every N Steps

```python
# New hyperparameter
svd_interval = 100  # How often to run SVD analysis and potentially switch modes
target_layers = ['attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj']  # Which layers to apply to

# In training loop (after gradient computation, before optimizer.step())
if step > 0 and step % svd_interval == 0:
    # Run SVD analysis on all target layers
    svd_decisions = analyze_all_layers(model, step)

    # Log decisions
    for layer_name, decision in svd_decisions.items():
        wandb_run.log({
            f"svd/{layer_name}/mode": decision['mode'],
            f"svd/{layer_name}/r": decision['r'],
            f"svd/{layer_name}/principal_alignment": decision['principal_alignment'],
            f"svd/{layer_name}/subspace_angle": decision['subspace_angle'],
            # ...other metrics...
        })
```

#### B. SVD Analysis Function

```python
def analyze_all_layers(model, step):
    """
    Analyze all target layers and make mode-switching decisions.

    Returns dict mapping layer_name -> decision_info
    """
    from nanochat.adaptive_svd import should_use_lowrank_training

    decisions = {}

    # Iterate through all transformer blocks
    for block_idx, block in enumerate(model.transformer.h):
        # Check each attention projection
        for layer_name in ['c_q', 'c_k', 'c_v', 'c_proj']:
            full_name = f"block{block_idx}.attn.{layer_name}"
            layer = getattr(block.attn, layer_name)

            # Skip if not AdaptiveSVDLinear
            if not isinstance(layer, AdaptiveSVDLinear):
                continue

            # Get current state
            if layer.mode == "full":
                W_current = layer.W.data
            else:
                W_current = layer.W_principal + layer.B @ layer.A

            # Get gradient
            grad_W = layer.W.grad if layer.W.grad is not None else torch.zeros_like(layer.W)

            # Need previous SVD for comparison
            if layer.U_prev is None:
                # First time - compute initial SVD
                layer.update_svd_tracking()
                decisions[full_name] = {
                    'mode': layer.mode,
                    'r': layer.r,
                    'reason': 'initial_svd_computation'
                }
                continue

            # Compute current SVD
            U_current, S_current, Vh_current = torch.linalg.svd(W_current, full_matrices=False)
            V_current = Vh_current.T

            # Get previous SVD
            U_prev = layer.U_prev
            S_prev = layer.S_prev
            V_prev = layer.V_prev
            W_prev = U_prev @ torch.diag(S_prev) @ V_prev.T

            # Determine current r (or default)
            r_current = layer.r if layer.r is not None else len(S_current) - 10

            # Make decision
            use_lowrank, r_next, diagnostics = should_use_lowrank_training(
                W_current, U_current, S_current, V_current,
                W_prev, U_prev, S_prev, V_prev,
                grad_W, r_current, step
            )

            # Execute decision
            if use_lowrank is True and layer.mode == "full":
                # Switch to low-rank
                layer.switch_to_lowrank(r_next)
                action = "switched_to_lowrank"
            elif use_lowrank is False and layer.mode == "lowrank":
                # Switch to full
                layer.switch_to_full()
                action = "switched_to_full"
            else:
                # Continue current mode
                action = "continue_current"
                if layer.mode == "lowrank" and r_next != layer.r:
                    # Update r for next cycle
                    layer.r = r_next

            # Update tracking
            layer.update_svd_tracking()

            # Record decision
            decisions[full_name] = {
                'mode': layer.mode,
                'r': layer.r,
                'action': action,
                **diagnostics
            }

    return decisions
```

#### C. Optimizer Handling

**Challenge**: Standard optimizers don't handle dynamically changing parameters well.

**Solution**: Use separate optimizer groups for W vs (B, A), and update them as needed.

```python
# During optimizer setup
def setup_adaptive_optimizers(model, learning_rate):
    """
    Create optimizers that can handle dynamic parameter switching.
    """
    # Collect all parameters into groups
    full_params = []   # W parameters (when in full mode)
    lowrank_params = [] # B and A parameters (when in lowrank mode)

    for module in model.modules():
        if isinstance(module, AdaptiveSVDLinear):
            full_params.append(module.W)
            # B and A will be added dynamically

    # Create optimizer with both groups
    # (initially, only full_params will have requires_grad=True)
    optimizer = torch.optim.Adam([
        {'params': full_params, 'lr': learning_rate, 'name': 'full'},
        {'params': [], 'lr': learning_rate, 'name': 'lowrank'}  # Start empty
    ])

    return optimizer

# After mode switching, update optimizer param groups
def update_optimizer_params(optimizer, model):
    """Update optimizer to track current trainable parameters."""
    full_params = []
    lowrank_params = []

    for module in model.modules():
        if isinstance(module, AdaptiveSVDLinear):
            if module.mode == "full" and module.W.requires_grad:
                full_params.append(module.W)
            elif module.mode == "lowrank":
                if module.B is not None and module.B.requires_grad:
                    lowrank_params.append(module.B)
                if module.A is not None and module.A.requires_grad:
                    lowrank_params.append(module.A)

    # Update param groups
    optimizer.param_groups[0]['params'] = full_params
    optimizer.param_groups[1]['params'] = lowrank_params
```

---

### Modification 3: Logging and Monitoring

**What to log** (per layer, every SVD interval):

1. **Decision metrics**:
   - `mode`: "full" or "lowrank"
   - `r`: Current rank
   - `action`: "switched_to_lowrank", "switched_to_full", or "continue_current"
   - `reason`: Decision rationale

2. **SVD comparison metrics**:
   - `subspace_angle`: How much principal subspace rotated
   - `principal_alignment`: Gradient alignment with principal components (DANGER METRIC)
   - `minor_alignment`: Gradient alignment with minor components
   - `reconstruction_error`: Safety of compression
   - `concentration_ratio`: Information concentration

3. **Singular value statistics**:
   - `sigma_max`: Largest singular value
   - `sigma_min`: Smallest singular value
   - `effective_rank`: Number of values for 95% energy
   - `spectral_gap`: Largest ratio σᵢ/σᵢ₊₁

4. **Training efficiency**:
   - `num_trainable_params`: How many parameters are being updated
   - `compression_ratio`: (num params in B+A) / (num params in W)

---

## Implementation Strategy

### Phase 1: Minimal Integration (Recommended First Step)

1. Create `AdaptiveSVDLinear` class in `nanochat/gpt.py`
2. Add `adaptive_svd=False` flag to GPTConfig
3. Modify only `c_q` in one attention layer as proof-of-concept
4. Run on depth=4, device_batch_size=1 for 100 steps
5. Verify metrics are computed correctly and mode switching works

### Phase 2: Full Attention Coverage

1. Apply to all attention matrices (c_q, c_k, c_v, c_proj)
2. Test on depth=4 for full training run
3. Verify final model still generates coherent text

### Phase 3: Full Deployment

1. Apply to all 20 layers in d20 model
2. Consider adding MLP matrices (c_fc, c_proj)
3. Run full base training on Ptolemy (8xA100, 7-8 hours)
4. Compare with baseline on CORE metric

---

## Expected Challenges and Solutions

### Challenge: SVD Computation Cost

**Issue**: SVD of 1280x1280 matrix every 100 steps × 80 matrices (4 per layer × 20 layers) = expensive

**Solutions**:
1. Only analyze subset of layers (e.g., layers 5, 10, 15, 20)
2. Increase svd_interval (e.g., 500 instead of 100)
3. Use randomized SVD for faster approximation
4. Run SVD on CPU while GPU trains next batch (pipelining)

### Challenge: Memory Usage

**Issue**: Storing U_prev, S_prev, V_prev for each layer doubles memory

**Solutions**:
1. Only store previous SVD, not full history
2. Use bfloat16 for SVD storage (half precision)
3. Compress: only store top-k singular values and vectors

### Challenge: Optimizer State

**Issue**: Adam stores momentum/variance for each parameter. Switching params breaks this.

**Solutions**:
1. Accept temporary optimizer state loss (minor impact)
2. Transfer momentum from W to B/A when switching (complex but possible)
3. Use SGD instead of Adam for adaptive layers (simpler state)

---

## Testing Plan

### Test 1: Single Layer Proof-of-Concept
- **Model**: depth=4, only c_q in layer 0 uses adaptive SVD
- **Duration**: 100 steps
- **Goal**: Verify mode switching happens and doesn't crash

### Test 2: Full Attention, Small Scale
- **Model**: depth=4, all attention layers use adaptive SVD
- **Duration**: Complete training (~2000 steps)
- **Goal**: Verify final model quality is comparable to baseline

### Test 3: Production Scale
- **Model**: depth=20, all attention layers
- **Duration**: Full base training (~21,400 steps, 7-8 hours)
- **Goal**: Compare CORE metric with baseline, analyze clobbering prevention

---

## Success Metrics

1. **Functionality**: Training completes without crashes
2. **Quality**: Final validation loss within 5% of baseline
3. **Clobbering detection**: At least one layer switches to low-rank due to high principal_alignment
4. **Efficiency**: r decreases over training (compression happening)
5. **CORE metric**: Comparable or better than baseline

---

## Next Steps

Before implementing, we should decide:

1. **Which layers to target first?** (recommendation: only attention, skip MLP initially)
2. **What svd_interval?** (recommendation: start with 100, increase if too slow)
3. **Phase 1 or full implementation?** (recommendation: Phase 1 proof-of-concept first)

This architectural foundation will enable the adaptive SVD experiment while maintaining code clarity and debuggability.
