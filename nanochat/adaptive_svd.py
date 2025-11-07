"""
Adaptive SVD-Based Training Module

This module implements adaptive SVD techniques for detecting and preventing
catastrophic forgetting during neural network training. It provides metrics
to compare SVD decompositions across training steps and determine when to
switch between full-matrix and low-rank training modes.

Key Features:
- Subspace angle computation for detecting principal component rotation
- Singular value concentration analysis for tracking information distribution
- Reconstruction error checking for compression safety
- Gradient alignment detection for clobbering prevention (THE KEY METRIC)
- Adaptive rank selection based on spectral properties
- Master decision logic for training mode selection

Based on the BASE_TRAIN_SVD_ENHANCEMENT experiment design.
"""

from typing import Dict, Optional, Tuple
import torch


def compute_subspace_angle(U1: torch.Tensor, U2: torch.Tensor, k: int) -> float:
    """
    Compute the principal angle between top-k subspaces of two SVD decompositions.

    This metric measures how much the principal subspace has rotated between two
    training steps. A large angle indicates significant structural changes in the
    learned patterns.

    WHY U (LEFT SINGULAR VECTORS)?
    ===============================
    For W = U Σ V^T, the columns of U represent the LEARNED OUTPUT PATTERNS:
    - U[:, 0] is the most important output direction (largest singular value)
    - U[:, k-1] is the k-th most important output direction
    - These are the patterns we want to protect from clobbering

    When the principal subspace (top-k columns of U) rotates significantly,
    it means the layer is learning fundamentally new output patterns that
    could overwrite previous learning.

    Args:
        U1: Left singular vectors from first SVD (shape: d_out x g)
            Columns are orthonormal output space basis vectors
        U2: Left singular vectors from second SVD (shape: d_out x g)
            Columns are orthonormal output space basis vectors
        k: Number of principal components to compare (top-k most important)

    Returns:
        angle: Principal angle in radians, range [0, π/2]
               0 = identical subspaces, larger values = more rotation

    Interpretation:
        - angle < 0.1 rad (5.7°): Stable subspace, safe for compression
        - angle > 0.1 rad: Significant rotation, principals are changing

    Example:
        >>> U_prev = model.layer.weight  # After SVD
        >>> # ... training step ...
        >>> U_curr = model.layer.weight  # After SVD
        >>> angle = compute_subspace_angle(U_prev, U_curr, k=100)
        >>> if angle > 0.1:
        >>>     print("Principal subspace has rotated significantly!")
    """
    # Extract top k columns
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]

    # Compute principal angles via SVD of cross-product
    # θ = arccos(σ) where σ are singular values of U1_k^T @ U2_k
    cross = U1_k.T @ U2_k
    singular_values = torch.linalg.svdvals(cross)

    # Principal angle (largest angle between subspaces)
    # Clamp to avoid numerical issues with arccos
    angle = torch.acos(torch.clamp(singular_values.min(), -1.0, 1.0))

    return angle.item()


def compute_singular_value_concentration(
    S: torch.Tensor,
    percentile: float = 0.95
) -> Tuple[float, int]:
    """
    Compute how concentrated the information is in the top singular values.

    This metric tracks whether information is becoming more concentrated
    (mature patterns forming) or more spread out (learning new patterns).

    Args:
        S: Singular values from SVD, sorted descending
        percentile: Energy threshold (default 95%, i.e., 0.95)

    Returns:
        concentration_ratio: Fraction of singular values needed to reach threshold
        effective_rank: Number of singular values needed to capture percentile energy

    Interpretation:
        - Higher concentration (lower ratio) = mature, stable patterns
        - Lower concentration (higher ratio) = learning new, complex patterns
        - Increasing effective_rank over time = model needs more capacity

    Example:
        >>> S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
        >>> ratio, rank = compute_singular_value_concentration(S, 0.95)
        >>> print(f"Need {rank}/{len(S)} values ({ratio:.1%}) for 95% energy")
    """
    total_energy = (S ** 2).sum()
    cumulative_energy = torch.cumsum(S ** 2, dim=0)

    # Find how many components needed for X% energy
    threshold = percentile * total_energy
    effective_rank = (cumulative_energy < threshold).sum().item() + 1
    concentration_ratio = effective_rank / len(S)

    return concentration_ratio, effective_rank


def compute_reconstruction_error(
    W: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    r: int
) -> Tuple[float, bool]:
    """
    Compute reconstruction error from using only top (g-r) principal components.

    This is a safety check before switching to low-rank training. It verifies
    that dropping the bottom r components doesn't lose too much information.

    Args:
        W: Full weight matrix
        U: Left singular vectors from SVD
        S: Singular values from SVD
        V: Right singular vectors from SVD (V, not V^T)
        r: Number of minor components (bottom r will be dropped)

    Returns:
        relative_error: ||W - W_reconstructed|| / ||W|| (Frobenius norm)
        safe_to_compress: True if error < 1% (0.01)

    Interpretation:
        - error < 0.01 (1%): Safe to compress, minimal information loss
        - error > 0.01: Risky, should keep more components or use full matrix

    Example:
        >>> error, safe = compute_reconstruction_error(W, U, S, V, r=64)
        >>> if safe:
        >>>     print(f"Safe to compress with r={r} (error: {error*100:.2f}%)")
        >>> else:
        >>>     print(f"Compression risky! Try smaller r")
    """
    g = len(S)  # Total rank
    k = g - r   # Number of principal components to keep

    # Reconstruct using only top k components
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = V[:, :k]
    W_reconstructed = U_k @ torch.diag(S_k) @ V_k.T

    # Relative Frobenius norm error
    error = torch.norm(W - W_reconstructed, p='fro') / torch.norm(W, p='fro')

    # Typically, <1% error is considered safe
    safe_to_compress = error.item() < 0.01

    return error.item(), safe_to_compress


def compute_gradient_alignment(
    grad_W: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    r: int
) -> Tuple[float, float]:
    """
    Measure how much gradient aligns with principal vs minor components.

    ⭐ THIS IS THE CLOBBERING DETECTOR ⭐

    If gradients heavily align with principal components, they're about to
    overwrite previously learned patterns. This is the early warning system
    for catastrophic forgetting.

    WHY U (LEFT SINGULAR VECTORS)?
    ===============================
    The gradient grad_W has shape (d_out, d_in), same as W.

    By projecting grad_W onto the subspaces defined by U, we're asking:
    "Which OUTPUT DIRECTIONS is the gradient trying to modify?"

    - U_principal = U[:, :k]: The k most important learned output patterns
    - U_minor = U[:, k:]: The less important output directions

    If grad_W aligns with U_principal:
      → Gradient wants to CHANGE our best learned patterns (DANGER!)
      → This is catastrophic forgetting about to happen
      → Switch to low-rank training to PROTECT these patterns

    If grad_W aligns with U_minor:
      → Gradient is learning in ORTHOGONAL directions (SAFE!)
      → New learning won't destroy old patterns
      → Continue current training mode

    This is the CORE INNOVATION of this approach: detecting clobbering
    BEFORE it happens by monitoring gradient alignment with learned subspaces.

    Args:
        grad_W: Gradient of weight matrix (shape: d_out x d_in)
        U: Left singular vectors from SVD (shape: d_out x g)
            Columns define the output space basis
        S: Singular values from SVD (shape: g,)
        r: Number of minor components (bottom r of g total)

    Returns:
        principal_alignment: Fraction of gradient affecting top (g-r) components [0, 1]
        minor_alignment: Fraction of gradient affecting bottom r components [0, 1]

    Interpretation:
        - principal_alignment > 0.4: DANGER! About to clobber learned patterns
        - minor_alignment > 0.6: SAFE! Gradients in orthogonal subspace
        - Switch to low-rank when principal_alignment is high

    Example:
        >>> principal, minor = compute_gradient_alignment(param.grad, U, S, r=64)
        >>> if principal > 0.4:
        >>>     print("WARNING: Gradients attacking principal components!")
        >>>     # Switch to B/A training immediately
        >>> elif minor > 0.7:
        >>>     print("Safe: gradients in minor subspace")
    """
    g = len(S)
    k = g - r  # Number of principal components

    # Project gradient onto principal subspace
    U_principal = U[:, :k]
    grad_principal_proj = U_principal @ (U_principal.T @ grad_W)
    principal_alignment = torch.norm(grad_principal_proj) / torch.norm(grad_W)

    # Project gradient onto minor subspace
    U_minor = U[:, k:]
    grad_minor_proj = U_minor @ (U_minor.T @ grad_W)
    minor_alignment = torch.norm(grad_minor_proj) / torch.norm(grad_W)

    return principal_alignment.item(), minor_alignment.item()


def determine_next_r(
    S: torch.Tensor,
    S_prev: Optional[torch.Tensor],
    r_current: int,
    r_min: int = 8
) -> Tuple[int, str]:
    """
    Adaptively determine r (number of minor components) for next training iteration.

    This function uses multiple strategies to select the optimal rank:
    1. Spectral gap analysis (natural cutoff points in singular values)
    2. Energy threshold (99.9% energy capture)
    3. Rate of change in bottom singular values

    Args:
        S: Current singular values (sorted descending)
        S_prev: Previous singular values (None for first iteration)
        r_current: Current r value
        r_min: Minimum allowed r (don't compress below this)

    Returns:
        r_next: Recommended r for next iteration
        rationale: String explaining the decision (for logging)

    Strategy:
        - r starts LARGE (nearly full rank) and DECREASES over time
        - If bottom values change rapidly, keep/increase r (new information)
        - If bottom values stable, decrease r (compression opportunity)

    Example:
        >>> r_next, reason = determine_next_r(S_curr, S_prev, r_current=128)
        >>> print(f"Next r={r_next}: {reason}")
    """
    g = len(S)

    # Strategy 1: Spectral gap (large drop in singular values)
    # If σ_i / σ_{i+1} is large, there's a natural "cutoff" point
    ratios = S[:-1] / (S[1:] + 1e-10)  # Add epsilon to avoid division by zero
    max_gap_idx = torch.argmax(ratios).item()
    natural_cutoff = g - max_gap_idx  # r that aligns with spectral gap

    # Strategy 2: Energy threshold
    # Use enough components to capture 99.9% of energy
    _, effective_rank = compute_singular_value_concentration(S, percentile=0.999)
    r_from_energy = g - effective_rank

    # Strategy 3: Relative change in singular values
    # If bottom singular values changed a lot, they contain new information
    if S_prev is not None and len(S_prev) == len(S):
        bottom_r = min(r_current, len(S) // 4)  # Check bottom 25%
        S_bottom_prev = S_prev[-bottom_r:]
        S_bottom_curr = S[-bottom_r:]
        relative_change = (
            torch.norm(S_bottom_curr - S_bottom_prev) /
            (torch.norm(S_bottom_prev) + 1e-10)
        )

        if relative_change > 0.1:  # Bottom values changed >10%
            # New information in minor components, keep current r or increase
            r_next = max(r_current, int(r_current * 1.2))
            rationale = (
                f"Minor components changing rapidly ({relative_change.item():.1%}), "
                f"keep/increase r"
            )
        else:
            # Minor components stable, can decrease r
            r_next = int(r_current * 0.8)
            rationale = (
                f"Minor components stable ({relative_change.item():.1%}), "
                f"decrease r"
            )
    else:
        # First iteration, start with conservative r
        r_next = min(natural_cutoff, r_from_energy, g // 2)
        rationale = "Initial r based on spectral structure"

    # Enforce bounds
    r_next = max(r_min, min(r_next, g - 1))

    return r_next, rationale


def should_use_lowrank_training(
    W_current: torch.Tensor,
    U_current: torch.Tensor,
    S_current: torch.Tensor,
    V_current: torch.Tensor,
    W_prev: torch.Tensor,
    U_prev: torch.Tensor,
    S_prev: torch.Tensor,
    V_prev: torch.Tensor,
    grad_W: torch.Tensor,
    r_current: int,
    step: int
) -> Tuple[Optional[bool], int, Dict[str, any]]:
    """
    Master decision function: Should we use full matrix or low-rank training?

    This function integrates all metrics to make the critical decision:
    - Continue training the full weight matrix W, OR
    - Switch to low-rank training (update only B and A matrices)

    Args:
        W_current: Current weight matrix
        U_current: Current left singular vectors
        S_current: Current singular values
        V_current: Current right singular vectors
        W_prev: Previous weight matrix
        U_prev: Previous left singular vectors
        S_prev: Previous singular values
        V_prev: Previous right singular vectors
        grad_W: Current gradient of weight matrix
        r_current: Current number of minor components
        step: Current training step number

    Returns:
        use_lowrank: True (switch to B/A), False (use full W), None (keep current mode)
        r_next: Recommended rank for next iteration
        diagnostics: Dict with all metrics for logging

    Decision Rules:
        1. DANGER (principal_align > 0.4): Switch to B/A immediately (clobbering risk)
        2. OPTIMAL (stable + safe gradients + low error): Use B/A (efficient)
        3. UNSAFE (reconstruction error high): Use full W (preserve information)
        4. NEUTRAL: Continue current mode

    Example:
        >>> use_lr, r_next, diag = should_use_lowrank_training(
        >>>     W_curr, U_curr, S_curr, V_curr,
        >>>     W_prev, U_prev, S_prev, V_prev,
        >>>     param.grad, r=64, step=100
        >>> )
        >>> print(f"Decision: {'Low-rank' if use_lr else 'Full matrix'}")
        >>> print(f"Reason: {diag['reason']}")
    """
    diagnostics = {}

    # Don't switch until we've trained for a few steps
    if step < 10:
        return False, r_current, {"reason": "warmup_period"}

    # Metric 1: Principal subspace stability
    k = len(S_current) - r_current  # Number of principal components
    angle = compute_subspace_angle(U_prev, U_current, k)
    diagnostics['subspace_angle'] = angle

    # Metric 2: Gradient alignment
    principal_align, minor_align = compute_gradient_alignment(
        grad_W, U_current, S_current, r_current
    )
    diagnostics['principal_alignment'] = principal_align
    diagnostics['minor_alignment'] = minor_align

    # Metric 3: Reconstruction safety
    error, safe = compute_reconstruction_error(
        W_current, U_current, S_current, V_current, r_current
    )
    diagnostics['reconstruction_error'] = error
    diagnostics['reconstruction_safe'] = safe

    # Metric 4: Determine next r
    r_next, r_rationale = determine_next_r(S_current, S_prev, r_current)
    diagnostics['r_next'] = r_next
    diagnostics['r_rationale'] = r_rationale

    # DECISION LOGIC

    # Rule 1: If subspace is stable AND gradients mostly in minor space
    stable_subspace = angle < 0.1  # <5.7 degrees
    gradients_safe = minor_align > 0.6

    # Rule 2: If gradients are attacking principal components
    gradients_dangerous = principal_align > 0.4

    # Rule 3: Reconstruction is safe
    reconstruction_ok = safe

    if gradients_dangerous:
        # DANGER: About to clobber principal components!
        diagnostics['reason'] = "gradient_clobbering_risk"
        use_lowrank = True
    elif stable_subspace and gradients_safe and reconstruction_ok:
        # SAFE: Good conditions for low-rank training
        diagnostics['reason'] = "optimal_lowrank_conditions"
        use_lowrank = True
    elif not reconstruction_ok:
        # UNSAFE: Would lose too much information
        diagnostics['reason'] = "reconstruction_unsafe"
        use_lowrank = False
    else:
        # NEUTRAL: Continue current mode
        diagnostics['reason'] = "continue_current_mode"
        use_lowrank = None  # Keep current mode

    return use_lowrank, r_next, diagnostics
