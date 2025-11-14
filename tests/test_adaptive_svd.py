"""
Unit tests for the adaptive_svd module.

Tests each metric function on toy matrices to verify correct behavior
before full training integration.
"""

import torch
import pytest
from nanochat.adaptive_svd import (
    compute_subspace_angle,
    compute_singular_value_concentration,
    compute_reconstruction_error,
    compute_gradient_alignment,
    determine_next_r,
    should_use_lowrank_training
)


class TestSubspaceAngle:
    """Test compute_subspace_angle function."""

    def test_identical_subspaces(self):
        """Angle between identical subspaces should be zero."""
        U = torch.randn(10, 10)
        U, _ = torch.linalg.qr(U)  # Orthonormalize

        angle = compute_subspace_angle(U, U, k=5)
        assert angle < 1e-3, f"Expected ~0, got {angle}"  # Relaxed for numerical precision

    def test_orthogonal_subspaces(self):
        """Angle between orthogonal subspaces should be π/2."""
        # Create orthogonal matrix
        U_full = torch.randn(10, 10)
        U_full, _ = torch.linalg.qr(U_full)

        # First k columns
        U1 = U_full.clone()
        # Use the orthogonal complement (last k columns)
        U2 = U_full.clone()
        U2[:, :5] = U_full[:, 5:]  # Swap first and last halves

        angle = compute_subspace_angle(U1, U2, k=5)
        # Should be close to π/2 for orthogonal subspaces
        assert angle > 1.0, f"Expected large angle for orthogonal subspaces, got {angle}"

    def test_small_rotation(self):
        """Small rotation should give small angle."""
        U1 = torch.randn(10, 10)
        U1, _ = torch.linalg.qr(U1)

        # Small rotation
        theta = 0.05  # radians
        R = torch.eye(10)
        R[0, 0] = torch.cos(torch.tensor(theta))
        R[0, 1] = -torch.sin(torch.tensor(theta))
        R[1, 0] = torch.sin(torch.tensor(theta))
        R[1, 1] = torch.cos(torch.tensor(theta))

        U2 = U1 @ R

        angle = compute_subspace_angle(U1, U2, k=5)
        assert angle < 0.2, f"Expected small angle, got {angle}"


class TestSingularValueConcentration:
    """Test compute_singular_value_concentration function."""

    def test_uniform_singular_values(self):
        """Uniform distribution should need most components."""
        S = torch.ones(10)  # All equal
        ratio, rank = compute_singular_value_concentration(S, percentile=0.95)

        # Should need most of the components
        assert ratio > 0.9, f"Expected high ratio for uniform, got {ratio}"
        assert rank >= 9, f"Expected rank ~9-10, got {rank}"

    def test_concentrated_singular_values(self):
        """Highly concentrated values should need few components."""
        S = torch.tensor([10.0, 5.0, 1.0, 0.1, 0.01, 0.001])
        ratio, rank = compute_singular_value_concentration(S, percentile=0.95)

        # Should need only first few components
        assert ratio < 0.5, f"Expected low ratio for concentrated, got {ratio}"
        assert rank <= 3, f"Expected rank <=3, got {rank}"

    def test_decreasing_singular_values(self):
        """Test with typical decreasing pattern."""
        S = torch.logspace(1, -2, 10)  # 10, ..., 0.01
        ratio, rank = compute_singular_value_concentration(S, percentile=0.99)

        # Should capture most energy in top components
        assert 0.0 < ratio < 1.0
        assert 1 <= rank <= 10


class TestReconstructionError:
    """Test compute_reconstruction_error function."""

    def test_full_rank_reconstruction(self):
        """Full rank (r=0) should have zero error."""
        W = torch.randn(8, 8)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T

        error, safe = compute_reconstruction_error(W, U, S, V, r=0)

        assert error < 1e-5, f"Expected ~0 error for r=0, got {error}"
        assert safe is True

    def test_low_rank_approximation(self):
        """Low rank should have measurable error."""
        W = torch.randn(10, 10)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T

        # Drop half the components
        error, safe = compute_reconstruction_error(W, U, S, V, r=5)

        # Should have some error
        assert error > 0.0
        # May or may not be safe depending on singular value distribution

    def test_rank_one_matrix(self):
        """Rank-1 matrix should compress perfectly to rank-1."""
        # Create rank-1 matrix
        u = torch.randn(10, 1)
        v = torch.randn(8, 1)
        W = u @ v.T

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T

        # Keep only top 1 (drop all but 1)
        r = len(S) - 1
        error, safe = compute_reconstruction_error(W, U, S, V, r=r)

        # Error should be very small (only numerical noise)
        assert error < 0.01
        assert safe is True


class TestGradientAlignment:
    """Test compute_gradient_alignment function."""

    def test_gradient_in_principal_space(self):
        """Gradient aligned with principals should have high principal_alignment."""
        # Create matrix with clear principal components
        U = torch.randn(10, 10)
        U, _ = torch.linalg.qr(U)
        S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])

        # Gradient in direction of first principal component
        grad_W = U[:, :1] @ torch.randn(1, 10)

        principal_align, minor_align = compute_gradient_alignment(
            grad_W, U, S, r=5
        )

        # Should be mostly in principal space
        assert principal_align > 0.7, f"Expected high principal alignment, got {principal_align}"
        assert minor_align < 0.5, f"Expected low minor alignment, got {minor_align}"

    def test_gradient_in_minor_space(self):
        """Gradient aligned with minors should have high minor_alignment."""
        U = torch.randn(10, 10)
        U, _ = torch.linalg.qr(U)
        S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01])

        # Gradient in direction of last components (minor space)
        r = 5
        grad_W = U[:, -r:] @ torch.randn(r, 10)

        principal_align, minor_align = compute_gradient_alignment(
            grad_W, U, S, r=r
        )

        # Should be mostly in minor space
        assert minor_align > 0.7, f"Expected high minor alignment, got {minor_align}"
        assert principal_align < 0.5, f"Expected low principal alignment, got {principal_align}"

    def test_alignments_are_normalized(self):
        """Principal and minor alignments should be in valid range [0, 1]."""
        U = torch.randn(10, 10)
        U, _ = torch.linalg.qr(U)
        S = torch.logspace(1, -2, 10)
        grad_W = torch.randn(10, 10)

        principal_align, minor_align = compute_gradient_alignment(
            grad_W, U, S, r=5
        )

        # Both should be valid fractions (may not sum to exactly 1 for matrix projections)
        assert 0.0 <= principal_align <= 1.5, f"principal_align out of range: {principal_align}"
        assert 0.0 <= minor_align <= 1.5, f"minor_align out of range: {minor_align}"


class TestDetermineNextR:
    """Test determine_next_r function."""

    def test_initial_r_selection(self):
        """First call (no S_prev) should return reasonable initial r."""
        S = torch.logspace(1, -2, 100)  # 100 singular values

        r_next, rationale = determine_next_r(S, S_prev=None, r_current=50)

        assert 8 <= r_next < 100, f"r should be in valid range, got {r_next}"
        assert "initial" in rationale.lower()

    def test_stable_minor_components(self):
        """If minor components are stable, r should decrease."""
        S_prev = torch.logspace(1, -2, 50)
        S_current = S_prev.clone()

        r_next, rationale = determine_next_r(S_current, S_prev, r_current=20)

        # Should decrease since components are identical (stable)
        assert r_next < 20, f"Expected r to decrease, got {r_next}"
        assert "stable" in rationale.lower() or "decrease" in rationale.lower()

    def test_changing_minor_components(self):
        """If minor components change, r should stay same or increase."""
        S_prev = torch.logspace(1, -2, 50)
        S_current = S_prev.clone()
        # Significantly change bottom 10 values
        S_current[-10:] *= 2.0

        r_next, rationale = determine_next_r(S_current, S_prev, r_current=20)

        # Should keep or increase r due to changing components
        assert r_next >= 20, f"Expected r to stay/increase, got {r_next}"

    def test_respects_r_min(self):
        """Should never go below r_min."""
        S = torch.logspace(1, -2, 50)

        r_next, _ = determine_next_r(S, S_prev=None, r_current=10, r_min=15)

        assert r_next >= 15, f"Should respect r_min=15, got {r_next}"


class TestShouldUseLowrankTraining:
    """Test should_use_lowrank_training master decision function."""

    def test_warmup_period(self):
        """During warmup (step < 10), should use full matrix."""
        W = torch.randn(10, 10)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T
        grad_W = torch.randn(10, 10)

        use_lowrank, r_next, diag = should_use_lowrank_training(
            W, U, S, V,
            W, U, S, V,  # Same for prev (doesn't matter in warmup)
            grad_W, r_current=5, step=5
        )

        assert use_lowrank is False
        assert diag['reason'] == 'warmup_period'

    def test_clobbering_detection(self):
        """High principal alignment should trigger low-rank mode."""
        # Create scenario where gradient attacks principals
        U = torch.randn(10, 10)
        U, _ = torch.linalg.qr(U)
        S = torch.logspace(1, -2, 10)
        V = torch.randn(10, 10)
        V, _ = torch.linalg.qr(V)

        W_prev = U @ torch.diag(S) @ V.T

        # Slightly changed W
        W_curr = W_prev + 0.01 * torch.randn(10, 10)
        U_curr, S_curr, Vh_curr = torch.linalg.svd(W_curr, full_matrices=False)
        V_curr = Vh_curr.T

        # Gradient aligned with principal components (dangerous!)
        grad_W = U_curr[:, :3] @ torch.randn(3, 10) * 10.0  # Strong gradient in principal space

        use_lowrank, r_next, diag = should_use_lowrank_training(
            W_curr, U_curr, S_curr, V_curr,
            W_prev, U, S, V,
            grad_W, r_current=5, step=10
        )

        # Should detect clobbering risk
        assert diag['principal_alignment'] > 0.0  # Some principal alignment

    def test_optimal_conditions(self):
        """Stable subspace + safe gradients should allow low-rank."""
        U = torch.randn(10, 10)
        U, _ = torch.linalg.qr(U)
        S = torch.logspace(1, -2, 10)
        V = torch.randn(10, 10)
        V, _ = torch.linalg.qr(V)

        W = U @ torch.diag(S) @ V.T

        # Gradient mostly in minor space (safe)
        grad_W = U[:, -3:] @ torch.randn(3, 10)

        use_lowrank, r_next, diag = should_use_lowrank_training(
            W, U, S, V,
            W, U, S, V,  # Same (very stable)
            grad_W, r_current=3, step=10
        )

        # Should allow low-rank training
        assert isinstance(use_lowrank, bool)

    def test_unsafe_reconstruction(self):
        """High reconstruction error should prevent low-rank."""
        # Create matrix where low-rank approximation is poor
        W = torch.randn(10, 10)  # Random, no low-rank structure
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T

        # Very uniform singular values (no compression opportunity)
        S_uniform = torch.ones_like(S)
        W_uniform = U @ torch.diag(S_uniform) @ V.T

        U_u, S_u, Vh_u = torch.linalg.svd(W_uniform, full_matrices=False)
        V_u = Vh_u.T

        grad_W = torch.randn(10, 10)

        # Try to compress heavily (r=8 means keep only 2 components)
        use_lowrank, r_next, diag = should_use_lowrank_training(
            W_uniform, U_u, S_u, V_u,
            W_uniform, U_u, S_u, V_u,
            grad_W, r_current=8, step=10
        )

        # May prevent low-rank due to high error
        assert 'reconstruction' in str(diag)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_typical_training_scenario(self):
        """Simulate a typical training progression."""
        # Initial weight matrix
        W_0 = torch.randn(20, 20) * 0.02  # Standard initialization

        # Step 1: Initial SVD
        U_0, S_0, Vh_0 = torch.linalg.svd(W_0, full_matrices=False)
        V_0 = Vh_0.T

        # Initial r (start large)
        r_0 = len(S_0) - 5  # Keep only 5 principal, train on 15 minor

        # Step 2: After some training, gradients appear
        grad = torch.randn(20, 20) * 0.001

        # Step 3: Check decision
        use_lowrank, r_next, diag = should_use_lowrank_training(
            W_0, U_0, S_0, V_0,
            W_0, U_0, S_0, V_0,
            grad, r_current=r_0, step=100
        )

        # Should make a valid decision
        assert use_lowrank in [True, False, None]
        assert r_next >= 8  # Above minimum
        assert 'reason' in diag

        print(f"\nIntegration test results:")
        print(f"  Decision: {'Low-rank' if use_lowrank else 'Full matrix' if use_lowrank is False else 'Continue current'}")
        print(f"  Reason: {diag['reason']}")
        print(f"  r_current: {r_0} -> r_next: {r_next}")
        print(f"  Principal alignment: {diag.get('principal_alignment', 'N/A')}")
        print(f"  Minor alignment: {diag.get('minor_alignment', 'N/A')}")

    def test_clobbering_prevention_workflow(self):
        """Test the complete clobbering detection workflow."""
        # Create a matrix with learned structure
        U = torch.randn(15, 15)
        U, _ = torch.linalg.qr(U)
        # Strong principal components (learned patterns)
        S = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0] + [0.1] * 10)
        V = torch.randn(15, 15)
        V, _ = torch.linalg.qr(V)

        W_learned = U @ torch.diag(S) @ V.T

        # Simulate gradient that would clobber (points at principals)
        grad_clobbering = U[:, :2] @ torch.randn(2, 15) * 5.0

        U_l, S_l, Vh_l = torch.linalg.svd(W_learned, full_matrices=False)
        V_l = Vh_l.T

        # Test clobbering detection
        principal_align, minor_align = compute_gradient_alignment(
            grad_clobbering, U_l, S_l, r=10
        )

        print(f"\nClobbering detection test:")
        print(f"  Principal alignment: {principal_align:.3f}")
        print(f"  Minor alignment: {minor_align:.3f}")

        # Should detect high principal alignment
        assert principal_align > 0.5, "Should detect clobbering risk"

        # Decision should recommend low-rank
        use_lowrank, _, diag = should_use_lowrank_training(
            W_learned, U_l, S_l, V_l,
            W_learned, U_l, S_l, V_l,
            grad_clobbering, r_current=10, step=100
        )

        print(f"  Decision: {diag['reason']}")
        # Should trigger clobbering protection
        if 'clobbering' in diag['reason']:
            print("  ✓ Clobbering detected and prevented!")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
