"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference

Norman's implementation - learning by implementing step by step
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW


@dataclass
class GPTConfig:
    """
    Configuration for GPT model architecture.

    TODO: Implement this dataclass with the following fields:
    - sequence_len: maximum sequence length (default: 1024)
    - vocab_size: vocabulary size (default: 50304)
    - n_layer: number of transformer layers (default: 12)
    - n_head: number of query heads (default: 6)
    - n_kv_head: number of key/value heads for MQA (default: 6)
    - n_embd: embedding dimension (default: 768)
    """

    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (MQA)
    n_embd: int = 768


def norm(x):
    """
    RMS normalization with no learnable parameters.

    TODO: Implement this function using F.rms_norm
    Args:
        x: input tensor
    Returns:
        normalized tensor

    DEBUG: This function is called before every attention and MLP layer
    """
    print0(f"[DEBUG] norm() called with input shape: {x.shape}")
    raise NotImplementedError(
        "TODO: Implement norm() using F.rms_norm(x, (x.size(-1),))"
    )


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary position embeddings to input tensor.

    TODO: Implement rotary embeddings
    Args:
        x: input tensor of shape (B, T, H, D) where D is head_dim
        cos: cosine values for rotary embedding
        sin: sine values for rotary embedding
    Returns:
        tensor with rotary embeddings applied

    DEBUG: This function applies positional information to queries and keys
    HINT: Split x into two halves along the last dimension, rotate, and concatenate
    """
    print0(f"[DEBUG] apply_rotary_emb() called with x.shape={x.shape}")
    print0(f"[DEBUG]   cos.shape={cos.shape}, sin.shape={sin.shape}")
    raise NotImplementedError("TODO: Implement rotary embeddings")


class CausalSelfAttention(nn.Module):
    """
    Multi-Query Attention (MQA) with rotary embeddings and QK normalization.

    TODO: Implement this class
    Key steps in __init__:
    1. Store config parameters (n_head, n_kv_head, n_embd, head_dim)
    2. Create linear projections for Q, K, V (c_q, c_k, c_v)
    3. Create output projection (c_proj)

    Key steps in forward:
    1. Project input to Q, K, V
    2. Apply rotary embeddings to Q and K
    3. Apply normalization to Q and K
    4. Handle KV cache if provided
    5. Compute scaled dot-product attention
    6. Project back to embedding dimension
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        print0(f"[DEBUG] Initializing CausalSelfAttention for layer {layer_idx}")
        self.layer_idx = layer_idx
        raise NotImplementedError("TODO: Implement CausalSelfAttention.__init__")

    def forward(self, x, cos_sin, kv_cache):
        print0(
            f"[DEBUG] CausalSelfAttention.forward() called for layer {self.layer_idx}"
        )
        print0(f"[DEBUG]   Input shape: {x.shape}")
        raise NotImplementedError("TODO: Implement CausalSelfAttention.forward")


class MLP(nn.Module):
    """
    MLP with relu^2 activation.

    TODO: Implement this class
    Key steps in __init__:
    1. Create first linear layer (c_fc) that expands by 4x
    2. Create second linear layer (c_proj) that projects back

    Key steps in forward:
    1. Apply first linear layer
    2. Apply relu activation and square it (relu^2)
    3. Apply second linear layer
    """

    def __init__(self, config):
        super().__init__()
        print0(f"[DEBUG] Initializing MLP")
        raise NotImplementedError("TODO: Implement MLP.__init__")

    def forward(self, x):
        print0(f"[DEBUG] MLP.forward() called with input shape: {x.shape}")
        raise NotImplementedError("TODO: Implement MLP.forward")


class Block(nn.Module):
    """
    Transformer block with attention and MLP.

    TODO: Implement this class
    Key steps in __init__:
    1. Create attention layer
    2. Create MLP layer

    Key steps in forward:
    1. Apply attention with pre-norm and residual connection
    2. Apply MLP with pre-norm and residual connection
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        print0(f"[DEBUG] Initializing Block {layer_idx}")

        # Each transformer block has two main components:

        # 1. Self-Attention: Allows the model to look at other positions in the sequence
        #    - "Causal" means it can only look at previous positions (left-to-right)
        #    - This is where the model learns relationships between tokens
        #    - Example: In "The cat sat on the", when processing "sat", attention
        #      can look back at "The" and "cat" to understand the subject
        self.attn = CausalSelfAttention(config, layer_idx)

        # 2. MLP (Multi-Layer Perceptron): Processes each position independently
        #    - Takes the output from attention and transforms it
        #    - Expands to 4x size internally (more capacity to learn patterns)
        #    - Then projects back to original size
        #    - This is where the model learns non-linear transformations
        self.mlp = MLP(config)

        # NOTE: We don't store layer_idx because we don't need it in forward()
        # The attention layer stores it for debugging purposes

    def forward(self, x, cos_sin, kv_cache):
        print0(f"[DEBUG] Block.forward() called")
        raise NotImplementedError("TODO: Implement Block.forward")


class GPT(nn.Module):
    """
    Main GPT model with embeddings, transformer blocks, and language modeling head.

    TODO: Implement this class step by step
    This is the most complex class - implement in this order:
    1. __init__: Set up embeddings, blocks, lm_head, and rotary embeddings
    2. _precompute_rotary_embeddings: Create rotary embedding cache
    3. init_weights: Initialize all model weights
    4. forward: Main forward pass for training/inference
    5. setup_optimizers: Create Muon and AdamW optimizers
    6. generate: Autoregressive text generation (optional for basic training)
    """

    def __init__(self, config):
        super().__init__()
        print0(f"[DEBUG] ========================================")
        print0(f"[DEBUG] Initializing GPT model")
        print0(f"[DEBUG]   n_layer: {config.n_layer}")
        print0(f"[DEBUG]   n_head: {config.n_head}")
        print0(f"[DEBUG]   n_kv_head: {config.n_kv_head}")
        print0(f"[DEBUG]   n_embd: {config.n_embd}")
        print0(f"[DEBUG]   vocab_size: {config.vocab_size}")
        print0(f"[DEBUG]   sequence_len: {config.sequence_len}")
        print0(f"[DEBUG] ========================================")

        # Store config for later use
        self.config = config

        # Create the main transformer components
        # ModuleDict is like a dictionary but registers modules properly
        self.transformer = nn.ModuleDict(
            {
                # wte = Word Token Embeddings: maps token IDs (0-vocab_size) to embedding vectors
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # h = Hidden layers: stack of transformer blocks
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )

        # # Language modeling head: projects embeddings back to vocabulary logits
        # # No bias needed - this is common in modern transformers
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # # Rotary embeddings: precompute cos/sin for positional encoding
        # # We overallocate by 10x to handle sequences longer than config.sequence_len
        # # This is cheaper than dynamically growing the cache during training
        # self.rotary_seq_len = config.sequence_len * 10  # 10x buffer
        # head_dim = config.n_embd // config.n_head

        # # Precompute the rotary embedding matrices
        # cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)

        # # Register as buffers (not parameters): they move with the model but aren't trained
        # # persistent=False means they're not saved in checkpoints (can be recomputed)
        # self.register_buffer("cos", cos, persistent=False)
        # self.register_buffer("sin", sin, persistent=False)

        # print0(f"[DEBUG] GPT.__init__ complete!")
        # print0(f"[DEBUG]   Created {config.n_layer} transformer blocks")
        # print0(f"[DEBUG]   Rotary cache size: {self.rotary_seq_len} positions")

    def init_weights(self):
        """Initialize model weights."""
        print0(f"[DEBUG] GPT.init_weights() called")
        raise NotImplementedError("TODO: Implement GPT.init_weights")

    def _init_weights(self, module):
        """Initialize weights for a single module."""
        print0(f"[DEBUG] GPT._init_weights() called for {type(module).__name__}")
        raise NotImplementedError("TODO: Implement GPT._init_weights")

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Precompute rotary embeddings for all positions."""
        print0(f"[DEBUG] _precompute_rotary_embeddings() called")
        print0(f"[DEBUG]   seq_len: {seq_len}, head_dim: {head_dim}")
        raise NotImplementedError("TODO: Implement GPT._precompute_rotary_embeddings")

    def get_device(self):
        """Get the device the model is on."""
        print0(f"[DEBUG] get_device() called")
        raise NotImplementedError(
            "TODO: Implement GPT.get_device - return device of wte embeddings"
        )

    def estimate_flops(self):
        """Estimate FLOPs per token."""
        print0(f"[DEBUG] estimate_flops() called")
        raise NotImplementedError("TODO: Implement GPT.estimate_flops")

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        """
        Setup Muon optimizer for linear layers and AdamW for embeddings/lm_head.

        TODO: Implement this method
        Key steps:
        1. Separate parameters into matrix_params, embedding_params, lm_head_params
        2. Create AdamW optimizer for embeddings and lm_head with scaled learning rates
        3. Create Muon optimizer for matrix parameters
        4. Return list of both optimizers
        """
        print0(f"[DEBUG] setup_optimizers() called")
        print0(f"[DEBUG]   unembedding_lr: {unembedding_lr}")
        print0(f"[DEBUG]   embedding_lr: {embedding_lr}")
        print0(f"[DEBUG]   matrix_lr: {matrix_lr}")
        raise NotImplementedError("TODO: Implement GPT.setup_optimizers")

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        """
        Forward pass through the model.

        TODO: Implement this method
        Key steps:
        1. Get rotary embeddings for current sequence length
        2. Embed tokens using wte
        3. Apply norm to embeddings
        4. Pass through all transformer blocks
        5. Apply final norm
        6. Compute logits through lm_head with softcapping
        7. If targets provided, compute cross-entropy loss
        """
        print0(f"[DEBUG] ========================================")
        print0(f"[DEBUG] GPT.forward() called")
        print0(f"[DEBUG]   idx.shape: {idx.shape}")
        print0(f"[DEBUG]   targets: {targets is not None}")
        print0(f"[DEBUG]   kv_cache: {kv_cache is not None}")
        print0(f"[DEBUG] ========================================")
        raise NotImplementedError("TODO: Implement GPT.forward")

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.

        TODO: Implement this method (optional for basic training)
        This is only needed for generation, not for training
        """
        print0(f"[DEBUG] generate() called")
        raise NotImplementedError(
            "TODO: Implement GPT.generate (optional - not needed for basic training)"
        )
