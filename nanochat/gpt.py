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


class AdaptiveSVDLinear(nn.Module):
    """
    A Linear layer that can switch between full-matrix and low-rank training.

    This layer supports dynamic switching between two training modes:
    1. Full Matrix Mode: Train all weights W directly (standard approach)
    2. Low-Rank Mode: Train only low-rank decomposition Bₘ @ Aₘ while freezing Wₚ

    The mode switching is determined by SVD analysis to prevent catastrophic
    forgetting (clobbering) of learned patterns.

    Attributes:
        W: Full weight matrix (out_features, in_features) - always present
        B: Low-rank left factor (out_features, r) - created when switching to low-rank
        A: Low-rank right factor (r, in_features) - created when switching to low-rank
        W_principal: Frozen principal components (out_features, in_features)
        bias: Optional bias (same as nn.Linear)

        mode: "full" or "lowrank" - current training mode
        r: Current rank for low-rank decomposition

        U_prev, S_prev, V_prev: Previous SVD for comparison between steps
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
        W_principal = U_p @ torch.diag(S_p) @ V_p.T
        self.register_buffer('W_principal', W_principal)

        # Wₘ = bottom-r components (trainable as B @ A)
        U_m = U[:, k:]
        S_m = S[k:]
        V_m = V[:, k:]

        # 3. Initialize B and A using MiLoRA approach
        # B = Uₘ √Σₘ,  A = √Σₘ Vₘᵀ
        sqrt_S_m = torch.sqrt(torch.diag(S_m))
        B = U_m @ sqrt_S_m
        A = sqrt_S_m @ V_m.T

        # Register as parameters
        self.B = nn.Parameter(B)
        self.A = nn.Parameter(A)

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
        if self.mode == "lowrank" and self.B is not None and self.A is not None:
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


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768
    adaptive_svd: bool = False  # Enable adaptive SVD training
    adaptive_svd_target_layer: int = 0  # Which layer to apply adaptive SVD to (Phase 1: single layer)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Phase 1: Only apply adaptive SVD to c_q in target layer
        use_adaptive_for_c_q = config.adaptive_svd and layer_idx == config.adaptive_svd_target_layer

        if use_adaptive_for_c_q:
            self.c_q = AdaptiveSVDLinear(self.n_embd, self.n_head * self.head_dim, bias=False)
        else:
            self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)

        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, AdaptiveSVDLinear):
            # Initialize W using same strategy as nn.Linear
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.W.size(0)
            fan_in = module.W.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.W, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        # IMPORTANT: Only include parameters that require gradients (filters out frozen W in low-rank mode)
        matrix_params = [p for p in self.transformer.h.parameters() if p.requires_grad]
        embedding_params = [p for p in self.transformer.wte.parameters() if p.requires_grad]
        lm_head_params = [p for p in self.lm_head.parameters() if p.requires_grad]
        # Assertion: verify all trainable params are accounted for
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        assert len(trainable_params) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

        # Debug logging: parameter counts
        if rank == 0:
            total_params = len(list(self.parameters()))
            print(f"[DEBUG] setup_optimizers() - Parameter counts:")
            print(f"  Total parameters (all):      {total_params}")
            print(f"  Trainable parameters:        {len(trainable_params)}")
            print(f"  Frozen parameters:           {total_params - len(trainable_params)}")
            print(f"  Matrix params (trainable):   {len(matrix_params)}")
            print(f"  Embedding params (trainable): {len(embedding_params)}")
            print(f"  LM head params (trainable):  {len(lm_head_params)}")

        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
