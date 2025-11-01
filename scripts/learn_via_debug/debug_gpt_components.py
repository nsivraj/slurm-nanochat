"""
Debug GPT Components During Training
This script instruments the GPT model to trace execution of key components:
- Token embeddings (wte)
- Positional embeddings (RoPE)
- Stack of Block layers
- RMSNorm operations
- Language modeling head
- Model scaling analysis

Run with:
python -m scripts.learn_via_debug.debug_gpt_components --depth=20 --device_batch_size=1 --total_batch_size=512 --num_iterations=5
"""

import os
import sys
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title):
    """Print a colored section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}\n")

def print_component(name, tensor, extra_info=""):
    """Print information about a tensor component"""
    if tensor is None:
        print(f"{Colors.YELLOW}[{name}] None{Colors.END}")
        return

    print(f"{Colors.CYAN}[{name}]{Colors.END}")
    print(f"  Shape: {Colors.GREEN}{tuple(tensor.shape)}{Colors.END}")
    print(f"  Dtype: {Colors.GREEN}{tensor.dtype}{Colors.END}")
    print(f"  Device: {Colors.GREEN}{tensor.device}{Colors.END}")
    print(f"  Mean: {Colors.GREEN}{tensor.float().mean().item():.6f}{Colors.END}")
    print(f"  Std: {Colors.GREEN}{tensor.float().std().item():.6f}{Colors.END}")
    print(f"  Min: {Colors.GREEN}{tensor.float().min().item():.6f}{Colors.END}")
    print(f"  Max: {Colors.GREEN}{tensor.float().max().item():.6f}{Colors.END}")
    if extra_info:
        print(f"  {Colors.BLUE}{extra_info}{Colors.END}")
    print()

class InstrumentedGPT(GPT):
    """
    Instrumented version of GPT that logs intermediate values
    """
    def __init__(self, config, verbose=True):
        super().__init__(config)
        self.verbose = verbose
        self.step_count = 0

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """Instrumented forward pass with detailed logging"""
        B, T = idx.size()

        if self.verbose:
            print_section(f"FORWARD PASS #{self.step_count}")
            print_component("Input tokens (idx)", idx, f"Batch size: {B}, Sequence length: {T}")
            if targets is not None:
                print_component("Target tokens", targets)

        # =============================================================================
        # 1. TOKEN EMBEDDINGS (wte)
        # =============================================================================
        if self.verbose:
            print_section("1. TOKEN EMBEDDINGS (wte)")
            print(f"{Colors.BLUE}The token embedding layer converts discrete token IDs to continuous vectors{Colors.END}")
            print(f"{Colors.BLUE}Embedding matrix shape: (vocab_size={self.config.vocab_size}, n_embd={self.config.n_embd}){Colors.END}\n")

        x = self.transformer.wte(idx)  # (B, T, n_embd)

        if self.verbose:
            print_component("Token embeddings (before norm)", x,
                          f"Each token is now a {self.config.n_embd}-dimensional vector")
            print(f"{Colors.YELLOW}Note: wte is learned during training{Colors.END}\n")

        # Norm after token embedding (architectural choice)
        x = norm(x)

        if self.verbose:
            print_component("Token embeddings (after RMSNorm)", x,
                          "Normalized to stabilize training")

        # =============================================================================
        # 2. POSITIONAL EMBEDDINGS (RoPE - Rotary Position Embeddings)
        # =============================================================================
        if self.verbose:
            print_section("2. POSITIONAL EMBEDDINGS (RoPE)")
            print(f"{Colors.BLUE}RoPE (Rotary Position Embeddings) encodes position information{Colors.END}")
            print(f"{Colors.BLUE}Unlike learned positional embeddings, RoPE is applied via rotation{Colors.END}")
            print(f"{Colors.BLUE}Rotary embeddings are precomputed and cached for efficiency{Colors.END}\n")

        # Grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos = self.cos[:, T0:T0+T]  # (1, T, 1, head_dim)
        sin = self.sin[:, T0:T0+T]  # (1, T, 1, head_dim)
        cos_sin = (cos, sin)

        if self.verbose:
            print_component("Rotary cos", cos, f"Precomputed cosine values for position encoding")
            print_component("Rotary sin", sin, f"Precomputed sine values for position encoding")
            print(f"{Colors.YELLOW}These will be applied to Q and K in attention via rotation{Colors.END}\n")
            print(f"{Colors.BLUE}RoPE formula: rotate pairs of dims using cos/sin{Colors.END}")
            print(f"{Colors.BLUE}  q_rotated = [q[:d/2] * cos + q[d/2:] * sin, q[:d/2] * (-sin) + q[d/2:] * cos]{Colors.END}\n")

        # =============================================================================
        # 3. STACK OF TRANSFORMER BLOCKS
        # =============================================================================
        if self.verbose:
            print_section(f"3. TRANSFORMER BLOCK STACK (depth={self.config.n_layer})")
            print(f"{Colors.BLUE}Model has {self.config.n_layer} layers stacked sequentially{Colors.END}")
            print(f"{Colors.BLUE}Each block contains: Attention + MLP with residual connections{Colors.END}\n")

        for layer_idx, block in enumerate(self.transformer.h):
            x_before = x.clone() if self.verbose else None

            # Each block does: x = x + attn(norm(x)) and x = x + mlp(norm(x))
            x = block(x, cos_sin, kv_cache)

            if self.verbose:
                print(f"{Colors.BOLD}--- Block {layer_idx}/{self.config.n_layer-1} ---{Colors.END}")

                # Show attention details for first few blocks
                if layer_idx < 3 or layer_idx == self.config.n_layer - 1:
                    print_component(f"  Input to Block {layer_idx}", x_before)

                    # Manually trace through attention for visibility
                    x_norm = norm(x_before)
                    print_component(f"  After RMSNorm (pre-attn)", x_norm, "Normalized for attention")

                    # Attention projections
                    q = block.attn.c_q(x_norm).view(B, T, block.attn.n_head, block.attn.head_dim)
                    k = block.attn.c_k(x_norm).view(B, T, block.attn.n_kv_head, block.attn.head_dim)
                    v = block.attn.c_v(x_norm).view(B, T, block.attn.n_kv_head, block.attn.head_dim)

                    print_component(f"  Queries (Q)", q,
                                  f"n_head={block.attn.n_head}, head_dim={block.attn.head_dim}")
                    print_component(f"  Keys (K)", k,
                                  f"n_kv_head={block.attn.n_kv_head} (MQA/GQA)")
                    print_component(f"  Values (V)", v)

                    # Apply RoPE to Q and K
                    q_rope = apply_rotary_emb(q, cos, sin)
                    k_rope = apply_rotary_emb(k, cos, sin)

                    print_component(f"  Queries (after RoPE)", q_rope, "Position info encoded via rotation")
                    print_component(f"  Keys (after RoPE)", k_rope)

                    # QK norm
                    q_norm = norm(q_rope)
                    k_norm = norm(k_rope)

                    print_component(f"  Queries (after QK norm)", q_norm, "Normalized Q for stability")
                    print_component(f"  Keys (after QK norm)", k_norm)

                    print_component(f"  Output from Block {layer_idx}", x, "After attn + MLP with residuals")
                else:
                    print(f"  {Colors.GREEN}[Skipping detailed trace for block {layer_idx}]{Colors.END}")

                print()

        # =============================================================================
        # 4. FINAL RMSNorm
        # =============================================================================
        if self.verbose:
            print_section("4. FINAL RMSNorm")
            print(f"{Colors.BLUE}Final normalization before the language modeling head{Colors.END}\n")

        x_before_final_norm = x.clone() if self.verbose else None
        x = norm(x)

        if self.verbose:
            print_component("Before final norm", x_before_final_norm)
            print_component("After final norm", x, "Ready for LM head projection")
            print(f"\n{Colors.YELLOW}RMSNorm formula: x / rms(x) * scale{Colors.END}")
            print(f"{Colors.YELLOW}  where rms(x) = sqrt(mean(x^2)){Colors.END}")
            print(f"{Colors.YELLOW}  No learnable parameters in this implementation{Colors.END}\n")

        # =============================================================================
        # 5. LANGUAGE MODELING HEAD
        # =============================================================================
        if self.verbose:
            print_section("5. LANGUAGE MODELING HEAD")
            print(f"{Colors.BLUE}Projects from model dimension to vocabulary size{Colors.END}")
            print(f"{Colors.BLUE}Shape: (n_embd={self.config.n_embd}) -> (vocab_size={self.config.vocab_size}){Colors.END}")
            print(f"{Colors.BLUE}Weights are UNTIED from token embeddings (separate parameters){Colors.END}\n")

        # Compute logits
        softcap = 15
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if self.verbose:
            print_component("Logits (before softcap)", logits,
                          f"Raw scores for each token in vocabulary")

        # Apply softcap
        logits = softcap * torch.tanh(logits / softcap)

        if self.verbose:
            print_component("Logits (after softcap)", logits,
                          f"Capped at ±{softcap} to prevent extreme values")
            print(f"\n{Colors.YELLOW}Softcap formula: {softcap} * tanh(logits / {softcap}){Colors.END}\n")

        # =============================================================================
        # 6. LOSS COMPUTATION (if training)
        # =============================================================================
        if targets is not None:
            if self.verbose:
                print_section("6. LOSS COMPUTATION (Training Mode)")
                print(f"{Colors.BLUE}Computing cross-entropy loss between predictions and targets{Colors.END}\n")

            logits = logits.float()  # Use fp32 for numerical stability
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )

            if self.verbose:
                print_component("Loss", loss.unsqueeze(0),
                              f"Cross-entropy loss (reduction={loss_reduction})")
                print(f"{Colors.YELLOW}Lower is better - measures how well model predicts next tokens{Colors.END}\n")

            self.step_count += 1
            return loss
        else:
            # Inference mode
            if self.verbose:
                print_section("INFERENCE MODE")
                print(f"{Colors.BLUE}Returning logits for next token prediction{Colors.END}\n")

            self.step_count += 1
            return logits

def analyze_model_scaling(config):
    """Analyze model scaling and parameter count"""
    print_section("MODEL SCALING ANALYSIS")

    # Count parameters for each component
    with torch.device("meta"):
        model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())

    # Break down by component
    wte_params = model.transformer.wte.weight.numel()
    lm_head_params = model.lm_head.weight.numel()

    # Attention parameters per layer
    sample_block = model.transformer.h[0]
    q_params = sample_block.attn.c_q.weight.numel()
    k_params = sample_block.attn.c_k.weight.numel()
    v_params = sample_block.attn.c_v.weight.numel()
    proj_params = sample_block.attn.c_proj.weight.numel()
    attn_params_per_layer = q_params + k_params + v_params + proj_params

    # MLP parameters per layer
    fc_params = sample_block.mlp.c_fc.weight.numel()
    mlp_proj_params = sample_block.mlp.c_proj.weight.numel()
    mlp_params_per_layer = fc_params + mlp_proj_params

    # Total for all layers
    total_block_params = (attn_params_per_layer + mlp_params_per_layer) * config.n_layer

    print(f"{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Depth (n_layer): {Colors.GREEN}{config.n_layer}{Colors.END}")
    print(f"  Model dimension (n_embd): {Colors.GREEN}{config.n_embd}{Colors.END}")
    print(f"  Number of heads (n_head): {Colors.GREEN}{config.n_head}{Colors.END}")
    print(f"  Head dimension: {Colors.GREEN}{config.n_embd // config.n_head}{Colors.END}")
    print(f"  Sequence length: {Colors.GREEN}{config.sequence_len}{Colors.END}")
    print(f"  Vocabulary size: {Colors.GREEN}{config.vocab_size}{Colors.END}")

    print(f"\n{Colors.BOLD}Parameter Breakdown:{Colors.END}")
    print(f"  Token embeddings (wte): {Colors.GREEN}{wte_params:,}{Colors.END} ({100*wte_params/total_params:.1f}%)")
    print(f"  LM head (unembedding): {Colors.GREEN}{lm_head_params:,}{Colors.END} ({100*lm_head_params/total_params:.1f}%)")
    print(f"  Transformer blocks: {Colors.GREEN}{total_block_params:,}{Colors.END} ({100*total_block_params/total_params:.1f}%)")
    print(f"    - Attention per layer: {Colors.GREEN}{attn_params_per_layer:,}{Colors.END}")
    print(f"    - MLP per layer: {Colors.GREEN}{mlp_params_per_layer:,}{Colors.END}")

    print(f"\n{Colors.BOLD}Total Parameters: {Colors.GREEN}{total_params:,}{Colors.END} ({total_params/1e6:.1f}M){Colors.END}")

    # Show target configuration
    print(f"\n{Colors.BOLD}Target Configuration (561M params):{Colors.END}")
    print(f"  depth=20, d_model=1280, n_heads=10")
    target_config = GPTConfig(n_layer=20, n_embd=1280, n_head=10, n_kv_head=10)
    with torch.device("meta"):
        target_model = GPT(target_config)
    target_params = sum(p.numel() for p in target_model.parameters())
    print(f"  Total parameters: {Colors.GREEN}{target_params:,}{Colors.END} ({target_params/1e6:.1f}M)")

    print()

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse arguments - defaults match local_cpu_train.sh (depth=4)
    # This ensures we stay within local CPU constraints
    depth = 4  # Same as local_cpu_train.sh (tiny model for learning)
    max_seq_len = 1024  # Same as local_cpu_train.sh
    device_batch_size = 1  # Same as local_cpu_train.sh
    total_batch_size = 1024  # Same as local_cpu_train.sh
    num_iterations = 3  # Just 3 iterations for debugging (vs 50 in full training)
    device_type = "cpu"  # Force CPU mode for safety

    # Allow CLI overrides
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open(os.path.join('nanochat', 'configurator.py')).read())

    # Setup - ensure CPU mode unless explicitly overridden
    if device_type == "":
        device_type = "cpu"  # Default to CPU for this debug script

    # Safety check: warn if trying to use GPU
    if device_type != "cpu":
        print(f"{Colors.YELLOW}WARNING: You specified device_type={device_type}{Colors.END}")
        print(f"{Colors.YELLOW}This debug script is designed for local CPU training.{Colors.END}")
        print(f"{Colors.YELLOW}Continuing anyway, but parameters may not be optimized for GPU.{Colors.END}\n")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Model configuration
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads

    print0(f"\n{Colors.BOLD}Initializing instrumented GPT model...{Colors.END}")
    print0(f"  depth={num_layers}, d_model={model_dim}, n_heads={num_heads}")
    print0(f"  Device: {device}, Device type: {device_type}\n")

    # Create model
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim
    )

    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = InstrumentedGPT(model_config, verbose=True)

    model.to_empty(device=device)
    model.init_weights()

    # Analyze scaling
    analyze_model_scaling(model_config)

    # Create data loader
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="train", device=device
    )

    # Training loop with detailed logging
    print_section("STARTING TRAINING WITH INSTRUMENTATION")
    print(f"{Colors.BLUE}Running {num_iterations} iterations with full component tracing{Colors.END}\n")

    model.train()

    for step in range(num_iterations):
        print(f"\n{Colors.BOLD}{Colors.RED}{'#'*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}ITERATION {step+1}/{num_iterations}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.RED}{'#'*80}{Colors.END}\n")

        # Get batch
        x, y = next(train_loader)

        # Forward pass (verbose for first iteration, quiet for rest)
        model.verbose = (step == 0)  # Only trace first iteration in detail

        with autocast_ctx:
            loss = model(x, y)

        # Backward pass
        loss.backward()

        if step == 0:
            print_section("BACKWARD PASS")
            print(f"{Colors.BLUE}Gradients computed via backpropagation{Colors.END}")
            print(f"{Colors.BLUE}Checking gradients for key components:{Colors.END}\n")

            # Check gradients
            if model.transformer.wte.weight.grad is not None:
                print_component("Token embedding gradients",
                              model.transformer.wte.weight.grad,
                              "Gradients for updating embedding matrix")

            if model.lm_head.weight.grad is not None:
                print_component("LM head gradients",
                              model.lm_head.weight.grad,
                              "Gradients for updating output projection")

        # Clear gradients
        model.zero_grad(set_to_none=True)

        print(f"{Colors.GREEN}✓ Iteration {step+1} complete - Loss: {loss.item():.6f}{Colors.END}\n")

    print_section("TRAINING COMPLETE")
    print(f"{Colors.GREEN}Successfully traced {num_iterations} iterations!{Colors.END}")
    print(f"{Colors.YELLOW}Key takeaways:{Colors.END}")
    print(f"  1. Token embeddings (wte) convert tokens to vectors")
    print(f"  2. RoPE adds positional info via rotation (not learned)")
    print(f"  3. Each Block has Attention + MLP with residuals")
    print(f"  4. RMSNorm stabilizes (no learnable params)")
    print(f"  5. LM head projects to vocab (untied from wte)")
    print(f"  6. Model scales with depth, d_model, and n_heads\n")

    compute_cleanup()
