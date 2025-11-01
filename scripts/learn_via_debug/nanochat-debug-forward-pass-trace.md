# GPT Component Debug Trace

**Generated**: 2025-11-01 14:32:09
**Configuration**: depth=4, d_model=256, n_heads=2
**Device**: cpu (cpu)
**Iterations**: 3


---

## <span style="color: #af87ff">MODEL SCALING ANALYSIS</span>


---

## <span style="color: #af87ff">STARTING TRAINING WITH INSTRUMENTATION</span>

<strong><span style="color: #d70000">ITERATION 1/3</span>
<strong><span style="color: #d70000">################################################################################</span>


---

## <span style="color: #af87ff">FORWARD PASS #0</span>

#### <span style="color: #5fafff">Input tokens (idx)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.int32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">4730.903320</span>
- **Std**: <span style="color: #5faf00">9284.364258</span>
- **Min**: <span style="color: #5faf00">10.000000</span>
- **Max**: <span style="color: #5faf00">65527.000000</span>

*<span style="color: #0087ff">Batch size: 1, Sequence length: 1024</span>*

#### <span style="color: #5fafff">Target tokens</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.int64`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">4670.597656</span>
- **Std**: <span style="color: #5faf00">9087.552734</span>
- **Min**: <span style="color: #5faf00">10.000000</span>
- **Max**: <span style="color: #5faf00">64929.000000</span>


---

## <span style="color: #af87ff">1. TOKEN EMBEDDINGS (wte)</span>

<span style="color: #0087ff">The token embedding layer converts discrete token IDs to continuous vectors</span>
<span style="color: #0087ff">Embedding matrix shape: (vocab_size=65536, n_embd=256)</span>

#### <span style="color: #5fafff">Token embeddings (before norm)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002861</span>
- **Std**: <span style="color: #5faf00">0.994997</span>
- **Min**: <span style="color: #5faf00">-4.497148</span>
- **Max**: <span style="color: #5faf00">4.629077</span>

*<span style="color: #0087ff">Each token is now a 256-dimensional vector</span>*

<span style="color: #d7af00">Note: wte is learned during training</span>

#### <span style="color: #5fafff">Token embeddings (after RMSNorm)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalized to stabilize training</span>*


---

## <span style="color: #af87ff">2. POSITIONAL EMBEDDINGS (RoPE)</span>

<span style="color: #0087ff">RoPE (Rotary Position Embeddings) encodes position information</span>
<span style="color: #0087ff">Unlike learned positional embeddings, RoPE is applied via rotation</span>
<span style="color: #0087ff">Rotary embeddings are precomputed and cached for efficiency</span>

#### <span style="color: #5fafff">Rotary cos</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1, 64)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.bfloat16`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.286139</span>
- **Std**: <span style="color: #5faf00">0.723427</span>
- **Min**: <span style="color: #5faf00">-1.000000</span>
- **Max**: <span style="color: #5faf00">1.000000</span>

*<span style="color: #0087ff">Precomputed cosine values for position encoding</span>*

#### <span style="color: #5fafff">Rotary sin</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1, 64)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.bfloat16`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.163450</span>
- **Std**: <span style="color: #5faf00">0.606759</span>
- **Min**: <span style="color: #5faf00">-1.000000</span>
- **Max**: <span style="color: #5faf00">1.000000</span>

*<span style="color: #0087ff">Precomputed sine values for position encoding</span>*

<span style="color: #d7af00">These will be applied to Q and K in attention via rotation</span>

<span style="color: #0087ff">RoPE formula: rotate pairs of dims using cos/sin</span>
<span style="color: #0087ff">  q_rotated = [q[:d/2] * cos + q[d/2:] * sin, q[:d/2] * (-sin) + q[d/2:] * cos]</span>


---

## <span style="color: #af87ff">3. TRANSFORMER BLOCK STACK (depth=4)</span>

<span style="color: #0087ff">Model has 4 layers stacked sequentially</span>
<span style="color: #0087ff">Each block contains: Attention + MLP with residual connections</span>

#### <span style="color: #5fafff">  Input to Block 0</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Input from previous layer (or token embeddings if layer 0)</span>*

<span style="color: #d7af00">  Formula: x = x + Attention(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 1a: RMSNorm(x) [pre-attention]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE attention (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Project to Q,K,V → RoPE → QK Norm → Scaled Dot-Product → Causal Mask → Softmax</span>

<span style="color: #d7af00">  [1b.i] Project to Queries, Keys, Values</span>
#### <span style="color: #5fafff">    Queries (Q)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.005007</span>
- **Std**: <span style="color: #5faf00">1.005951</span>
- **Min**: <span style="color: #5faf00">-4.526191</span>
- **Max**: <span style="color: #5faf00">4.440659</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Keys (K)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.004547</span>
- **Std**: <span style="color: #5faf00">1.001035</span>
- **Min**: <span style="color: #5faf00">-4.371484</span>
- **Max**: <span style="color: #5faf00">4.612371</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Values (V)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000568</span>
- **Std**: <span style="color: #5faf00">0.996983</span>
- **Min**: <span style="color: #5faf00">-4.165663</span>
- **Max**: <span style="color: #5faf00">4.348375</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

<span style="color: #d7af00">  [1b.ii] Apply RoPE (Rotary Position Embeddings)</span>
<span style="color: #0087ff">    RoPE encodes position via rotation (no learned parameters)</span>
<span style="color: #0087ff">    Formula: rotate(x, θ) where θ depends on position</span>
<span style="color: #0087ff">    Applied to Q and K only (not V)</span>

#### <span style="color: #5fafff">    Q before RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.005007</span>
- **Std**: <span style="color: #5faf00">1.005951</span>
- **Min**: <span style="color: #5faf00">-4.526191</span>
- **Max**: <span style="color: #5faf00">4.440659</span>

*<span style="color: #0087ff">Original query vectors</span>*

#### <span style="color: #5fafff">    Q after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.003880</span>
- **Std**: <span style="color: #5faf00">1.006000</span>
- **Min**: <span style="color: #5faf00">-4.604209</span>
- **Max**: <span style="color: #5faf00">4.628593</span>

*<span style="color: #0087ff">Position-encoded query vectors via rotation</span>*

#### <span style="color: #5fafff">    K after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.001734</span>
- **Std**: <span style="color: #5faf00">1.001088</span>
- **Min**: <span style="color: #5faf00">-4.352281</span>
- **Max**: <span style="color: #5faf00">4.694830</span>

*<span style="color: #0087ff">Position-encoded key vectors via rotation</span>*

<span style="color: #d7af00">    Note: RoPE allows relative position encoding</span>
<span style="color: #d7af00">    The dot product Q·K naturally captures relative distances</span>

<span style="color: #d7af00">  [1b.iii] QK Normalization</span>
<span style="color: #0087ff">    Normalize Q and K separately for training stability</span>
<span style="color: #0087ff">    Uses RMSNorm (no learnable parameters)</span>

#### <span style="color: #5fafff">    Q after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.003523</span>
- **Std**: <span style="color: #5faf00">0.999996</span>
- **Min**: <span style="color: #5faf00">-4.534496</span>
- **Max**: <span style="color: #5faf00">4.560542</span>

*<span style="color: #0087ff">Normalized queries for stable training</span>*

#### <span style="color: #5fafff">    K after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.001801</span>
- **Std**: <span style="color: #5faf00">1.000000</span>
- **Min**: <span style="color: #5faf00">-4.497032</span>
- **Max**: <span style="color: #5faf00">4.184650</span>

*<span style="color: #0087ff">Normalized keys for stable training</span>*

<span style="color: #0087ff">    Reshape: [B, T, H, D] → [B, H, T, D]</span>
<span style="color: #0087ff">    Makes head dimension the batch dimension for parallel processing</span>

#### <span style="color: #5fafff">    Q transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.003523</span>
- **Std**: <span style="color: #5faf00">0.999996</span>
- **Min**: <span style="color: #5faf00">-4.534496</span>
- **Max**: <span style="color: #5faf00">4.560542</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    K transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.001801</span>
- **Std**: <span style="color: #5faf00">1.000000</span>
- **Min**: <span style="color: #5faf00">-4.497032</span>
- **Max**: <span style="color: #5faf00">4.184650</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    V transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000568</span>
- **Std**: <span style="color: #5faf00">0.996983</span>
- **Min**: <span style="color: #5faf00">-4.165663</span>
- **Max**: <span style="color: #5faf00">4.348375</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Formula: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V</span>
<span style="color: #0087ff">    Causal mask ensures token i can only attend to tokens ≤ i</span>
<span style="color: #0087ff">    This maintains autoregressive property for language modeling</span>

#### <span style="color: #5fafff">    Attention output (multi-head)</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.003387</span>
- **Std**: <span style="color: #5faf00">0.177209</span>
- **Min**: <span style="color: #5faf00">-2.612478</span>
- **Max**: <span style="color: #5faf00">2.507700</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Reshape: [B, H, T, D] → [B, T, H*D] = [B, T, d_model]</span>
<span style="color: #0087ff">    Then apply output projection to residual stream</span>

#### <span style="color: #5fafff">    Concatenated heads</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.003387</span>
- **Std**: <span style="color: #5faf00">0.177209</span>
- **Min**: <span style="color: #5faf00">-2.612478</span>
- **Max**: <span style="color: #5faf00">2.507700</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    Final attention output (after c_proj)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">  Step 1c: x + Attention(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Add attention output to input (residual connection)</span>*

<span style="color: #d7af00">  Formula: x = x + MLP(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 2a: RMSNorm(x) [pre-MLP]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE MLP (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Two linear layers with ReLU² activation</span>
<span style="color: #0087ff">  Expansion pattern: d_model → 4*d_model → d_model</span>
<span style="color: #0087ff">  Formula: MLP(x) = c_proj(ReLU²(c_fc(x)))</span>

<span style="color: #d7af00">  [2b.i] First Linear Layer: Expansion to 4x dimension</span>
<span style="color: #0087ff">    c_fc: Linear(256, 1024, bias=False)</span>
<span style="color: #0087ff">    Expands representation to wider intermediate dimension</span>

#### <span style="color: #5fafff">    Input to MLP</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    After c_fc (expansion)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.000038</span>
- **Std**: <span style="color: #5faf00">0.997663</span>
- **Min**: <span style="color: #5faf00">-4.531831</span>
- **Max**: <span style="color: #5faf00">4.659674</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, 4*d_model=1024] - Expanded to 4x</span>*

<span style="color: #0087ff">    Formula: ReLU²(x) = (max(0, x))²</span>
<span style="color: #0087ff">    Why ReLU²? Provides smoother gradients than ReLU</span>
<span style="color: #0087ff">    Alternative to: GELU, SwiGLU, or other activations</span>

#### <span style="color: #5fafff">    After ReLU (before squaring)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.397704</span>
- **Std**: <span style="color: #5faf00">0.583067</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">4.659674</span>

*<span style="color: #0087ff">All negative values zeroed out</span>*

#### <span style="color: #5fafff">    After ReLU² (squared)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.498136</span>
- **Std**: <span style="color: #5faf00">1.119545</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">21.712563</span>

*<span style="color: #0087ff">Non-linearity: enhances positive activations</span>*

<span style="color: #d7af00">    Sparsity: 524845/1048576 (50.1%) zeros after ReLU²</span>
<span style="color: #d7af00">    Original negative values: 524845/1048576 (50.1%)</span>

<span style="color: #d7af00">  [2b.iii] Second Linear Layer: Projection back to d_model</span>
<span style="color: #0087ff">    c_proj: Linear(1024, 256, bias=False)</span>
<span style="color: #0087ff">    Projects back to original dimension for residual connection</span>

#### <span style="color: #5fafff">    After c_proj (output)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256] - Back to original dim</span>*

<span style="color: #d7af00">  Key insight: 4x expansion provides capacity for complex transformations</span>
<span style="color: #d7af00">  No bias terms (bias=False) - simpler, often works just as well</span>

<span style="color: #d7af00">  [2c] Residual Connection</span>
<span style="color: #0087ff">    Formula: x_new = x_old + MLP(RMSNorm(x_old))</span>
<span style="color: #0087ff">    Allows gradient flow directly through the network</span>

#### <span style="color: #5fafff">    Input to MLP path (x_after_attn)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">State after attention residual</span>*

#### <span style="color: #5fafff">    MLP output</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Transformation to be added</span>*

#### <span style="color: #5fafff">    x + MLP(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Combined via addition (residual connection)</span>*

<span style="color: #d7af00">  Key insight: Pre-norm means we normalize BEFORE each operation</span>
<span style="color: #d7af00">  Residuals allow gradients to flow directly through the network</span>

#### <span style="color: #5fafff">  Input to Block 1</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Input from previous layer (or token embeddings if layer 0)</span>*

<span style="color: #d7af00">  Formula: x = x + Attention(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 1a: RMSNorm(x) [pre-attention]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE attention (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Project to Q,K,V → RoPE → QK Norm → Scaled Dot-Product → Causal Mask → Softmax</span>

<span style="color: #d7af00">  [1b.i] Project to Queries, Keys, Values</span>
#### <span style="color: #5fafff">    Queries (Q)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.005793</span>
- **Std**: <span style="color: #5faf00">0.995982</span>
- **Min**: <span style="color: #5faf00">-4.125390</span>
- **Max**: <span style="color: #5faf00">5.065095</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Keys (K)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.007150</span>
- **Std**: <span style="color: #5faf00">0.993268</span>
- **Min**: <span style="color: #5faf00">-4.249630</span>
- **Max**: <span style="color: #5faf00">4.414808</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Values (V)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.003871</span>
- **Std**: <span style="color: #5faf00">1.001604</span>
- **Min**: <span style="color: #5faf00">-4.378510</span>
- **Max**: <span style="color: #5faf00">4.375952</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

<span style="color: #d7af00">  [1b.ii] Apply RoPE (Rotary Position Embeddings)</span>
<span style="color: #0087ff">    RoPE encodes position via rotation (no learned parameters)</span>
<span style="color: #0087ff">    Formula: rotate(x, θ) where θ depends on position</span>
<span style="color: #0087ff">    Applied to Q and K only (not V)</span>

#### <span style="color: #5fafff">    Q before RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.005793</span>
- **Std**: <span style="color: #5faf00">0.995982</span>
- **Min**: <span style="color: #5faf00">-4.125390</span>
- **Max**: <span style="color: #5faf00">5.065095</span>

*<span style="color: #0087ff">Original query vectors</span>*

#### <span style="color: #5fafff">    Q after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.001950</span>
- **Std**: <span style="color: #5faf00">0.996038</span>
- **Min**: <span style="color: #5faf00">-4.141755</span>
- **Max**: <span style="color: #5faf00">4.761730</span>

*<span style="color: #0087ff">Position-encoded query vectors via rotation</span>*

#### <span style="color: #5fafff">    K after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.000120</span>
- **Std**: <span style="color: #5faf00">0.993334</span>
- **Min**: <span style="color: #5faf00">-4.198006</span>
- **Max**: <span style="color: #5faf00">4.275361</span>

*<span style="color: #0087ff">Position-encoded key vectors via rotation</span>*

<span style="color: #d7af00">    Note: RoPE allows relative position encoding</span>
<span style="color: #d7af00">    The dot product Q·K naturally captures relative distances</span>

<span style="color: #d7af00">  [1b.iii] QK Normalization</span>
<span style="color: #0087ff">    Normalize Q and K separately for training stability</span>
<span style="color: #0087ff">    Uses RMSNorm (no learnable parameters)</span>

#### <span style="color: #5fafff">    Q after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002017</span>
- **Std**: <span style="color: #5faf00">1.000000</span>
- **Min**: <span style="color: #5faf00">-4.196603</span>
- **Max**: <span style="color: #5faf00">4.850463</span>

*<span style="color: #0087ff">Normalized queries for stable training</span>*

#### <span style="color: #5fafff">    K after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000040</span>
- **Std**: <span style="color: #5faf00">1.000002</span>
- **Min**: <span style="color: #5faf00">-4.403380</span>
- **Max**: <span style="color: #5faf00">4.211719</span>

*<span style="color: #0087ff">Normalized keys for stable training</span>*

<span style="color: #0087ff">    Reshape: [B, T, H, D] → [B, H, T, D]</span>
<span style="color: #0087ff">    Makes head dimension the batch dimension for parallel processing</span>

#### <span style="color: #5fafff">    Q transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002017</span>
- **Std**: <span style="color: #5faf00">1.000000</span>
- **Min**: <span style="color: #5faf00">-4.196603</span>
- **Max**: <span style="color: #5faf00">4.850463</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    K transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000040</span>
- **Std**: <span style="color: #5faf00">1.000002</span>
- **Min**: <span style="color: #5faf00">-4.403380</span>
- **Max**: <span style="color: #5faf00">4.211719</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    V transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.003871</span>
- **Std**: <span style="color: #5faf00">1.001604</span>
- **Min**: <span style="color: #5faf00">-4.378510</span>
- **Max**: <span style="color: #5faf00">4.375952</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Formula: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V</span>
<span style="color: #0087ff">    Causal mask ensures token i can only attend to tokens ≤ i</span>
<span style="color: #0087ff">    This maintains autoregressive property for language modeling</span>

#### <span style="color: #5fafff">    Attention output (multi-head)</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.001724</span>
- **Std**: <span style="color: #5faf00">0.178429</span>
- **Min**: <span style="color: #5faf00">-2.778253</span>
- **Max**: <span style="color: #5faf00">2.833642</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Reshape: [B, H, T, D] → [B, T, H*D] = [B, T, d_model]</span>
<span style="color: #0087ff">    Then apply output projection to residual stream</span>

#### <span style="color: #5fafff">    Concatenated heads</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.001724</span>
- **Std**: <span style="color: #5faf00">0.178429</span>
- **Min**: <span style="color: #5faf00">-2.778253</span>
- **Max**: <span style="color: #5faf00">2.833642</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    Final attention output (after c_proj)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">  Step 1c: x + Attention(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Add attention output to input (residual connection)</span>*

<span style="color: #d7af00">  Formula: x = x + MLP(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 2a: RMSNorm(x) [pre-MLP]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE MLP (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Two linear layers with ReLU² activation</span>
<span style="color: #0087ff">  Expansion pattern: d_model → 4*d_model → d_model</span>
<span style="color: #0087ff">  Formula: MLP(x) = c_proj(ReLU²(c_fc(x)))</span>

<span style="color: #d7af00">  [2b.i] First Linear Layer: Expansion to 4x dimension</span>
<span style="color: #0087ff">    c_fc: Linear(256, 1024, bias=False)</span>
<span style="color: #0087ff">    Expands representation to wider intermediate dimension</span>

#### <span style="color: #5fafff">    Input to MLP</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    After c_fc (expansion)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.000874</span>
- **Std**: <span style="color: #5faf00">0.999181</span>
- **Min**: <span style="color: #5faf00">-5.231340</span>
- **Max**: <span style="color: #5faf00">4.550129</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, 4*d_model=1024] - Expanded to 4x</span>*

<span style="color: #0087ff">    Formula: ReLU²(x) = (max(0, x))²</span>
<span style="color: #0087ff">    Why ReLU²? Provides smoother gradients than ReLU</span>
<span style="color: #0087ff">    Alternative to: GELU, SwiGLU, or other activations</span>

#### <span style="color: #5fafff">    After ReLU (before squaring)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.397364</span>
- **Std**: <span style="color: #5faf00">0.583206</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">4.550129</span>

*<span style="color: #0087ff">All negative values zeroed out</span>*

#### <span style="color: #5fafff">    After ReLU² (squared)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.498028</span>
- **Std**: <span style="color: #5faf00">1.114632</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">20.703672</span>

*<span style="color: #0087ff">Non-linearity: enhances positive activations</span>*

<span style="color: #d7af00">    Sparsity: 525688/1048576 (50.1%) zeros after ReLU²</span>
<span style="color: #d7af00">    Original negative values: 525688/1048576 (50.1%)</span>

<span style="color: #d7af00">  [2b.iii] Second Linear Layer: Projection back to d_model</span>
<span style="color: #0087ff">    c_proj: Linear(1024, 256, bias=False)</span>
<span style="color: #0087ff">    Projects back to original dimension for residual connection</span>

#### <span style="color: #5fafff">    After c_proj (output)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256] - Back to original dim</span>*

<span style="color: #d7af00">  Key insight: 4x expansion provides capacity for complex transformations</span>
<span style="color: #d7af00">  No bias terms (bias=False) - simpler, often works just as well</span>

<span style="color: #d7af00">  [2c] Residual Connection</span>
<span style="color: #0087ff">    Formula: x_new = x_old + MLP(RMSNorm(x_old))</span>
<span style="color: #0087ff">    Allows gradient flow directly through the network</span>

#### <span style="color: #5fafff">    Input to MLP path (x_after_attn)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">State after attention residual</span>*

#### <span style="color: #5fafff">    MLP output</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Transformation to be added</span>*

#### <span style="color: #5fafff">    x + MLP(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Combined via addition (residual connection)</span>*

<span style="color: #d7af00">  Key insight: Pre-norm means we normalize BEFORE each operation</span>
<span style="color: #d7af00">  Residuals allow gradients to flow directly through the network</span>

#### <span style="color: #5fafff">  Input to Block 2</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Input from previous layer (or token embeddings if layer 0)</span>*

<span style="color: #d7af00">  Formula: x = x + Attention(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 1a: RMSNorm(x) [pre-attention]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE attention (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Project to Q,K,V → RoPE → QK Norm → Scaled Dot-Product → Causal Mask → Softmax</span>

<span style="color: #d7af00">  [1b.i] Project to Queries, Keys, Values</span>
#### <span style="color: #5fafff">    Queries (Q)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001989</span>
- **Std**: <span style="color: #5faf00">1.000418</span>
- **Min**: <span style="color: #5faf00">-4.408793</span>
- **Max**: <span style="color: #5faf00">4.299022</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Keys (K)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001357</span>
- **Std**: <span style="color: #5faf00">1.016430</span>
- **Min**: <span style="color: #5faf00">-4.155736</span>
- **Max**: <span style="color: #5faf00">4.549500</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Values (V)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000018</span>
- **Std**: <span style="color: #5faf00">0.994930</span>
- **Min**: <span style="color: #5faf00">-4.475419</span>
- **Max**: <span style="color: #5faf00">4.554348</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

<span style="color: #d7af00">  [1b.ii] Apply RoPE (Rotary Position Embeddings)</span>
<span style="color: #0087ff">    RoPE encodes position via rotation (no learned parameters)</span>
<span style="color: #0087ff">    Formula: rotate(x, θ) where θ depends on position</span>
<span style="color: #0087ff">    Applied to Q and K only (not V)</span>

#### <span style="color: #5fafff">    Q before RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001989</span>
- **Std**: <span style="color: #5faf00">1.000418</span>
- **Min**: <span style="color: #5faf00">-4.408793</span>
- **Max**: <span style="color: #5faf00">4.299022</span>

*<span style="color: #0087ff">Original query vectors</span>*

#### <span style="color: #5fafff">    Q after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.003251</span>
- **Std**: <span style="color: #5faf00">1.000451</span>
- **Min**: <span style="color: #5faf00">-4.479077</span>
- **Max**: <span style="color: #5faf00">4.329697</span>

*<span style="color: #0087ff">Position-encoded query vectors via rotation</span>*

#### <span style="color: #5fafff">    K after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002011</span>
- **Std**: <span style="color: #5faf00">1.016466</span>
- **Min**: <span style="color: #5faf00">-4.411456</span>
- **Max**: <span style="color: #5faf00">4.350796</span>

*<span style="color: #0087ff">Position-encoded key vectors via rotation</span>*

<span style="color: #d7af00">    Note: RoPE allows relative position encoding</span>
<span style="color: #d7af00">    The dot product Q·K naturally captures relative distances</span>

<span style="color: #d7af00">  [1b.iii] QK Normalization</span>
<span style="color: #0087ff">    Normalize Q and K separately for training stability</span>
<span style="color: #0087ff">    Uses RMSNorm (no learnable parameters)</span>

#### <span style="color: #5fafff">    Q after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.003627</span>
- **Std**: <span style="color: #5faf00">0.999995</span>
- **Min**: <span style="color: #5faf00">-4.342686</span>
- **Max**: <span style="color: #5faf00">4.084086</span>

*<span style="color: #0087ff">Normalized queries for stable training</span>*

#### <span style="color: #5fafff">    K after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001954</span>
- **Std**: <span style="color: #5faf00">1.000000</span>
- **Min**: <span style="color: #5faf00">-4.208494</span>
- **Max**: <span style="color: #5faf00">4.139342</span>

*<span style="color: #0087ff">Normalized keys for stable training</span>*

<span style="color: #0087ff">    Reshape: [B, T, H, D] → [B, H, T, D]</span>
<span style="color: #0087ff">    Makes head dimension the batch dimension for parallel processing</span>

#### <span style="color: #5fafff">    Q transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.003627</span>
- **Std**: <span style="color: #5faf00">0.999995</span>
- **Min**: <span style="color: #5faf00">-4.342686</span>
- **Max**: <span style="color: #5faf00">4.084086</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    K transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001954</span>
- **Std**: <span style="color: #5faf00">1.000000</span>
- **Min**: <span style="color: #5faf00">-4.208494</span>
- **Max**: <span style="color: #5faf00">4.139342</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    V transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000018</span>
- **Std**: <span style="color: #5faf00">0.994930</span>
- **Min**: <span style="color: #5faf00">-4.475419</span>
- **Max**: <span style="color: #5faf00">4.554348</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Formula: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V</span>
<span style="color: #0087ff">    Causal mask ensures token i can only attend to tokens ≤ i</span>
<span style="color: #0087ff">    This maintains autoregressive property for language modeling</span>

#### <span style="color: #5fafff">    Attention output (multi-head)</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001405</span>
- **Std**: <span style="color: #5faf00">0.175041</span>
- **Min**: <span style="color: #5faf00">-2.410661</span>
- **Max**: <span style="color: #5faf00">2.889322</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Reshape: [B, H, T, D] → [B, T, H*D] = [B, T, d_model]</span>
<span style="color: #0087ff">    Then apply output projection to residual stream</span>

#### <span style="color: #5fafff">    Concatenated heads</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.001405</span>
- **Std**: <span style="color: #5faf00">0.175041</span>
- **Min**: <span style="color: #5faf00">-2.410661</span>
- **Max**: <span style="color: #5faf00">2.889322</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    Final attention output (after c_proj)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">  Step 1c: x + Attention(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Add attention output to input (residual connection)</span>*

<span style="color: #d7af00">  Formula: x = x + MLP(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 2a: RMSNorm(x) [pre-MLP]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE MLP (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Two linear layers with ReLU² activation</span>
<span style="color: #0087ff">  Expansion pattern: d_model → 4*d_model → d_model</span>
<span style="color: #0087ff">  Formula: MLP(x) = c_proj(ReLU²(c_fc(x)))</span>

<span style="color: #d7af00">  [2b.i] First Linear Layer: Expansion to 4x dimension</span>
<span style="color: #0087ff">    c_fc: Linear(256, 1024, bias=False)</span>
<span style="color: #0087ff">    Expands representation to wider intermediate dimension</span>

#### <span style="color: #5fafff">    Input to MLP</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    After c_fc (expansion)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002114</span>
- **Std**: <span style="color: #5faf00">1.000856</span>
- **Min**: <span style="color: #5faf00">-4.645262</span>
- **Max**: <span style="color: #5faf00">4.834872</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, 4*d_model=1024] - Expanded to 4x</span>*

<span style="color: #0087ff">    Formula: ReLU²(x) = (max(0, x))²</span>
<span style="color: #0087ff">    Why ReLU²? Provides smoother gradients than ReLU</span>
<span style="color: #0087ff">    Alternative to: GELU, SwiGLU, or other activations</span>

#### <span style="color: #5fafff">    After ReLU (before squaring)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.400218</span>
- **Std**: <span style="color: #5faf00">0.584865</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">4.834872</span>

*<span style="color: #0087ff">All negative values zeroed out</span>*

#### <span style="color: #5fafff">    After ReLU² (squared)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.502242</span>
- **Std**: <span style="color: #5faf00">1.123516</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">23.375990</span>

*<span style="color: #0087ff">Non-linearity: enhances positive activations</span>*

<span style="color: #d7af00">    Sparsity: 523794/1048576 (50.0%) zeros after ReLU²</span>
<span style="color: #d7af00">    Original negative values: 523794/1048576 (50.0%)</span>

<span style="color: #d7af00">  [2b.iii] Second Linear Layer: Projection back to d_model</span>
<span style="color: #0087ff">    c_proj: Linear(1024, 256, bias=False)</span>
<span style="color: #0087ff">    Projects back to original dimension for residual connection</span>

#### <span style="color: #5fafff">    After c_proj (output)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256] - Back to original dim</span>*

<span style="color: #d7af00">  Key insight: 4x expansion provides capacity for complex transformations</span>
<span style="color: #d7af00">  No bias terms (bias=False) - simpler, often works just as well</span>

<span style="color: #d7af00">  [2c] Residual Connection</span>
<span style="color: #0087ff">    Formula: x_new = x_old + MLP(RMSNorm(x_old))</span>
<span style="color: #0087ff">    Allows gradient flow directly through the network</span>

#### <span style="color: #5fafff">    Input to MLP path (x_after_attn)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">State after attention residual</span>*

#### <span style="color: #5fafff">    MLP output</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Transformation to be added</span>*

#### <span style="color: #5fafff">    x + MLP(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Combined via addition (residual connection)</span>*

<span style="color: #d7af00">  Key insight: Pre-norm means we normalize BEFORE each operation</span>
<span style="color: #d7af00">  Residuals allow gradients to flow directly through the network</span>

#### <span style="color: #5fafff">  Input to Block 3</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Input from previous layer (or token embeddings if layer 0)</span>*

<span style="color: #d7af00">  Formula: x = x + Attention(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 1a: RMSNorm(x) [pre-attention]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE attention (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Project to Q,K,V → RoPE → QK Norm → Scaled Dot-Product → Causal Mask → Softmax</span>

<span style="color: #d7af00">  [1b.i] Project to Queries, Keys, Values</span>
#### <span style="color: #5fafff">    Queries (Q)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002573</span>
- **Std**: <span style="color: #5faf00">1.006546</span>
- **Min**: <span style="color: #5faf00">-4.603458</span>
- **Max**: <span style="color: #5faf00">4.590590</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Keys (K)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.011672</span>
- **Std**: <span style="color: #5faf00">1.004807</span>
- **Min**: <span style="color: #5faf00">-3.942575</span>
- **Max**: <span style="color: #5faf00">4.120285</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

#### <span style="color: #5fafff">    Values (V)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.008103</span>
- **Std**: <span style="color: #5faf00">1.009636</span>
- **Min**: <span style="color: #5faf00">-4.318686</span>
- **Max**: <span style="color: #5faf00">4.124578</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, n_kv_head=2, head_dim=128]</span>*

<span style="color: #d7af00">  [1b.ii] Apply RoPE (Rotary Position Embeddings)</span>
<span style="color: #0087ff">    RoPE encodes position via rotation (no learned parameters)</span>
<span style="color: #0087ff">    Formula: rotate(x, θ) where θ depends on position</span>
<span style="color: #0087ff">    Applied to Q and K only (not V)</span>

#### <span style="color: #5fafff">    Q before RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002573</span>
- **Std**: <span style="color: #5faf00">1.006546</span>
- **Min**: <span style="color: #5faf00">-4.603458</span>
- **Max**: <span style="color: #5faf00">4.590590</span>

*<span style="color: #0087ff">Original query vectors</span>*

#### <span style="color: #5fafff">    Q after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.003068</span>
- **Std**: <span style="color: #5faf00">1.006581</span>
- **Min**: <span style="color: #5faf00">-4.426324</span>
- **Max**: <span style="color: #5faf00">4.519401</span>

*<span style="color: #0087ff">Position-encoded query vectors via rotation</span>*

#### <span style="color: #5fafff">    K after RoPE</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002832</span>
- **Std**: <span style="color: #5faf00">1.004913</span>
- **Min**: <span style="color: #5faf00">-4.240757</span>
- **Max**: <span style="color: #5faf00">4.261254</span>

*<span style="color: #0087ff">Position-encoded key vectors via rotation</span>*

<span style="color: #d7af00">    Note: RoPE allows relative position encoding</span>
<span style="color: #d7af00">    The dot product Q·K naturally captures relative distances</span>

<span style="color: #d7af00">  [1b.iii] QK Normalization</span>
<span style="color: #0087ff">    Normalize Q and K separately for training stability</span>
<span style="color: #0087ff">    Uses RMSNorm (no learnable parameters)</span>

#### <span style="color: #5fafff">    Q after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002965</span>
- **Std**: <span style="color: #5faf00">0.999997</span>
- **Min**: <span style="color: #5faf00">-4.345214</span>
- **Max**: <span style="color: #5faf00">4.388961</span>

*<span style="color: #0087ff">Normalized queries for stable training</span>*

#### <span style="color: #5fafff">    K after QK norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 2, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002700</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.020113</span>
- **Max**: <span style="color: #5faf00">4.146988</span>

*<span style="color: #0087ff">Normalized keys for stable training</span>*

<span style="color: #0087ff">    Reshape: [B, T, H, D] → [B, H, T, D]</span>
<span style="color: #0087ff">    Makes head dimension the batch dimension for parallel processing</span>

#### <span style="color: #5fafff">    Q transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002965</span>
- **Std**: <span style="color: #5faf00">0.999997</span>
- **Min**: <span style="color: #5faf00">-4.345214</span>
- **Max**: <span style="color: #5faf00">4.388961</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    K transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002700</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.020113</span>
- **Max**: <span style="color: #5faf00">4.146988</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

#### <span style="color: #5fafff">    V transposed</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.008103</span>
- **Std**: <span style="color: #5faf00">1.009636</span>
- **Min**: <span style="color: #5faf00">-4.318686</span>
- **Max**: <span style="color: #5faf00">4.124578</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Formula: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V</span>
<span style="color: #0087ff">    Causal mask ensures token i can only attend to tokens ≤ i</span>
<span style="color: #0087ff">    This maintains autoregressive property for language modeling</span>

#### <span style="color: #5fafff">    Attention output (multi-head)</span>

- **Shape**: <span style="color: #5faf00">`(1, 2, 1024, 128)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.010063</span>
- **Std**: <span style="color: #5faf00">0.180329</span>
- **Min**: <span style="color: #5faf00">-3.038945</span>
- **Max**: <span style="color: #5faf00">2.587185</span>

*<span style="color: #0087ff">Shape: [B=1, H=2, T=1024, D=128]</span>*

<span style="color: #0087ff">    Reshape: [B, H, T, D] → [B, T, H*D] = [B, T, d_model]</span>
<span style="color: #0087ff">    Then apply output projection to residual stream</span>

#### <span style="color: #5fafff">    Concatenated heads</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.010063</span>
- **Std**: <span style="color: #5faf00">0.180329</span>
- **Min**: <span style="color: #5faf00">-3.038945</span>
- **Max**: <span style="color: #5faf00">2.587185</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    Final attention output (after c_proj)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">  Step 1c: x + Attention(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Add attention output to input (residual connection)</span>*

<span style="color: #d7af00">  Formula: x = x + MLP(RMSNorm(x))</span>

#### <span style="color: #5fafff">  Step 2a: RMSNorm(x) [pre-MLP]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Normalize BEFORE MLP (pre-norm architecture)</span>*

<span style="color: #0087ff">  Input/Output: [batch=1, seq_len=1024, d_model=256] → [batch=1, seq_len=1024, d_model=256]</span>
<span style="color: #0087ff">  Architecture: Two linear layers with ReLU² activation</span>
<span style="color: #0087ff">  Expansion pattern: d_model → 4*d_model → d_model</span>
<span style="color: #0087ff">  Formula: MLP(x) = c_proj(ReLU²(c_fc(x)))</span>

<span style="color: #d7af00">  [2b.i] First Linear Layer: Expansion to 4x dimension</span>
<span style="color: #0087ff">    c_fc: Linear(256, 1024, bias=False)</span>
<span style="color: #0087ff">    Expands representation to wider intermediate dimension</span>

#### <span style="color: #5fafff">    Input to MLP</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256]</span>*

#### <span style="color: #5fafff">    After c_fc (expansion)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.002479</span>
- **Std**: <span style="color: #5faf00">1.000838</span>
- **Min**: <span style="color: #5faf00">-5.011234</span>
- **Max**: <span style="color: #5faf00">4.840754</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, 4*d_model=1024] - Expanded to 4x</span>*

<span style="color: #0087ff">    Formula: ReLU²(x) = (max(0, x))²</span>
<span style="color: #0087ff">    Why ReLU²? Provides smoother gradients than ReLU</span>
<span style="color: #0087ff">    Alternative to: GELU, SwiGLU, or other activations</span>

#### <span style="color: #5fafff">    After ReLU (before squaring)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.397704</span>
- **Std**: <span style="color: #5faf00">0.584432</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">4.840754</span>

*<span style="color: #0087ff">All negative values zeroed out</span>*

#### <span style="color: #5fafff">    After ReLU² (squared)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 1024)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.499729</span>
- **Std**: <span style="color: #5faf00">1.122004</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">23.432899</span>

*<span style="color: #0087ff">Non-linearity: enhances positive activations</span>*

<span style="color: #d7af00">    Sparsity: 526773/1048576 (50.2%) zeros after ReLU²</span>
<span style="color: #d7af00">    Original negative values: 526773/1048576 (50.2%)</span>

<span style="color: #d7af00">  [2b.iii] Second Linear Layer: Projection back to d_model</span>
<span style="color: #0087ff">    c_proj: Linear(1024, 256, bias=False)</span>
<span style="color: #0087ff">    Projects back to original dimension for residual connection</span>

#### <span style="color: #5fafff">    After c_proj (output)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Shape: [B=1, T=1024, d_model=256] - Back to original dim</span>*

<span style="color: #d7af00">  Key insight: 4x expansion provides capacity for complex transformations</span>
<span style="color: #d7af00">  No bias terms (bias=False) - simpler, often works just as well</span>

<span style="color: #d7af00">  [2c] Residual Connection</span>
<span style="color: #0087ff">    Formula: x_new = x_old + MLP(RMSNorm(x_old))</span>
<span style="color: #0087ff">    Allows gradient flow directly through the network</span>

#### <span style="color: #5fafff">    Input to MLP path (x_after_attn)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">State after attention residual</span>*

#### <span style="color: #5fafff">    MLP output</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Transformation to be added</span>*

#### <span style="color: #5fafff">    x + MLP(RMSNorm(x)) [RESIDUAL]</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Combined via addition (residual connection)</span>*

<span style="color: #d7af00">  Key insight: Pre-norm means we normalize BEFORE each operation</span>
<span style="color: #d7af00">  Residuals allow gradients to flow directly through the network</span>


---

## <span style="color: #af87ff">4. FINAL RMSNorm</span>

<span style="color: #0087ff">Final normalization before the language modeling head</span>

#### <span style="color: #5fafff">Before final norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

#### <span style="color: #5fafff">After final norm</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.002904</span>
- **Std**: <span style="color: #5faf00">0.999998</span>
- **Min**: <span style="color: #5faf00">-4.750214</span>
- **Max**: <span style="color: #5faf00">4.510354</span>

*<span style="color: #0087ff">Ready for LM head projection</span>*

<span style="color: #d7af00">  where rms(x) = sqrt(mean(x^2))</span>
<span style="color: #d7af00">  No learnable parameters in this implementation</span>


---

## <span style="color: #af87ff">5. LANGUAGE MODELING HEAD</span>

<span style="color: #0087ff">Projects from model dimension to vocabulary size</span>
<span style="color: #0087ff">Shape: (n_embd=256) -> (vocab_size=65536)</span>
<span style="color: #0087ff">Weights are UNTIED from token embeddings (separate parameters)</span>

#### <span style="color: #5fafff">Logits (before softcap)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 65536)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Raw scores for each token in vocabulary</span>*

#### <span style="color: #5fafff">Logits (after softcap)</span>

- **Shape**: <span style="color: #5faf00">`(1, 1024, 65536)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Capped at ±15 to prevent extreme values</span>*


---

## <span style="color: #af87ff">6. LOSS COMPUTATION (Training Mode)</span>

#### <span style="color: #5fafff">Loss</span>

- **Shape**: <span style="color: #5faf00">`(1,)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">11.090358</span>
- **Std**: <span style="color: #5faf00">nan</span>
- **Min**: <span style="color: #5faf00">11.090358</span>
- **Max**: <span style="color: #5faf00">11.090358</span>

*<span style="color: #0087ff">Cross-entropy loss (reduction=mean)</span>*


---

## <span style="color: #af87ff">BACKWARD PASS</span>

<span style="color: #0087ff">Gradients computed via backpropagation</span>
<span style="color: #0087ff">Checking gradients for key components:</span>

#### <span style="color: #5fafff">Token embedding gradients</span>

- **Shape**: <span style="color: #5faf00">`(65536, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">0.000000</span>
- **Std**: <span style="color: #5faf00">0.000000</span>
- **Min**: <span style="color: #5faf00">0.000000</span>
- **Max**: <span style="color: #5faf00">0.000000</span>

*<span style="color: #0087ff">Gradients for updating embedding matrix</span>*

#### <span style="color: #5fafff">LM head gradients</span>

- **Shape**: <span style="color: #5faf00">`(65536, 256)`</span>
- **Dtype**: <span style="color: #5faf00">`torch.float32`</span>
- **Device**: <span style="color: #5faf00">`cpu`</span>
- **Mean**: <span style="color: #5faf00">-0.000000</span>
- **Std**: <span style="color: #5faf00">0.000153</span>
- **Min**: <span style="color: #5faf00">-0.028741</span>
- **Max**: <span style="color: #5faf00">0.034515</span>

*<span style="color: #0087ff">Gradients for updating output projection</span>*

<span style="color: #5faf00">✓ Iteration 1 complete - Loss: 11.090358</span>

<strong><span style="color: #d70000">ITERATION 2/3</span>
<strong><span style="color: #d70000">################################################################################</span>

<span style="color: #5faf00">✓ Iteration 2 complete - Loss: 11.090358</span>

<strong><span style="color: #d70000">ITERATION 3/3</span>
<strong><span style="color: #d70000">################################################################################</span>

<span style="color: #5faf00">✓ Iteration 3 complete - Loss: 11.090358</span>


---

## <span style="color: #af87ff">TRAINING COMPLETE</span>


## Key Takeaways

1. **Token embeddings (wte)** convert discrete tokens to continuous vectors
2. **RoPE** adds positional info via rotation (not learned parameters)
3. Each **Block** has Attention + MLP with residual connections
4. **RMSNorm** stabilizes training (no learnable parameters)
5. **LM head** projects to vocabulary (untied from wte)
6. Model scales with depth, d_model, and n_heads
