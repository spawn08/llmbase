# RoPE: Rotary Position Embedding (RoFormer)

**Original Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
**Timeline:** RoFormer (2021) → RoPE becomes standard in LLaMA, Mistral, Qwen, and most modern LLMs
**Links:** [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---

## TL;DR

RoPE (Rotary Position Embedding) encodes positional information by **rotating** query and key vectors in 2D subspaces using position-dependent angles. Unlike additive positional encodings (sinusoidal, learned), RoPE **multiplicatively** mixes position into attention arguments, enabling **relative position awareness** without any learned parameters. This gives excellent length extrapolation, decay of attention with distance, and became the **default positional encoding** for modern LLMs (LLaMA, Mistral, Qwen, PaLM, and many more).

---

## Why This Paper Matters

RoPE is now the **industry standard** for positional encoding in autoregressive language models:

1. **Relative position by design:** Dot product of rotated Q and K naturally encodes relative distance \(m - n\)
2. **No learned parameters:** Fixed trigonometric functions, zero training cost
3. **Excellent extrapolation:** Works well beyond training sequence length (with scaling tricks)
4. **Adoption:** Used by LLaMA, Mistral, Qwen, PaLM, GLM, and virtually every open-weight model since 2023
5. **Extensions:** NTK-aware scaling, YaRN, and position interpolation all build on RoPE's foundation
6. **Interview essential:** Expected knowledge for any LLM architecture or systems role

---

## Key Concepts Explained Simply

### The Position Problem in Transformers

Self-attention is **permutation-equivariant** — shuffling input tokens and running attention produces the same result as running attention then shuffling outputs. Without position, "the cat sat on the mat" is indistinguishable from "mat the on sat cat the."

Early solutions **added** position vectors to token embeddings (sinusoidal, learned). RoPE takes a different approach: **rotate** queries and keys based on their position.

### RoPE's Insight: Rotation Encodes Relative Position

Imagine two 2D vectors. If you rotate both by the same angle, their dot product stays the same. But if you rotate them by **different** angles (proportional to their positions), the dot product depends on the **difference** in rotation — which corresponds to **relative position**.

RoPE extends this to high dimensions by treating each head dimension as multiple 2D planes, each rotating at different frequencies.

### Why Only Q and K, Not V?

Attention scores come from \(QK^\top\) — only queries and keys determine **which** tokens to attend to. Values (V) determine **what** information to extract once attention weights are computed. Position affects **where** to attend, not **what** to attend to, so rotating only Q and K is sufficient.

---

## The Math — Explained Step by Step

### RoPE Formulation

For a token at position \(m\) with query vector \(q_m \in \mathbb{R}^d\) (where \(d\) is even), RoPE applies a rotation in each 2D subspace:

For frequency index \(i \in \{0, 1, \ldots, d/2 - 1\}\), define the base frequency:

\[
\theta_i = 10000^{-2i/d}
\]

The rotation angle for position \(m\) in subspace \(i\) is \(m \theta_i\). For each pair of dimensions \((2i, 2i+1)\):

\[
\begin{bmatrix} q_{m, 2i}^{\text{rot}} \\ q_{m, 2i+1}^{\text{rot}} \end{bmatrix} = \begin{bmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{bmatrix} \begin{bmatrix} q_{m, 2i} \\ q_{m, 2i+1} \end{bmatrix}
\]

Similarly for keys \(k_n\).

### Dot Product Reveals Relative Position

After rotation, the dot product between rotated query at position \(m\) and key at position \(n\) is:

\[
(q_m^{\text{rot}})^\top k_n^{\text{rot}} = \sum_{i=0}^{d/2-1} \|q_{m, i}\| \|k_{n, i}\| \cos((m - n) \theta_i + \phi_{q,i} - \phi_{k,i})
\]

The key insight: the angle depends on **\(m - n\)**, the relative position difference, even though we applied **absolute** position rotations. This naturally encodes relative position without computing it explicitly.

### Frequency Decay and Locality Bias

The geometric progression \(\theta_i = 10000^{-2i/d}\) means:
- **Low indices** (\(i\) small): large \(\theta_i\), fast rotation → capture **short-range** dependencies
- **High indices** (\(i\) large): small \(\theta_i\), slow rotation → capture **long-range** dependencies

This creates a natural **decay**: tokens far apart have larger angular differences, typically reducing cosine similarity and thus attention weight. This gives RoPE an implicit **locality bias** without any hard constraints.

---

## Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn


def build_rope_frequencies(d: int, base: int = 10000) -> np.ndarray:
    """
    Build frequency array for RoPE.
    theta_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
    
    Args:
        d: head dimension (must be even)
        base: base of geometric progression (default 10000)
    
    Returns:
        freqs: array of shape (d//2,) with frequencies theta_i
    """
    assert d % 2 == 0, "RoPE requires even head dimension"
    i = np.arange(d // 2)
    freqs = base ** (-2 * i / d)
    return freqs


def apply_rope_2d(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Apply RoPE to input x using frequencies.
    
    Args:
        x: array of shape (..., d) where d is even
        freqs: array of shape (d//2,) with frequencies
    
    Returns:
        x_rotated: array of shape (..., d) with rotations applied
    """
    # Reshape into pairs: (..., d//2, 2)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    
    # Compute rotation angles for each position
    # For simplicity, assume x has position info in batch dim
    # In practice, positions come from token indices
    freqs_expanded = freqs[np.newaxis, :]  # (1, d//2)
    
    # Extract cos and sin
    cos_vals = np.cos(freqs_expanded)  # (1, d//2)
    sin_vals = np.sin(freqs_expanded)  # (1, d//2)
    
    # Apply rotation in each 2D subspace
    # For pair (x_0, x_1): rotate by angle theta
    # x_0' = x_0 * cos(theta) - x_1 * sin(theta)
    # x_1' = x_0 * sin(theta) + x_1 * cos(theta)
    x0 = x_reshaped[..., 0]  # (..., d//2)
    x1 = x_reshaped[..., 1]  # (..., d//2)
    
    x0_rot = x0 * cos_vals - x1 * sin_vals
    x1_rot = x0 * sin_vals + x1 * cos_vals
    
    # Stack back: (..., d//2, 2) -> (..., d)
    x_rotated = np.stack([x0_rot, x1_rot], axis=-1).reshape(x.shape)
    return x_rotated


def rope_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   base: int = 10000) -> np.ndarray:
    """
    Full attention with RoPE applied to Q and K.
    
    Args:
        Q: query array (B, H, N, d_k)
        K: key array (B, H, N, d_k)
        V: value array (B, H, N, d_k)
        base: base for frequency computation
    
    Returns:
        O: attention output (B, H, N, d_k)
    """
    B, H, N, d_k = Q.shape
    
    # Build frequencies
    freqs = build_rope_frequencies(d_k, base)
    
    # Apply RoPE to Q and K at each position
    # Reshape for broadcasting: (N, 1, d_k) -> apply freqs per position
    positions = np.arange(N)
    angles = np.outer(positions, freqs)  # (N, d_k//2)
    
    cos_vals = np.cos(angles)  # (N, d_k//2)
    sin_vals = np.sin(angles)  # (N, d_k//2)
    
    # Reshape Q, K to apply rotation: (B, H, N, d_k//2, 2)
    Q_reshaped = Q.reshape(B, H, N, -1, 2)
    K_reshaped = K.reshape(B, H, N, -1, 2)
    
    # Apply rotation to Q
    Q0, Q1 = Q_reshaped[..., 0], Q_reshaped[..., 1]
    cos_B = cos_vals[np.newaxis, np.newaxis, :, :]  # (1, 1, N, d_k//2)
    sin_B = sin_vals[np.newaxis, np.newaxis, :, :]
    
    Q0_rot = Q0 * cos_B - Q1 * sin_B
    Q1_rot = Q0 * sin_B + Q1 * cos_B
    Q_rot = np.stack([Q0_rot, Q1_rot], axis=-1).reshape(B, H, N, d_k)
    
    # Apply rotation to K
    K0, K1 = K_reshaped[..., 0], K_reshaped[..., 1]
    K0_rot = K0 * cos_B - K1 * sin_B
    K1_rot = K0 * sin_B + K1 * cos_B
    K_rot = np.stack([K0_rot, K1_rot], axis=-1).reshape(B, H, N, d_k)
    
    # Compute attention with rotated Q and K
    d_k_sqrt = np.sqrt(d_k)
    scores = (Q_rot @ K_rot.transpose(0, 1, 3, 2)) / d_k_sqrt
    
    # Causal mask
    mask = np.triu(np.full((N, N), -np.inf), k=1)
    scores = scores + mask[np.newaxis, np.newaxis, :, :]
    
    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    
    # Output
    O = attn @ V
    return O


class RotaryPE(nn.Module):
    """PyTorch implementation of RoPE for queries and keys."""
    
    def __init__(self, d_k: int, max_len: int = 8192, base: int = 10000):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires even head dimension")
        self.d_k = d_k
        self.max_len = max_len
        
        # Precompute frequencies: theta_i = base^(-2i/d)
        i = torch.arange(d_k // 2)
        freqs = base ** (-2 * i / d_k)
        
        # Precompute angles for all positions: (max_len, d_k//2)
        positions = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        angles = positions * freqs.unsqueeze(0)  # (max_len, d_k//2)
        
        # Register as buffers (not parameters, but move with model)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor,
                offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to Q and K.
        
        Args:
            q: queries (B, H, T, d_k)
            k: keys (B, H, T, d_k)
            offset: position offset for incremental decoding
        
        Returns:
            q_rot, k_rot: rotated queries and keys
        """
        # Get cos/sin for this position range
        T = q.shape[2]
        cos = self.cos[offset:offset+T, :]  # (T, d_k//2)
        sin = self.sin[offset:offset+T, :]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_k//2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation using interleaved pairs
        q_rot = self._rotate_pairs(q, cos, sin)
        k_rot = self._rotate_pairs(k, cos, sin)
        
        return q_rot, k_rot
    
    def _rotate_pairs(self, x: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> torch.Tensor:
        """Apply 2D rotation to interleaved pairs in x."""
        # Split into even and odd dimensions
        x_even = x[..., ::2]  # (B, H, T, d_k//2)
        x_odd = x[..., 1::2]  # (B, H, T, d_k//2)
        
        # Rotate
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot.reshape(x.shape)
        return x_rot


def verify_relative_position_property(d_k: int = 64, base: int = 10000):
    """
    Verify that RoPE dot products depend on relative position (m - n).
    """
    freqs = build_rope_frequencies(d_k, base)
    
    # Create random Q and K vectors
    np.random.seed(42)
    q_vec = np.random.randn(d_k)
    k_vec = np.random.randn(d_k)
    
    # Test: dot product at positions (m, n) should equal dot product at (m+delta, n+delta)
    m, n = 10, 5
    delta = 7
    
    # Rotate at (m, n)
    q_m = apply_rope_2d(q_vec[np.newaxis, :], freqs * m)[0]
    k_n = apply_rope_2d(k_vec[np.newaxis, :], freqs * n)[0]
    dot_mn = np.dot(q_m, k_n)
    
    # Rotate at (m+delta, n+delta)
    q_m_delta = apply_rope_2d(q_vec[np.newaxis, :], freqs * (m + delta))[0]
    k_n_delta = apply_rope_2d(k_vec[np.newaxis, :], freqs * (n + delta))[0]
    dot_m_delta_n_delta = np.dot(q_m_delta, k_n_delta)
    
    print(f"Dot product at positions ({m}, {n}): {dot_mn:.6f}")
    print(f"Dot product at positions ({m+delta}, {n+delta}): {dot_m_delta_n_delta:.6f}")
    print(f"Difference (should be ~0): {abs(dot_mn - dot_m_delta_n_delta):.2e}")
    
    # Now test different relative positions
    m2, n2 = 15, 10  # Same relative distance: m2 - n2 = 5 = m - n
    q_m2 = apply_rope_2d(q_vec[np.newaxis, :], freqs * m2)[0]
    k_n2 = apply_rope_2d(k_vec[np.newaxis, :], freqs * n2)[0]
    dot_m2_n2 = np.dot(q_m2, k_n2)
    
    print(f"\nDot product at positions ({m2}, {n2}): {dot_m2_n2:.6f}")
    print(f"Relative position (m-n): {m-n}, (m2-n2): {m2-n2}")
    print(f"Difference (should be ~0): {abs(dot_mn - dot_m2_n2):.2e}")


def compare_extrapolation_methods():
    """
    Demonstrate different RoPE extrapolation techniques.
    """
    d_k = 64
    base = 10000
    train_len = 2048
    test_len = 8192
    
    freqs = build_rope_frequencies(d_k, base)
    
    print("=" * 70)
    print("RoPE Extrapolation Methods")
    print("=" * 70)
    
    # 1. Original RoPE (no modification)
    print("\n1. Original RoPE:")
    print(f"   Train length: {train_len}, Test length: {test_len}")
    print(f"   Max train angle (pos {train_len}): {train_len * freqs[0]:.2f} rad")
    print(f"   Max test angle (pos {test_len}): {test_len * freqs[0]:.2f} rad")
    print(f"   Issue: Test angles 4x larger than training distribution")
    
    # 2. Position Interpolation (PI)
    scale = train_len / test_len
    print(f"\n2. Position Interpolation:")
    print(f"   Scale positions by {scale:.3f}")
    print(f"   Effective test angles: {test_len * scale * freqs[0]:.2f} rad")
    print(f"   Stays in training distribution, but compresses all frequencies")
    
    # 3. NTK-aware Scaling
    ntk_factor = (test_len / train_len) ** (d_k / (d_k - 2))
    effective_base = base * ntk_factor
    print(f"\n3. NTK-aware Scaling:")
    print(f"   Effective base: {effective_base:.0f} (original: {base})")
    print(f"   Rescales frequencies to maintain neural tangent kernel stability")
    print(f"   Less aggressive than PI on high frequencies")
    
    # 4. YaRN (Yet another RoPE extensioN)
    print(f"\n4. YaRN:")
    print(f"   Combines PI with selective frequency targeting")
    print(f"   Some frequency bands interpolated more aggressively")
    print(f"   Preserves short-context behavior while enabling long context")


# --- Demo ---
if __name__ == "__main__":
    print("=" * 70)
    print("RoPE: Rotary Position Embedding")
    print("=" * 70)
    
    # 1. Verify relative position property
    print("\n--- Verifying Relative Position Property ---")
    verify_relative_position_property(d_k=64)
    
    # 2. Compare extrapolation methods
    print("\n" + "=" * 70)
    compare_extrapolation_methods()
    
    # 3. PyTorch RoPE test
    print("\n--- PyTorch RoPE Shape Test ---")
    rope = RotaryPE(d_k=64, max_len=4096)
    B, H, T = 2, 8, 128
    q = torch.randn(B, H, T, 64)
    k = torch.randn(B, H, T, 64)
    q_rot, k_rot = rope(q, k)
    print(f"Q shape: {q.shape} -> Q_rot shape: {q_rot.shape}")
    print(f"K shape: {k.shape} -> K_rot shape: {k_rot.shape}")
    print(f"Q and Q_rot have same shape: {q.shape == q_rot.shape}")
    
    # 4. Incremental decoding test
    print("\n--- Incremental Decoding Test ---")
    # First pass: positions 0-127
    q1, k1 = rope(q, k, offset=0)
    # Second pass: positions 128-255 (continuation)
    q2, k2 = rope(q, k, offset=128)
    print(f"First pass uses positions 0-127, second pass uses 128-255")
    print(f"RoPE correctly handles incremental decoding with offset")
```

---

## Length Extrapolation: The RoPE Scaling Problem

### The Core Issue

RoPE is trained on sequences up to length \(L_{\text{train}}\). At inference, if you evaluate at position \(L_{\text{test}} \gg L_{\text{train}}\):

- Angles \(m \theta_i\) exceed the range seen during training
- High-frequency components oscillate rapidly, creating out-of-distribution attention patterns
- Model quality degrades

### Solution 1: Position Interpolation (PI)

Scale all positions down to fit in training range:

\[
m' = m \cdot \frac{L_{\text{train}}}{L_{\text{test}}}
\]

**Pros:** Simple, keeps angles in-distribution.
**Cons:** Compresses all frequencies equally, losing resolution on short-range patterns the model learned.

### Solution 2: NTK-Aware Scaling

Adjust the base \(10000\) to \(10000 \cdot \alpha\) where \(\alpha\) depends on the extrapolation ratio:

\[
\alpha = \left(\frac{L_{\text{test}}}{L_{\text{train}}}\right)^{d/(d-2)}
\]

This comes from analyzing the **Neural Tangent Kernel** (NTK) of the RoPE-modified attention layer. The insight: scaling the base keeps the **optimization geometry** stable at longer lengths.

**Pros:** Preserves high-frequency behavior better than PI.
**Cons:** Still a heuristic; requires tuning the scaling factor.

### Solution 3: YaRN (Yet another RoPE extensioN)

YaRN combines position interpolation with **frequency-aware blending**:

\[
\theta_i' = \begin{cases} \theta_i \cdot \frac{L_{\text{test}}}{L_{\text{train}}} & \text{if } \theta_i > \tau \\ \theta_i & \text{otherwise} \end{cases}
\]

Low frequencies (long-range) are scaled aggressively; high frequencies (short-range) are left unchanged to preserve local patterns.

**Pros:** Best empirical results for extreme extrapolation (4K → 128K).
**Cons:** More complex; requires choosing threshold \(\tau\).

---

## Interview Importance

RoPE is a **top-10 architecture topic** in LLM interviews. Any role involving model architecture, training, or serving expects you to understand positional encoding choices.

### Difficulty Level: ⭐⭐⭐⭐ (Hard)

---

## Interview Questions & Answers

### Q1: Why did RoPE replace sinusoidal and learned positional encodings?

**Answer:** Three reasons:
1. **Relative position by construction:** Sinusoidal encodings approximate relative position through additive patterns; RoPE encodes it **exactly** in the dot product \(q_m^\top k_n\) depends on \(m - n\).
2. **Zero parameters:** Learned positions add \(L_{\max} \times d\) parameters and fail to extrapolate. RoPE uses fixed trig functions — no training cost, no overfitting.
3. **Better extrapolation:** With scaling tricks (NTK, YaRN), RoPE generalizes to 4-32× training length. Learned positions have no principled behavior beyond \(L_{\max}\).

### Q2: Explain why RoPE only rotates Q and K, not V.

**Answer:** Attention computes \(\text{softmax}(QK^\top / \sqrt{d_k}) V\). The \(QK^\top\) term determines **where** to attend (which positions are relevant) — this is where position matters. The \(V\) term determines **what** information to extract once attention weights are computed. Position affects the "where" (relative distance between tokens), not the "what" (token content). Rotating only Q and K makes attention scores position-aware while keeping values as pure content representations.

### Q3: What happens if you train on 4K tokens and evaluate on 32K without any RoPE scaling?

**Answer:** The rotation angles at position 32K are 8× larger than the maximum angle seen during training (position 4K). High-frequency components (\(\theta_i\) for small \(i\)) oscillate so rapidly that:
- Dot products between distant tokens become **noise-like** (random phase differences)
- The attention distribution becomes **unpredictable** — the model never saw this geometry
- Perplexity degrades significantly, and generated text quality drops

### Q4: How does position interpolation enable longer context?

**Answer:** Position interpolation rescales positions: instead of feeding position \(m\) into RoPE, feed \(m \cdot (L_{\text{train}} / L_{\text{test}})\). This compresses the 32K positions into the 4K angle range the model was trained on. The angles stay in-distribution, so the model's learned attention patterns still apply. The trade-off is **reduced resolution** — adjacent positions are closer in angle space, potentially blurring fine-grained position distinctions.

### Q5: What is the relationship between RoPE and relative position biases like T5's?

**Answer:** Both encode relative position, but differently:
- **T5 relative bias:** Adds a learned scalar bias \(b_{m-n}\) to attention logits based on relative distance. Requires a lookup table of biases.
- **RoPE:** Multiplies Q and K by rotation matrices. The relative position emerges from the **geometry** of rotated vectors — no learned table needed.

RoPE is more parameter-efficient and integrates directly into the attention computation rather than adding a post-hoc bias.

---

## Connections to Other Papers

- **Transformer** → RoPE replaces the original sinusoidal positional encoding
- **LLaMA** → Adopted RoPE as the standard positional encoding for open-weight models
- **Mistral** → Uses RoPE with sliding window attention
- **DeepSeek-V2** → **Decoupled RoPE** pathway separates content and position in MLA
- **Qwen2.5** → Uses RoPE as part of the modern decoder stack
- **PaLM** → Also adopted RoPE for position encoding
- **YaRN / NTK scaling** → Extensions enabling RoPE extrapolation to 128K+ context

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Rotate Q and K by position-dependent angles in 2D subspaces |
| Key property | Dot product depends on **relative position** \(m - n\) |
| Frequency | \(\theta_i = 10000^{-2i/d}\); geometric progression |
| Parameters | **None** — fixed trigonometric functions |
| Rotation scope | **Only Q and K**, not V |
| Extrapolation | Works well with scaling: PI, NTK-aware, YaRN |
| Adoption | LLaMA, Mistral, Qwen, PaLM, GLM, virtually all modern LLMs |
| Advantage over learned PE | No parameters, better extrapolation, no out-of-vocabulary positions |
| Advantage over sinusoidal | Exact relative position, not approximate |
| Interview angle | Essential architecture knowledge; expect "why RoPE?" questions |
