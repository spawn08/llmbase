# 2.3 — Positional Encoding

## Intuition

Self-attention is **permutation-equivariant**: if you shuffle the input tokens, the attention weights rearrange accordingly but the function doesn't "know" which token was first. Without positional information, "dog bites man" and "man bites dog" produce identical representations. **Positional encodings** inject order into the model. The choice of encoding affects context length, extrapolation, and computational cost — making this one of the most actively researched components.

---

## Core concepts

### 1. Sinusoidal positional encoding (Vaswani et al., 2017)

The original Transformer adds fixed sinusoidal signals to embeddings:

\[
\text{PE}(pos, 2i) = \sin\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr), \quad
\text{PE}(pos, 2i+1) = \cos\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr)
\]

Each dimension oscillates at a different frequency — low dimensions change slowly (capturing global position), high dimensions change rapidly (capturing local position).

**Properties:**

- Deterministic — no learned parameters.
- Relative positions can be expressed as linear transformations: \(\text{PE}(pos + k)\) is a linear function of \(\text{PE}(pos)\).
- **Does not extrapolate** well beyond training lengths in practice.

### 2. Learned positional embeddings

BERT and GPT-2 replace sinusoidal functions with a **learned embedding table** \(P \in \mathbb{R}^{L_{\max} \times d}\):

\[
\mathbf{x}_i = \text{TokenEmbed}(w_i) + P[i]
\]

**Properties:**

- More flexible — the model can learn any positional pattern.
- **Hard length limit** — cannot process sequences longer than \(L_{\max}\) without re-training or interpolation.
- Used in: GPT-2, BERT, ViT.

### 3. Rotary Position Embedding (RoPE) — Su et al., 2021

RoPE encodes position by **rotating** query and key vectors in 2D subspaces. For dimension pairs \((2i, 2i+1)\):

\[
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\]

where \(m\) is the position index and \(\theta_i = 10000^{-2i/d}\).

**Key insight:** After rotation, the dot product \(\mathbf{q}_m \cdot \mathbf{k}_n\) depends only on the **relative** position \(m - n\), not the absolute positions. This gives:

- **Relative position encoding** without extra parameters.
- Applied to Q and K only (not V, not embeddings).
- **Extrapolation** — better than learned or sinusoidal, and can be extended with techniques like **NTK-aware scaling** or **YaRN** for longer contexts.
- Used in: LLaMA, Mistral, GPT-NeoX, PaLM, Qwen.

### 4. ALiBi (Attention with Linear Biases) — Press et al., 2022

ALiBi adds a **static linear bias** to attention scores based on the distance between query and key positions:

\[
\text{score}_{i,j} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m \cdot |i - j|
\]

where \(m\) is a head-specific slope (set geometrically, not learned). Closer tokens get higher scores; distant tokens are penalized linearly.

**Properties:**

- No parameters — the bias is deterministic.
- **Excellent extrapolation** — trains on 1K tokens, works on 2K+ without fine-tuning.
- Simple to implement — just add a bias matrix to attention scores.
- Used in: BLOOM, MPT.

### Comparison

| Method | Type | Extrapolation | Where applied | Used by |
| --- | --- | --- | --- | --- |
| Sinusoidal | Absolute, fixed | Poor | Added to embeddings | Original Transformer |
| Learned | Absolute, learned | None (hard limit) | Added to embeddings | BERT, GPT-2 |
| RoPE | Relative, fixed | Good (with scaling) | Applied to Q, K | LLaMA, Mistral, Qwen |
| ALiBi | Relative, fixed | Excellent | Bias on attention scores | BLOOM, MPT |

---

## Code — All four positional encodings implemented

```python
"""
Four positional encoding schemes in PyTorch:
sinusoidal, learned, RoPE, and ALiBi.
Includes visualization of encoding patterns.
"""
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# ── 1. Sinusoidal ──────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ── 2. Learned ─────────────────────────────────────────────────────────
class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_emb(positions)


# ── 3. RoPE ────────────────────────────────────────────────────────────
class RotaryPE(nn.Module):
    def __init__(self, d_k: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        t = torch.arange(max_len).float()
        angles = torch.outer(t, freqs)  # (max_len, d_k/2)
        self.register_buffer("cos_cached", angles.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """Apply rotary embedding to q and k. Shape: (B, H, T, d_k)."""
        T = q.size(2)
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
            return x * cos.repeat(1, 1, 1, 2) + rotated * sin.repeat(1, 1, 1, 2)

        return rotate(q), rotate(k)


# ── 4. ALiBi ───────────────────────────────────────────────────────────
class ALiBi(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        ratio = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
        slopes = torch.tensor(
            [ratio ** i for i in range(1, n_heads + 1)], dtype=torch.float32
        )
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))

    def forward(self, T: int) -> torch.Tensor:
        """Return bias matrix of shape (1, n_heads, T, T)."""
        positions = torch.arange(T, dtype=torch.float32)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        return -self.slopes * distance.unsqueeze(0).unsqueeze(0)


# ── Visualization ──────────────────────────────────────────────────────
def plot_encodings() -> None:
    d, T = 64, 128

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sinusoidal
    spe = SinusoidalPE(d, T)
    ax = axes[0, 0]
    ax.imshow(spe.pe[0, :T, :d].numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_title("Sinusoidal PE")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Position")

    # Learned (random init for visualization)
    lpe = LearnedPE(d, T)
    ax = axes[0, 1]
    ax.imshow(lpe.pos_emb.weight.detach()[:T, :d].numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_title("Learned PE (random init)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Position")

    # RoPE frequency pattern
    rope = RotaryPE(d, T)
    ax = axes[1, 0]
    ax.imshow(rope.cos_cached[0, 0, :T].numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_title("RoPE cos(mθ) pattern")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Position")

    # ALiBi bias for 4 heads
    alibi = ALiBi(4)
    bias = alibi(T)
    ax = axes[1, 1]
    ax.imshow(bias[0, 0, :T, :T].numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_title("ALiBi bias (head 0)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    plt.suptitle("Positional Encoding Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("positional_encodings.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D, H = 1, 16, 64, 4

    x = torch.randn(B, T, D)

    # Sinusoidal
    spe = SinusoidalPE(D)
    print(f"Sinusoidal: input {x.shape} → output {spe(x).shape}")

    # Learned
    lpe = LearnedPE(D, max_len=512)
    print(f"Learned:    input {x.shape} → output {lpe(x).shape}")

    # RoPE — applied to Q and K after projection
    rope = RotaryPE(D // H)
    q = torch.randn(B, H, T, D // H)
    k = torch.randn(B, H, T, D // H)
    q_rot, k_rot = rope(q, k)
    print(f"RoPE:       Q {q.shape} → Q_rot {q_rot.shape}")

    # Verify RoPE relative property
    dot_original = (q[:, :, 3:4] * k[:, :, 1:2]).sum(-1)
    dot_rotated = (q_rot[:, :, 3:4] * k_rot[:, :, 1:2]).sum(-1)
    print(f"  Dot (original): {dot_original.item():.4f}")
    print(f"  Dot (rotated):  {dot_rotated.item():.4f}  (different — position-dependent)")

    # ALiBi
    alibi = ALiBi(H)
    bias = alibi(T)
    print(f"ALiBi:      bias shape {bias.shape}, range [{bias.min():.1f}, {bias.max():.1f}]")

    plot_encodings()
    print("Saved: positional_encodings.png")
```

---

## Interview takeaways

1. **Why needed** — self-attention is permutation-equivariant. Without positional info, word order is lost entirely. Be ready to explain this clearly.
2. **Sinusoidal vs. learned** — sinusoidal is deterministic and compact; learned is more flexible but has a hard max length. Most modern models use neither (they use RoPE).
3. **RoPE** — the dominant choice in 2024+ open models. Know the key property: the Q·K dot product depends on **relative** position \(m - n\). This comes from the rotation matrix algebra.
4. **RoPE scaling for long context** — NTK-aware scaling (reduce the base frequency), YaRN, and Position Interpolation allow extending trained context windows. LLaMA went from 2K to 128K with these techniques.
5. **ALiBi** — simplest approach: just subtract \(m \cdot |i - j|\) from attention scores. Excellent extrapolation but less common in latest models. Know it was used in BLOOM.
6. **Where applied** — sinusoidal/learned: added to token embeddings. RoPE: applied to Q and K inside attention. ALiBi: added as a bias to attention scores. The "where" matters architecturally.

---

## References

- Vaswani et al. (2017), *Attention Is All You Need* — Section 3.5 (sinusoidal)
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Press et al. (2022), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization*
- Chen et al. (2023), *Extending Context Window of Large Language Models via Positional Interpolation*
