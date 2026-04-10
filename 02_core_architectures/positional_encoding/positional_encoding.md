# Positional Encoding

## Why This Matters for LLMs

Attention compares every token to every other token using inner products that are **label-invariant** with respect to position unless you inject positional structure. Without that injection, the sentence “dog bites man” and the sentence “man bites dog” yield the same attention score matrix up to permutation of rows and columns paired with the same permutation of value vectors. Order is the carrier of syntax and semantics in language, so models that cannot represent order cannot model grammar, scope, or discourse structure at the depth required for useful language technologies.

Positional encoding is therefore not an optional garnish: it is the family of methods that breaks permutation symmetry in a controlled way. Different encodings trade off parameter count, length extrapolation, compatibility with relative position reasoning, and implementation complexity. When you read that a model uses **rotary position embedding** or **ALiBi**, you are reading choices about how position interacts with attention scores at inference time and how aggressively the system generalizes beyond training length.

Finally, positional methods interact directly with **long-context scaling**. Techniques such as **NTK-aware scaling** and **YaRN** are not separate from architecture: they modify how frequencies or biases behave as sequence length grows. Interviewers who care about production LLMs will expect you to connect positional encoding with both training stability and serving constraints, because context window upgrades are often achieved by revisiting these mechanisms rather than by changing token embeddings alone.

---

## Why Position Matters: Permutation Equivariance

Let \(X \in \mathbb{R}^{T \times d}\) be a matrix of token vectors. Let \(\Pi\) be a permutation matrix that reorders rows. Self-attention built purely from \(X\) without positional information satisfies:

\[
\text{Attn}(\Pi X) = \Pi \, \text{Attn}(X)
\]

when value projections and softmax are applied consistently with the same permutation on keys and queries.

!!! math-intuition "In Plain English"
    If you shuffle the input rows and shuffle attention outputs with the **same** permutation, you get the same numerical relationships recombined. The model has no canonical notion of “first token” versus “last token” unless you break the symmetry. Adding positional information breaks that equivalence in the way you choose: absolute indices in learned embeddings, sinusoidal signals, rotations in query and key space, or biases on attention logits.

---

## Sinusoidal Positional Encoding

The original Transformer uses fixed sinusoids of varying frequency added to token embeddings:

\[
\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

\[
\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

for dimension indices \(2i\) and \(2i+1\) paired from \(i = 0, 1, \ldots, d_{\text{model}}/2 - 1\).

!!! math-intuition "In Plain English"
    Each pair of dimensions behaves like a clock hand: low \(i\) uses a slow angular progression as position increases, encoding coarse location. Higher \(i\) uses faster oscillations, encoding finer variation. The division by \(10000^{2i/d_{\text{model}}}\) sets a geometric spacing of wavelengths across the spectrum of dimensions. You add these vectors to embeddings so that every position carries a unique fingerprint without introducing new learnable parameters.

!!! example "Worked Example: Sinusoidal PE at Position 3 with \(d_{\text{model}} = 8\)"
    Here \(d_{\text{model}} = 8\), so \(i \in \{0,1,2,3\}\) pairs dimensions \((0,1)\), \((2,3)\), \((4,5)\), \((6,7)\). The denominator for pair index \(i\) is \(10000^{2i/d_{\text{model}}}\). The angle for both \(\sin\) and \(\cos\) in that pair is \(\text{pos}\) divided by that denominator.

    - \(i=0\): exponent \(0\), denominator \(1\), angle \(3\) radians. \(\sin(3) \approx 0.141120008\), \(\cos(3) \approx -0.989992497\).
    - \(i=1\): exponent \(2/8 = 0.25\), denominator \(10000^{0.25} = 10\), angle \(3/10 = 0.3\) radians. \(\sin(0.3) \approx 0.295520207\), \(\cos(0.3) \approx 0.955336489\).
    - \(i=2\): exponent \(4/8 = 0.5\), denominator \(100\), angle \(3/100 = 0.03\) radians. \(\sin(0.03) \approx 0.029995500\), \(\cos(0.03) \approx 0.999550034\).
    - \(i=3\): exponent \(6/8 = 0.75\), denominator \(10000^{0.75} \approx 1000\), angle \(3/1000 = 0.003\) radians. \(\sin(0.003) \approx 0.002999996\), \(\cos(0.003) \approx 0.999995500\).

    The full eight-dimensional vector for position \(3\) is:

    \[
    [0.141120008,\ -0.989992497,\ 0.295520207,\ 0.955336489,\ 0.029995500,\ 0.999550034,\ 0.002999996,\ 0.999995500]
    \]

    Display rounding is for readability; implementations keep full floating-point precision end to end.

---

## Learned Positional Embeddings

Learned positional embeddings allocate a table \(P \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}\) and add \(P[pos]\) to token embeddings:

\[
\mathbf{x}_i = \text{Embed}(w_i) + P[i]
\]

!!! math-intuition "In Plain English"
    The model learns any pattern of position usage that helps the objective. Flexibility is high: the representation can encode absolute location, bucketed ranges, or task-specific biases. The downside is a **hard ceiling** at \(L_{\max}\): indices beyond the trained maximum have no row unless you interpolate, extrapolate, or allocate a larger table and continue training.

**When learned embeddings tend to help:** shorter contexts where data are abundant relative to maximum length, and tasks where flexible positional priors beat hand-crafted waves. **When they tend to hurt extrapolation:** you need zero-shot longer contexts without additional adaptation, because unseen rows were never optimized.

---

## Rotary Position Embedding (RoPE)

RoPE applies position-dependent rotations to queries and keys in two-dimensional subspaces. For a frequency index \(i\) and position \(m\), the \(2 \times 2\) rotation block uses angle \(m \theta_i\) with \(\theta_i = 10000^{-2i/d}\) in the usual formulation.

\[
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\]

Keys receive the same structure with their own position index \(n\).

!!! math-intuition "In Plain English"
    Rotating query and key vectors mixes their two coordinates in a way that depends on absolute position. When you take dot products across positions, the relative displacement \(m - n\) governs the effective angle between rotated vectors, which is why RoPE encodes **relative** relationships while being applied as a fixed function of absolute indices. Values are typically **not** rotated; only queries and keys participate in the attention score.

!!! example "Worked Example: RoPE in Two Dimensions with Position 5"
    Take \(d_k = 2\) for a single head pair so one rotation angle suffices. Let \(\theta = 1\) radian for a toy illustration (real models use \(\theta_i = 10000^{-2i/d}\)). Position \(m = 5\) uses

    \[
    R(5) =
    \begin{pmatrix}
    \cos(5) & -\sin(5) \\
    \sin(5) & \cos(5)
    \end{pmatrix}
    \approx
    \begin{pmatrix}
    0.283662185 & 0.958924274 \\
    -0.958924274 & 0.283662185
    \end{pmatrix}
    \]

    Let an unrotated query vector be \(\mathbf{q}_{\text{raw}} = [1, 0]^\top\). Then

    \[
    \mathbf{q}^{(5)} = R(5)\, \mathbf{q}_{\text{raw}} \approx [0.283662185,\ -0.958924274]^\top
    \]

    Let an unrotated key vector at another position be \(\mathbf{k}_{\text{raw}} = [0, 1]^\top\). For position \(n = 2\),

    \[
    R(2) \approx
    \begin{pmatrix}
    -0.416146837 & -0.909297427 \\
    0.909297427 & -0.416146837
    \end{pmatrix}
    \]

    \[
    \mathbf{k}^{(2)} = R(2)\, \mathbf{k}_{\text{raw}} \approx [-0.909297427,\ -0.416146837]^\top
    \]

    The attention score contribution from this pair before scaling is \(\mathbf{q}^{(5)} \cdot \mathbf{k}^{(2)} \approx -0.2579\). The crucial interview point is not the numeric value but the mechanism: **both** vectors are rotated by angles tied to their positions, and the resulting dot product depends on the **difference** of angles, linking \(m - n\) to a consistent relative phase.

---

## ALiBi (Attention with Linear Biases)

ALiBi subtracts a head-specific linear penalty proportional to distance between query and key positions:

\[
\text{score}_{i,j} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} - m_h \cdot |i - j|
\]

where \(m_h > 0\) is a slope associated with attention head \(h\).

!!! math-intuition "In Plain English"
    Closer positions keep more of the raw dot product; distant positions pay a larger penalty that grows linearly with separation. No extra parameters are stored for positions themselves: distance is the only input to the bias. The inductive bias favors local context while still allowing heads with smaller slopes to attend farther away.

!!! example "Worked Example: ALiBi Bias Matrix for Four Tokens (Single Head)"
    Ignore the dot-product term and examine only the bias \(-m |i - j|\) for \(T = 4\) with slope \(m = 0.5\). Distances \(|i - j|\) form

    \[
    D =
    \begin{pmatrix}
    0 & 1 & 2 & 3 \\
    1 & 0 & 1 & 2 \\
    2 & 1 & 0 & 1 \\
    3 & 2 & 1 & 0
    \end{pmatrix}
    \]

    Multiply by \(-0.5\):

    \[
    B =
    \begin{pmatrix}
    0 & -0.5 & -1.0 & -1.5 \\
    -0.5 & 0 & -0.5 & -1.0 \\
    -1.0 & -0.5 & 0 & -0.5 \\
    -1.5 & -1.0 & -0.5 & 0
    \end{pmatrix}
    \]

    Each row \(i\) shows how key positions \(j\) are penalized when producing scores from query \(i\). The penalty is **linear** in distance with slope \(-m\). Multi-head ALiBi assigns different positive slopes \(m_h\) per head following a geometric schedule so some heads remain sharply local while others spread attention more widely.

---

## Extrapolation Comparison

| Method | Train-time parameters for position | Extrapolation to longer sequences | Typical application point |
| --- | --- | --- | --- |
| Sinusoidal additive | None | Mixed in practice; not a panacea | Original Transformer |
| Learned absolute | \(L_{\max} \times d\) | Poor without retraining or interpolation | BERT, GPT-2 |
| RoPE | None (fixed trig) | Good; further improved by scaling laws | LLaMA, Mistral, Qwen |
| ALiBi | None (fixed slopes) | Strong linear bias inductive prior | BLOOM, MPT |

Extrapolation quality always depends on fine-tuning data and attention sparsity patterns; the table captures **inductive biases**, not guarantees.

---

## NTK-Aware Scaling and YaRN (Brief)

**NTK-aware scaling** adjusts the effective base of geometric progressions used in RoPE (or related frequency parameterizations) so that the neural tangent kernel perspective remains stable when context length increases. Practically, researchers rescale frequencies so that **longer** sequences do not force attention logits into regimes where relative phase changes too abruptly at large positions.

**YaRN** (Yet another RoPE extensioN) combines **position interpolation** with selective frequency targeting: some bands of frequencies are interpolated more aggressively than others, and a blending strategy preserves behavior on shorter contexts while enabling longer ones. The method acknowledges that not all frequency components extrapolate equally.

!!! math-intuition "In Plain English"
    Both families of techniques recognize that RoPE’s rotations spin faster in some dimensions than others. If you train on 4096 tokens and suddenly evaluate at 131072 tokens, raw angles can leave the regime the optimizer saw. NTK-inspired rescaling tries to keep optimization geometry coherent; YaRN-style methods **reshape** how aggressively different frequencies are stretched when you extend the window.

---

??? deep-dive "Deep Dive: Where to Apply RoPE Versus Additive PE"
    Additive encodings modify the residual stream before attention. RoPE modifies **queries and keys inside attention** after projections. That placement matters for how gradients flow and for how weight tying interacts with positional structure. In interviews, stating “RoPE acts on Q and K, not on V” is a crisp differentiator.

---

??? deep-dive "Deep Dive: Relative Position Without Relying on Table Lookup"
    RoPE and ALiBi both avoid O(L) embedding rows. RoPE bakes relative structure into rotations of attention arguments; ALiBi bakes locality into logits. Learned absolute embeddings instead memorize positions as rows, which is powerful within range but brittle outside it unless augmented.

---

## Code

The listing implements **sinusoidal**, **learned**, **RoPE**, and **ALiBi**. RoPE uses interleaved pairs within each head dimension, matching the common open-weight layout. The `if __name__` block runs quick shape checks and a sinusoidal numerical consistency check.

```python
"""
Positional encoding schemes: sinusoidal, learned, RoPE (interleaved pairs), and ALiBi.
Dependencies: PyTorch only.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Vaswani-style sinusoids added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        return x + self.pe[:, :t, :]


class LearnedPE(nn.Module):
    """Learned absolute positions added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        idx = torch.arange(t, device=x.device)
        return x + self.pos_emb(idx)


class RotaryPE(nn.Module):
    """
    RoPE on queries and keys: (B, H, T, d_k). Interleaved pair rotation.
    """

    def __init__(self, d_k: int, max_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE expects even d_k")
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        t = torch.arange(max_len).float()
        angles = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", angles.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = q.size(2)
        cos = self.cos_cached[:, :, :seq, :]
        sin = self.sin_cached[:, :, :seq, :]
        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)

        def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
            return x * cos + rotated * sin

        return rotate_interleaved(q), rotate_interleaved(k)


class ALiBiBias(nn.Module):
    """Per-head linear bias on attention logits: -slope_h * |i - j|."""

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        ratio = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
        slopes = torch.tensor([ratio ** i for i in range(1, n_heads + 1)], dtype=torch.float32)
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))

    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float32, device=self.slopes.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return -self.slopes * dist.unsqueeze(0).unsqueeze(0)


def apply_alibi_to_scores(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """scores (B, H, T, T) plus bias (1, H, T, T)."""
    return scores + bias


if __name__ == "__main__":
    torch.manual_seed(0)
    b, t, d_model, n_heads = 2, 16, 64, 4
    d_k = d_model // n_heads
    x = torch.randn(b, t, d_model)

    spe = SinusoidalPE(d_model, max_len=128)
    y1 = spe(x)
    assert y1.shape == x.shape

    lpe = LearnedPE(d_model, max_len=128)
    y2 = lpe(x)
    assert y2.shape == x.shape

    rope = RotaryPE(d_k, max_len=128)
    q = torch.randn(b, n_heads, t, d_k)
    k = torch.randn(b, n_heads, t, d_k)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape

    alibi = ALiBiBias(n_heads)
    bias = alibi(t)
    assert bias.shape == (1, n_heads, t, t)
    scores = torch.randn(b, n_heads, t, t) / math.sqrt(d_k)
    scores2 = apply_alibi_to_scores(scores, bias)
    assert scores2.shape == scores.shape

    d8 = 8
    pe8 = SinusoidalPE(d8, max_len=16)
    row = pe8.pe[0, 3].tolist()
    div_term = torch.exp(torch.arange(0, d8, 2).float() * (-math.log(10000.0) / d8))
    pos = torch.tensor([[3.0]])
    manual = torch.zeros(d8)
    manual[0::2] = torch.sin(pos * div_term)
    manual[1::2] = torch.cos(pos * div_term)
    max_err = max(abs(row[i] - manual[i].item()) for i in range(d8))
    print("Sinusoidal row max error:", max_err)
    print("All positional modules ran: shapes consistent.")
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Explain permutation equivariance of attention and how positional encodings break it.
    *Answer:* Without position, if you permute input rows and apply the same permutation to outputs, attention computes the same pairwise relationships—order is invisible. **Additive** or **multiplicative** position info (sinusoidal, learned, RoPE on Q/K) makes each index carry a distinct vector or rotation so “first token” is not equivalent to “last token.”
    2. Write the sinusoidal formula and interpret wavelength progression across dimensions.
    *Answer:* \(\mathrm{PE}(pos,2i)=\sin(pos/10000^{2i/d})\), \(\mathrm{PE}(pos,2i+1)=\cos(\cdots)\). Small \(i\) uses **long** wavelengths (slow change with \(pos\), coarse position); large \(i\) uses **short** wavelengths (fine-grained distinctions)—a multi-scale code over the same \(d_{\text{model}}\) channels.
    3. Contrast learned absolute positions with RoPE regarding length extrapolation.
    *Answer:* Learned rows \(P[0..L_{\max}-1]\) have **no** parameters for \(pos \ge L_{\max}\) unless you interpolate or continue training. RoPE uses fixed functions of \((m,n)\) through rotations on Q/K, so relative geometry often **generalizes** better to longer sequences (especially with scaling tricks like NTK-aware or YaRN).
    4. Show how RoPE relates dot products to relative offsets \(m - n\) at a conceptual level.
    *Answer:* Rotating \(q\) by angle \(m\theta\) and \(k\) by \(n\theta\) in each 2D subspace turns \(\langle R_m q, R_n k\rangle\) into a function where **relative** phase \(m-n\) appears in the inner product—so attention scores depend on distance \(m-n\), not only absolute \(m\).
    5. Write ALiBi’s bias and explain the linear distance penalty intuition.
    *Answer:* \(\mathrm{score}_{ij} = (q_i\cdot k_j)/\sqrt{d_k} - m_h |i-j|\) with slope \(m_h>0\) per head. Farther keys pay a larger linear penalty, biasing heads toward **local** context while still allowing small-slope heads to attend broadly.
    6. Describe where sinusoidal and learned encodings attach versus where RoPE attaches.
    *Answer:* Sinusoidal and learned PE are **added to embeddings** at the input to the stack (residual stream carries position from the start). **RoPE** applies **after** Q/K projections inside attention—it rotates query and key vectors per position, not the value stream.
    7. Explain why values are typically not rotated in RoPE implementations.
    *Answer:* Attention scores need position only in \(q_i^\top k_j\); rotating both \(q\) and \(k\) encodes relative position there. **Values** are mixed **after** weights are fixed; rotating \(V\) would not change the softmax weights and would complicate the semantics of “what content is blended” without helping the positional term.
    8. Name two practical methods for extending RoPE-trained models beyond initial context.
    *Answer:* **Position interpolation** (compress positions into the trained range—e.g. scale positions down), and **NTK-aware** or **YaRN**-style frequency rescaling that adjusts RoPE base/wavelength schedules so large indices stay in-distribution.
    9. Give one strength of ALiBi in extrapolation and one reason modern stacks might still pick RoPE.
    *Answer:* ALiBi adds only a **distance-based** logit bias—no learned position table—often **train short, test long** friendly. RoPE still dominates in open LMs because it pairs well with **GQA**, fused kernels, and extensive **ecosystem** tuning (scaling laws, long-context recipes).
    10. Compute a single sinusoidal pair for a toy \(pos\) and \(i\) to demonstrate you can turn formulas into numbers.
    *Answer:* Example: \(d_{\text{model}}=8\), \(i=1\): divisor \(=10000^{2/8}=10\), angle \(=pos/10\). For \(pos=3\): \(\sin(0.3)\approx 0.296\), \(\cos(0.3)\approx 0.955\)—one pair of PE dimensions at that position; other \(i\) repeat with different divisors (1, 100, 1000…).

!!! interview "Follow-up Probes"
    - “What breaks if you double the sequence length without changing any positional hyperparameters?” Expect discussion of unseen learned rows or out-of-distribution rotations.
    - “How does ALiBi interact with causal masking?” Expect additive combination on logits before softmax.
    - “Why might NTK-aware scaling help RoPE models?” Expect frequency stability and optimization geometry references.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Attention is permutation-equivariant without positional structure.”
    - “RoPE injects relative position by rotating queries and keys in paired subspaces.”
    - “ALiBi biases attention logits linearly with token distance per head.”
    - “Learned embeddings are flexible within \(L_{\max}\) but do not generalize length for free.”

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — sinusoidal positional encodings
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Press et al., *Train Short, Test Long: Attention with Linear Biases* (2022)
- Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023)
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023)
