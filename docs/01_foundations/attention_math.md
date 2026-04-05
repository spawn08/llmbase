# Mathematics of Attention

## Why This Matters for LLMs

Every decoder-only LLM (GPT-class), encoder-only model (BERT-class), and encoder–decoder (T5, translation) **is** attention stacks plus MLPs plus norms. **Scaled dot-product attention** \(\text{softmax}(QK^\top/\sqrt{d_k})V\) is the **atomic** operation interviewers whiteboard first. Understanding **scaling**, **masks**, **multi-head** splitting, and **\(O(T^2)\)** cost is table stakes for systems roles (KV cache, FlashAttention) and research roles (linear attention, state-space layers). This page ties **Bahdanau → dot product → scaled softmax → Transformer** into one quantitative thread.

---

## Core Concepts

### From Bahdanau to Dot-Product

Bahdanau (additive) attention:

\[
e_{t,j} = \mathbf{v}^\top \tanh(W_1 \mathbf{h}_j^{\text{enc}} + W_2 \mathbf{s}_t)
\]

**Dot-product** attention (Luong / Transformer style) replaces the small MLP score with **compatibility** between **query** and **key** vectors:

\[
e_{i,j} = \mathbf{q}_i^\top \mathbf{k}_j
\]

!!! math-intuition "In Plain English"
    - **Query** \(\mathbf{q}_i\): “what I am looking for at position \(i\).”
    - **Key** \(\mathbf{k}_j\): “what is offered at position \(j\).”
    - **Dot product**: similarity if vectors are unit-norm—large positive → **align**; negative → anti-align.

### Scaled Dot-Product Attention

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr) V
\]

Shapes: \(Q \in \mathbb{R}^{T \times d_k}\), \(K \in \mathbb{R}^{T \times d_k}\), \(V \in \mathbb{R}^{T \times d_v}\). Output: \(\mathbb{R}^{T \times d_v}\).

!!! math-intuition "In Plain English"
    - \(QK^\top\): **all-pairs** similarity scores between queries (rows of \(Q\)) and keys (rows of \(K\)).
    - **Softmax** turns each query row into **weights** over positions.
    - **Multiply by \(V\)**: **blend** value vectors—**differentiable** weighted sum.

### Why \(\sqrt{d_k}\)? — Numerical Stabilization

If components of \(\mathbf{q}, \mathbf{k}\) are i.i.d. with variance 1 and mean 0, then

\[
\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = 0,\quad \mathrm{Var}(\mathbf{q}^\top \mathbf{k}) = d_k
\]

Thus dot products **grow** like \(\sqrt{d_k}\) in typical magnitude. **Softmax** of huge logits \(\to\) **nearly one-hot** \(\to\) **vanishing gradients** through other positions. Dividing by \(\sqrt{d_k}\) **re-scales** logits to \(\mathrm{Var} \approx 1\).

!!! example "Numerical Demo: \(d_k = 512\)"
    Rough i.i.d. heuristic: **unscaled** dot \(\mathbf{q}^\top\mathbf{k}\) has **standard deviation** \(\approx \sqrt{d_k} = \sqrt{512} \approx 22.6\). Softmax on five logits around **22** vs **0** is effectively **argmax**—gradients through non-argmax positions \(\approx 0\).

    **Scaled** logits use \(\mathbf{q}^\top\mathbf{k}/\sqrt{512}\): typical std \(\approx 22.6 / 22.6 = 1\). Softmax is **smoother**; training signal reaches multiple positions.

    (Real networks learn non-i.i.d. statistics; **learned** LayerNorm and projections matter—but the **variance argument** is the textbook reason for the scale.)

### Softmax and Temperature

\[
\text{softmax}(z_i; \tau) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
\]

!!! math-intuition "In Plain English"
    - \(\tau \to 0^+\): **sharper** distribution (approaches one-hot).
    - \(\tau \to \infty\): **uniform** mixing.
    - **Decoding temperature** in LMs applies the same idea to **next-token** logits—not identical to attention temperature, but the **same function family**.

---

## Worked Example: Four Tokens, \(d_k = 4\)

**Tokens:** `["The", "cat", "sat", "."]` — indices \(0..3\).

### 1. Toy \(Q\), \(K\), \(V\) (each \(4 \times 4\))

Use **simple integers** (pedagogical, not trained weights):

\[
Q = 2 I_4 = \begin{bmatrix}
2&0&0&0\\0&2&0&0\\0&0&2&0\\0&0&0&2
\end{bmatrix}
\]

\[
K = \begin{bmatrix}
1&1&0&0\\
1&0&1&0\\
0&1&1&0\\
0&0&1&1
\end{bmatrix},
\quad
V = \begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}
\]

(Here \(V=I\) so **output row** is literally the softmax **weight vector** over positions—easy to read.)

### 2. Compute \(S = Q K^\top\) (every cell)

\(K^\top\) is:

\[
K^\top = \begin{bmatrix}
1&1&0&0\\
1&0&1&0\\
0&1&1&0\\
0&0&1&1
\end{bmatrix}
\]

Since \(Q = 2I\), **\(S = Q K^\top = 2 K^\top\)**:

\[
S = \begin{bmatrix}
2&2&0&0\\
2&0&2&0\\
0&2&2&0\\
0&0&2&2
\end{bmatrix}
\]

**Cell check (row 0 · col 2):** Row0 of \(Q\) is \([2,0,0,0]\), col2 of \(K^\top\) is \([0,1,1,0]^\top\), dot \(=0\). Matches \(S_{0,2}=0\).

### 3. Scale: \(S / \sqrt{d_k} = S / 2\)

\[
\frac{S}{2} = \begin{bmatrix}
1&1&0&0\\
1&0&1&0\\
0&1&1&0\\
0&0&1&1
\end{bmatrix}
\]

### 4. Softmax **row-wise** — show **row 0** fully

Row 0 logits: \([1, 1, 0, 0]\).

\[
e^1 \approx 2.718,\quad e^0 = 1
\]

Numerators: \([2.718,\, 2.718,\, 1,\, 1]\). Sum \(\approx 7.436\).

\[
w_{0,0} \approx 2.718/7.436 \approx 0.366,\quad
w_{0,1} \approx 0.366,\quad
w_{0,2} \approx 0.134,\quad
w_{0,3} \approx 0.134
\]

(Weights sum to 1; positions 0 and 1 tie for **highest** mass because logits tied at 1.)

### 5. Output row 0: \(w_0 V\)

Because \(V=I\), **output row 0** \(\approx [0.366,\, 0.366,\, 0.134,\, 0.134]\)—the **context vector** for token “The” is a **blend** of positional value vectors with those weights.

!!! math-intuition "In Plain English"
    - Row \(i\) of the attention output is: “**re-read** all positions, mixing their **values** by how well **keys** match **query** \(i\).”
    - With \(V=I\), you literally see the **attention distribution** as the row vector.

---

## Causal (Autoregressive) Masking

For **decoder** self-attention, position \(i\) must **not** depend on \(j > i\). Take the **scaled** score matrix \(Z = S/\sqrt{d_k}\). **Causal mask** sets \(Z_{i,j} = -\infty\) for \(j > i\) **before** softmax.

!!! example "Mask Walkthrough (same \(Z\) as above)"
    For **row 3** (token “.”), without mask, logits were \([0,0,1,1]\). With **causal** constraint, positions \(j>3\) do not exist; row 3 only has \(j \le 3\). For **row 0**, mask out \(j>0\): keep only column 0 → softmax over a **single** finite logit → weight \(1\) on self (often combined with **causal** + **additive** pos encodings in real models).

    **Typical 4×4 causal \(Z'\)** (set upper triangle to \(-\infty\); shown symbolically):

    \[
    Z'_{i,j} = \begin{cases}
    Z_{i,j} & j \le i \\
    -\infty & j > i
    \end{cases}
    \]

    After softmax, **masked** positions have **weight 0**—no information flows from the future.

!!! math-intuition "In Plain English"
    - **\(-\infty\)** + softmax = **0** probability—clean masking without “almost zero” numerical hacks (implementation uses large negative floats).

### Multi-Head Intuition

\[
\text{head}_h = \text{softmax}\!\Bigl(\frac{Q_h K_h^\top}{\sqrt{d_k}}\Bigr) V_h, \quad Q_h = X W_h^Q,\ \ldots
\]

!!! math-intuition "In Plain English"
    - Each **head** projects into a **subspace** where a different similarity makes sense.
    - **Possible specialization (story, not guaranteed):** Head A attends to **local** neighbors (syntax / n-grams); Head B attends to **distant** coreferent mentions. In practice, heads are **mixed**, but **multi-head** increases **capacity** vs. one attention pool.

---

## Complexity Analysis

For sequence length \(T\), head dimension \(d_k\), value dimension \(d_v\), **one** attention layer (single head):

- **Form \(QK^\top\):** \(O(T^2 d_k)\) multiply–accumulates (each of \(T^2\) scores needs \(d_k\) ops).
- **Softmax:** \(O(T^2)\).
- **Multiply by \(V\):** \(O(T^2 d_v)\).

**Dominant** term is often \(O(T^2 \cdot \max(d_k, d_v))\). **Memory** for full scores: \(O(T^2)\)—the **KV cache** and **FlashAttention** story for long contexts.

!!! example "Plug in Numbers: \(T=2048\), \(d_k=64\)"
    - Rough MACs for \(QK^\top\): \(T^2 d_k = 2048^2 \times 64 = 4{,}194{,}304 \times 64 \approx 2.68 \times 10^8\) MACs **per head per layer** (order-of-magnitude; constants omitted).
    - This quadratic **\(T^2\)** term is why **long-context** inference stresses **memory bandwidth** and why **subquadratic** alternatives (linear attention, state-space models, sliding windows) matter.

---

## Code (existing implementation, with inline comments)

```python
"""
Scaled Dot-Product Attention from scratch in PyTorch.
Includes: basic attention, causal masking, multi-head attention,
and an attention weight heatmap.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        Q: (B, ..., n, d_k)
        K: (B, ..., m, d_k)
        V: (B, ..., m, d_v)
        mask: broadcastable to (B, ..., n, m), True = keep, False = mask out
    Returns:
        output: (B, ..., n, d_v)
        weights: (B, ..., n, m)
    """
    d_k = Q.size(-1)
    # Raw affinities: each query row vs all key rows
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,...,n,m)

    if mask is not None:
        # Set masked positions to -inf so softmax -> 0 there
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)  # (B,...,n,m) — convex combo per query row
    output = torch.matmul(weights, V)    # (B,...,n,d_v)
    return output, weights


def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular boolean mask: position i can attend to j <= i."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V projections."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n, _ = Q.shape
        m = K.size(1)

        # Split last dim into heads: (B, n, H, d_k) -> (B, H, n, d_k) for batched matmuls
        q = self.W_q(Q).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, m, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, m, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, m)

        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, n, -1)
        return self.W_o(out), weights


def plot_attention(weights: torch.Tensor, tokens: list[str]) -> None:
    """Visualize a single-head attention weight matrix as a heatmap."""
    w = weights.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title("Attention Weights")
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
    print("Saved: attention_heatmap.png")


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    T, D = len(tokens), 16
    N_HEADS = 2

    x = torch.randn(1, T, D)

    # 1) Basic self-attention (no mask)
    out, w = scaled_dot_product_attention(x, x, x)
    print(f"Self-attention output shape: {out.shape}")
    print(f"Weights shape: {w.shape}")
    print(f"Weights sum per query row: {w.sum(dim=-1)}")

    # 2) Causal self-attention
    mask = causal_mask(T)
    out_causal, w_causal = scaled_dot_product_attention(x, x, x, mask=mask)
    print(f"\nCausal attention — last query attends to all? "
          f"{(w_causal[0, -1] > 0).all().item()}")
    print(f"Causal attention — first query attends only to itself? "
          f"{(w_causal[0, 0, 1:] == 0).all().item()}")

    # 3) Multi-head attention
    mha = MultiHeadAttention(D, N_HEADS)
    mha_out, mha_w = mha(x, x, x, mask=mask)
    print(f"\nMulti-head output shape: {mha_out.shape}")
    print(f"MHA weights shape: {mha_w.shape}  (B, heads, n, m)")

    # 4) Visualize head 0
    plot_attention(mha_w[0, 0], tokens)

    # 5) Show scaling effect: unscaled vs scaled dot products -> softmax saturation
    d_k_large = 512
    q = torch.randn(1, 1, d_k_large)
    k = torch.randn(1, 5, d_k_large)
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = raw_scores / math.sqrt(d_k_large)
    print(f"\nUnscaled score std: {raw_scores.std().item():.2f}")
    print(f"Scaled score std:   {scaled_scores.std().item():.2f}")
    print(f"Softmax(unscaled):  {F.softmax(raw_scores, -1).detach().numpy().round(3)}")
    print(f"Softmax(scaled):    {F.softmax(scaled_scores, -1).detach().numpy().round(3)}")
```

---

## Deep Dive

??? deep-dive "Attention as Low-Rank Kernel Approximation (sketch)"
    Writing \(A = \text{softmax}(QK^\top/\sqrt{d_k})\) is **not** linear, but **before softmax**, scores are **rank-\(\le d_k\)** bilinear forms. Some **linear attention** methods replace softmax with feature maps \(\phi(Q)\phi(K)^\top\) to get **subquadratic** or recurrent forms—useful context for “**alternatives to softmax attention**” questions.

??? deep-dive "FlashAttention — What Problem It Solves"
    **Standard** attention materializes \(T \times T\) scores in **HBM**; **FlashAttention** tiles computation to use **SRAM**, reducing **memory traffic**. Know: **asymptotic** \(O(T^2)\) unchanged, **wall-clock** and **memory footprint** improved—critical at **long** \(T\).

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Write scaled dot-product attention** and identify \(Q\), \(K\), \(V\).
    2. **Why divide by \(\sqrt{d_k}\)?** — Variance of dot products grows with \(d_k\); softmax saturates without scale.
    3. **What is causal masking and where applied?** — Before softmax, set forbidden \(j>i\) to \(-\infty\); used in **decoder** self-attention.
    4. **Self-attention vs. cross-attention?** — Same math; **cross** uses \(Q\) from decoder, \(K,V\) from encoder in **seq2seq** Transformers.
    5. **Compute complexity of attention?** — \(O(T^2 d)\) dominant for \(QK^\top\) and softmax matmuls; **memory** \(O(T^2)\) for scores unless fused/blockwise.
    6. **What does multi-head buy?** — Multiple **subspaces** / relationship types; more parameters and rank capacity.
    7. **Temperature vs. attention sharpness?** — Same softmax family; low temperature → peaked distributions.
    8. **Why not additive Bahdanau everywhere?** — Dot-product is **fast** on GPUs/TPUs—matrix multiply friendly.
    9. **Gradient flow:** what happens if softmax is almost one-hot? — Small gradients to non-selected positions.
    10. **KV cache in autoregressive decoding?** — Reuse past **keys/values** for new queries—saves recomputation; **still linear** memory in \(T\) for cache size.

!!! interview "Follow-up Probes"
    - “**Relative position** encodings—where do they enter?” — often **bias** to \(QK^\top\) or alternate RPE layers (Transformer-XL, T5 biases).
    - “**AliBi** vs. rotary?” — both inject position info; know **high-level** tradeoffs only if you claim expertise.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Scaled dot-product** keeps logits \(O(1)\) so **softmax** doesn’t **saturate**.”
    - “**Causal mask** enforces **autoregressive** factorization—no **future** tokens.”
    - “**Attention is \(O(T^2)\)** in sequence length—that’s the **long-context** bottleneck.”
    - “**Multi-head** learns **multiple** compatibility functions in **parallel subspaces**.”

---

## References

- Vaswani et al. (2017), *Attention Is All You Need* — Section 3.2
- Luong et al. (2015), *Effective Approaches to Attention-Based NMT*
- Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- Dao et al. — FlashAttention (IO-aware exact attention)
