# 1.6 — Mathematics of Attention

## Intuition

Attention answers one question: **given a set of values, which ones should I focus on right now?** It computes a *weighted sum* of values, where the weights are determined by how relevant each value is to the current query. This mechanism — going from Bahdanau's additive score (Part 1.4) to the scaled dot-product used in every Transformer — is arguably the single most important equation in modern deep learning.

---

## Core concepts

### From Bahdanau to dot-product

Recall from Part 1.4 that Bahdanau attention uses a learned scoring network:

\[
e_{t,j} = \mathbf{v}^T \tanh(W_1 \mathbf{h}_j + W_2 \mathbf{s}_t)
\]

This is **additive attention** — expressive but slow, requiring a forward pass through a small network for every (query, key) pair.

**Dot-product attention** (Luong, 2015) replaces this with:

\[
e_{t,j} = \mathbf{s}_t^T \mathbf{h}_j
\]

This is a simple inner product — fast and parallelizable, but its magnitude grows with the dimension of the vectors.

### Scaled dot-product attention (Vaswani et al., 2017)

The Transformer's attention divides by \(\sqrt{d_k}\) to stabilize gradients:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr) V
\]

where:

- \(Q \in \mathbb{R}^{n \times d_k}\) — **queries** (what am I looking for?)
- \(K \in \mathbb{R}^{m \times d_k}\) — **keys** (what do I contain?)
- \(V \in \mathbb{R}^{m \times d_v}\) — **values** (what information do I provide?)
- \(n\) = number of query positions, \(m\) = number of key/value positions

### Why scale by \(\sqrt{d_k}\)?

If entries of \(Q\) and \(K\) are i.i.d. with mean 0 and variance 1, then:

\[
\text{Var}\bigl(\mathbf{q}^T \mathbf{k}\bigr) = d_k
\]

For large \(d_k\) (e.g., 64 or 128), the dot products become very large in magnitude. When fed into softmax, this pushes outputs toward one-hot vectors — the softmax saturates, and gradients vanish. Dividing by \(\sqrt{d_k}\) restores unit variance:

\[
\text{Var}\!\Bigl(\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}\Bigr) = 1
\]

### Softmax as a soft argmax

The softmax function converts raw scores into a probability distribution:

\[
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
\]

**Properties:**

- Output sums to 1 (valid probability distribution).
- Differentiable — unlike hard argmax.
- **Temperature** — dividing logits by \(\tau\) before softmax controls sharpness: small \(\tau\) → peaky (near argmax), large \(\tau\) → uniform. This is exactly the temperature parameter used during LLM decoding (Part 4.1).

### Attention as soft dictionary lookup

Think of attention as a **differentiable dictionary**:

| Concept | Standard dict | Attention |
| --- | --- | --- |
| Lookup | Exact key match | Soft similarity to all keys |
| Output | Single value | Weighted sum of all values |
| Gradient | None (discrete) | Flows through weights |

The query "looks up" the most relevant keys, and the output is a blend of their values proportional to relevance.

### Multi-head attention (preview)

Instead of one attention function, the Transformer runs \(h\) parallel heads with separate learned projections:

\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\]

Each head can learn a different type of relationship (positional, syntactic, semantic). The full treatment is in Part 2.2.

### Masking

In decoder self-attention (autoregressive), we prevent position \(i\) from attending to positions \(> i\). This is done by adding \(-\infty\) to those entries in the score matrix before softmax:

\[
\text{score}_{i,j} = \begin{cases} \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
\]

After softmax, \(-\infty\) entries become 0 — the model cannot "see the future."

---

## Code — Scaled dot-product attention from scratch

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
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,...,n,m)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)  # (B,...,n,m)
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

        q = self.W_q(Q).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, m, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, m, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, m)

        out, weights = scaled_dot_product_attention(q, k, v, mask)
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

    # 5) Show scaling effect
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

## Interview takeaways

1. **The attention equation** — be ready to write \(\text{softmax}(QK^T / \sqrt{d_k}) \, V\) on a whiteboard and explain each term: queries select, keys are compared, values are retrieved, scaling prevents saturation.
2. **Why \(\sqrt{d_k}\)?** — dot-product variance grows linearly with \(d_k\). Without scaling, softmax saturates and gradients vanish. This is the single most-asked "why" question about attention.
3. **Softmax temperature** — same math as decoding temperature. Low temp → sharper attention → more like hard lookup. High temp → uniform → more like averaging.
4. **Causal mask** — autoregressive models (GPT) add a triangular mask so token \(i\) cannot attend to token \(j > i\). Know that this is applied *before* softmax by adding \(-\infty\).
5. **Complexity** — standard attention is \(O(n^2 d)\) in compute and \(O(n^2)\) in memory (the score matrix). This quadratic cost is what FlashAttention, linear attention, and state-space models address.
6. **Multi-head = multiple subspaces** — each head projects Q/K/V into a smaller subspace. Different heads learn different patterns (e.g., one head for position, another for syntax). Outputs are concatenated and projected.
7. **Self-attention vs. cross-attention** — self: Q, K, V all come from the same sequence. Cross: Q from decoder, K/V from encoder. The Transformer uses both.

---

## References

- Vaswani et al. (2017), *Attention Is All You Need* — Section 3.2
- Luong et al. (2015), *Effective Approaches to Attention-Based NMT*
- Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
