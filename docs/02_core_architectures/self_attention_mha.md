# 2.2 — Self-Attention & Multi-Head Attention

## Intuition

Part 1.6 derived scaled dot-product attention mathematically. This section focuses on the **engineering**: how self-attention and multi-head attention are implemented inside a Transformer, what each head learns, and the masking patterns that distinguish encoders from decoders.

---

## Core concepts

### Self-attention revisited

In **self-attention**, the queries, keys, and values all come from the same sequence. For input \(X \in \mathbb{R}^{T \times d}\):

\[
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
\]

\[
\text{SelfAttn}(X) = \text{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr) V
\]

Each token's output is a weighted combination of **all tokens' values**, where the weights reflect pairwise compatibility (via Q-K dot products).

### Why multi-head?

A single attention head uses one set of Q/K/V projections and can only capture one type of relationship. **Multi-head attention** runs \(h\) parallel heads, each with its own projections into a lower-dimensional subspace:

\[
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V), \quad W_i^Q, W_i^K \in \mathbb{R}^{d \times d_k}, \quad d_k = d/h
\]

\[
\text{MHA}(X) = [\text{head}_1; \ldots; \text{head}_h] \, W^O
\]

**What different heads learn (empirically observed):**

| Head type | Pattern | Example |
| --- | --- | --- |
| Positional | Attends to fixed offset (e.g., previous token) | "Token at position \(i\) attends to \(i-1\)" |
| Syntactic | Attends to syntactic parent/child | Verb → subject, adjective → noun |
| Semantic | Attends to co-referent or related entity | Pronoun → its antecedent |
| Induction | Copies from matching prior context | "AB...A" → predict "B" |

### Cross-attention

In encoder–decoder models (T5, original Transformer), the decoder has an additional **cross-attention** layer where:

- **Q** comes from the decoder
- **K, V** come from the encoder output

This lets the decoder "look at" the source sequence at every generation step — the attention-based analogue of Bahdanau attention from Part 1.4.

### Masking patterns

| Type | Shape | Purpose | Used in |
| --- | --- | --- | --- |
| No mask | Full \(T \times T\) | Bidirectional attention | BERT encoder, T5 encoder |
| Causal | Lower-triangular | Prevent future tokens | GPT, LLaMA decoder |
| Padding | Per-sample binary | Ignore `[PAD]` tokens | All models with variable-length batches |
| Prefix | Bidirectional prefix + causal suffix | Allow prompt tokens to see each other | PrefixLM (e.g., U-PaLM) |

### Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

Standard MHA requires \(h\) separate K and V projections, which is expensive for KV cache at inference. Modern models reduce this:

- **Multi-Query Attention (MQA):** All heads share a **single** K and V projection. Only Q is per-head.
- **Grouped-Query Attention (GQA):** Heads are divided into \(g\) groups; heads within a group share K/V. When \(g = 1\) → MQA; when \(g = h\) → standard MHA.

LLaMA 2 (70B) and Mistral use GQA. This reduces KV cache memory by a factor of \(h/g\) with minimal quality loss.

---

## Code — Multi-Head Attention with GQA support

```python
"""
Multi-Head Attention with standard MHA, GQA, and MQA modes.
Includes: causal masking, padding masking, and head visualization.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Literal


class FlexibleMHA(nn.Module):
    """
    Multi-Head Attention supporting MHA, GQA, and MQA.

    Args:
        d_model: hidden dimension
        n_heads: number of query heads
        n_kv_heads: number of key/value heads
            - n_kv_heads == n_heads → standard MHA
            - n_kv_heads == 1 → MQA
            - 1 < n_kv_heads < n_heads → GQA
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads  # heads per KV group
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """(B, n_kv_heads, T, d_k) → (B, n_heads, T, d_k) by repeating."""
        if n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        src = kv if kv is not None else x
        S = src.size(1)

        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(src).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(src).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out), weights


def causal_mask(T: int) -> torch.Tensor:
    return torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)


def padding_mask(lengths: list[int], max_len: int) -> torch.Tensor:
    """(B, 1, 1, T) mask where True = valid token."""
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return mask.unsqueeze(1).unsqueeze(2)


def visualize_heads(weights: torch.Tensor, tokens: list[str], title: str = "") -> None:
    """Plot attention weights for all heads in a single figure."""
    n_heads = weights.size(1)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    for h in range(n_heads):
        w = weights[0, h].detach().cpu().numpy()
        ax = axes[h]
        ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Head {h}")
    fig.suptitle(title or "Multi-Head Attention Weights")
    plt.tight_layout()
    plt.savefig("mha_heads.png", dpi=150)
    plt.show()


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    D, H = 64, 4
    tokens = ["The", "cat", "sat", "on", "a", "mat"]
    T = len(tokens)
    x = torch.randn(1, T, D)

    # Standard MHA (4 heads, 4 KV heads)
    mha = FlexibleMHA(D, n_heads=H, n_kv_heads=H)
    out_mha, w_mha = mha(x, mask=causal_mask(T))
    print(f"MHA output: {out_mha.shape}, weights: {w_mha.shape}")

    # GQA (4 query heads, 2 KV groups)
    gqa = FlexibleMHA(D, n_heads=H, n_kv_heads=2)
    out_gqa, w_gqa = gqa(x, mask=causal_mask(T))
    print(f"GQA output: {out_gqa.shape}")
    print(f"GQA KV params: {sum(p.numel() for p in [gqa.W_k.weight, gqa.W_v.weight]):,}")
    print(f"MHA KV params: {sum(p.numel() for p in [mha.W_k.weight, mha.W_v.weight]):,}")

    # MQA (4 query heads, 1 KV head)
    mqa = FlexibleMHA(D, n_heads=H, n_kv_heads=1)
    out_mqa, _ = mqa(x, mask=causal_mask(T))
    print(f"MQA output: {out_mqa.shape}")

    # Cross-attention example
    encoder_out = torch.randn(1, 10, D)
    cross_attn = FlexibleMHA(D, n_heads=H)
    cross_out, cross_w = cross_attn(x, kv=encoder_out)
    print(f"\nCross-attention: Q from decoder {x.shape}, KV from encoder {encoder_out.shape}")
    print(f"Cross output: {cross_out.shape}, weights: {cross_w.shape}")

    visualize_heads(w_mha, tokens, "Causal Self-Attention (4 heads)")
```

---

## Interview takeaways

1. **Self vs. cross attention** — self: Q/K/V from same sequence. Cross: Q from decoder, K/V from encoder. Every encoder–decoder model uses both.
2. **Why multi-head?** — a single head can only learn one attention pattern. Multiple heads capture positional, syntactic, and semantic patterns in parallel.
3. **GQA/MQA** — the primary inference optimization for KV cache. GQA with 8 KV groups vs 64 query heads reduces KV cache by 8× with <1% quality loss. Know that LLaMA 2 70B and Mistral use GQA.
4. **Head dimension** — \(d_k = d_{\text{model}} / n_{\text{heads}}\). Typical values: 64 or 128. More heads with smaller \(d_k\) = more diverse patterns; fewer heads with larger \(d_k\) = more expressive per head.
5. **Padding mask vs. causal mask** — padding mask prevents attention to `[PAD]` tokens (batch efficiency). Causal mask prevents attending to future tokens (autoregressive generation). They are combined via element-wise AND.
6. **Attention complexity** — \(O(T^2 d_k)\) per head, \(O(T^2 d_{\text{model}})\) total. The \(T^2\) factor is the bottleneck for long sequences.

---

## References

- Vaswani et al. (2017), *Attention Is All You Need*
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head Is All You Need* (MQA)
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- Olsson et al. (2022), *In-context Learning and Induction Heads*
