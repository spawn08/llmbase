# 2.1 — The Transformer

## Intuition

The Transformer (Vaswani et al., 2017) replaced recurrence with **self-attention**, allowing the model to relate every token to every other token in a single step — and critically, to do so in **parallel**. It is the architecture behind GPT, BERT, T5, LLaMA, and virtually every modern LLM.

The original paper proposed an **encoder–decoder** model for machine translation, but the key insight — stacking self-attention and feedforward layers — generalizes to every variant used today.

---

## Architecture walkthrough

### High-level structure

```
Input tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌──────────────────────────┐
│  Transformer Block × N   │
│  ┌────────────────────┐  │
│  │  Multi-Head Attn   │──┐
│  └────────────────────┘  │ Add & LayerNorm (residual)
│  ┌────────────────────┐  │
│  │  Feed-Forward Net  │──┐
│  └────────────────────┘  │ Add & LayerNorm (residual)
└──────────────────────────┘
    ↓
Output projection (linear + softmax)
```

### Components in detail

**1. Input embedding + positional encoding**

Tokens are mapped to dense vectors via an embedding table \(E \in \mathbb{R}^{V \times d_{\text{model}}}\), then summed with positional encodings (covered in depth in Part 2.3):

\[
\mathbf{x}_i = \text{Embed}(w_i) + \text{PE}(i)
\]

**2. Multi-head self-attention (MHA)**

Covered in detail in Part 2.2. The core operation:

\[
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\]

\[
\text{head}_i = \text{softmax}\!\Bigl(\frac{(XW_i^Q)(XW_i^K)^T}{\sqrt{d_k}}\Bigr)(XW_i^V)
\]

**3. Feed-forward network (FFN)**

A position-wise two-layer MLP applied independently to each token:

\[
\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2
\]

- Original paper: \(\text{activation} = \text{ReLU}\), inner dim = \(4 \times d_{\text{model}}\).
- Modern LLMs: **SwiGLU** or **GeGLU** (LLaMA, PaLM) — a gated variant that outperforms ReLU.

The FFN is where the model stores "knowledge" — factual associations live in these weights.

**4. Residual connections + Layer normalization**

Each sub-layer (attention, FFN) is wrapped in a **residual connection** followed by **layer norm**:

\[
\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))
\]

This is **Post-Norm** (original paper). Modern models typically use **Pre-Norm** — applying LayerNorm *before* the sub-layer:

\[
\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))
\]

Pre-Norm trains more stably at scale and is used by GPT-2+, LLaMA, and most open models. Some recent work uses **RMSNorm** (a simpler variant without the mean-centering step).

**5. The residual stream**

A powerful mental model: the input flows through a **residual stream**, and each attention/FFN layer *writes into* the stream additively. The final output is the sum of the input plus all layer contributions. This view explains:

- Why deeper models generalize better (more additive refinement).
- Why layers can be pruned (some contribute little).
- Why attention heads can be interpreted (each adds a specific pattern to the stream).

### Encoder vs. decoder vs. encoder–decoder

| Variant | Attention mask | Used by | Training objective |
| --- | --- | --- | --- |
| Encoder-only | Bidirectional (no mask) | BERT, RoBERTa | Masked LM |
| Decoder-only | Causal (lower-triangular) | GPT, LLaMA, Mistral | Next-token prediction |
| Encoder–decoder | Encoder: bidir; Decoder: causal + cross-attn | T5, BART | Seq2seq / span corruption |

### Key hyperparameters

| Symbol | Meaning | Typical range |
| --- | --- | --- |
| \(d_{\text{model}}\) | Hidden dimension | 768 (BERT-base) → 8192 (LLaMA-70B) |
| \(n_{\text{heads}}\) | Number of attention heads | 12 → 64 |
| \(d_k = d_{\text{model}} / n_{\text{heads}}\) | Per-head dimension | 64 → 128 |
| \(d_{\text{ff}}\) | FFN inner dimension | \(4 \times d_{\text{model}}\) typically |
| \(N\) | Number of layers | 6 (original) → 80+ (GPT-4-class) |

---

## Code — Transformer encoder block from scratch

```python
"""
Single Transformer encoder block in PyTorch — the building block of every LLM.
Implements: Pre-Norm, Multi-Head Self-Attention, SwiGLU FFN, residual connections.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.W_qkv(x)                                       # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)                                   # each (B, H, T, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                # (B, H, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, D)                # (B, T, D)
        return self.W_o(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (Shazeer, 2020) — used in LLaMA, PaLM."""

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)  # SwiGLU convention
        # Round to nearest multiple of 64 for hardware efficiency
        d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block with RMSNorm and SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)   # residual + attention
        x = x + self.ffn(self.ffn_norm(x))            # residual + FFN
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks with token + positional embedding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x)  # (B, T, V)


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    V, D, H, L = 1000, 256, 4, 4
    model = TransformerEncoder(V, D, H, L)

    tokens = torch.randint(0, V, (2, 32))
    causal = torch.tril(torch.ones(32, 32, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    logits = model(tokens, mask=causal)
    print(f"Input:  {tokens.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Verify residual stream: output changes when we zero out FFN weights
    with torch.no_grad():
        baseline = model(tokens, mask=causal)
        for layer in model.layers:
            layer.ffn.w1.weight.zero_()
        no_ffn = model(tokens, mask=causal)
        diff = (baseline - no_ffn).abs().mean().item()
        print(f"Mean diff (with vs. without FFN): {diff:.4f}")
```

---

## Interview takeaways

1. **Draw the block** — be ready to sketch: Input → Norm → MHA → Add → Norm → FFN → Add. Know Pre-Norm vs. Post-Norm and why Pre-Norm is preferred at scale.
2. **Residual stream** — the model additively refines representations layer by layer. This is why attention heads and FFN layers can be analyzed independently.
3. **FFN stores knowledge** — factual recall (e.g., "Paris is the capital of France") is localized in FFN weight matrices. Attention heads handle routing/retrieval.
4. **SwiGLU over ReLU** — modern LLMs use gated activations. Know that SwiGLU has three projections (gate, up, down) vs. two in vanilla FFN.
5. **RMSNorm over LayerNorm** — simpler (no mean subtraction), faster, and works as well. Used in LLaMA, Mistral.
6. **Scaling** — going from BERT-base (12 layers, 110M params) to GPT-3 (96 layers, 175B params) is "just" stacking more blocks with wider dimensions. The architecture is identical.
7. **Why Transformers beat RNNs** — parallelism (no sequential dependency during training), direct long-range connections (every token attends to every other), and easier optimization (residual connections + normalization).

---

## References

- Vaswani et al. (2017), *Attention Is All You Need*
- Shazeer (2020), *GLU Variants Improve Transformer*
- Zhang & Sennrich (2019), *Root Mean Square Layer Normalization*
- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
