# Attention Is All You Need (Transformer)

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Year:** 2017 &nbsp;|&nbsp; **Venue:** NeurIPS  
**Link:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## TL;DR

The paper introduces the **Transformer**, a sequence-to-sequence architecture that replaces recurrence entirely with **self-attention**. Every token attends to every other token in a single layer, weighted by learned compatibility scores. Multi-head attention runs several attention functions in parallel so the model captures different relational patterns (syntax, coreference, long-range agreement). Positional encodings inject ordering information because attention itself is permutation-invariant.

---

## Why This Paper Matters

The Transformer is the architectural backbone of virtually every modern LLM — GPT, BERT, T5, LLaMA, Mistral, and beyond. Every serving optimization (FlashAttention, KV cache, speculative decoding) targets the same attention primitive. If you understand this paper deeply, you understand the foundation of everything that followed.

---

## Key Concepts Explained Simply

### Self-Attention: The Core Idea

Imagine you're reading a sentence and for every word, you ask: **"Which other words should I pay attention to in order to understand this word?"**

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I give back?"

Each word creates a Q, K, and V vector. The attention score between word *i* and word *j* is the dot product of word *i*'s query with word *j*'s key. High dot product = high relevance.

### Multi-Head Attention

Instead of running attention once, the model runs it **h times in parallel** (typically h=8 or h=16), each with different learned projections. One head might learn syntactic relationships, another might learn coreference, another might learn positional patterns. The outputs are concatenated and projected.

### Positional Encoding

Attention is permutation-invariant — it doesn't know word order. The paper adds sinusoidal position signals to the input embeddings:

- Even dimensions: \(\sin(pos / 10000^{2i/d})\)
- Odd dimensions: \(\cos(pos / 10000^{2i/d})\)

This lets the model learn relative positions because any fixed offset can be represented as a linear function of the encoding.

---

## The Math — Explained Step by Step

### Scaled Dot-Product Attention

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

**Breaking it down:**

1. **\(QK^\top\):** Matrix of dot products between every query and every key. Shape: \([n \times n]\) where \(n\) is sequence length. Each entry measures how much token *i* should attend to token *j*.

2. **\(\div \sqrt{d_k}\):** Scaling factor. Without it, when \(d_k\) is large (e.g., 64), dot products grow large, pushing softmax into regions with tiny gradients. Dividing by \(\sqrt{d_k}\) keeps values in a "nice" range for softmax.

3. **\(\mathrm{softmax}\):** Converts raw scores into a probability distribution over positions. Each row sums to 1 — it's a weighted "where to look" distribution.

4. **\(\times V\):** Weighted sum of value vectors using the attention weights. The output for each position is a mixture of all value vectors, weighted by relevance.

### Multi-Head Attention

\[
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O
\]

where each head is:

\[
\mathrm{head}_i = \mathrm{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\]

Each head has its own learned projection matrices \(W_i^Q, W_i^K, W_i^V\) of shape \([d_{\text{model}} \times d_k]\). The final projection \(W^O\) maps the concatenated heads back to \(d_{\text{model}}\).

### Complexity

- **Time:** \(O(n^2 \cdot d)\) — quadratic in sequence length
- **Space:** \(O(n^2)\) for the attention matrix
- This quadratic cost is the bottleneck that FlashAttention, sparse attention, and linear attention methods target

---

## Python Implementation

```python
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [seq_len_q, d_k]
    K: [seq_len_k, d_k]
    V: [seq_len_k, d_v]
    mask: [seq_len_q, seq_len_k] — 0 where attention is blocked
    Returns: [seq_len_q, d_v]
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    return attn_weights @ V, attn_weights


def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """
    W_q, W_k, W_v: [d_model, d_model] (split across heads internally)
    W_o: [d_model, d_model]
    """
    d_model = Q.shape[-1]
    d_k = d_model // n_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    heads = []
    for i in range(n_heads):
        start, end = i * d_k, (i + 1) * d_k
        head_out, _ = scaled_dot_product_attention(
            Q_proj[:, start:end],
            K_proj[:, start:end],
            V_proj[:, start:end],
            mask
        )
        heads.append(head_out)

    concat = np.concatenate(heads, axis=-1)
    return concat @ W_o


def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding from the paper."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def causal_mask(seq_len):
    """Lower-triangular mask for autoregressive (decoder) attention."""
    return np.tril(np.ones((seq_len, seq_len)))


def transformer_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
                       gamma1, beta1, gamma2, beta2, n_heads, mask=None):
    """Single Transformer encoder block: MHA + Add&Norm + FFN + Add&Norm."""
    # Multi-head self-attention
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, n_heads, mask)
    x = layer_norm(x + attn_out, gamma1, beta1)

    # Position-wise FFN: Linear -> ReLU -> Linear
    ffn_out = np.maximum(0, x @ W1 + b1) @ W2 + b2
    x = layer_norm(x + ffn_out, gamma2, beta2)
    return x


def layer_norm(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model, n_heads = 6, 16, 4

    x = np.random.randn(seq_len, d_model)
    x = x + positional_encoding(seq_len, d_model)

    Q = K = V = x
    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights (row 0):", np.round(weights[0], 3))
    print("Weights sum per row:", np.round(weights.sum(axis=-1), 6))

    # With causal mask (decoder-style)
    mask = causal_mask(seq_len)
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask)
    print("\nCausal attention weights (row 3):", np.round(weights_causal[3], 3))
```

---

## Interview Importance

This is the **#1 most asked** paper in LLM interviews. Every concept in modern LLMs traces back here.

### Difficulty Level: ⭐⭐⭐ (Medium-High)

Interviewers expect you to be able to write the attention formula on a whiteboard, explain every term, and discuss practical implications.

---

## Interview Questions & Answers

### Q1: Walk through the attention formula. Why divide by \(\sqrt{d_k}\)?

**Answer:** The dot product \(q \cdot k\) between two random vectors with \(d_k\) independent components has variance proportional to \(d_k\). When \(d_k = 64\), raw dot products can easily reach values like ±16, which pushes softmax into saturation — outputs near 0 or 1 with near-zero gradients. Dividing by \(\sqrt{d_k}\) normalizes the variance to approximately 1, keeping softmax in its informative gradient regime.

### Q2: What is the difference between self-attention in an encoder vs. a decoder?

**Answer:** In an **encoder**, each token attends to all tokens (bidirectional). In a **decoder**, each token can only attend to itself and previous tokens — future tokens are masked with \(-\infty\) before softmax. This causal masking ensures the model can't "peek ahead," which is essential for autoregressive generation.

### Q3: Why is attention \(O(n^2)\) and what are the consequences?

**Answer:** The attention matrix is \(n \times n\) (every token vs. every token). For a 128K context window, that's ~16 billion entries per layer per head. This drives memory costs (storing the matrix) and compute costs (matrix multiplications). Solutions include sparse attention, sliding window attention, linear attention approximations, and FlashAttention (which is exact but IO-aware to avoid materializing the full matrix in HBM).

### Q4: Why use multi-head attention instead of single-head with larger dimension?

**Answer:** Different heads can learn to attend to different types of relationships simultaneously. Empirically, one head might specialize in local syntactic patterns while another captures long-range dependencies. A single head with the same total dimension would be forced to compress all these patterns into one set of weights. Multi-head also doesn't cost more compute — the per-head dimension is \(d_k = d_{\text{model}}/h\), so total FLOPs are equivalent.

### Q5: How do positional encodings work and why are they needed?

**Answer:** Self-attention is permutation-equivariant — shuffling the input tokens and then running attention gives the same result as running attention then shuffling outputs. Without position signals, the model can't distinguish "the cat sat on the mat" from "mat the on sat cat the." Sinusoidal encodings add unique position-dependent signals. Later models replaced these with learned positional embeddings or Rotary Position Embeddings (RoPE), which encode relative positions directly into the attention computation.

### Q6: In the encoder-decoder architecture, how does cross-attention work?

**Answer:** In cross-attention, queries come from the decoder and keys/values come from the encoder output. This lets each decoder position "look at" the entire encoded input to decide what source information is relevant for the current generation step. It's the mechanism that connects the "understanding" (encoder) with the "generation" (decoder) in tasks like translation.

---

## Connections to Other Papers

- **BERT** → Uses the encoder side with bidirectional attention
- **GPT-2/3** → Uses the decoder side with causal masking
- **T5** → Uses the full encoder-decoder architecture
- **FlashAttention** → Optimizes the exact same computation for GPU memory hierarchy
- **Mamba/SSMs** → Alternative to attention with linear complexity

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core formula | \(\mathrm{softmax}(QK^\top / \sqrt{d_k}) \cdot V\) |
| Scaling reason | Prevent softmax saturation with large \(d_k\) |
| Multi-head purpose | Parallel attention patterns (syntax, semantics, position) |
| Positional encoding | Sinusoidal — needed because attention is permutation-invariant |
| Complexity | \(O(n^2 d)\) time, \(O(n^2)\) space |
| Architecture | Encoder-decoder with 6 layers each in original paper |
