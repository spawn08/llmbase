# Self-Attention and Multi-Head Attention

## Why This Matters for LLMs

Self-attention is the mechanism that lets every token **look at** every other token in one parallel pass. Without mastering it, you cannot explain training cost, inference KV-cache sizing, or why long-context research focuses on quadratic scaling in sequence length. Every interview that goes beyond surface level will ask you to derive or narrate the attention map, because that map is the operational heart of what Transformers **do** with language: assign credit across positions.

Multi-head attention exists because one attention pattern is rarely enough. Language carries positional regularities, syntactic dependencies, semantic coreference, and memorized patterns like induction. Heads provide subspaces with separate projections so the model can implement multiple relationship types **in parallel** before mixing them back together. When you read papers on “induction heads” or “attention head ablations,” they are discussing the same engineering object you implement with reshape and softmax.

Grouped-query attention and multi-query attention matter because deployment is not the same as training. During autoregressive decoding, keys and values for past tokens are stored and reused. Reducing the number of distinct KV projections shrinks memory bandwidth and footprint with limited quality impact when done carefully. Modern model cards from major labs explicitly report GQA configurations because this is a first-order serving concern alongside model quality.

---

## Scaled Dot-Product Attention (Recap)

For queries \(Q\), keys \(K\), values \(V\) (with shapes compatible for batched matrix multiply), attention is:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
\]

!!! math-intuition "In Plain English"
    Each query row asks a question: “How much should I take from each key row’s value?” The dot product \(Q K^\top\) scores compatibility between queries and keys. Softmax turns those scores into nonnegative weights that sum to one along each query position. Multiplying by \(V\) mixes value vectors according to those weights. The scale \(\sqrt{d_k}\) prevents dot products from growing too large as head dimension increases, which would push softmax into extremely peaked regions with vanishing gradients.

---

## Self-Attention

In **self-attention**, queries, keys, and values all come from the same sequence \(X \in \mathbb{R}^{B \times T \times d_{\text{model}}}\):

\[
Q = X W^Q,\quad K = X W^K,\quad V = X W^V
\]

\[
\text{SelfAttn}(X) = \text{Attention}(Q, K, V)
\]

!!! math-intuition "In Plain English"
    The same sentence supplies both the questions and the answers. A pronoun at position \(i\) can attend strongly to a noun at position \(j\) because the learned projections place compatible vectors in \(Q\) and \(K\) space. Because every position can attend to every position, the operation captures **global** dependencies in one step, unlike a convolution with fixed locality or an RNN with fixed recurrence depth per step.

!!! example "Worked Example: Self-Attention on a Short Sentence"
    Consider tokens `["The", "cat", "sat"]` embedded to vectors of dimension \(d_{\text{model}} = 6\) for a toy illustration. Self-attention does not care about the English words; it cares about numerical vectors. Suppose after projections a single head has \(Q\) rows \(q_0, q_1, q_2\) and key rows \(k_0, k_1, k_2\) with head dimension \(d_k = 2\). The score matrix before scaling is \(S_{ij} = q_i \cdot k_j\). After dividing by \(\sqrt{2}\) and applying softmax along the key index \(j\), position \(1\) might place most mass on index \(0\) if the learned geometry encodes determiner–noun affinity. The output row \(1\) becomes a convex combination of value vectors \(v_0, v_1, v_2\) using those weights. The narrative you give in an interview is exactly this: **mixing values according to learned pairwise scores**.

---

## Cross-Attention

In encoder–decoder models, **cross-attention** uses queries from the **decoder** sequence and keys and values from the **encoder** sequence:

\[
Q = X_{\text{dec}} W^Q,\quad K = X_{\text{enc}} W^K,\quad V = X_{\text{enc}} W^V
\]

\[
\text{CrossAttn}(X_{\text{dec}}, X_{\text{enc}}) = \text{Attention}(Q, K, V)
\]

!!! math-intuition "In Plain English"
    The decoder token at position \(i\) asks: “Which source tokens from the encoder should I read right now?” Translation is the classic picture: an English word in the decoder attends to French words in the encoder output. The operation is the same softmax-weighted mix, but the **key and value bank** is the entire encoder representation, while **queries** come from the autoregressive side. This is the attention analogue of the older RNN attention mechanisms, implemented as batched matrix multiplies.

!!! example "Worked Example: Cross-Attention Shapes"
    Let batch \(B = 2\), decoder length \(T_{\text{dec}} = 5\), encoder length \(T_{\text{enc}} = 7\), and \(d_{\text{model}} = 512\). The decoder hidden states have shape \((2, 5, 512)\) and the encoder memory has shape \((2, 7, 512)\). After projections, \(Q\) has shape \((2, 5, 512)\) which reshapes to heads as \((2, h, 5, d_k)\). Keys and values have shape \((2, h, 7, d_k)\). The attention score tensor has shape \((2, h, 5, 7)\): each of the five decoder positions assigns seven weights over encoder positions. The output has shape \((2, 5, 512)\) again, ready for the decoder FFN. In conversation, emphasize that **time axes differ** between decoder query length and encoder memory length.

---

## Multi-Head Attention

Multiple heads use separate learned projections into subspaces:

\[
\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)
\]

\[
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O
\]

!!! math-intuition "In Plain English"
    Each head gets its own angles in \(Q\), \(K\), and \(V\) space. One head might specialize in attending to the previous token. Another might track subject–verb agreement at longer distance. Concatenation stacks those specialized mixes along the feature axis, and \(W^O\) fuses them back into the model width. Saying “each head can focus on different relationship types” is accurate as intuition; the formal statement is that heads parameterize different low-rank factorizations of attention patterns.

---

## Grouped-Query Attention and Multi-Query Attention

Standard multi-head attention uses \(h\) distinct \(K\) and \(V\) projections. **Multi-query attention** shares one \(K\) and one \(V\) across all heads; only queries remain per-head. **Grouped-query attention** partitions heads into \(g\) groups that share \(K\) and \(V\). When \(g = h\) you recover full MHA; when \(g = 1\) you recover MQA.

!!! math-intuition "In Plain English"
    During autoregressive decoding, past tokens’ keys and values are cached. If you duplicate fewer unique KV vectors per layer, you spend less memory and bandwidth re-reading those tensors. Queries stay per-head so expressive routing remains, while KV sharing trades a controlled amount of representational freedom for efficiency.

!!! example "Worked Example: GQA Memory Savings"
    Take \(h = 32\) query heads and \(g = 8\) KV head groups, a common “GQA-4” style ratio because \(32 / 8 = 4\). Let sequence length \(T = 2048\) and head dimension \(d_{\text{head}} = 128\). For each layer, the KV cache stores two tensors (keys and values) for the prefix. With full multi-head attention you store \(32\) distinct \(K\) heads and \(32\) distinct \(V\) heads across positions. Counting scalar entries for keys only at one layer: \(32 \times 2048 \times 128 = 8{,}388{,}608\) floats. Values duplicate that count, so keys plus values together give \(2 \times 8{,}388{,}608 = 16{,}777{,}216\) floats for the key–value side at one layer in this toy accounting. With \(8\) KV heads instead of \(32\), each of keys and values contributes \(8 \times 2048 \times 128 = 2{,}097{,}152\) floats, and together \(4{,}194{,}304\) floats. The ratio is exactly \(16{,}777{,}216 / 4{,}194{,}304 = 4\), so KV cache size drops by **four times** at this level of head grouping. Real systems also count batch size, layers, and precision, but the **fourfold** reduction is the headline number you want in an interview.

---

## Masking Patterns

Attention supports arbitrary boolean masks over key positions. Three recurring patterns are **causal** (decoder), **padding** (batched sequences of unequal length), and **prefix** (bidirectional prefix with causal continuation).

### Causal Mask

Decoder-only models forbid position \(i\) from attending to positions \(j > i\). For \(T = 4\), allowed positions have \(j \le i\). Using `True` for allowed entries:

\[
M_{\text{causal}} =
\begin{pmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{pmatrix}
\]

!!! math-intuition "In Plain English"
    The lower triangle encodes **time direction**: information flows from past and present to the present, never from the future. During training, the entire matrix of targets is processed in parallel, but each position’s attention sees only compatible keys. Implementation-wise, disallowed entries are filled with large negative values before softmax so their weights become numerically zero.

### Padding Mask

When sequences in a batch have different lengths, shorter sequences are padded to a common \(T\). Padding positions must not be attended to. For a single sequence of logical length \(3\) padded to \(T = 4\), mark valid positions as \(1\) along the key axis:

\[
M_{\text{padding}} =
\begin{pmatrix}
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 0
\end{pmatrix}
\]

Rows correspond to queries; columns to keys. Here the fourth column is disallowed for **all** queries because the fourth token is padding.

!!! math-intuition "In Plain English"
    Padding masks protect the model from treating artificial zeros as real content. In bidirectional encoders, both queries and keys for padded positions are typically ignored by additional masking on the query side as well, depending on implementation. The invariant you should articulate: **never let softmax assign mass to keys that do not correspond to real tokens**.

### Prefix Mask

Some models allow bidirectional attention inside a prompt prefix while keeping causal attention on generated continuation. For prefix length \(2\) and total \(T = 4\), positions \(0\) and \(1\) attend to each other freely; after that, causal structure applies:

\[
M_{\text{prefix}} =
\begin{pmatrix}
1 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{pmatrix}
\]

!!! math-intuition "In Plain English"
    The upper-left block is fully visible because the prompt tokens should mutually condition. The lower-right behaves like a causal decoder for the generated suffix. This pattern appears in models trained with **prefix language modeling** objectives and in certain instruction-tuning setups where the instruction is bidirectional within itself but generation remains autoregressive.

---

??? deep-dive "Deep Dive: Why Multiple Heads Beat One Wide Head"
    A single attention head with dimension \(d_{\text{model}}\) could in principle learn any attention matrix of rank limited by \(d_k\). Multiple heads factor the representation into \(h\) separate subspaces of dimension \(d_k = d_{\text{model}} / h\). Empirically, this factorization improves optimization and interpretability: heads specialize. The story you tell should not claim orthogonality; specialization emerges from training, not from linear algebra alone.

---

??? deep-dive "Deep Dive: Attention Complexity"
    Computing \(Q K^\top\) for sequences of length \(T\) costs \(O(T^2 d_k)\) per head before softmax and the multiply with \(V\). Across heads, costs scale with \(d_{\text{model}}\). The quadratic term in \(T\) is why long-context systems explore sparse attention, linear attention, or hardware-aware approximations. Mentioning KV-cache linear growth with \(T\) is equally important for production.

---

## Code

The module `FlexibleMHA` implements self-attention and cross-attention with **MHA**, **GQA**, or **MQA** depending on `n_kv_heads`. Mask broadcasting follows PyTorch rules: a mask of shape \((B, 1, T_q, T_k)\) broadcasts over heads. Padding-style masks can be built per batch. The demo avoids plotting libraries so the script runs in minimal environments.

```python
"""
Flexible multi-head attention: MHA, GQA, and MQA in one implementation.
Includes causal, padding-style, and prefix masks as explicit 4x4 arrays in docs
and constructors for tensors used in attention.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask_square(t: int) -> torch.Tensor:
    """Lower-triangular boolean mask of shape (1, 1, t, t), True = allow."""
    m = torch.tril(torch.ones(t, t, dtype=torch.bool))
    return m.view(1, 1, t, t)


def padding_mask_from_lengths(lengths: list[int], total_len: int) -> torch.Tensor:
    """
    Build (B, 1, 1, total_len) mask for KEY positions: True where token is real.
    Broadcasts over queries and heads.
    """
    b = len(lengths)
    m = torch.zeros(b, total_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        if length < 0 or length > total_len:
            raise ValueError("invalid sequence length")
        m[i, :length] = True
    return m.view(b, 1, 1, total_len)


def prefix_mask_square(prefix_len: int, total_len: int) -> torch.Tensor:
    """
    PrefixLM-style mask: first prefix_len tokens fully bidirectional among themselves;
    remaining positions causal relative to all earlier tokens.
    """
    if prefix_len < 0 or prefix_len > total_len:
        raise ValueError("invalid prefix length")
    m = torch.zeros(total_len, total_len, dtype=torch.bool)
    m[:prefix_len, :prefix_len] = True
    for i in range(prefix_len, total_len):
        m[i, : i + 1] = True
    return m.view(1, 1, total_len, total_len)


class FlexibleMHA(nn.Module):
    """
    Multi-head attention with configurable KV head count.

    Self-attention: pass x only.
    Cross-attention: pass x as decoder states and kv as encoder memory.

    Mask convention: boolean tensor broadcastable to (B, 1, T_q, T_k); True keeps score.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        if self.n_kv_heads < 1 or n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.n_rep = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """(B, n_kv, T, d_k) -> (B, n_heads, T, d_k) by repeating along head axis."""
        if n_rep == 1:
            return x
        b, h_kv, t, d_k = x.shape
        x = x[:, :, None, :, :].expand(b, h_kv, n_rep, t, d_k)
        return x.reshape(b, h_kv * n_rep, t, d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t_q, _ = x.shape
        src = kv if kv is not None else x
        t_k = src.size(1)

        q = self.W_q(x).view(b, t_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(src).view(b, t_k, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(src).view(b, t_k, self.n_kv_heads, self.d_k).transpose(1, 2)

        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, t_q, self.n_heads * self.d_k)
        return self.W_o(out), weights


def count_kv_cache_floats(
    n_layers: int,
    batch: int,
    seq_len: int,
    n_kv_heads: int,
    d_head: int,
    bytes_per_float: int = 4,
) -> int:
    """
    Return approximate KV cache bytes for keys+values for inference-style caching.
    Keys and values each contribute n_kv_heads * seq_len * d_head entries per layer.
    """
    entries_per_layer = 2 * n_kv_heads * seq_len * d_head
    total = n_layers * batch * entries_per_layer * bytes_per_float
    return total


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model = 64
    n_heads = 4
    t = 6
    x = torch.randn(1, t, d_model)

    causal = causal_mask_square(t)
    mha = FlexibleMHA(d_model, n_heads=n_heads, n_kv_heads=n_heads)
    y_mha, w_mha = mha(x, mask=causal)
    print("MHA output:", tuple(y_mha.shape), "weights:", tuple(w_mha.shape))

    gqa = FlexibleMHA(d_model, n_heads=n_heads, n_kv_heads=2)
    y_gqa, _ = gqa(x, mask=causal)
    print("GQA output:", tuple(y_gqa.shape))

    mqa = FlexibleMHA(d_model, n_heads=n_heads, n_kv_heads=1)
    y_mqa, _ = mqa(x, mask=causal)
    print("MQA output:", tuple(y_mqa.shape))

    kv_params_mha = mha.W_k.weight.numel() + mha.W_v.weight.numel()
    kv_params_gqa = gqa.W_k.weight.numel() + gqa.W_v.weight.numel()
    print("KV parameter tensors (MHA vs GQA):", kv_params_mha, kv_params_gqa)

    enc = torch.randn(1, 10, d_model)
    dec = torch.randn(1, t, d_model)
    cross = FlexibleMHA(d_model, n_heads=n_heads, n_kv_heads=n_heads)
    mask_cross = torch.ones(1, 1, t, 10, dtype=torch.bool)
    y_cross, w_cross = cross(dec, mask=mask_cross, kv=enc)
    print("Cross-attn output:", tuple(y_cross.shape), "weights:", tuple(w_cross.shape))

    lengths = [3, 5, 4]
    total_len = 6
    pad_mask = padding_mask_from_lengths(lengths, total_len)
    x_pad = torch.randn(len(lengths), total_len, d_model)
    attn_pad = FlexibleMHA(d_model, n_heads=n_heads, n_kv_heads=n_heads)
    y_pad, _ = attn_pad(x_pad, mask=pad_mask)
    print("Padding-masked output:", tuple(y_pad.shape))

    floats_full = 32 * 2 * 2048 * 128
    floats_gqa8 = 8 * 2 * 2048 * 128
    print("Toy KV float count full MHA:", floats_full)
    print("Toy KV float count GQA-8:", floats_gqa8)
    print("Ratio:", floats_full / floats_gqa8)

    bytes_cached = count_kv_cache_floats(
        n_layers=32, batch=1, seq_len=2048, n_kv_heads=8, d_head=128
    )
    print("Approx KV cache bytes (illustrative):", bytes_cached)
```

The matrices printed in the narrative for \(T = 4\) are:

**Causal**

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

**Padding** (valid length 3, padded to 4)

```
1 1 1 0
1 1 1 0
1 1 1 0
1 1 1 0
```

**Prefix** (prefix length 2, total 4)

```
1 1 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Derive the attention output as softmax-weighted values and explain numerical scaling by \(\sqrt{d_k}\).
    *Answer:* \(\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V\): row \(i\) of the softmax matrix is weights over keys, and multiplying by \(V\) mixes value vectors. Scaling by \(\sqrt{d_k}\) keeps dot-product variance near \(O(1)\) when components of \(q,k\) are \(O(1)\), since \(\mathrm{Var}(q\cdot k)\) grows with \(d_k\); without it softmax saturates and gradients vanish.
    2. Contrast self-attention with cross-attention with concrete tensor shapes.
    *Answer:* Self-attention: \(Q,K,V\) all from the same tensor `(B, T, d)` → scores `(B, H, T, T)`. Cross-attention: \(Q\) from decoder `(B, T_dec, d)`, \(K,V\) from encoder `(B, T_enc, d)` → scores `(B, H, T_dec, T_enc)`; each decoder step attends over **source** positions only.
    3. Explain why multi-head attention helps optimization compared with one head of full width.
    *Answer:* Splitting \(d_{\text{model}}\) into \(h\) heads gives \(h\) separate low-rank attention maps in parallel (`d_k = d_{\text{model}}/h`), which factorizes different relation types (syntax, coreference, etc.) and typically optimizes better than one monolithic head—empirically heads **specialize**, improving both loss and interpretability.
    4. Compute KV-cache memory implications when moving from 32 KV heads to 8 KV heads at fixed sequence length.
    *Answer:* KV cache stores two tensors proportional to `n_kv_heads × T × d_head` per layer (times batch, layers, bytes). Going from 32 to 8 KV heads at fixed \(T\) and \(d_{\text{head}}\) cuts KV **fourfold** (e.g. ~16.8M vs ~4.2M floats per layer in a toy 32-head vs 8-head count at \(T=2048\), \(d_{\text{head}}=128\)).
    5. Describe causal masking and how parallel teacher forcing still respects autoregressive constraints.
    *Answer:* Causal mask zeros (or \(-\infty\) before softmax) attention from position \(i\) to keys \(j>i\). In training, **all** positions’ targets are known, but each position’s representation only sees the **past** keys, so the model never “cheats” with future tokens while you still batch parallel matmuls over the triangle—this is teacher forcing with a lower-triangular mask.
    6. Combine padding and causal masks: which logical operation combines them?
    *Answer:* Both are boolean “allowed” masks; you combine with **elementwise AND** (logical intersection): a key must be both **not future** (causal) and **not padding** to receive nonzero attention mass.
    7. Explain induction heads at a high level and how attention enables copying.
    *Answer:* Induction heads (from mechanistic interpretability) attend to **previous occurrences** of a pattern and promote **continuation**—supporting in-context copying like `[A][B] … [A] → [B]`. Attention implements this by high weights from the query position to earlier matching keys, pulling associated values forward.
    8. State the asymptotic complexity of attention in sequence length and when sparse approximations matter.
    *Answer:* Dense attention is \(O(T^2 d)\) per layer (dominant term: \(QK^\top\)). When \(T\) exceeds a few thousand to tens of thousands, quadratic cost and memory push toward **sparse** patterns (local windows, block-sparse), **linear** attention approximations, or alternative layers (SSMs) for long sequences.
    9. Describe how grouped-query attention repeats KV heads to match query head count.
    *Answer:* You compute `n_kv_heads` distinct \(K,V\) projections, then **repeat** each KV head `n_heads / n_kv_heads` times along the head axis so every query head sees a consistent key-value pair (same as MHA grouping). Queries stay fully per-head; only KV is shared within groups.
    10. Give a deployment scenario where MQA is attractive and a scenario where full MHA is still used.
    *Answer:* **MQA** (one KV for all heads): maximum KV-cache savings and bandwidth wins for **long-context, high-throughput inference** (serving chat at scale). **Full MHA** is still preferred when **quality** and flexible head specialization matter most and KV memory is not the bottleneck (e.g. short-context training or research baselines).

!!! interview "Follow-up Probes"
    - “What happens if you apply a causal mask in the encoder?” Expect discussion of information leakage and training–inference mismatch.
    - “How does attention differ from a convolution with kernel size one?” Expect locality and weight sharing contrast.
    - “Why not use one attention head with dimension \(d_{\text{model}}\)?” Expect optimization and specialization arguments.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Self-attention mixes value vectors using softmax-normalized QK compatibility scores.”
    - “Cross-attention lets the decoder query encoder memory at every step.”
    - “GQA reduces KV cache footprint by sharing keys and values across head groups.”
    - “Causal masking enforces autoregressive structure during parallel training.”

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Shazeer, *Fast Transformer Decoding: One Write-Head Is All You Need* (2019)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)
- Olsson et al., *In-Context Learning and Induction Heads* (2022)
