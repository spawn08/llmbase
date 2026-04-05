# KV Cache

## Why This Matters for LLMs

Transformer decoders compute, for each layer and head, **queries**, **keys**, and **values** from hidden states. In **training**, the full sequence is known and attention can be computed in parallel across positions (subject to causal masking). In **inference**, you generate **one new token** at a time. Without caching, you would re-run the entire prefix through every layer on every step, making the cost of generating \(T\) tokens **quadratic** in \(T\) in total FLOPs for attention-dominated regions—and **quadratic** in a painful way for memory traffic.

The **KV cache** stores, for each layer, the **key** and **value** tensors for **all past tokens** so that at step \(t\) you only compute \(Q_t, K_t, V_t\) for the **new** position and **concatenate** \(K,V\) along the sequence dimension. That reduces per-step compute from \(O(t^2)\) back toward **linear** in total sequence length for the attention block (with caveats for very long context and memory bandwidth). Every systems interview that mentions “**inference optimization**” expects KV cache fluency.

A second reason is **memory planning**. The cache size grows with **layers**, **heads**, **head dimension**, **batch size**, and **sequence length**. Serving frameworks run out of **GPU HBM** long before FLOPs if you batch many long prompts—this is why **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)** exist: they shrink the **per-token** \(K,V\) footprint by sharing heads across queries.

Third, **eviction** and **paging** (see continuous batching) treat the KV cache like **GPU virtual memory**—understanding the tensor layout is prerequisite for **PagedAttention** and long-context products.

---

## Core Concepts

### Autoregressive Attention Without Cache — \(O(T^2)\) Total Work

Consider a single layer, single head (omit batch). At generation step when the prefix has length \(t\), attention maps need scores between the **new** query at position \(t\) and **all** keys \(1..t\). If you **recomputed** keys and values for positions \(1..t-1\) from scratch each time by projecting **full** hidden rows, you would repeat work proportional to **\(1 + 2 + \cdots + T = O(T^2)\)** across steps for that layer’s attention pathway.

!!! math-intuition "In Plain English"
    - **Naive** “full forward on full prefix every step” is wasteful: earlier positions’ representations do not change once generated (in standard sampling) except through **new** tokens appended—so **K,V** for old positions are **invariant** and should be **memoized**.

### KV Cache Mechanism

At step \(t\), let hidden state for new token be \(\mathbf{h}_t\). For layer \(\ell\),

\[
\mathbf{q}_t^{(\ell)} = W_Q^{(\ell)} \mathbf{h}_t,\quad
\mathbf{k}_t^{(\ell)} = W_K^{(\ell)} \mathbf{h}_t,\quad
\mathbf{v}_t^{(\ell)} = W_V^{(\ell)} \mathbf{h}_t.
\]

The cache stores tensors \(K^{(\ell)}, V^{(\ell)}\) with shapes growing along sequence dimension:

\[
K^{(\ell)} \in \mathbb{R}^{t \times d_h},\quad V^{(\ell)} \in \mathbb{R}^{t \times d_h}
\]

(for one head; **multi-head** stacks a head dimension). You **append** \(\mathbf{k}_t, \mathbf{v}_t\) to the cache instead of recomputing from old hidden states.

Attention for the new token (single head) becomes:

\[
\mathbf{a}_t = \mathrm{softmax}\left(\frac{\mathbf{q}_t K^{(\ell)\top}}{\sqrt{d_h}}\right) V^{(\ell)}.
\]

!!! math-intuition "In Plain English"
    - **Query** always for the **current** position only (when generating one token).
    - **Keys/values** for past positions are **frozen** in cache—**no** backward pass through earlier layers for old tokens on this step.

### Memory Footprint Formula

For **one** layer, **one** sample in batch, **multi-head** attention with \(H\) heads, head dim \(d_h\), sequence length \(L\), **separate** \(K\) and \(V\) per head:

\[
\text{bytes} \approx 2 \times H \times L \times d_h \times \text{bytes\_per\_elem}
\]

(2 for \(K\) and \(V\)). With \(N_{\text{layer}}\) layers, multiply by \(N_{\text{layer}}\). With batch \(B\), multiply by \(B\) if each sequence has its own cache (padded to common \(L_{\max}\) in static batching).

**Common compact form** (element size suppressed):

\[
\text{KV cache} \propto 2 \cdot N_{\text{layer}} \cdot H \cdot d_h \cdot L \cdot B.
\]

!!! math-intuition "In Plain English"
    - **FP16**: 2 bytes/elem → **halve** vs FP32; **INT8 KV** halves again (with accuracy trade-offs).
    - Long \(L\) (128k context) explodes **linearly** in \(L\)—this motivates **compression**, **quantized KV**, and **offloading** to CPU.

### Multi-Query Attention (MQA)

**MQA** (Shazeer, 2019) uses **one** shared key/value head across all query heads:

\[
\text{\#KV heads} = 1,\quad \text{\#Q heads} = H.
\]

Cache size for \(K,V\) drops by ~\(H\) vs standard MHA (per layer), while queries stay multi-head.

!!! math-intuition "In Plain English"
    - **Pros**: much smaller KV cache; faster **inference** memory bandwidth.
    - **Cons**: slightly reduced **expressivity**; training may need **up-knowledge** from MHA teacher or careful regularization—many models are **trained** with MQA from scratch now.

### Grouped-Query Attention (GQA)

**GQA** (Ainslie et al., 2023) interpolates: partition \(H\) query heads into \(G\) groups, each group shares one \(K,V\) pair. **MHA** is \(G=H\); **MQA** is \(G=1\).

\[
\text{KV heads} = G,\quad 1 \le G \le H.
\]

!!! math-intuition "In Plain English"
    - **Sweet spot** for LLaMA 2/3-class models: **near-MHA quality** with **MQA-like** memory wins.

### Cache Eviction Strategies

When context exceeds hardware limits, systems may:

- **Windowing**: keep only last \(W\) tokens’ KV (forget early history)—**approximate** for true long-doc QA.
- **Sliding attention patterns**: some long-context models use **sparse** masks; caches align with **block** structure.
- **Offloading**: store older \(K,V\) in **CPU RAM** or **NVMe**, prefetch windows—adds **latency** but raises **effective** context.
- **Compression / quantization**: INT8 or **FP8** KV; **per-channel** scales for \(K,V\).

!!! math-intuition "In Plain English"
    - **Eviction** is always a **policy** on **what information to drop**—interacts with **attention sink** phenomena and **system** SLOs.

---

## Manual KV Cache in PyTorch

Below is a **minimal** educational implementation: prefill several tokens, then decode one step at a time, **caching** \(K,V\) per layer for a **single-head** simplified block. A full GPT uses **stacked** layers and **rotary** positions; the **cache layout** concept is identical.

```python
"""
Minimal single-head Transformer block with KV cache for autoregressive steps.
Educational: no RoPE, one layer, batch size 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    """Stores K, V tensors appended along sequence dim: shape (1, T, d_head)."""
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        # k_new, v_new: (1, 1, d_head) for one new token
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=1)
            self.v = torch.cat([self.v, v_new], dim=1)


class TinyDecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.d_head = d_head
        self.wq = nn.Linear(d_model, d_head, bias=False)
        self.wk = nn.Linear(d_model, d_head, bias=False)
        self.wv = nn.Linear(d_model, d_head, bias=False)
        self.wo = nn.Linear(d_head, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        x: (1, T_new, d_model) — either full prefill (T_new>1) or one token (T_new=1)
        """
        if cache is None:
            cache = KVCache()

        h = self.norm(x)
        q = self.wq(h)
        k_new = self.wk(h)
        v_new = self.wv(h)

        if use_cache:
            # Append new keys/values
            cache.append(k_new, v_new)
            K, V = cache.k, cache.v
            assert K is not None and V is not None
            # Attention for ALL positions in this forward (prefill) or last pos (decode)
            # scores: (1, T_q, T_total)
            scores = torch.matmul(q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
            # Causal mask: for prefill, standard lower-triangular; for 1-token decode, no future
            T_total = K.size(1)
            T_q = q.size(1)
            mask = torch.ones(T_q, T_total, device=x.device).tril(diagonal=T_total - T_q)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, V)
        else:
            T = q.size(1)
            scores = torch.matmul(q, k_new.transpose(-2, -1)) / math.sqrt(self.d_head)
            causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v_new)

        out = self.wo(ctx)
        return x + out, cache if use_cache else None


def demo_prefill_then_decode() -> None:
    torch.manual_seed(0)
    d_model = 64
    d_head = 64
    block = TinyDecoderBlock(d_model, d_head)

    # Prefill prompt of length 4
    x_prefill = torch.randn(1, 4, d_model)
    y, cache = block(x_prefill, cache=None, use_cache=True)
    assert cache is not None
    print("After prefill, cache length:", cache.k.shape[1])

    # Decode 3 steps with single-token forwards
    x_t = y[:, -1:, :]  # last hidden from upstream (toy: identity path omitted)
    for step in range(3):
        x_step = torch.randn(1, 1, d_model)  # would be embedding of new token
        x_t, cache = block(x_step, cache=cache, use_cache=True)
        print(f"step {step}: cache len {cache.k.shape[1]}")


if __name__ == "__main__":
    demo_prefill_then_decode()
```

!!! math-intuition "In Plain English"
    - Real **Hugging Face** models return `past_key_values` as a nested tuple per layer; **FlashAttention-2** fuses the **append** path for speed.
    - **GQA/MQA** change **how many** \(K,V\) tensors you cache per layer—**not** the **append** semantics.

### Complexity Summary

Let \(L\) be total sequence length after \(L\) steps, \(N\) layers, hidden size \(d\), head dim \(d_h\), \(H\) heads.

- **Per new token** (with cache): attention is \(O(H \cdot L \cdot d_h)\) for QK\(^\top\) against **full** past length \(L\) per layer → **linear in \(L\)** per step, **quadratic** in \(L\) **total** if you sum \(1..L\) (still true globally).
- **Memory**: \(O(N \cdot H \cdot L \cdot d_h)\) for \(K,V\) storage dominates many deployments.

### Multi-Head Tensor Shapes (Batch View)

For batch \(B\), \(H\) heads, head dimension \(d_h\), sequence length \(L\), separate KV per head (**MHA**):

\[
K, V \in \mathbb{R}^{B \times N_{\text{layer}} \times H \times L \times d_h}
\]

Some implementations transpose to **(B, L, N, H, d_h)** for fused kernels—**logical** content is the same.

!!! math-intuition "In Plain English"
    - **Layout** matters for **memory bandwidth**: contiguous access along `L` or `H` changes **which** kernels win on A100 vs H100.
    - **GQA**: replace \(H\) in the **KV** slot with \(G \ll H\) and **broadcast** \(K,V\) across the **query** heads in each group.

### Rotary Embeddings (RoPE) and Cached Keys

Modern LLaMA-class models apply **RoPE** to \(Q,K\) **before** attention. Positions \(m\) and \(n\) enter through a rotation that depends on **absolute** index. At inference, **cached** \(K\) tensors store **already rotated** keys for each absolute position—when you append token at position \(L\), you compute \(K_L\) with RoPE angle for index \(L\) only.

!!! math-intuition "In Plain English"
    - You **do not** “re-rotate” old keys when the sequence grows—each row was stored **final** for its index.
    - **Relative** schemes still need **consistent** position bookkeeping across **prefill** and **decode**—bugs here show up as **garbled** long outputs.

### Worked Example: Memory for 7B-Class Model (Order-of-Magnitude)

Assume **FP16** (2 bytes), **\(N_{\text{layer}} = 32\)**, **\(H_{\text{kv}} = 8\)** (GQA with 8 KV heads), **\(d_h = 128\)**, **\(L = 8192\)**, **\(B = 1\)**:

\[
\text{KV bytes} \approx 2 \cdot N_{\text{layer}} \cdot 2 \cdot H_{\text{kv}} \cdot L \cdot d_h \cdot 2
\]

Here the middle **2** is **\(K\) and \(V\)**. Plugging numbers:

\[
\approx 2 \times 32 \times 2 \times 8 \times 8192 \times 128 \times 2 \ \text{bytes} \approx 2^{31}\ \text{bytes} \approx 2\ \text{GiB}
\]

(order-of-magnitude; real checkpoints differ). Scale **linearly** with **batch** and **context**.

!!! math-intuition "In Plain English"
    - This back-of-envelope is how **serving** teams decide **max batch × max context** on an **80GB** card—often **KV** not **FLOPs** binds first.

### Interaction with FlashAttention

**FlashAttention** (training or prefill) computes attention **in tiles** without materializing full \(L \times L\) scores in HBM. With **KV cache** at decode, **FlashDecoding** variants batch **query length 1** against **long** \(K,V\) efficiently. The **asymptotic** \(O(L)\) per step remains, but **constants** improve.

### When Cache Does Not Help as Much

- **Encoder-decoder** models: **cross-attention** to encoder states also needs **caching** of encoder-side projections—similar ideas, different tensor names.
- **Sliding-window** models: only last \(W\) \(K,V\) **exist** by construction—memory **caps** at \(W\).

### Prefix Caching (Share Prompt KV Across Users)

If many requests share the **same system prompt** or **RAG** prefix, **prefix KV** tensors are **identical** until the first **divergent** token. Serving systems can **hash** the prefix and **reuse** cached \(K,V\) blocks—dramatically cutting **TTFT** and **compute** for templated assistants.

!!! math-intuition "In Plain English"
    - This is **not** a new attention math—purely a **storage** + **scheduling** optimization.
    - **Security**: shared prefixes must not leak **cross-tenant** state—**isolation** is implementation detail, not tensor math.

### Batch Dimension and Padding Waste

Static batching pads shorter sequences to \(L_{\max}\) in the batch. **KV cache** for padded positions is **wasted HBM** unless **masking** removes their contribution (they still **consume** memory in naive layouts). **PagedAttention** (Part 4 sibling page) removes **physical** contiguity requirements so padding waste drops.

### Numerical Stability and Mixed Precision

Caches in **FP16** can **overflow** rare attention logits if **scale** drifts; frameworks often keep **master** weights in FP16 but apply **K/V** in FP16 with **softmax** in FP32. **INT8 KV** requires **per-tensor** or **per-channel** scales updated during **calibration**.

\[
K_{\text{int8}} = \mathrm{round}\bigl(K / s_K\bigr),\quad s_K > 0.
\]

Dequantize at attention matmul:

\[
Q K^\top \approx (Q_{\text{fp16}}) (s_K \cdot \mathrm{dequant}(K_{\text{int8}}))^\top.
\]

!!! math-intuition "In Plain English"
    - **Quantized KV** is a **compression** codec for HBM—**perplexity** impact usually smaller than **weight** INT4, but **mis-calibration** causes **catastrophic** attention mistakes.

### Checklist: What to Say in a Systems Interview

1. **Problem**: autoregressive decode **recomputes** past \(K,V\) without cache → **waste**.
2. **Fix**: **append-only** cache per layer; **Q** for **new** token only.
3. **Cost**: **linear** in \(L\) **per step** attention vs **quadratic** aggregate over generation; **memory** **linear** in \(L\).
4. **GQA/MQA**: reduce **KV head** count → **linear** memory savings in **KV heads**.
5. **PagedAttention**: **non-contiguous** physical blocks for **logical** sequences—see [Continuous Batching](continuous_batching.md).

### Algebraic Identity: Cached Step vs Full Reforward

Let \(f_\theta\) be the full LM map from tokens to next logits. **Caching** exploits

\[
f_\theta(w_{1:t}) = g_\theta\bigl(w_t,\, \mathrm{Cache}(w_{1:t-1})\bigr),
\]

where \(g\) applies **one** block update using **stored** \(K,V\). The **equality** holds for **Transformer** decoders without **recurrent** state besides attention cache—**LayerNorm** and **RoPE** must be applied **consistently** with position indices.

*(End of core concepts — see continuous batching for how schedulers multiplex many caches on one GPU.)*

---

## Interview Takeaways

- **Without** KV reuse, autoregressive inference repeats \(O(t)\) work at each step \(t\) → **\(O(T^2)\)** aggregate attention cost for length \(T\).
- **KV cache** stores past **keys** and **values** per layer (and per head group in GQA); **queries** are computed only for **new** positions.
- **Cache memory** scales as **\(2 \times N_{\text{layer}} \times (\text{KV heads}) \times L \times d_{\text{head}} \times B\)** (times dtype bytes).
- **MQA** shares one \(K,V\) across all query heads; **GQA** uses **groups**—interpolates memory vs quality.
- **Eviction/offload/quantize** are **policies** when \(L\) exceeds GPU RAM—each has **latency** and **quality** trade-offs.
- Connect to **PagedAttention**: physical KV blocks need not be contiguous—**virtual** contiguous **logical** sequences.

## References

- Vaswani et al. (2017), *Attention Is All You Need*
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* — MQA
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models* — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Pope et al. (2023), *Efficiently Scaling Transformer Inference* — [MLSys](https://arxiv.org/abs/2211.05102)
- Dao et al., *FlashAttention-2* — fused attention with IO-awareness
