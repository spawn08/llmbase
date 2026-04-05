# KV Cache

## Why This Matters for LLMs

The **KV cache** is the single most important **algorithmic** optimization for autoregressive Transformer inference at scale. Without caching past keys and values, each new token would force the model to **re-project** the entire prefix through every attention layer, and the total attention work over generating \(T\) tokens scales like **\(O(T^2)\)** in the naive accounting that dominates wall-clock for long outputs. With a cache, each decode step **appends** one new \(K_t, V_t\) pair per layer (per KV head group) and runs attention with **query length one** against **all past** keys—turning the per-step cost from “recompute everything” into “**update** and **attend** linearly in current context length.” Every **systems** and **ML** interview that touches LLM serving expects you to explain this clearly, quantify **memory**, and connect to **GQA/MQA** and **PagedAttention**.

Second, **memory** not **FLOPs** often caps batch size and context length in production. The KV cache occupies **two** tensors per layer (\(K\) and \(V\)), each proportional to **number of KV heads**, **sequence length**, **head dimension**, **batch size**, and **dtype bytes**. A single long-context request can consume **multiple gigabytes** of HBM for KV alone; multiply by concurrent users and **quantization** of weights does not automatically shrink KV unless you **also** quantize or compress the cache. Interviewers ask you to **size** a deployment: “How many layers? FP16 or INT8 KV? GQA ratio?”—this page gives the **back-of-envelope** algebra and the vocabulary to defend answers.

Third, **prefill** versus **decode** phases have different **bottlenecks**. **Prefill** processes the entire prompt in parallel (often **compute-bound** on tensor cores). **Decode** generates one token at a time (often **memory-bandwidth bound** moving weights and **KV** tensors). **PagedAttention**-style runtimes treat KV like **virtual memory pages** to reduce fragmentation and enable **non-contiguous** storage for variable-length sequences. Understanding **layout** \((B, H, L, d)\) vs \((B, L, H, d)\) matters for **kernel** performance and for **correct** integration with **FlashAttention**-class fused kernels. This page ties the **math** of attention to the **systems** story you need on a whiteboard.

---

## Core Concepts

### Why Cache \(K\) and \(V\)?

At decoder layer \(\ell\), for hidden state \(\mathbf{h}_t\) at position \(t\), projections yield

\[
\mathbf{q}_t^{(\ell)} = W_Q^{(\ell)} \mathbf{h}_t,\quad
\mathbf{k}_t^{(\ell)} = W_K^{(\ell)} \mathbf{h}_t,\quad
\mathbf{v}_t^{(\ell)} = W_V^{(\ell)} \mathbf{h}_t.
\]

!!! math-intuition "In Plain English"
    **Q** asks “what am I looking for at this new position?” **K** and **V** are **content** that past positions expose to the attention mechanism. For **causal** decoding, **past** positions **do not change** when you append a new token—so their **projected** \(K\) and \(V\) rows are **reusable** once computed.

For **causal** self-attention on a sequence of length \(L\), attention weights for position \(t\) use **only** keys and values from positions \(\le t\). When generating token-by-token, the **new** token \(t\) introduces **new** rows \(K_t, V_t\); all **previous** rows \(K_{1:t-1}, V_{1:t-1}\) are **identical** to what they were at step \(t-1\) (given fixed weights and deterministic ops). Hence **store** them.

### Attention With Cached \(K, V\) (Single Head)

Let \(K \in \mathbb{R}^{L \times d_h}\) and \(V \in \mathbb{R}^{L \times d_h}\) store **all** positions up to length \(L\). For the **new** query \(\mathbf{q}_L \in \mathbb{R}^{1 \times d_h}\):

\[
\mathbf{a}_L = \mathrm{softmax}\left(\frac{\mathbf{q}_L K^\top}{\sqrt{d_h}}\right) V.
\]

!!! math-intuition "In Plain English"
    The **new** token only needs **one** query vector, but it **attends** to **all** past keys. The **softmax** is over **length \(L\)**—that is **linear in \(L\)** for this matmul **given** materialized \(K, V\).

### Naive Reforward Cost Sketch

If each step \(t\) recomputed \(K_{1:t}\) and \(V_{1:t}\) from **scratch** by re-running layers on **length-\(t\)** prefixes, the **sum** of work across \(t=1..T\) behaves like **\(O(T^2)\)** in the **attention-dominated** regime (each step pays \(\sim t\) work). **Caching** avoids recomputation of **past** \(K,V\) so **per-step** attention cost scales **\(O(t)\)** at step \(t\) for that layer.

!!! math-intuition "In Plain English"
    Think of paying **\(1+2+3+\cdots+T = O(T^2)\)** if you redo all prefix lengths every time. **KV cache** stops **re-paying** old positions.

### Memory Footprint (MHA)

For **one** layer, **batch** \(B\), **\(H\)** KV heads (standard multi-head attention, **\(H\)** key heads and **\(H\)** value heads), head dimension \(d_h\), sequence length \(L\), **FP16** element size \(s=2\) bytes:

\[
\text{Bytes}_{K,V} \approx B \cdot 2 \cdot H \cdot L \cdot d_h \cdot s.
\]

The leading **2** counts **both** \(K\) and \(V\).

!!! math-intuition "In Plain English"
    **Linear** in **\(L\)** and **linear** in **\(H\)**. Doubling context **doubles** KV bytes for that layer (same \(B\) and dtype). **GQA** shrinks **effective \(H\)** for **KV** only.

### Multi-Query Attention (MQA)

MQA uses **one** shared key head and **one** shared value head for all query heads:

\[
H_Q = H,\quad H_{KV} = 1.
\]

!!! math-intuition "In Plain English"
    **All** query heads **read** the **same** cached \(K,V\) (broadcast along head dimension in the implementation). **KV** memory drops by ~\(H\) vs MHA; **quality** may drop if not trained for it—modern models often **train** with MQA/GQA from the start.

### Grouped-Query Attention (GQA)

GQA splits \(H\) query heads into \(G\) **groups**; each group shares one KV head:

\[
H_Q = H,\quad H_{KV} = G,\quad 1 \le G \le H.
\]

!!! math-intuition "In Plain English"
    **MHA** is \(G=H\); **MQA** is \(G=1\). **GQA** is the **interpolating** design that recovers much of MHA quality with **substantial** KV savings.

### KV Cache With GQA — Memory Scales With \(H_{KV}\)

Replace \(H\) in the KV storage formula with **\(H_{KV}\)**:

\[
\text{Bytes}_{K,V} \approx B \cdot 2 \cdot N_{\text{layer}} \cdot H_{KV} \cdot L \cdot d_h \cdot s.
\]

!!! math-intuition "In Plain English"
    **Query** heads stay **\(H\)** for expressivity; **KV** cache **only** stores **\(H_{KV}\)** heads—this is why **LLaMA 2**-class models can run long contexts **without** linearly growing KV with **\(H\)**.

!!! example "Worked Example: KV Cache Memory (LLaMA-2 7B–Style Order of Magnitude)"
    Use **FP16** (\(s=2\) bytes). Suppose **\(N_{\text{layer}} = 32\)**, **\(H = 32\)** query heads, **\(d_h = 128\)**, **\(L = 4096\)**, **\(B = 1\)**.

    **Per layer, MHA** (full KV heads = 32):

    \[
    2 \times 32 \times 4096 \times 128 \times 2 \ \text{bytes} = 67{,}108{,}864\ \text{bytes} \approx 64\ \text{MiB}.
    \]

    **Total KV cache (all layers)**:

    \[
    32 \times 64\ \text{MiB} \approx 2{,}048\ \text{MiB} \approx 2.0\ \text{GiB}.
    \]

    **Batch \(B=8\)** (naive independent caches, same \(L\)): multiply by **8** → about **16 GiB** **just** for KV in this toy accounting (real checkpoints differ slightly in \(H, d_h, N\)).

    **Takeaway**: **KV** dominates **GPU RAM** for long contexts at moderate batch.

!!! example "Worked Example: GQA Cache Savings (8 KV Heads)"
    Keep **\(N_{\text{layer}} = 32\)**, **\(L = 4096\)**, **\(d_h = 128\)**, **FP16**, **\(B=1\)**, but set **\(H_{KV} = 8\)** (8 KV heads instead of 32).

    Per layer:

    \[
    2 \times 8 \times 4096 \times 128 \times 2 = 16{,}777{,}216\ \text{bytes} \approx 16\ \text{MiB}.
    \]

    Total:

    \[
    32 \times 16\ \text{MiB} = 512\ \text{MiB}.
    \]

    Ratio vs MHA 32-head KV: **\(64\ \text{MiB} / 16\ \text{MiB} = 4\)** per layer → **~4×** KV reduction **for the cache** when moving from 32 KV heads to 8.

### PagedAttention (vLLM)

PagedAttention allocates KV storage in **fixed-size blocks** (pages). Logical token positions map to **non-contiguous** physical blocks via a **block table**, analogous to **virtual memory**.

!!! math-intuition "In Plain English"
    **Pre-allocating** a dense tensor of shape **\(L_{\max}\)** wastes memory for short sequences; **pages** reuse **physical** memory flexibly and reduce **fragmentation** when **batching** many requests of different lengths.

??? deep-dive "Deep Dive: Copy-on-Write and Shared Prefixes"
    When two requests share an **identical prompt prefix**, **KV blocks** for that prefix can be **reference-counted** and **shared** until the first **divergent** token. This is **prefix caching** at the **block** level—reduces **TTFT** and **memory** for templated system prompts.

### KV Cache Compression (Overview)

Techniques include **INT8/FP8 KV** with per-tensor scales, **token dropping** (approximate), and **low-rank** or **quantized** factorizations of \(K,V\) blocks.

\[
K_{\text{quant}} = \mathrm{round}(K / s),\quad s > 0.
\]

!!! math-intuition "In Plain English"
    Quantization is **lossy compression**. The **risk** is attention **score** errors that accumulate across layers—**calibration** on representative workloads matters.

??? deep-dive "Deep Dive: StreamingLLM and Attention Sinks"
    Some long-context systems use **windowed** KV or **retention** policies that keep **initial** tokens plus a **sliding** window because **attention sink** phenomena concentrate mass on **early** positions—**eviction** policies must be validated **per model family**.

### Prefill vs Decode Phases

- **Prefill**: process all prompt tokens (often **high** parallel arithmetic intensity).
- **Decode**: **one** new token per step; **memory-bound** on weights + KV **moves** at small batch.

\[
T_{\text{decode,step}} \approx f_{\text{MLP}} + f_{\text{Attn}}(L)
\]

where \(f_{\text{Attn}}(L)\) grows with **context length** \(L\) due to growing **KV** and longer softmax vectors.

!!! math-intuition "In Plain English"
    **Optimizers** pick different **kernel** strategies for **prefill** (large matmuls) vs **decode** (thin matmuls + bandwidth-heavy memory reads).

### RoPE and Cached Keys

With **rotary** position embeddings \( \mathrm{RoPE}(\cdot, \text{pos}) \), keys are stored **after** rotation for their **absolute** position index \(m\):

\[
\mathbf{k}_m \leftarrow W_K \mathbf{h}_m \text{ then RoPE at } m.
\]

!!! math-intuition "In Plain English"
    When you **append** position \(L\), you **only** compute RoPE for **that** row—**cached** rows remain **valid** for their **original** indices.

---

## Code

Below is a **self-contained** PyTorch module demonstrating **prefill** (sequence length \(>1\)) then **decode** (one token per step) with an explicit **`KVCache`** that **concatenates** along sequence length. It uses **one** decoder block and **single-head** attention for clarity; production models stack **\(N\)** layers and use **multi-head** + **RoPE** + **FlashAttention** kernels.

```python
"""
kv_cache_demo.py — minimal KV cache for causal self-attention (educational).
Run: python kv_cache_demo.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    """Stores K, V with shape (batch, seq_len, d_head)."""

    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        # k_new: (B, T_new, d_head)
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=1)
            self.v = torch.cat([self.v, v_new], dim=1)

    @property
    def length(self) -> int:
        return 0 if self.k is None else int(self.k.size(1))


class TinyDecoderBlock(nn.Module):
    """Single-head causal self-attention + residual (no RoPE — teaching layout only)."""

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
        x: (B, T_new, d_model)
        If use_cache: append new K,V to cache and compute attention for all queries in this pass.
        """
        if cache is None:
            cache = KVCache()

        h = self.norm(x)
        q = self.wq(h)
        k_new = self.wk(h)
        v_new = self.wv(h)

        if use_cache:
            cache.append(k_new, v_new)
            K, V = cache.k, cache.v
            assert K is not None and V is not None
            scores = torch.matmul(q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
            # Causal: for prefill T_new>1, mask future positions; for decode T_new=1, allow all past keys
            B, Tq, _ = q.shape
            Tk = K.size(1)
            mask = torch.tril(torch.ones(Tq, Tk, device=x.device, dtype=torch.bool))
            # When Tq==1 (decode), mask is all ones up to Tk
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, V)
        else:
            T = q.size(1)
            scores = torch.matmul(q, k_new.transpose(-2, -1)) / math.sqrt(self.d_head)
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v_new)

        out = self.wo(ctx)
        return x + out, cache if use_cache else None


def bytes_for_kv_mha(
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    batch: int,
    bytes_per_elem: int = 2,
) -> int:
    """Order-of-magnitude KV bytes (K+V) for all layers."""
    per_layer = 2 * num_kv_heads * seq_len * head_dim * bytes_per_elem
    return batch * num_layers * per_layer


def demo() -> None:
    torch.manual_seed(0)
    B, d_model, d_head = 1, 64, 64
    block = TinyDecoderBlock(d_model, d_head)

    # Prefill prompt length 4
    x_prefill = torch.randn(B, 4, d_model)
    y, cache = block(x_prefill, cache=None, use_cache=True)
    assert cache is not None
    print("Prefill cache length:", cache.length)

    # Decode 3 steps (one token each)
    for step in range(3):
        x_step = torch.randn(B, 1, d_model)
        y, cache = block(x_step, cache=cache, use_cache=True)
        print(f"decode step {step}: cache len {cache.length}, y shape {tuple(y.shape)}")

    # Report memory formula (MHA num_kv_heads = 1 in this toy single-head block)
    b = bytes_for_kv_mha(
        num_layers=1, num_kv_heads=1, seq_len=cache.length, head_dim=d_head, batch=B
    )
    print("Approx KV bytes (single layer, this toy head):", b)


if __name__ == "__main__":
    demo()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Explain** why autoregressive Transformer inference without KV caching leads to **\(O(T^2)\)** aggregate attention work for length \(T\).
    2. **Write** the formula for KV cache memory with **batch**, **layers**, **KV heads**, **head dim**, **sequence length**, and **dtype**.
    3. **How** does **GQA** reduce KV cache size compared to **MHA** without changing query head count?
    4. **Contrast** **prefill** and **decode** bottlenecks on a modern GPU.
    5. **What** is **PagedAttention** solving, and why does **contiguous** preallocation waste memory?
    6. **How** do **RoPE**-equipped models store **cached keys** as the sequence grows?
    7. **Why** might **INT8 KV** be attractive, and what is the main **accuracy** risk?
    8. **Describe** **prefix caching** / shared KV blocks for identical prompts across users.
    9. **What** happens to **KV** memory when you **double** context length from 4k to 8k (same batch)?
    10. **How** does **beam search** interact with **KV** memory (multiple hypotheses)?

!!! interview "Follow-up Probes"
    - **Probe A**: “If HBM is the limit, do you quantize **weights** or **KV** first—and why?”
    - **Probe B**: “How does **continuous batching** interact with **KV** block allocation?”
    - **Probe C**: “Why does **FlashAttention** not remove the need for a **KV cache** at decode?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**KV cache stores past keys and values so we do not recompute attention for earlier tokens.**”
    - “**Per-step attention cost grows linearly with context length given materialized KV.**”
    - “**GQA reduces KV heads, not query heads—memory scales with \(H_{KV}\).**”
    - “**PagedAttention maps logical sequences to physical KV blocks to reduce fragmentation.**”
    - “**Prefill is often compute-bound; decode is often memory-bandwidth bound.**”

---

## References

1. **Vaswani et al., “Attention Is All You Need” (NeurIPS 2017)** — attention and autoregressive decoding foundations: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
2. **Noam Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need” (2019)** — Multi-Query Attention: [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150).
3. **Joshua Ainslie et al., “GQA: Training Generalized Multi-Query Transformer Models” (2023)** — grouped-query attention: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
4. **Reiner Pope et al., “Efficiently Scaling Transformer Inference” (2023)** — KV cache and multi-query serving: [arXiv:2211.05102](https://arxiv.org/abs/2211.05102).
5. **Woosuk Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention” (SOSP 2023)** — vLLM paging: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
6. **Tri Dao et al., “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning” (2023)** — IO-aware attention: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
