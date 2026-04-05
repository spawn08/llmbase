# Long-Context Modeling

## Why This Matters for LLMs

Attention-based Transformers have **quadratic** cost in sequence length \(T\): the attention matrix materializes scores between all pairs of positions. For long documents, code repositories, or multi-turn chats, that complexity and memory footprint dominate training and inference budgets. **Long-context modeling** is the umbrella for exact and approximate methods—FlashAttention kernels, sliding windows, sparse patterns, positional extrapolation, and distributed sequence parallelism—that make \(T\) in the hundreds of thousands feasible.

Second, **product requirements** increasingly ask for “whole book” or “whole repo” reasoning. Retrieval helps, but some tasks need **cross-paragraph coreference** that only a single forward pass over a long context can provide cheaply. Interviewers probe whether you understand **KV cache** growth \(O(L \cdot T \cdot d_{\text{head}})\) per layer and why **context length** is not free even when FLOPs are optimized.

Third, **positional encoding** assumptions break when you train at length \(L_{\text{train}}\) and deploy at \(L_{\text{test}} \gg L_{\text{train}}\). **RoPE scaling** methods (NTK-aware, YaRN) are now standard talking points. You should connect math (rotation frequencies) to engineering (which hyperparameters are tuned on validation perplexity).

---

## Core Concepts

### Standard Attention Complexity

For hidden size \(d\), heads \(h\), head dimension \(d_h = d/h\), batch \(B\), and length \(T\), self-attention costs:

- **FLOPs** (matmul-dominant, ignoring softmax): roughly \(O(B \cdot h \cdot T^2 \cdot d_h)\) for \(QK^\top\) and softmax\(\cdot V\).
- **Memory**: storing \(S \in \mathbb{R}^{T \times T}\) per head for backprop is prohibitive at large \(T\).

Thus both **compute** and **HBM bandwidth** become bottlenecks.

!!! math-intuition "In Plain English"
    Long-context pain is not “more tokens to embed”—it is the **all-pairs dating game** between positions. Shrink that pairwise work (sparsity, locality) or make it **IO-efficient** (kernel fusion) or **distribute** it across devices.

### FlashAttention: IO-Aware Exact Attention

**FlashAttention** (Dao et al.) computes exact softmax attention **without materializing** the full \(T \times T\) matrix in slow GPU high-bandwidth memory (HBM). The algorithm:

1. **Tiles** \(Q\), \(K\), \(V\) blocks in on-chip SRAM (fast).
2. Computes softmax **incrementally** using the **log-sum-exp** trick for numerical stability across blocks.
3. Fuses softmax with the \(V\) multiply to reduce HBM round trips.

For query block \(Q_i \in \mathbb{R}^{B_q \times d_h}\) and key blocks \(K_j\), the attention weights satisfy:

\[
\text{softmax}(Q_i K^\top) V = \sum_j \text{softmax}(Q_i K_j^\top) V_j
\]

FlashAttention streams over \(j\) while maintaining running statistics for the softmax denominator on SRAM-resident tiles.

Let \(m \in \mathbb{R}^{B_q}\) be per-row maxima and \(\ell \in \mathbb{R}^{B_q}\) be cumulative exponentials for partial blocks. When merging block \(j+1\), update:

\[
m' = \max(m, m_{j+1}),\quad
\ell' = e^{m - m'}\ell + e^{m_{j+1} - m'}\ell_{j+1}
\]

and adjust outputs accordingly—this is the **online softmax** algebra that makes blockwise computation exact.

!!! math-intuition "In Plain English"
    Standard attention writes a huge \(T \times T\) score matrix to GPU RAM repeatedly. FlashAttention keeps partial summaries in **fast scratchpad** memory and only spills smaller tiles—same math answer, fewer memory joules.

### Complexity Analysis: Arithmetic vs Memory

Define:

- \(F\) = total FLOPs for attention (still \(O(T^2)\) for full attention).
- \(M\) = HBM accesses.

FlashAttention reduces **\(M\)** dramatically; wall-clock improves when **memory-bound**. For very long \(T\) on modern accelerators, attention may become **compute-bound** again—profile before assuming kernels fix everything.

### Sliding Window Attention (e.g., Mistral)

**Sliding window** restricts each token to attend only to the previous \(w\) tokens (per layer or per head). Complexity drops to \(O(T \cdot w \cdot d)\) per head vs \(O(T^2 d)\).

\[
S_{ij} = \begin{cases}
\frac{\exp(q_i^\top k_j / \sqrt{d_h})}{\sum_{k \in \mathcal{W}(i)} \exp(q_i^\top k_j / \sqrt{d_h})} & j \in \mathcal{W}(i) \\
0 & \text{otherwise}
\end{cases}
\]

where \(\mathcal{W}(i) = \{i-w,\ldots,i\}\) (causal). **Stacking** layers yields **effective receptive field** roughly \(L \cdot w\) for \(L\) layers—global information propagates via depth.

!!! math-intuition "In Plain English"
    Sliding window is **local gossip**: each position only talks to neighbors, but gossip spreads through layers like telephone chains—cheap and surprisingly effective when combined with full attention on **some** layers.

### RoPE and Extrapolation: NTK-Aware, YaRN

**Rotary Position Embedding (RoPE)** encodes relative position by rotating query/key pairs in 2D subspaces:

\[
q_m' = R_{\Theta,m} q,\quad k_n' = R_{\Theta,n} k
\]

Attention logits depend on \(m-n\) through \(\langle R_{\Theta,m} q, R_{\Theta,n} k \rangle\). Base frequencies \(\theta_i\) control how quickly phase wraps with distance.

When extrapolating to \(T_{\text{test}} > T_{\text{train}}\), naive RoPE **overshoots** trained phases. **NTK-aware** scaling adjusts base frequencies or uses **non-linear** scaling maps; **YaRN** (Yet another RoPE extensioN) blends scaled and unscaled RoPE with a ramp to stabilize perplexity on long sequences.

### Ring Attention (Distributed Long Sequences)

For model-parallel or context-parallel training, **ring attention** circulates **key/value blocks** around a ring of devices so each GPU computes a shard of attention while receiving \(K,V\) from neighbors—avoiding a single device holding full \(T\) for all heads. Complexity per device improves relative to naive replication; collective communication patterns must hide latency.

### Sparse Attention Patterns: BigBird, Longformer

**Longformer** uses **local windows** + **global tokens** (e.g., CLS, special tokens attend everywhere). **BigBird** adds **random** sparse edges—enough for universal approximation properties under certain graph connectivity conditions.

Sparse patterns reduce \(T^2\) to \(T \cdot s\) for average sparsity \(s \ll T\). Implementation requires **block-sparse** kernels or custom CUDA; not all frameworks ship equal performance.

### Memory vs Compute at Different Context Lengths

| Regime | Dominant cost | Mitigations |
|--------|----------------|-------------|
| Short \(T\) | Compute under-utilization | Batch up |
| Medium \(T\) | HBM bandwidth | FlashAttention, fused kernels |
| Long \(T\) | KV cache size | GQA/MQA, quantization, paged attention (vLLM) |
| Very long \(T\) | Cross-device memory | Ring attention, context parallelism |

**Grouped-query attention (GQA)** shares \(K,V\) across head groups, shrinking KV cache roughly by group size—critical for serving.

### PagedAttention and Prefix Caching (Serving)

**vLLM** introduced **PagedAttention**: KV cache blocks are stored in **non-contiguous** pages like OS virtual memory, reducing fragmentation when batching requests of different lengths. **Prefix caching** reuses KV blocks for shared system prompts across users—critical for chat templates and tool schemas that repeat identically.

### ALiBi as an Alternative Bias

**Attention with Linear Biases (ALiBi)** adds a **head-specific** linear penalty to logits based on distance \(i-j\) instead of explicit position embeddings. It extrapolates to longer \(T\) without retraining positional tables in some setups—compare with RoPE-heavy stacks where YaRN is more common in open LLaMA-class models.

### Training vs Inference Trade-offs

During **training**, activation checkpointing trades compute for memory across layers; **tensor parallelism** shards \(d\) across GPUs. During **inference**, batching increases throughput but blows KV cache—**continuous batching** (iteration-level scheduling) interleaves prefill and decode steps to keep GPUs full.

### Memory Accounting: Attention Materialization

Naive attention **materializes** the score matrix \(S \in \mathbb{R}^{T \times T}\) per head before softmax. The following estimates **HBM footprint** for storing \(S\) and post-softmax weights (ignoring fusion):

```python
from __future__ import annotations


def attention_score_bytes(
    batch: int,
    heads: int,
    seq: int,
    bytes_per_elem: int = 2,
    include_softmax_output: bool = True,
) -> int:
    """
    FP16/BF16 = 2 bytes. Counts S = QK^T / sqrt(d) and optionally P = softmax(S).
    """
    per_head = seq * seq * bytes_per_elem
    layers = 2 if include_softmax_output else 1
    return batch * heads * per_head * layers


def kv_cache_bytes(
    batch: int,
    layers: int,
    heads_kv: int,
    seq: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    """K and V tensors cached per autoregressive step during decoding."""
    per_layer = 2 * batch * heads_kv * seq * head_dim * bytes_per_elem
    return layers * per_layer


if __name__ == "__main__":
    B, H, T, L, Dh = 1, 32, 8192, 80, 128
    print("Score+softmax bytes (FP16):", attention_score_bytes(B, H, T))
    print("KV cache bytes (FP16):", kv_cache_bytes(B, L, H, T, Dh))
```

FlashAttention avoids materializing full \(S\) in HBM by **fusing** softmax with the \(V\) multiply in SRAM-sized tiles—your profiler should show reduced **memory traffic**, not necessarily fewer FLOPs.

### Backward Pass and Recomputation

Training attention requires gradients through softmax. FlashAttention **recomputes** forward activations in tiles during the backward pass instead of storing full \(S\) in HBM—trading **extra compute** for **memory** (classic **rematerialization**). The net effect lowers **peak** memory, enabling longer \(T\) or larger batch sizes.

### Mistral-Style Sliding Window: Effective Receptive Field

If layer \(\ell\) uses window \(w\) and there are \(L\) layers, information can propagate roughly **\(O(L \cdot w)\)** steps along the sequence—still linear in \(T\) for fixed \(L,w\), but **global** mixing emerges across depth. Some architectures alternate **full** and **windowed** attention layers to balance quality and cost.

### Context Lengths in the Wild (Illustrative)

| Family | Typical context (order of magnitude) | Notes |
|--------|----------------------------------------|-------|
| GPT-4 class | 8k–128k tokens | Product-dependent tiers |
| Claude 3 | 200k tokens | Long document workflows |
| Open LLaMA / Mistral | 4k–32k extensible | YaRN/NTK extrapolation |

Always read **vendor system cards**—**usable** context may be lower after system prompts and tool schemas.

### Kernel Fusion and Roofline Perspective

Attention is often **memory-bound** on the **roofline model**:

\[
\text{Attainable FLOPs/s} = \min \bigl(\text{Peak FLOPs/s},\; \beta \cdot \text{Arithmetic Intensity}\bigr)
\]

where \(\beta\) is memory bandwidth. FlashAttention increases **arithmetic intensity** of the attention kernel by keeping tiles resident in SRAM—moving the bottleneck toward **compute** on some shapes.

### When Not to Use Exact Long Attention

If your task is **retrieval-heavy** (open-book QA), **RAG** plus moderate context may beat **naive** full attention over a million tokens on cost and quality. **Long-context** models shine when **cross-span reasoning** within a single document is unavoidable.

### Causal Masking and Memory Footprint

Decoder-only models apply a **causal** mask: position \(i\) attends only to \(j \le i\). **Training** still scales as \(O(T^2)\) in naive implementations for **compute**, but **FlashAttention** reduces **memory** traffic. **Inference** uses **KV cache** storing **past** keys and values—**memory grows linearly** with \(T\) per layer.

### Multi-Query and Grouped-Query Attention

**MQA** shares one \(K,V\) pair across all heads; **GQA** shares within **groups**. This shrinks **KV cache** for serving:

\[
\text{KV memory} \propto L \cdot B \cdot T \cdot d_{\text{kv}}
\]

where \(d_{\text{kv}}\) shrinks with sharing—critical for **long** chats at scale.

### YaRN: Blending Scaled and Unscaled RoPE

**YaRN** interpolates RoPE frequencies with a **ramp** across dimensions so **low** frequencies (long wavelengths) extrapolate more aggressively than **high** frequencies—stabilizing perplexity when extending context. Hyperparameters (\(\alpha, \beta\) scalars, **cutoff** dimensions) are tuned on **validation** long sequences.

### Ring Attention Communication Pattern

In **context parallelism**, devices hold **shards** of \(Q,K,V\) along sequence dimension and **rotate** \(K,V\) blocks in a ring. **Attention** is **mathematically** identical if all-to-all communication completes; **implementation** must **overlap** communication with compute to hide latency.

### Long-Context Debugging Tips

- **Spike** in loss at long positions → **positional** extrapolation issue (try YaRN/NTK).
- **OOM** at **prefill** → reduce **batch** or use **FlashAttention** / **chunked** prefill.
- **OOM** at **decode** → **KV cache** pressure—enable **GQA**, **KV quantization**, or **paged** attention.

### BigBird: Random, Window, and Global Attention

**BigBird** uses three edge types between tokens: **local windows**, a small set of **global** tokens attending everywhere, and **random** edges. The combination yields **universal approximation** properties under certain graph connectivity assumptions—**sparse** but **expressive** enough for some **long-sequence** tasks.

### Longformer: Dilated Sliding Windows

**Longformer** adds **dilated** patterns to sliding windows to widen receptive fields without dense \(T^2\) cost—implementation detail matters for **GPU** efficiency (block-sparse kernels).

### Positional Extrapolation Experiments

When evaluating **YaRN**/**NTK** scaling, measure:

1. **Perplexity** on **held-out** long documents.
2. **Needle-in-a-haystack** retrieval accuracy (does the model attend to **distant** facts?).
3. **Latency** and **memory** at target \(T\).

### Interaction with Quantization

**KV cache INT8/FP8** quantization reduces **memory** but can **degrade** long-range attention—**calibrate** on representative **chat** transcripts.

### FlashAttention Forward Pass (Step Outline)

1. **Tile** \(Q\) rows and \(K,V\) columns to SRAM-sized blocks.
2. For each \(Q\) tile, iterate \(K,V\) tiles along sequence dimension.
3. Maintain running **max** and **sum-exp** statistics for **numerically stable** softmax.
4. Accumulate **output** contributions from each \(V\) tile **without** writing full \(S\) to HBM.

This is **exact** (up to floating-point non-associativity) compared to **materialized** attention.

### Sliding Window + Full Attention Hybrids

Some models use **windowed** attention on **most** layers and **full** attention on **selected** layers or **tokens**—balancing **global** mixing with **efficiency**.

### Research Directions: Linear Attention

**Linear attention** approximations replace softmax kernels with **feature maps** enabling \(O(T)\) recurrence—trade-offs include **expressivity** and **hardware** support compared to **FlashAttention**-optimized softmax.

### Needle-in-a-Haystack (NIAH) Testing

**NIAH** embeds a **secret** fact deep in a long prompt and checks whether the model **retrieves** it in the answer—useful for **validating** positional extrapolation **beyond** training length. **False negatives** may indicate **attention** dilution or **positional** encoding failure.

### Document Pretraining vs Chat Fine-Tunes

**Base** models trained on **documents** may handle **long** contexts better than **chat-only** fine-tunes—**distribution** shift matters for **long-context** claims.

### Hardware Mapping: Tensor Cores and Shapes

Attention **kernels** are sensitive to **head** dimensions and **sequence** multiples of **8/16**—**padding** strategies can change **throughput** without changing **semantics**.

### Summary Formula Sheet

- **Attention FLOPs** (rough): \(O(B \cdot H \cdot T^2 \cdot d_h)\)
- **KV cache size** (rough): \(O(L \cdot B \cdot T \cdot d_{\text{kv}})\)
- **RoPE** encodes **relative** position via **rotations** in **2D** subspaces

---

## Interview Takeaways

- Full attention is \(O(T^2)\) in FLOPs **and** memory traffic; **FlashAttention** attacks memory bandwidth while preserving exact attention (up to numerics).
- **Sliding windows** and **sparse patterns** trade global pairwise attention for subquadratic structure; **depth** increases receptive field.
- **RoPE extrapolation** requires scaling tricks (**NTK-aware**, **YaRN**)—validate on long-held-out perplexity, not toy tasks only.
- **Ring attention** and **context parallelism** distribute \(T\) across devices when activations do not fit.
- **KV cache** dominates serving memory—**GQA/MQA** and **paged KV** (vLLM) are standard optimizations.
- Always distinguish **training** bottlenecks (activation checkpointing, FSDP) from **inference** (cache, batching).

## References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022): [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023): [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Jiang et al., *Mistral 7B* (2023) — sliding window discussion: [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Beltagy et al., *Longformer: The Long-Document Transformer* (2020): [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- Zaheer et al., *Big Bird: Transformers for Longer Sequences* (2020): [arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023): [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023): [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
