# Long-Context Modeling

## Why This Matters for LLMs

Transformer **self-attention** is **quadratic** in sequence length \(T\): every position can attend to every other position, so FLOPs and memory traffic scale as \(O(T^2)\) per layer (before IO-aware kernels). **Long-context modeling** is the engineering and algorithm stack—FlashAttention, sliding windows, sparse patterns, RoPE scaling, distributed sequence parallelism—that makes **128K–1M+** token contexts feasible on modern hardware. Interviewers expect you to connect **math** (what grows with \(T\)) to **symptoms** (OOM during prefill, KV cache exhaustion during decode) and **mitigations** (GQA, paged attention, YaRN).

Second, **product requirements** increasingly assume “read the whole PDF” or “reason across an entire repo” in one forward pass. **RAG** mitigates many knowledge tasks, but some **coreference** and **cross-paragraph** reasoning is awkward to chunk—long contexts reduce **retrieval orchestration** at the cost of **compute and memory**. You must articulate **effective context**: models may **accept** 128K tokens yet **under-use** the middle (“lost in the middle”)—**needle-in-a-haystack** tests exist precisely to measure this.

Third, **positional encodings** trained at \(L_{\text{train}}\) often **fail** at \(L_{\text{test}} \gg L_{\text{train}}\) unless you apply **RoPE scaling** (linear, NTK-aware, YaRN). Understanding **why** phases drift and **how** scaling preserves local structure separates senior systems answers from “we used a bigger window.”

---

## Core Concepts

### Quadratic Attention Cost

For batch size \(B\), heads \(H\), sequence \(T\), head dimension \(d_h\), self-attention is dominated by \(QK^\top\) and \(\mathrm{softmax}(QK^\top)V\). **FLOPs** scale approximately as:

\[
O(B \cdot H \cdot T^2 \cdot d_h)
\]

**Memory** for materialized score matrices (without fusion) scales as \(O(B \cdot H \cdot T^2)\) for storage of logits per head.

!!! math-intuition "In Plain English"
    Doubling context from 32K to 64K roughly **quadruples** attention work **per layer**. Long context is expensive even if **embedding** layers are linear in \(T\).

### FlashAttention: IO-Aware Exact Attention

**FlashAttention** computes **mathematically exact** softmax attention (up to floating-point non-associativity) **without** writing the full \(T \times T\) matrix to HBM. It tiles \(Q\), \(K\), \(V\) into SRAM-sized blocks and fuses softmax with the \(V\) multiply.

For a query block \(Q_i\) and key blocks \(K_j\), attention output is accumulated across blocks:

\[
O_i = \sum_j \mathrm{softmax}(Q_i K_j^\top) V_j
\]

!!! math-intuition "In Plain English"
    Conceptually this is the **same** attention as the fused matmul—just written as a **sum over key/value tiles** \(j\) so the implementation can stream blocks through SRAM instead of materializing the full \(T \times T\) matrix.

**Online softmax** maintains running per-row maxima \(m\) and cumulative exponentials \(\ell\) so partial blocks merge **exactly**:

\[
m' = \max(m, m_{j+1}),\quad
\ell' = e^{m - m'}\ell + e^{m_{j+1} - m'}\ell_{j+1}
\]

!!! math-intuition "In Plain English"
    You still do \(O(T^2)\) **arithmetic** for full attention—but you **stop spilling** giant \(T \times T\) tensors to slow memory. When attention is **memory-bound**, wall-clock drops sharply.

### Sliding Window Attention

Restrict attention to a local neighborhood \(\mathcal{W}(i)\) of width \(w\):

\[
\alpha_{ij} = 0 \quad \text{if } j \notin \mathcal{W}(i)
\]

!!! math-intuition "In Plain English"
    **Masking** enforces sparsity: most pairs \((i,j)\) are **forbidden**, so you skip computing or adding those logits—this is where **subquadratic** behavior comes from in sliding-window kernels.

For causal models, \(\mathcal{W}(i) = \{i-w+1,\ldots,i\}\). Complexity drops to \(O(T \cdot w \cdot d_h)\) per head vs \(O(T^2 d_h)\).

!!! math-intuition "In Plain English"
    Each layer only **mixes locally**, but **stacking** \(L\) layers propagates information roughly **\(O(L \cdot w)\)** steps along the sequence—**telephone chain** effect.

!!! example "Worked Example: Effective Receptive Field Sketch"
    If \(w=4096\) and \(L=32\), a **rough** propagation bound is **131072** tokens **along the chain**—not a formal theorem, but intuition for why **deep** local attention can still **reach** far. In practice, **global tokens** or **full** layers are often mixed in so one path does not carry all responsibility.

### RoPE and Long Context Extrapolation

**Rotary Position Embedding (RoPE)** applies rotations to \(q,k\) in 2D subspaces so attention depends on **relative** offset \(m-n\). For linear scaling to extend from trained length \(L\) to target \(L'\), a simple heuristic **divides** positions by factor \(\rho = L'/L\):

\[
\text{pos}' = \frac{\text{pos}}{\rho}
\]

!!! math-intuition "In Plain English"
    You **squeeze** the position ruler so the model never “runs out” of trained angles—but **resolution** at short distances can blur—**NTK-aware** methods adjust **frequencies** non-uniformly.

**NTK-aware** scaling adjusts the **base** \(\theta_0\) of RoPE (conceptually):

\[
\theta_0' = \theta_0 \cdot \rho^{d/(d-2)}
\]

(for common formulations; implementations vary by codebase).

!!! math-intuition "In Plain English"
    High-frequency components encode **fine** local distinctions; **NTK** scaling tries to stretch **long** wavelengths for far positions without destroying **near** behavior—better than naive linear scaling alone on many models.

### YaRN (Conceptual)

**YaRN** blends scaled and unscaled RoPE across dimensions with a **ramp** so low-frequency (long-wavelength) components extrapolate more aggressively—stabilizing perplexity on long documents when extending context.

### Grouped-Query Attention and KV Cache

KV cache memory scales with the number of **key/value heads** stored. With **grouped-query attention (GQA)**, heads share \(K,V\), shrinking cache roughly by the **group size** factor:

\[
\text{Memory}_{\text{KV}} \propto L_{\text{layers}} \cdot B \cdot T \cdot d_{\text{kv-per-token}}
\]

!!! math-intuition "In Plain English"
    At long \(T\), **KV** dominates serving memory—**GQA/MQA** and **KV quantization** are standard mitigations alongside **FlashAttention** for prefill.

### Ring Attention (Distributed)

**Ring attention** circulates \(K,V\) blocks around devices in a ring so no single GPU holds the full sequence for all heads—**communication** overlaps with **compute** when implemented well.

### Sparse Patterns: Longformer and BigBird

**Longformer**: sliding window + **global** tokens (e.g., special tokens attend everywhere). **BigBird**: window + global + **random** edges—sparse \(O(T)\) edges per token under chosen hyperparameters.

### Needle in a Haystack (NIAH)

**NIAH** places a **secret fact** at a controlled depth in a long prompt and measures whether the model **retrieves** it—sanity check for **positional extrapolation** and **attention dilution**.

### Lost in the Middle

Empirically, models may attend more to **beginning** and **end** of long contexts; **important** facts in the **middle** can be **under-weighted**—design prompts and **retrieval** accordingly.

### Roofline Model (Compute vs Memory)

A simplified **roofline** bound:

\[
\text{Attainable FLOPs/s} = \min\left(\text{Peak}_{\text{FLOPs/s}},\; \beta \cdot \text{AI}\right)
\]

where \(\beta\) is memory bandwidth (bytes/s) and **AI** (arithmetic intensity) is FLOPs per byte moved.

!!! math-intuition "In Plain English"
    If attention is **memory-starved**, **FlashAttention** raises AI by keeping tiles in SRAM—your kernel moves from the **left** roofline slope to a **higher** plateau before you hit peak FLOPs.

### ALiBi: Linear Biases on Distances

**ALiBi** adds a head-specific penalty proportional to distance \(i-j\) instead of explicit rotary tables:

\[
\text{score}_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}} + m_h \cdot (i - j)
\]

with learned or fixed slopes \(m_h\) per head.

!!! math-intuition "In Plain English"
    ALiBi **punishes** far-away keys with a linear ramp—simple extrapolation story for longer \(T\) in some setups; compare with **RoPE**-heavy stacks where **YaRN** is more common in open LLaMA-class models.

### Context Window vs Effective Context

**Advertised** \(T_{\max}\) is not **usable** \(T_{\max}\): system prompts, tool schemas, safety prefixes, and user formatting consume tokens. Define **usable** budget:

\[
T_{\text{usable}} = T_{\max} - T_{\text{system}} - T_{\text{tools}} - T_{\text{safety}}
\]

!!! math-intuition "In Plain English"
    Even if the model **accepts** 128K tokens, **effective** reasoning over all positions may be lower—validate with **NIAH** and task-specific evals, not parameter counts alone.

!!! example "Worked Example: RoPE Linear Scale (4K → 32K)"
    Suppose \(\rho = L'/L = 8\). Position index **32000** maps to **\(32000/8 = 4000\)**—inside the trained 4K regime. **Trade-off**: distant positions become **crowded** into fewer effective indices; **NTK/YaRN** exist to reduce **blur** at short distances.

### Multi-Query Attention (MQA)

**MQA** shares one \(K,V\) pair across all \(H\) heads. Attention still uses \(H\) query heads, but **KV** bandwidth and cache shrink:

\[
\text{KV heads} = 1 \implies \text{KV cache} \approx \frac{1}{H} \times \text{MHA KV cache}
\]

!!! math-intuition "In Plain English"
    You **duplicate** queries for expressivity but **share** keys/values—serving memory wins; some **quality** trade-off vs full MHA depending on architecture and training.

### Chunked Prefill

For very long user prompts, **chunked prefill** splits the sequence into segments and **accumulates** KV cache segment-wise to avoid **peak** activation OOM—**latency** may increase vs one-shot fused kernel, but **peak** memory drops.

### Training vs Inference

| Phase | Dominant concern | Typical mitigations |
|-------|------------------|---------------------|
| Training | Activation memory at large \(T\) | Checkpointing, FlashAttention, sequence parallel |
| Inference (prefill) | \(T^2\) attention + activations | FA, shorter batch |
| Inference (decode) | KV cache growth | GQA, KV quant, paged KV |

### Interaction with Quantization

**Weight-only** INT4/INT8 cuts model size; **KV cache** INT8/FP8 reduces **long-chat** RAM but can **harm** **long-range** fidelity—**calibrate** on representative transcripts.

??? deep-dive "Deep Dive: FlashAttention-2 vs FlashAttention-1"
    FlashAttention-2 improves **work partitioning** across warps, better **sequence parallelism**, and reduces **non-matmul** overhead—often ~2× faster than FA-1 on A100/H100-class GPUs for typical shapes. Always **profile** your \((B,T,H,d_h)\).

??? deep-dive "Deep Dive: PagedAttention (vLLM)"
    **PagedAttention** stores KV cache in **non-contiguous** blocks like virtual memory, reducing **fragmentation** when batching variable-length requests—critical for high-throughput serving with long chats.

## Code

### Memory estimator (runs everywhere)

```python
"""Attention score tensor and KV cache byte estimates (FP16/BF16)."""
from __future__ import annotations


def attention_score_bytes(
    batch: int,
    heads: int,
    seq: int,
    bytes_per_elem: int = 2,
    include_softmax: bool = True,
) -> int:
    per_head = seq * seq * bytes_per_elem
    layers_mem = 2 if include_softmax else 1
    return batch * heads * per_head * layers_mem


def kv_cache_bytes(
    batch: int,
    layers: int,
    heads_kv: int,
    seq: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> int:
    per_layer = 2 * batch * heads_kv * seq * head_dim * bytes_per_elem
    return layers * per_layer


if __name__ == "__main__":
    B, H, T, L, Dh = 1, 32, 32768, 80, 128
    print("S and P tensors (FP16, naive):", attention_score_bytes(B, H, T))
    print("KV cache (FP16):", kv_cache_bytes(B, L, H, T, Dh))
```

### Optional FlashAttention forward (GPU + package required)

```python
"""pip install flash-attn torch -- requires compatible CUDA GPU."""
from __future__ import annotations

import torch


def run_flash_attn_if_available() -> None:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("flash-attn not installed; skip GPU kernel demo.")
        return
    if not torch.cuda.is_available():
        print("CUDA required for flash_attn_func demo.")
        return
    B, T, H, Dh = 2, 512, 8, 64
    q = torch.randn(B, T, H, Dh, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, H, Dh, device="cuda", dtype=torch.float16)
    v = torch.randn(B, T, H, Dh, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v, causal=True)
    print("out shape:", tuple(out.shape))


if __name__ == "__main__":
    run_flash_attn_if_available()
```

### Sliding-window causal mask (educational)

```python
"""O(T*w) attention mask construction — use fused kernels in production."""
from __future__ import annotations

import torch


def sliding_window_mask(seq: int, window: int, device: str = "cpu") -> torch.Tensor:
    """Boolean mask: True = allow attend. Causal + last w positions."""
    idx = torch.arange(seq, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    causal = diff >= 0
    local = diff < window
    return causal & local


if __name__ == "__main__":
    m = sliding_window_mask(8, window=3)
    print(m.int())
```

### Needle-in-a-haystack style check (string match)

```python
"""Minimal NIAH-style test using substring retrieval from generated filler."""
from __future__ import annotations


def build_long_prompt(needle: str, depth_chars: int, filler_unit: str = "blah ") -> str:
    prefix = filler_unit * (depth_chars // len(filler_unit))
    suffix = filler_unit * 200
    return prefix + needle + suffix


def needle_found(model_output: str, needle: str) -> bool:
    return needle.strip() in model_output


if __name__ == "__main__":
    secret = "THE_SECRET_CODE_IS_4242"
    prompt = build_long_prompt(secret, depth_chars=5000)
    fake_model_out = f"I found: {secret} in the text."
    print("prompt length chars:", len(prompt))
    print("recovered:", needle_found(fake_model_out, secret))
```

!!! interview "FAANG-Level Questions"
    1. Why is standard attention \(O(T^2)\), and what exactly does FlashAttention change vs approximate sparse attention?
    *Answer:* Each of \(T\) queries attends to \(T\) keys, so per-layer attention FLOPs and naive materialization scale **\(O(T^2)\)** (times heads and batch). **FlashAttention** keeps the **same** softmax attention math but tiles computation to reduce **HBM traffic**—exact attention, faster in practice when memory-bound. **Sparse/window** attention changes the **math** by zeroing distant pairs—subquadratic work but approximate long-range mixing unless you add globals or deep stacks.
    2. Explain sliding-window attention and how depth increases receptive field.
    *Answer:* Each position only attends to a **local neighborhood** of width \(w\), cutting per-layer cost to \(O(Tw)\). Information still flows “along the chain” over layers—roughly, effective reach grows with **layer depth × window** (plus any global tokens). Very deep local stacks can cover long ranges, but early layers see only local context; hybrid models often add periodic full or global attention.
    3. What breaks when you extrapolate RoPE from 4K to 128K training lengths?
    *Answer:* RoPE phases were fit for trained positions; at much longer spans **relative** angles for far tokens leave the training manifold—attention degrades (“wobbly” position sensitivity) and perplexity/NIAH scores suffer. Naive linear scaling squeezes indices but can blur fine local distinctions; **NTK/YaRN** re-tune frequency bases to stretch long wavelengths without trashing short-range behavior.
    4. Compare linear RoPE scaling vs NTK-aware scaling in one sentence each.
    *Answer:* **Linear scaling** divides position indices by a factor so longer spans map into the trained range—simple but can over-compress distant positions and hurt local resolution. **NTK-aware scaling** adjusts RoPE frequency bases (not just positions) so high-frequency components still encode nearby detail while low frequencies extrapolate—usually better perplexity on long docs at the cost of more hyperparameters.
    5. How does KV cache memory scale with \(T\), and what reduces it at inference?
    *Answer:* KV cache size grows **linearly** with sequence length \(T\) (per layer, per batch, per KV head dimension). **GQA/MQA** shrink KV head count; **KV quantization** (INT8/FP8), **paged KV** (vLLM), and smaller batch/long-context serving configs cut RAM. Long prompts dominate **prefill** compute; long generation dominates **KV** footprint.
    6. What is the “lost in the middle” phenomenon, and how might you mitigate it?
    *Answer:* Models often **underweight** evidence placed in the **middle** of long prompts—retrieval and needle tests show U-shaped attention bias. Mitigate by **reranking** to put key facts at the start or end, **structured** sections, repeating critical constraints, or **chunking+RAG** instead of one giant dump. Measure with needle-in-haystack tasks for your actual model and template.
    7. Describe ring attention or context parallelism at a high level.
    *Answer:* **Ring attention** (and similar context-parallel schemes) **shard the sequence** across devices: each GPU holds a **block** of tokens and passes \(K,V\) blocks around a ring so attention can be computed without one GPU owning the full \(T\times T\) for all tokens. It trades **network bandwidth** for **memory** savings—essential for million-token training or very long prefills when model parallel alone is insufficient.
    8. When would you prefer RAG over stuffing an enormous context?
    *Answer:* Prefer **RAG** when the corpus is **larger than the usable window**, updates frequently, or needs **ACL/filtered** retrieval—RAG pins cost to relevant slices. Stuffing helps when you need **global** reasoning over a single artifact that fits (e.g., one repo) and chunk boundaries would break coreference—but it burns prefill FLOPs and may still “lose” middle content. Hybrid: retrieve candidates, then selectively expand.
    9. What does YaRN modify compared to vanilla RoPE extrapolation?
    *Answer:* **YaRN** **interpolates** between scaled and unscaled RoPE across **frequency dimensions** with a ramp—low-frequency (long-wavelength) components extrapolate more, preserving stability on long documents, while higher frequencies keep local behavior. It targets perplexity gains when extending context versus only linearly rescaling positions.
    10. How does grouped-query attention differ from multi-head attention for serving?
    *Answer:* **MHA** stores separate \(K,V\) per head—maximum expressivity, largest KV cache. **GQA** shares one \(K,V\) across a **group** of query heads, shrinking cache and memory bandwidth with minor quality impact in many architectures. Serving teams pick GQA to scale concurrent long chats; training must co-design attention patterns.

!!! interview "Follow-up Probes"
    - “At what \(T\) does your workload become memory-bound vs compute-bound?”
    - “How do you validate long-context quality beyond perplexity?”
    - “What’s your prefill vs decode memory story for 128K user prompts?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “FlashAttention reduces HBM traffic; it’s exact attention, not a sparse approximation.”
    - “Sliding window is subquadratic per layer; depth propagates information.”
    - “RoPE extrapolation needs frequency scaling—linear divide is a blunt instrument.”
    - “Serving is KV-cache bound—GQA and paged KV are first-class optimizations.”

### Linear Attention (Research Pointer)

**Linear attention** variants replace \(\mathrm{softmax}(QK^\top)V\) with kernel feature maps \(\phi(Q)\phi(K)^\top V\) that admit **recurrent** \(O(T)\) forms—**different** trade-offs vs **FlashAttention**-optimized softmax on modern GPUs:

\[
\text{Attention}_{\text{lin}}(Q,K,V) = \bigl(\phi(Q)\bigl(\phi(K)^\top V\bigr)\bigr)
\]

!!! math-intuition "In Plain English"
    If \(\phi\) maps to a **small** feature dimension, you can **stream** tokens with fixed state—**infinite** context in principle—but **expressivity** and **hardware** maturity vary; profile against **FA2** baselines.

## References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022): [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023): [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021): [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023): [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Jiang et al., *Mistral 7B* (sliding window): [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Beltagy et al., *Longformer* (2020): [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- Zaheer et al., *Big Bird* (2020): [arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
- Liu et al., *Ring Attention with Blockwise Transformers* (2023): [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023): [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
