# Continuous Batching and PagedAttention

## Why This Matters for LLMs

Static batching—the pattern familiar from training—forces every request in a batch to advance in lockstep. When one sequence needs two hundred decode steps and another needs ten, the short request still occupies a slot until the longest finishes, and padding masks burn memory bandwidth on tokens that contribute nothing to useful output. Continuous batching, also called iteration-level scheduling, treats each decoding step as a scheduling quantum: sequences can complete, depart, and be replaced by newly admitted work between iterations. That behavior is what lets production servers keep GPUs busy under skewed workloads instead of idling while waiting for stragglers.

PagedAttention addresses a different failure mode of naive serving: the key-value cache for autoregressive attention is enormous, and reserving a single contiguous maximum-length buffer per request leaves most of that reservation empty for typical prompts. External fragmentation appears when many variable-size allocations churn through GPU memory—free space exists, but not in one contiguous slab. PagedAttention stores \(K\) and \(V\) in fixed-size blocks and maps logical token indices to physical blocks with per-sequence page tables, much like virtual memory in an operating system. Together with continuous batching, it underpins vLLM, Hugging Face TGI, TensorRT-LLM’s paged KV paths, and other stacks that interviewers treat as the default answer for “how would you serve an LLM at scale?”

In system-design interviews you are expected to name the two bottlenecks—compute scheduling and KV memory—and explain why they are orthogonal: batching improves utilization of tensor cores on short vs long mixes, while paging improves bytes-per-request and pool reuse. You should also connect prefill (prompt processing) versus decode (one-token steps), because continuous batching mainly reshapes the decode steady state, and PagedAttention mainly reshapes how KV bytes are laid out in HBM. The sections below make those relationships quantitative.

---

## Core Concepts

### Static vs Dynamic Batching

In **static** batching, a batch of \(B\) requests is formed once, tensors are padded to length \(L = \max_i \ell_i\), and the model runs until all sequences reach end-of-sequence or a cap. The number of “active” token positions per layer is \(B \cdot L\), but many positions are padding. A simple measure of padding overhead is:

\[
\eta_{\text{pad}} = 1 - \frac{\sum_{i=1}^{B} \ell_i}{B L}.
\]

!!! math-intuition "In Plain English"
    - The numerator is how many real tokens you actually care about this step; the denominator is what you pay for if every row is padded to the longest sequence.
    - When lengths are wildly different, \(\eta_{\text{pad}}\) approaches values that would be unacceptable in a training pipeline—but serving often accepted it until iteration-level methods matured.

**Continuous** (dynamic) batching changes the set of sequences that participate in each forward. After iteration \(t\), any sequence that finished leaves the batch; new requests whose prefill completed join. The effective batch size \(B_t\) can vary every step.

!!! example "Worked Example: Throughput Comparison"
    Consider four decode jobs with remaining lengths (including the current step) of \(10, 50, 100, 200\) tokens until EOS. Assume one decode forward handles one token per sequence per iteration.

    **Static batch:** Every sequence stays until the longest finishes. Wall-clock steps \(\approx 200\). Total token-steps processed \(= 4 \times 200 = 800\) token-steps (many of them “empty” for finished sequences unless you mask aggressively and still pay memory).

    **Continuous batch (idealized):** The 10-token job finishes at step 10 and frees a slot. If a new job is always ready to fill the slot, the GPU keeps four live sequences longer, but short jobs do not wait idly for 200 steps *inside the same batch membership*—the scheduler replaces them. Counting only these four jobs without arrivals, total steps until all four complete is still dominated by the 200-token job, but **average time-to-completion** for the short job drops from waiting for the batch’s slowest member to roughly its own length when scheduling is per-sequence. Under open-loop arrivals, freed slots raise **throughput** (tokens per second per GPU) because fewer decode iterations waste capacity on padding-only rows.

    Numerical summary: static grouping ties tail latency to \(\max_i \ell_i\); continuous scheduling decouples per-request completion from unrelated long requests.

### Iteration-Level Scheduling

The Orca paper (OSDI 2022) popularized **iteration-level** scheduling for transformer generation. The server maintains a set of active sequences. Each iteration:

1. Run one decode step (or a chunk of prefill) for every active sequence.
2. Remove sequences that hit EOS or length limits.
3. Admit new requests from the queue when KV memory and policy allow.

Let \(A_t\) be the set of active sequence indices at step \(t\). A trivial count of token forwards per step is:

\[
N_t = \sum_{i \in A_t} 1 = |A_t|
\]

for pure decode with one new token per sequence per iteration (ignoring tensor parallelism duplication).

!!! math-intuition "In Plain English"
    - This says decode throughput scales with how many sequences you can keep live *this* step—not with the maximum length anyone ever had in the past.
    - Prefill often processes many prompt tokens in one shot; decode repeats many times with batch size equal to concurrent sequences.

**Prefill vs decode** split metrics:

\[
\mathrm{TTFT} \approx T_{\text{prefill}}(L_{\text{prompt}}), \qquad
\mathrm{ITL} \approx T_{\text{decode}}.
\]

!!! math-intuition "In Plain English"
    - **Time-to-first-token (TTFT)** is dominated by prompt processing (attention over the prompt, sometimes chunked).
    - **Inter-token latency (ITL)** is the steady-state cost per new token for a single user once generation has started—often memory-bound.

!!! example "Worked Example: Prefill Chunking"
    Suppose a request has a 8{,}192-token prompt but the engine caps **prefill chunk** size at 2{,}048 tokens to avoid starving other users’ decode steps.

    Chunks required: \(\lceil 8192 / 2048 \rceil = 4\) prefill forwards before decode begins for that request.

    If each chunk takes 15 ms GPU time, **best-case** prefill time \(\approx 4 \times 15 = 60\) ms before the first decode token can be scheduled (ignoring queueing). Chunking trades a slightly higher TTFT for **fairness**: one user cannot monopolize the GPU for a single giant matmul without yielding.

### PagedAttention

Traditional KV cache allocation reserves, for each layer and head, contiguous storage for \(L_{\max}\) positions. Logical token index \(t\) maps by simple stride arithmetic into a dense buffer. PagedAttention instead splits the sequence into **blocks** of \(B_{\text{block}}\) tokens. Logical token \(t\) maps to:

\[
b = \left\lfloor \frac{t}{B_{\text{block}}} \right\rfloor, \qquad
o = t \bmod B_{\text{block}}.
\]

!!! math-intuition "In Plain English"
    - \(b\) is the logical block index for this sequence; the **block table** stores which physical block ID holds that logical block.
    - \(o\) tells you where inside the 16- or 32-token physical block to read or write \(K\) and \(V\).

The number of physical blocks allocated for a sequence of length \(L\) is:

\[
N_{\text{blk}}(L) = \left\lceil \frac{L}{B_{\text{block}}} \right\rceil.
\]

!!! math-intuition "In Plain English"
    - You pay for storage in multiples of the block size. The last block may be partially filled—**internal fragmentation** inside that block.
    - Total physical pages across layers and heads is what matters for HBM pressure.

??? deep-dive "Deep Dive: Copy-on-Write and Beam Search"
    Beam search maintains multiple hypotheses that share a common prefix. PagedAttention-style block tables allow **reference counting** on physical blocks: beams split only when they diverge, at which point a **copy-on-write** path allocates fresh physical blocks for the differing suffix. Without paging, duplicating full contiguous KV tensors per beam multiplies memory by beam width. Interview tip: name **reference counting** and **COW** as the reason paging helps **beam width** without a linear blow-up in KV bytes for shared prefixes.

### Memory Efficiency

For one layer, one head, with head dimension \(d_h\), storing \(K\) and \(V\) in FP16 for \(L\) positions uses:

\[
M_{\text{KV,layer,head}}(L) = 2 \times L \times d_h \times \text{bytes per element}
\]

for the pair \((K,V)\).

!!! math-intuition "In Plain English"
    - The factor 2 is **K and V**; some accounting splits across QKV projections, but the cache story is “two tensors of shape roughly \((L, d_h)\) per head per layer.”
    - Multiply by number of heads and layers for the full model.

Contiguous reservation for \(L_{\max}\) when only \(L\) tokens are live wastes:

\[
\frac{L_{\max} - L}{L_{\max}}
\]

of that head’s KV slice (first-order sketch).

!!! math-intuition "In Plain English"
    - If average \(L\) is 500 and \(L_{\max}\) is 4096, you waste about \(88\%\) of the reserved KV for that sequence—**before** counting fragmentation effects.

!!! example "Worked Example: Memory Savings with PagedAttention"
    **Setup:** \(L_{\max} = 4096\), **block size** \(B_{\text{block}} = 16\), **head dim** \(d_h = 128\), **32 layers**, **32 heads**, **FP16** (2 bytes per element). Consider **one request** with actual length \(L = 500\) tokens.

    **Contiguous reservation per layer (all heads):** For each head, KV size \(\propto 2 \cdot L_{\max} \cdot d_h\). Per layer:

    \[
    M_{\text{layer}} = 32 \cdot 2 \cdot 4096 \cdot 128 \cdot 2 \ \text{bytes} = 67{,}108{,}864\ \text{bytes} \approx 64\ \text{MiB}.
    \]

    !!! math-intuition "In Plain English"
        - \(32\) heads; each head stores \(K\) and \(V\) (\(\times 2\)) for \(4096\) positions; \(128\) dims; FP16 is 2 bytes.
        - Multiply by 32 layers: \(\approx 2\) GiB **reserved** for KV alone in this toy accounting.

    **Paged allocation for \(L=500\):** Blocks per head:

    \[
    N_{\text{blk}} = \left\lceil \frac{500}{16} \right\rceil = 32.
    \]

    Per block per head, KV bytes (one block holds \(16\) tokens):

    \[
    2 \times 16 \times 128 \times 2 = 8{,}192\ \text{bytes} = 8\ \text{KiB}.
    \]

    !!! math-intuition "In Plain English"
        - Here the leading 2 is still **K and V**; \(16\) tokens \(\times\) \(128\) dims \(\times\) 2 bytes.

    Per layer (32 heads):

    \[
    32\ \text{heads} \times 32\ \text{blocks} \times 8\ \text{KiB} = 8192\ \text{KiB} = 8\ \text{MiB}.
    \]

    Over 32 layers:

    \[
    32 \times 8\ \text{MiB} = 256\ \text{MiB}.
    \]

    **Comparison:** \(\approx 2\) GiB reserved contiguously vs \(\approx 256\) MiB actually needed for \(L=500\) in this block model—an order-of-magnitude illustration of why paging matters. Real systems add metadata, alignment, and cross-layer layouts; trends remain.

### Prefix Caching and RadixAttention

When many requests share a long system prompt, the KV for that prefix is identical. **Prefix caching** stores KV blocks keyed by prompt hash or trie path so later requests **reuse** physical blocks instead of recomputing attention over the prefix.

A **radix tree** over token sequences (as in SGLang’s **RadixAttention**) lets shared nodes represent shared prefixes; divergent suffixes branch. The **hit rate** for prefix blocks in an API serving the same template prompt to many users can be modeled roughly as:

\[
\mathbb{E}[\text{saved prefill FLOPs}] \approx p_{\text{hit}} \cdot L_{\text{prefix}} \cdot C_{\text{prefill}}.
\]

!!! math-intuition "In Plain English"
    - \(p_{\text{hit}}\) is the probability a new request’s prefix matches a cached path; \(C_{\text{prefill}}\) is cost per token for prefill.
    - This is **orthogonal** to PagedAttention: paging is **allocation**; radix/prefix structures are **deduplication** of **compute** and **KV reuse**.

??? deep-dive "Deep Dive: Interaction with FlashAttention"
    FlashAttention reduces HBM traffic for attention **kernels**; PagedAttention reduces **allocator** waste and enables **non-contiguous** KV. Production kernels may **gather** \(K,V\) from physical blocks into warps efficiently. The bottleneck can still be **memory bandwidth** on decode—batching raises arithmetic intensity; paging raises **practical** batch sizes by fitting more sequences.

---

## Code

The first script is **pure Python**: a toy **continuous batching** simulator that runs without GPU. The second uses **NumPy** to compute KV memory budgets. The third shows **vLLM**’s `LLM` API behind a **try**/**except** so importing this file in environments without CUDA still works.

```python
"""
Educational continuous batching + PagedAttention memory estimator.

- continuous_batching_demo(): pure Python, no GPU.
- kv_paged_bytes(): NumPy numeric illustration.
- vllm_optional_demo(): runs only if vllm is installed (CUDA machine).

Dependencies for full run: numpy; optional vllm + GPU.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(order=True)
class Request:
    """Simulates one decode stream: remaining tokens including current."""

    sort_index: int
    rid: int
    remaining: int = field(compare=False)

    def step(self) -> bool:
        """Returns True if still active after this token."""
        self.remaining -= 1
        return self.remaining > 0


class ContinuousBatchScheduler:
    """
    Greedy iteration scheduler: each step processes all active requests once.
    When a request completes, optionally pull a new one from the queue.
    """

    def __init__(self, max_batch: int) -> None:
        self.max_batch = max_batch
        self.time = 0
        self.active: List[Request] = []
        self._next_id = 0
        self._waitq: List[Tuple[int, int]] = []  # (remaining_tokens, rid)

    def submit(self, num_tokens: int) -> int:
        rid = self._next_id
        self._next_id += 1
        heapq.heappush(self._waitq, (num_tokens, rid))
        return rid

    def _admit_from_queue(self) -> None:
        while len(self.active) < self.max_batch and self._waitq:
            rem, rid = heapq.heappop(self._waitq)
            heapq.heappush(
                self.active,
                Request(sort_index=-rem, rid=rid, remaining=rem),
            )

    def run_until_empty(self) -> Dict[str, float]:
        self._admit_from_queue()
        total_token_steps = 0
        peak_active = 0
        while self.active or self._waitq:
            peak_active = max(peak_active, len(self.active))
            if not self.active:
                self._admit_from_queue()
            # one decode iteration: one token per active request
            total_token_steps += len(self.active)
            still: List[Request] = []
            while self.active:
                req = heapq.heappop(self.active)
                if req.step():
                    heapq.heappush(still, req)
            self.active = still
            self.time += 1
            self._admit_from_queue()
        return {
            "iterations": float(self.time),
            "total_token_steps": float(total_token_steps),
            "peak_batch": float(peak_active),
        }


def kv_paged_bytes(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    block_tokens: int,
    bytes_per_el: int = 2,
) -> Tuple[int, int, int]:
    """
    Returns (paged_bytes_for_seq_len, contiguous_bytes_for_exact_seq_len,
              contiguous_bytes_if_reserved_to_Lmax where Lmax=len*block ceiling from paging).

    Per-block KV = 2 * block_tokens * head_dim * bytes_per_el * num_heads per layer.
    """
    blocks = int(np.ceil(seq_len / block_tokens))
    per_block = 2 * block_tokens * head_dim * bytes_per_el * num_heads
    paged_total = num_layers * blocks * per_block
    contiguous_actual = num_layers * 2 * seq_len * head_dim * bytes_per_el * num_heads
    # Worst-case contiguous reservation per request (common serving pattern):
    lmax = int(blocks * block_tokens)
    contiguous_at_least_lmax = num_layers * 2 * lmax * head_dim * bytes_per_el * num_heads
    return int(paged_total), int(contiguous_actual), int(contiguous_at_least_lmax)


def continuous_batching_demo() -> None:
    sched = ContinuousBatchScheduler(max_batch=4)
    # lengths [10, 50, 100, 200] as in the text
    for t in (10, 50, 100, 200):
        sched.submit(t)
    stats = sched.run_until_empty()
    print("Continuous scheduler stats:", stats)

    pb, cb, _ = kv_paged_bytes(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        seq_len=500,
        block_tokens=16,
    )
    _, _, naive_lmax = kv_paged_bytes(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        seq_len=4096,
        block_tokens=16,
    )
    print(f"Paged KV bytes for L=500 (toy): {pb / 1e6:.2f} MB")
    print(f"Dense contiguous KV for L=500 (toy): {cb / 1e6:.2f} MB")
    print(f"Dense contiguous reserve for L_max=4096 (toy): {naive_lmax / 1e9:.3f} GiB")


def vllm_optional_demo() -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        print("vLLM not installed; skip GPU demo:", exc)
        return

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64)
    prompts = [
        "Explain continuous batching in one paragraph.",
        "What is PagedAttention?",
    ]
    outputs = llm.generate(prompts, params)
    for i, o in enumerate(outputs):
        print("=== Output", i, "===")
        print(o.outputs[0].text)


if __name__ == "__main__":
    continuous_batching_demo()
    vllm_optional_demo()
```

!!! math-intuition "In Plain English"
    - The scheduler’s `total_token_steps` counts **useful** parallel token generations across iterations—compare with padded static batching metrics in your own benchmarks.
    - `kv_paged_bytes` is a **planning** helper; production frameworks add **GQA/MQA**, **FP8 KV**, and **alignment**.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Explain why static batching inflates tail latency when request lengths are skewed, and how continuous batching changes the scheduling unit.
    2. Write the formula for logical token index to block id and offset; why does internal fragmentation appear at most \(B_{\text{block}}-1\) tokens per sequence?
    3. How does PagedAttention reduce **external** fragmentation compared to contiguous per-request buffers?
    4. Contrast **TTFT** and **per-token latency**; which phase is more compute-bound vs memory-bound for typical decoder-only transformers?
    5. How would you implement **fairness** between a long prefill and many short chat messages?
    6. What breaks naive **CUDA graph** capture when batch membership changes every iteration?
    7. How does **tensor parallelism** interact with per-sequence KV block tables?
    8. Describe **prefix caching** vs **paged allocation**—which saves compute, which saves memory layout overhead?
    9. What metrics would you put on a dashboard to detect KV pool exhaustion before OOM?
    10. How does **Little’s law** relate offered load, queue depth, and latency for an LLM API?

!!! interview "Follow-up Probes"
    - “We enabled continuous batching but p95 latency got worse—what could explain that?” (admission control, batch too large, prefill starvation, network, quantization errors.)
    - “How would you shard KV across GPUs in pipeline parallelism without duplicating full sequences?”
    - “When is RadixAttention strictly better than hashing whole prompts for prefix cache?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Iteration-level scheduling decouples sequence completion from the longest member of a static batch.”
    - “PagedAttention is virtual memory for KV cache: logical blocks map to physical pages in a pool.”
    - “Decode is often memory-bandwidth bound; raising effective batch size increases arithmetic intensity.”
    - “Chunked prefill prevents one huge prompt from starving everyone else’s decode.”
    - “Prefix caching and paging compose: dedup saves compute; paging saves fragmentation.”

---

## References

- Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models* — [USENIX OSDI](https://www.usenix.org/conference/osdi22/presentation/yu)
- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- vLLM documentation — [https://docs.vllm.ai](https://docs.vllm.ai)
- Pope et al. (2023), *Efficiently Scaling Transformer Inference* — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- Hugging Face, *Text Generation Inference* — [https://huggingface.co/docs/text-generation-inference](https://huggingface.co/docs/text-generation-inference)
- NVIDIA, *TensorRT-LLM* — [https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
