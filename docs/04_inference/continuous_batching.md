# Continuous Batching & PagedAttention

## Why This Matters for LLMs

Serving an LLM is not “run `forward` on a tensor” once—it is a **multi-tenant queue** of requests with **different** prompt lengths, **different** output lengths, and **different** arrival times. **Static batching** waits until \(N\) requests arrive, pads them to **\(L_{\max}\)**, runs one **fat** forward, and wastes **compute** on **padding** tokens while **blocking** short jobs behind long ones. **Continuous batching** (iteration-level batching, popularized by **Orca**) **adds** and **removes** sequences **between** decoding steps so the GPU **always** processes a **full** micro-batch of **real** tokens, improving **throughput** under variable load.

**PagedAttention** (vLLM) attacks the **other** bottleneck: **KV cache memory fragmentation**. Naive implementations allocate **contiguous** GPU buffers sized for **worst-case** context per request, leaving **holes** when sequences finish early or grow unevenly—**OOM** despite **average** utilization being fine. PagedAttention stores KV in **fixed-size blocks** (like **OS pages**) and maps **logical** token positions to **physical** blocks via **tables**, enabling **non-contiguous** storage with **little** external fragmentation.

Interviewers at **systems** and **ML infra** roles expect you to connect **scheduler** (continuous batching) with **memory allocator** (paging)—they are **orthogonal** optimizations that **compose**.

---

## Core Concepts

### Static Batching and Padding Waste

Suppose batch size \(B\), true lengths \(\ell_1,\ldots,\ell_B\), padded length \(L = \max_i \ell_i\). **Attention FLOPs** scale \(\propto B L^2\) in naive implementations (or \(B L\) per layer with proper masking—but **padding** still consumes **memory** and **bandwidth** for **masked-out** positions unless **fully** skipped).

\[
\text{waste ratio} \approx 1 - \frac{\sum_i \ell_i^2}{B L^2}
\]

(sketch—exact waste depends on kernel implementation).

!!! math-intuition "In Plain English"
    - **Tail latency**: one **long** request forces **everyone** in the batch to wait for **max** length in naive static scheduling.
    - **Padding** is pure **overhead**—continuous batching tries to **eliminate** fake tokens from **steady-state** decode.

### Continuous Batching (Orca-Style)

Instead of fixing the set of requests for the whole **generation**, the **scheduler** **re-batches** at **each** forward **iteration**:

- When a sequence **emits EOS**, **drop** it from the GPU batch and **admit** a **new** waiting request (often after **prefill** completes in a separate **phase**).
- **Micro-batch** size may change every step—throughput **tracks** offered load.

!!! math-intuition "In Plain English"
    - You trade **implementation complexity** (dynamic CUDA graphs, ref counting KV blocks) for **higher** **GPU** **occupancy** and **fairer** latency.

### Prefill vs Decode Phases

**Prefill**: process the **entire prompt** at once—compute-heavy, **high** tensor parallelism **efficiency**.

**Decode**: **one** (or few) new tokens per request per step—**memory-bandwidth** bound, **small** batches hurt **utilization**.

Orchestrators often **separate** pools: **prefill** servers vs **decode** servers, or **interleave** phases with **careful** **scheduling**.

\[
T_{\text{TTFT}} \approx T_{\text{prefill}}(L_{\text{prompt}}),\quad
T_{\text{token}} \approx T_{\text{decode\_step}}.
\]

!!! math-intuition "In Plain English"
    - **TTFT** dominated by **prefill** attention over **prompt** length.
    - **Tokens/sec** dominated by **decode** **steady state**—continuous batching targets **decode**.

### PagedAttention: Virtual Memory for KV Cache

PagedAttention (Kwon et al., 2023) partitions each sequence’s KV tensors into **fixed-size blocks** of \(B_{\text{block}}\) tokens. A **block table** stores **pointers** to **physical** blocks in a **preallocated** **pool**.

Logical position \(t\) maps to:

\[
\text{block\_id} = \left\lfloor \frac{t}{B_{\text{block}}} \right\rfloor,\quad
\text{offset} = t \bmod B_{\text{block}}.
\]

!!! math-intuition "In Plain English"
    - Same idea as **virtual memory**: **logical** contiguous **address** space, **physical** pages **anywhere** in RAM.
    - **Free** a **page** when a sequence **shrinks** or **ends**—**reuse** for **new** requests.

### Memory Fragmentation

**External fragmentation**: enough **total** free bytes, but **no single** contiguous chunk satisfies a **large** allocation. **Paging** reduces **external** frag by **standardizing** block size.

**Internal fragmentation**: last **partial** block wastes space inside the block—bounded by \(B_{\text{block}} - 1\) tokens worst-case per sequence.

!!! math-intuition "In Plain English"
    - **Smaller** blocks → less internal waste, **larger** **metadata** overhead.
    - **Larger** blocks → **coarser** reuse, more internal waste.

### Throughput vs Latency Tradeoffs

Increasing batch size improves **throughput** (tokens/s) but **queues** requests longer—**Little’s law** intuition: \(W \approx \lambda / \mu\) under stability. **p95 latency** explodes if **admission control** is absent.

\[
\text{throughput} \approx \frac{B_{\text{eff}}}{\text{time per decode step}}.
\]

!!! math-intuition "In Plain English"
    - **SLO-driven** autoscaling should use **latency** **curves**, not **average** tokens/s alone.

---

## vLLM API Usage Example (Python)

The following **minimal** snippet shows **offline** batched inference with the **`LLM`** class. Production deployments use **OpenAI-compatible** servers (`vllm serve`). Requires `pip install vllm` and a compatible GPU.

```python
"""
Minimal vLLM offline inference example (API surface only).
Requires a CUDA GPU with sufficient memory for the chosen model.
"""
from __future__ import annotations

from typing import List, Optional

# vLLM imports are lazy-documented here so static linters without vllm still parse.
try:
    from vllm import LLM, SamplingParams
except ImportError as e:  # pragma: no cover - doc-only path
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def run_vllm_example(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    prompts: Optional[List[str]] = None,
) -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Install vLLM in a CUDA environment to run this example: pip install vllm"
        ) from _IMPORT_ERROR

    if prompts is None:
        prompts = [
            "Explain KV cache in three sentences.",
            "Write a Python function that adds two numbers.",
        ]

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    outputs = llm.generate(prompts, sampling)
    for i, out in enumerate(outputs):
        print("--- prompt", i, "---")
        print(out.outputs[0].text)


if __name__ == "__main__":
    run_vllm_example()
```

!!! math-intuition "In Plain English"
    - **`gpu_memory_utilization`** trades **reserved** KV **pool** vs **headroom** for **CUDA** **fragmentation**.
    - **`max_model_len`** caps **PagedAttention** **tables**—raising it increases **worst-case** **memory**.

### Block Size and CUDA Kernels

PagedAttention kernels **gather** \(K,V\) **blocks** using **indices**—**random** access patterns vs **pure** contiguous **stride**-\(L\) tensors. The **win** is **allocator** **efficiency**; the **cost** is **indirection** and **less** **predictable** **memory** **coalescing**—still a **net** win at scale.

---

## Scheduling Policies (Conceptual)

| Policy | Behavior | Risk |
|--------|----------|------|
| FIFO | first-come | head-of-line blocking |
| Shortest-job-first | minimize mean latency | starvation of long jobs |
| Maximal batch packing | fill GPU | complex preemption |
| Chunked prefill | cap prefill chunks | more iterations |

!!! math-intuition "In Plain English"
    - **Production** stacks combine **chunked prefill** with **decode** **batching** to avoid **starving** **decode** when a **huge** prompt arrives.

### Little’s Law for Request Queues

Let \(\lambda\) be arrival rate (req/s), \(\bar{W}\) average time in system, \(L_q\) average **number** of requests **queued**:

\[
L_q = \lambda \bar{W}.
\]

Implication: **halving** per-request **latency** at fixed \(\lambda\) **halves** **average** queue **depth**—**latency** and **queueing** are **coupled**.

### Orca: Iteration-Level Scheduling (Sketch)

Classic **sequence-level** batching runs **until completion** for **all** sequences in the batch. **Orca** introduces **iteration-level** scheduling:

1. Maintain a **running** batch of **active** sequences.
2. After **each** **forward** **iteration** (one new token per sequence), **remove** finished sequences.
3. **Start** **prefill** for **new** requests when slots **free**—**merge** into **decode** batch when **prefill** completes.

**Ragged** tensors or **flattened** token lists with **metadata** represent **variable** **active** lengths per step.

!!! math-intuition "In Plain English"
    - **Key idea**: **GPU** **work** should scale with **total** **live** tokens **this** step, not **max** length over **all** time.

### Copy-on-Write Blocks and Forking

Some systems **fork** a **conversation** branch (A/B tests, **retry** with **different** sampling). **Paged** **KV** blocks can be **shared** read-only until **one** branch **writes**—**copy-on-write** **saves** **memory** for **shared** **prefixes**.

### Connection to KV Cache Formula (Recall)

Total **physical** KV memory **\(M\)** with **page** pool size \(N_{\text{pages}}\), **block** bytes \(S_b\):

\[
M \approx N_{\text{pages}} \cdot S_b.
\]

**Logical** sequence length \(L\) needs **\(\lceil L / B_{\text{block}} \rceil\)** **blocks**—**internal** waste bounded as **before**.

### Worked Numbers: Internal Fragmentation

If \(B_{\text{block}} = 16\) tokens and a sequence has length \(L = 100\), **blocks** required \(= \lceil 100/16 \rceil = 7\). **Capacity** \(= 7 \cdot 16 = 112\) tokens—**12** **token-slots** of **internal** waste (\(\approx 10.7\%\)) unless **partial** blocks are **packed** with **other** sequences (advanced).

!!! math-intuition "In Plain English"
    - **Tuning** \(B_{\text{block}}\) is a **knob** between **metadata** overhead and **packing** efficiency.

### Multi-GPU and Tensor Parallelism

**Tensor parallel** splits **layers** across GPUs—**PagedAttention** **block** tables are **still** **per** **sequence**, but **physical** memory is **sharded**. **Pipeline** parallel adds **microbatch** **complexity**—continuous batching **must** **coordinate** **across** stages.

### Determinism and CUDA Graphs

**Dynamic** shapes **break** **static** **CUDA** **graphs**. Frameworks may **bucket** sequence lengths or **pad** to **graph-friendly** **sizes**—a **tension** with **pure** **minimal** **work**.

### Fairness, Starvation, and Priority Classes

Interactive **chat** users expect **low** **TTFT**; **offline** **batch** jobs tolerate **queueing**. A **single** FIFO queue **starves** **short** chats behind **massive** **prompts** unless **chunked prefill** or **priority** **queues** exist.

**Weighted fair queueing** intuition: allocate **GPU** **slots** proportional to **service** **class** weights \(w_i\):

\[
\text{share}_i = \frac{w_i}{\sum_j w_j}.
\]

!!! math-intuition "In Plain English"
    - **SRE** reality: **95th** percentile **latency** **SLO** drives **admission** **rejections** and **autoscaling**—not **mean** throughput alone.

### Ragged Batches and Flattened Attention

Some kernels **concatenate** **all** **active** tokens into **one** **flat** tensor `tokens` of length \(N_{\text{tot}} = \sum_i \ell_i^{(\text{active})}\) and carry **segment** **IDs** so attention **masks** **restrict** **cross-talk** between **sequences**. This **removes** **padding** **compute** entirely at the **cost** of **more** **complex** **indexing**.

\[
\text{Attention}(Q, K, V)\ \text{with block-sparse mask by segment.}
\]

### Comparison: Static vs Continuous (Qualitative)

| Aspect | Static batching | Continuous batching |
|--------|-----------------|---------------------|
| Scheduler | Simple | Complex |
| Padding | High worst-case | Lower steady-state |
| CUDA graphs | Easier | Harder |
| Tail latency | Worse | Often better |
| Implementation | Training-like | Serving-specific |

### Memory Pool Sizing

Let **\(R\)** be **max concurrent** requests, **\(L_{\max}\)** **context** cap, **KV bytes/token** \(=\eta\). **Naive** upper bound:

\[
M_{\text{naive}} \approx R \cdot L_{\max} \cdot \eta.
\]

**Paged** pools allocate closer to **actual** \(\sum_i L_i\) plus **fragmentation** overhead—**constants** matter when **\(R\)** is **large**.

### Observability: What to Monitor

- **Batch size distribution** per decode step
- **KV cache utilization** (% of **page** pool **allocated**)
- **Preemption** counts (if supported)
- **End-to-end** **tokens/s** vs **per-GPU** **tokens/s** (**scaling** efficiency)

### Interaction with FlashAttention / FlashInfer

**FlashAttention** speeds **attention** **kernels**; **PagedAttention** speeds **allocation**. They **compose**: **FlashAttention-2** can read **\(K,V\)** from **non-contiguous** **blocks** via **gather** **indices**—the **bottleneck** shifts between **compute** and **HBM** **traffic** depending on **model** **width** and **batch**.

### Pseudocode: Iteration Scheduler (Educational)

```text
state: active = Map[request_id -> seq_state]
loop forever:
    # 1) Pull finished requests (EOS or max length)
    active -= finished

    # 2) Admit waiting requests whose prefill completed -> merge into decode batch
    active += ready_from_prefill_queue

    # 3) Build tensors: ragged or padded; attach block tables for KV
    logits = model.step(active)

    # 4) Sample / argmax next tokens; append to sequences
    update(active, logits)

    # 5) Metrics: tokens processed this iteration, KV pool usage
    emit_metrics()
```

!!! math-intuition "In Plain English"
    - Real engines **pipeline** **prefill** and **decode** on **different** **streams** or **GPUs**—this pseudocode **merges** both for **clarity**.
    - **Back-pressure**: when **KV** **pool** is **full**, **admit** **fewer** **new** **requests**—**PagedAttention** raises **ceiling**, does not **remove** **capacity** **planning**.

### Cost Model (Very Rough)

Let **\(N_{\text{tok}}\)** be **total** **live** tokens **this** iteration across **all** sequences. A **first-order** **FLOP** model for **decoder** **layers**:

\[
\text{FLOPs} \propto N_{\text{layer}} \cdot N_{\text{tok}} \cdot d_{\text{model}}^2
\]

(ignores **attention** **quadratic** **in** per-sequence length when **flattened**—**still** **subquadratic** in **\(N_{\text{tok}}\)** **if** **sequences** **do not** **interact**).

### When Continuous Batching Helps Least

- **Single** **user**, **single** **sequence**, **latency**-critical: batch size \(=1\) → **no** **multiplexing** **gain**.
- **Huge** **tensor-parallel** **overhead** **dominates** **micro-batches**: **need** **bigger** **local** batches or **pipeline** **parallelism**.

### Radix Attention (Name Drop)

**SGLang** introduces **radix** **trees** over **shared** **prompt** **prefixes**—**stronger** than **plain** **paging** when **many** **requests** **diverge** **late**. Interview one-liner: **radix** **caches** **entire** **shared** **subtrees** of **tokens** for **KV** **reuse**, complementary to **PagedAttention**’s **physical** **allocation**.

---

## Interview Takeaways

- **Static** batching **pads** to \(L_{\max}\)—wastes **compute** and **inflates** **latency** for **short** jobs.
- **Continuous** batching **mutates** the **active** set **each** decode **iteration**—higher **GPU** **utilization** under **skew**.
- **PagedAttention** implements **KV** as **logical→physical** **blocks**—reduces **external** **fragmentation** and enables **large** **shared** pools.
- **Prefill** vs **decode** have **different** **bottlenecks**—split **metrics** (**TTFT** vs **tokens/s**).
- **Throughput** ↑ often **hurts** **p99 latency** without **admission** **control**—know **Little’s law** intuition.
- **vLLM** exposes **`LLM`**, **`AsyncLLMEngine`**, and **OpenAI** **compatible** **HTTP**—same **PagedAttention** **core**.
- **Radix/SGLang**-style **prefix** **trees** reduce **duplicate** **KV** **compute** for **shared** **prompts**—orthogonal to paging.
- **Chunked prefill** avoids **head-of-line** blocking when **one** **prompt** is **extremely** **long**—pairs naturally with **continuous** **decode** **batching**.

## References

- Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models* — [OSDI](https://www.usenix.org/conference/osdi22/presentation/yu)
- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* — [SOSP / arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- vLLM documentation — [https://docs.vllm.ai](https://docs.vllm.ai)
- Pope et al. (2023), *Efficiently Scaling Transformer Inference* — [MLSys](https://arxiv.org/abs/2211.05102)
- Aminabadi et al., *DeepSpeed-FastGen* — chunked prefill + speculative decode
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) — radix attention + scheduling
- NVIDIA Triton Inference Server documentation — [https://docs.nvidia.com/deeplearning/triton-inference-server/](https://docs.nvidia.com/deeplearning/triton-inference-server/) — model orchestration patterns
