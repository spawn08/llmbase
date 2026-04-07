# LLM Serving Systems

## Why This Matters for LLMs

Knowing how to **serve** a model is as important as understanding its architecture. Training produces static weights; serving turns them into **reliable**, **metered**, **secure** APIs under bursty traffic. The stack you choose—vLLM, Text Generation Inference (TGI), TensorRT-LLM, Ollama, or NVIDIA Triton—determines not only peak tokens per second but also operational complexity: how you load LoRA adapters, stream tokens to clients, enforce rate limits, and observe queue depths when GPUs saturate.

Inference economics are brutal: accelerators rent for dollars per hour, KV cache grows with concurrent users, and autoregressive decoding is often **memory-bandwidth** bound before it is FLOP bound. A design that ignores **continuous batching**, **paged KV**, **quantization**, or **tensor parallelism** can miss SLOs while burning budget. Interviewers therefore ask “design an LLM serving system” to see whether you connect **model** choices to **systems** primitives: batching, scheduling, caching, autoscaling, and failure modes (OOM, preemption, straggler GPUs).

Third, the ecosystem moves quickly—new kernels, new quant formats, new routing layers—but **metrics** remain stable. **Time-to-first-token (TTFT)**, **inter-token latency (ITL)**, **throughput**, **queue depth**, and **cost per million tokens** are how teams compare revisions. You should be able to sketch a **Little’s law** argument, a **horizontal scaling** story with sticky sessions if needed, and where a **CDN** or **edge** runtime (Ollama, mobile) fits versus a **datacenter** GPU fleet.

---

## Core Concepts

### What Makes LLM Serving Hard?

Let \(N\) be the number of parameters, \(d_{\text{model}}\) hidden size, \(L\) context length, batch \(B\). A coarse **memory** budget for weights alone is:

\[
M_{\text{weights}} \approx 2 \cdot N \cdot \text{bytes per parameter}
\]

for FP16/BF16 (two bytes per parameter).

!!! math-intuition "In Plain English"
    - **7B** params \(\times\) **2** bytes \(\approx\) **14** GiB weights—before **KV**, **activations**, **CUDA** context, and **fragmentation**.
    - **Quantization** attacks this term directly; **multi-GPU** **tensor parallel** shards it.

Decode **step** time per layer often scales roughly linearly with **active tokens** \(B\) times **model width** for matmul-dominated regions, but **attention** can add **per-sequence** terms. A simplified **roofline** intuition: if you are **memory bound**, raising **batch** increases **arithmetic intensity** until you become **compute bound**.

\[
\text{tokens/sec} \approx \frac{B_{\text{eff}}}{\tau_{\text{step}}},
\]

where \(\tau_{\text{step}}\) is wall-clock per decode iteration for the engine.

!!! math-intuition "In Plain English"
    - **\(B_{\text{eff}}\)** is how many sequences participate **this** iteration—continuous batching raises it under load.
    - **Throughput** is not the same as **per-user** **latency**—see queueing below.

### Key Metrics

- **TTFT**: time until first generated token is available (dominated by **prefill** and **queue** wait).
- **ITL** (inter-token latency): spacing between tokens **after** generation starts—dominated by **decode** steady state.
- **Aggregate throughput**: sum of **new** tokens per second across all users—**capacity** planning metric.
- **Per-user TPS**: inverse of ITL for streaming—**UX** metric.

\[
\mathrm{ITL} \approx \tau_{\text{decode}}, \qquad
\mathrm{TPS}_{\text{user}} \approx \frac{1}{\mathrm{ITL}}.
\]

!!! math-intuition "In Plain English"
    - Users **feel** ITL as “typing speed” of the model; TTFT is “time before the first word appears.”

!!! example "Worked Example: Throughput vs Per-User TPS"
    Suppose one GPU sustains **2000** new tokens per second aggregate in decode for a 7B model with continuous batching, and **50** concurrent streaming users each receive tokens fairly.

    **Naive** average per user: \(2000 / 50 = 40\) tokens/s if perfectly fair and decode-bound.

    If instead one **large** batch job consumes **half** the slots, interactive users see **worse** ITL—this is why **priority queues** and **chunked prefill** exist.

    Numerically: **40 TPS/user** \(\Rightarrow\) **25 ms** ITL on average; if ITL rises to **50 ms** (20 TPS), users perceive **sluggish** streaming even if aggregate throughput is unchanged—**fairness** and **scheduling** matter.

### vLLM

**vLLM** combines **PagedAttention**, **continuous batching**, fused CUDA kernels, and optional **OpenAI-compatible** HTTP servers. Core API surfaces include offline `LLM`, async `AsyncLLMEngine`, and `vllm serve` for production.

!!! math-intuition "In Plain English"
    - Think **vLLM** when you want **maximum** **GPU** **efficiency** for **variable-length** **chat** on **NVIDIA** **datacenter** cards with **Python** **ecosystem** **integration**.

### Text Generation Inference (TGI)

**TGI** is Hugging Face’s Rust-centric server with **Flash Attention**, **continuous batching**, and tight **Hub** integration (model cards, safetensors). It supports features like **grammar** constraints and **watermarking** in many builds.

### TensorRT-LLM

**TensorRT-LLM** compiles models to **highly** optimized **CUDA** graphs with **in-flight batching**, **INT4/INT8/FP8** weights, and **multi-GPU** **tensor** and **pipeline** parallelism. It targets **maximum** throughput on **NVIDIA** hardware when you can pay **build**/**compile** complexity.

\[
\text{latency} \approx f(\text{engine},\ \text{KV layout},\ \text{batch},\ \text{precision}).
\]

!!! math-intuition "In Plain English"
    - **Latency** is not a single number—it is a **distribution** over batch, sequence lengths, and **hardware** **topology**.

### Ollama

**Ollama** packages **GGUF** models with a simple **CLI** and **REST** API, using **llama.cpp**-style runtimes under the hood. It optimizes for **developer** **experience** and **local** deployment rather than **maximum** **multi-tenant** **throughput**.

### Triton Inference Server

**NVIDIA Triton** is **backend-agnostic**: TensorRT engines, ONNX, Python backends, **TensorRT-LLM** backend, or custom C++ plugins. It is valuable when one **platform** must serve **vision**, **recsys**, and **LLMs** behind a **unified** **routing** layer.

### Comparison Table

| System | Batching | Quantization | Multi-GPU | API | Ease of Use | Best For |
|--------|----------|--------------|-----------|-----|-------------|----------|
| vLLM | Continuous | GPTQ / AWQ / FP8 | TP / PP | OpenAI-compatible | Medium | Production GPU |
| TGI | Continuous | GPTQ / AWQ / EETQ | TP | Custom / HF | Medium | HF ecosystem |
| TensorRT-LLM | In-flight | INT4 / INT8 / FP8 | TP / PP | Triton / C++ | Hard | Peak NVIDIA perf |
| Ollama | Basic / llama.cpp | GGUF | Limited | REST | Easy | Local dev |
| Triton | Backend-defined | Backend-defined | Yes | gRPC/HTTP | Medium–Hard | Multi-model ops |

!!! math-intuition "In Plain English"
    - Treat this table as **orientation**—your **model**, **GPU**, and **precision** dominate **A/B** results.

### Deployment Patterns

| Pattern | Strength | Risk |
|---------|----------|------|
| Single GPU + quant | Simple | OOM at large context |
| Tensor parallel | Fits large models | NCCL overhead at small batch |
| Pipeline parallel | Very deep / wide stacks | Bubble inefficiency |
| Multi-node | Frontier sizes | Fault tolerance harder |

\[
\text{cost per million tokens} \propto \frac{\text{\$/hour}}{\text{tokens/hour}}.
\]

!!! math-intuition "In Plain English"
    - **Anything** that raises **tokens/hour** on the same rent **lowers** **$/token**—batching and quant are **first-class** **financial** levers.

Let sustainable capacity be **\(C = 1.2 \times 10^{6}\)** tokens/hour per **vLLM** replica at **p95 latency** target. Offered load **\(\lambda = 3.0 \times 10^{6}\)** tokens/hour.

\[
R = \left\lceil \frac{\lambda}{C} \right\rceil = \left\lceil \frac{3.0}{1.2} \right\rceil = 3 \ \text{replicas (ignoring safety margin)}.
\]

!!! math-intuition "In Plain English"
    - Add **headroom** (e.g. **1.2–1.5×**) for **spikes** and **rolling** **deploys**.
    - **HPA** on **GPU** **utilization** alone is **noisy**—prefer **queue** **depth** or **SLO** **breach** **rate**.

!!! example "Worked Example: Replicas from Offered Load"
    **Given:** \(C = 1.2 \times 10^{6}\) tokens/hour per replica, \(\lambda = 3.0 \times 10^{6}\) tokens/hour offered.

    **Step 1:** Divide load by capacity: \(\lambda / C = 3.0 / 1.2 = 2.5\).

    **Step 2:** Take the ceiling because fractional replicas are not allowed: \(R = \lceil 2.5 \rceil = 3\).

    **Step 3:** Add a **safety factor** (not shown in the formula)—ops teams often provision **\(4\)** replicas if they need **N+1** fault tolerance or expect **30%** **traffic** **spikes**.

### Little’s Law for Queues

For stable queues:

\[
L = \lambda W,
\]

where \(L\) is average number of requests **in system**, \(\lambda\) is arrival rate, \(W\) is average **time in system**.

!!! math-intuition "In Plain English"
    - If **\(\lambda\)** doubles and **service** time stays fixed, **average** **queue** **depth** doubles—**latency** rises **unless** you **scale** **capacity** or **shed** **load**.

??? deep-dive "Deep Dive: Speculative Decoding in Serving Stacks"
    **Speculative decoding** runs a **small** **draft** model (or additional heads) to propose tokens verified by the **large** model, improving **tokens/sec** under some regimes. It interacts with **batching**: verification steps may **reshape** effective batch sizes. Mention **Medusa**/**EAGLE** when interviewer asks how to break the **serial** **decode** barrier—tradeoffs include **memory** for **extra** heads and **acceptance** **rate** variability.

??? deep-dive "Deep Dive: Multi-LoRA and Adapter Routing"
    One **base** model can serve **many** **LoRA** adapters selected per request. Routing overhead stays small if **weights** are **preloaded**; **KV** remains **shared** architecture. Risk: **adapter** **contention** on **single** GPU—**shard** **tenants** or **cap** **concurrent** **distinct** adapters.

### Security and Multi-Tenancy (Production Checklist)

- **Authentication**: API keys, **mTLS** between gateway and model pods, **OIDC** for human operators.
- **Authorization**: per-tenant **rate limits** (token buckets keyed by tenant id).
- **Data handling**: avoid logging **raw** prompts in production; **redact** PII; store **hashes** of policy-violating inputs if needed for **abuse** response.
- **Isolation**: separate **namespaces** or **clusters** for regulated workloads; **sidecar** validators for **tool** calls.
- **Supply chain**: verify **model** **checksums** (safetensors hashes) on deploy; pin **container** **digests**.

### Kubernetes and Autoscaling Patterns

Common pattern: **Ingress** \(\to\) **API gateway** \(\to\) **vLLM** **Deployment** with **HPA** on **custom** metrics (queue depth from **Prometheus** exporter). For **multi-node** **tensor parallel**, schedule **all** **ranks** of a **model** **replica** as one **pod group** or **co-located** containers to keep **NVLink** locality.

\[
\text{desired replicas} = \left\lceil \frac{\text{queue depth}}{\text{target depth per replica}} \right\rceil.
\]

!!! math-intuition "In Plain English"
    - **Target queue depth** is a **tunable** **SLO** **knob**: too **low** \(\Rightarrow\) **over-provisioning**; too **high** \(\Rightarrow\) **latency** **SLO** **miss**.

Allow **\(r = 1000\)** tokens per **minute** per tenant with **burst** **\(b = 500\)** tokens. A **token bucket** updates:

\[
\text{bucket}_{t+1} = \min\left(b,\ \text{bucket}_t + r \cdot \Delta t - \text{requested tokens}\right).
\]

!!! math-intuition "In Plain English"
    - **Burst** absorbs **short** spikes; **sustained** rate is still capped by **\(r\)**—prevents one tenant from **starving** others when **KV** memory is **shared**.

!!! example "Worked Example: Token Bucket Rate Limit"
    **Setup:** \(r = 1000\) tokens/minute refill, burst capacity \(b = 500\), current bucket \(= 500\).

    **Request 1:** Client asks **300** tokens. New bucket \(= \min(500,\ 500 - 300) = 200\). **Allowed.**

    **Request 2 (same minute):** Client asks **250** tokens. Need **250** but only **200** remain—**reject** or **queue** depending on policy.

    **After 60 seconds:** Bucket refills by \(r \cdot \Delta t = 1000\) tokens/min \(\times 1\) min in the discrete minute tick \(\Rightarrow\) capped at **500**: bucket returns to **500**.

---

## Code

The script below **always** runs: it defines a **local mock HTTP handler** (stdlib only) that mimics OpenAI **streaming** chunks, includes **requests**-based client code guarded for missing dependency, optional **vLLM** offline generation, and optional **Hugging Face** `InferenceClient` when `huggingface_hub` is installed.

```python
"""
LLM serving examples: mock OpenAI stream, optional vLLM, optional HF client.

Dependencies: standard library + threading; `requests` for the client line (optional).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def mock_openai_stream_handler() -> None:
    """Single-endpoint mock: POST /v1/chat/completions returns SSE token deltas."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            tokens = ["Hello", ",", " ", "world", "!"]
            for i, tok in enumerate(tokens):
                chunk = {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": tok},
                            "finish_reason": None if i < len(tokens) - 1 else "stop",
                        }
                    ]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        stream_chat_local(port)
    finally:
        server.shutdown()


def stream_chat_local(port: int) -> None:
    try:
        import requests
    except ImportError:
        print("Install requests to exercise HTTP client: pip install requests")
        return

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "mock-llm",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True, timeout=10) as r:
        r.raise_for_status()
        acc: list[str] = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                acc.append(delta)
    print("Mock stream assembled:", "".join(acc))


def try_vllm_offline() -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        print("vLLM not installed; skip offline LLM:", exc)
        return
    try:
        llm = LLM(
            model="meta-llama/Llama-3.2-1B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=2048,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=32)
        out = llm.generate(["Serving systems combine batching and memory."], sp)
        print(out[0].outputs[0].text)
    except Exception as exc:  # noqa: BLE001 — doc demo
        print("vLLM runtime skipped (CUDA/model):", exc)


def try_hf_inference_client() -> None:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("huggingface_hub not installed; skip TGI-style client demo.")
        return
    # Requires HF_TOKEN and deployed endpoint for real calls; document only.
    print("InferenceClient ready — set endpoint=... and token=... for production TGI.")


if __name__ == "__main__":
    mock_openai_stream_handler()
    try_vllm_offline()
    try_hf_inference_client()
```

!!! math-intuition "In Plain English"
    - The **mock** server proves **SSE** parsing—swap **URL** for **vLLM**/**TGI** when running locally.
    - **Streaming** reduces **time-to-first** **visible** **character** even when total compute is fixed.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Compare **TTFT** vs **ITL**—what system components dominate each?
        *Answer:* **TTFT** spans queue wait, scheduler admission, **prefill** over the prompt (often large attention matmuls), and first-token serialization—dominated by prefill compute + queueing under load. **ITL** (inter-token latency) is steady-state **decode**: one token per step per stream, dominated by memory bandwidth to stream weights and KV and by batch contention—users perceive ITL as “typing speed” after the first token.
    2. Why is LLM decode often **memory-bandwidth** bound, and how does **batching** change the roofline?
        *Answer:* Each decode step touches full model weights and growing KV for roughly **one** new token per sequence—arithmetic intensity (FLOPs/byte) is low at \(B=1\), so HBM bandwidth caps tokens/sec. Increasing **batch** \(B\) amortizes weight reads across more tokens per step, raising arithmetic intensity until the kernel becomes **compute**-bound on tensor cores—classic roofline move from memory-left to compute-limited.
    3. Explain **continuous batching** vs **static** batching for serving; name one downside of continuous batching operationally.
        *Answer:* **Static** batching pads to \(\max\) length and runs the group to completion—simple but wastes slots and ties short jobs to long ones. **Continuous** batching swaps sequences every iteration for higher GPU utilization. **Downside:** dynamic shapes complicate CUDA graphs, kernel specialization, and debugging; p95 latency can **worse** if admission control or batch caps are wrong (starvation, oversized batches).
    4. How does **PagedAttention** help **multi-tenant** serving **without** changing model math?
        *Answer:* PagedAttention only changes **KV storage layout** (logical tokens → physical blocks in a pool)—attention math and softmax are identical to dense KV if gathers are correct. It reduces over-reservation and fragmentation so **more** concurrent sequences fit in the same HBM, improving **capacity** (tenants/sec) without altering logits.
    5. When would you pick **TensorRT-LLM** over **vLLM**, and what do you trade away?
        *Answer:* Choose **TensorRT-LLM** when you need maximum **throughput** on NVIDIA hardware with compiled engines, FP8/INT4 kernels, aggressive **CUDA graphs**, and in-flight batching—typical for large-scale production after engineering invest in build pipelines. You trade **flexibility** (long compile times, version pinning, harder dynamic research workflows) versus **vLLM**’s faster iteration and Python ecosystem at sometimes lower peak perf.
    6. How does **Little’s law** relate autoscaling decisions to **queue depth** SLOs?
        *Answer:* \(L=\lambda W\): for stable load, average queue depth scales linearly with arrival rate times mean latency. If **SLO** caps end-to-end latency \(W\), rising \(\lambda\) forces either more replicas (lower per-replica \(\lambda\)) or admission shedding—**autoscale on queue depth** tracks Little’s law directly: target depth corresponds to target \(W\) at observed \(\lambda\).
    7. What is **in-flight batching** in TensorRT-LLM, and how does it relate to continuous batching conceptually?
        *Answer:* **In-flight batching** lets the engine **add/remove** requests between kernel launches while a batch of generations is “in flight”—similar spirit to iteration-level scheduling: the effective batch tensor changes step to step without draining all sequences. Conceptually it is vendor **continuous batching** inside a compiled TRT engine, paired with KV block managers like vLLM’s paging.
    8. Why might **Ollama** be unsuitable for a **10k RPS** **public** API without additional architecture?
        *Answer:* Ollama targets **local** dev ergonomics and llama.cpp-class single-node throughput—not horizontal multi-tenant isolation, SLA dashboards, or global load balancing. **10k RPS** needs replicated GPU fleets, autoscaling, rate limiting, multi-zone failover, and observability—wrap Ollama-like runtimes in **Kubernetes**, gateways, and queues rather than exposing one daemon.
    9. How would you add **multi-LoRA** routing to a **single** **base** model deployment?
        *Answer:* Load **N** small LoRA adapters into GPU memory (or tiered storage) keyed by tenant/task; each request carries `adapter_id` resolved at prefill to select the correct **delta** for forward passes while **sharing** base weights and KV layout. Optimize: **batched** LoRA for same adapter, cap distinct adapters per GPU to avoid thrash, and reference-count KV for shared prefixes across adapters.
    10. What metrics would you alert on before **GPU** **OOM** during **traffic** spikes?
        *Answer:* **GPU HBM utilization %**, **KV block pool** saturation, **pending prefill** queue depth, **OOM** / **CUDA malloc** failure rate, **eviction** or **admission reject** counts, and **time-to-grant** KV allocation. Early warning: sustained **>85%** memory + growing wait times—scale out or tighten **max_model_len** / concurrency before hard failures.

!!! interview "Follow-up Probes"
    - “Throughput doubled but p95 latency worsened—what happened?” (batch too aggressive, queueing, network, cache misses.)
    - “How do you roll out a new quantized weight format without downtime?” (blue/green, shadow traffic, canary.)
    - “Where does **prefix caching** sit in the architecture?” (KV reuse layer—often collocated with scheduler.)

!!! key-phrases "Key Phrases to Use in Interviews"
    - “We separate **prefill** and **decode** metrics because they hit different bottlenecks.”
    - “**Continuous batching** maximizes **active** tokens per step; **PagedAttention** maximizes **concurrent** sequences per GB.”
    - “**TensorRT-LLM** trades **compile** complexity for **kernel** fusion and **in-flight** batching on NVIDIA.”
    - “**Triton** is our **router**; the **LLM** is one **backend** among **CV** and **recsys** models.”
    - “Autoscale on **queue** **depth** and **p95** **TTFT**, not raw GPU **utilization** alone.”

---

## References

- Kwon et al. (2023), *PagedAttention / vLLM* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- vLLM documentation — [https://docs.vllm.ai](https://docs.vllm.ai)
- Hugging Face, *Text Generation Inference* — [https://huggingface.co/docs/text-generation-inference](https://huggingface.co/docs/text-generation-inference)
- NVIDIA, *TensorRT-LLM* — [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- NVIDIA, *Triton Inference Server* — [https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)
- Ollama — [https://ollama.com](https://ollama.com)
- Yu et al. (2022), *Orca* — continuous batching foundations — [USENIX OSDI](https://www.usenix.org/conference/osdi22/presentation/yu)
- Lewis et al. (2020), *Retrieval-Augmented Generation* — serving + retrieval patterns (related systems) — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
