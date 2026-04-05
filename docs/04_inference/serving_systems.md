# LLM Serving Systems

Production LLM APIs are **assembled** from **schedulers**, **kernels**, **network** **frontends**, and **observability**—this page names the **major** **stacks** and **where** each **fits**.

## Why This Matters for LLMs

Training produces **weights**; **serving** turns them into **reliable APIs** under **latency** **SLOs**, **multi-tenant** **load**, and **hardware** **heterogeneity**. The ecosystem spans **Python-first** stacks (**vLLM**, **TGI**), **C++/CUDA** **optimized** runtimes (**TensorRT-LLM**), **portable** **local** clients (**Ollama**), and **generic** **model** **servers** (**Triton**) that compose **ensemble** **graphs** with **custom** **backends**.

Interviewers expect you to **map** workloads to stacks: **offline** **batch** **embedding** jobs differ from **chat** **streaming**; **single** **GPU** **home** **lab** differs from **tensor-parallel** **multi-node** **clusters**.

A second dimension is **features**: **OpenAI-compatible** **HTTP**, **LoRA** **hot-swapping**, **speculative** **decoding**, **prefix** **caching**, **structured** **outputs**, **grammar** **constraints**—feature matrices change **monthly**, but **architectural** **patterns** (**continuous** **batching**, **paged** **KV**) are **stable**.

Third, **observability** and **autoscaling** tie to **queueing** **theory**—**tokens/s** is **not** **latency**; **p95** **TTFT** drives **UX** for **chat**.

---

## Core Concepts

### vLLM: Architecture Highlights

**vLLM** couples **PagedAttention** with **high-throughput** **CUDA** **kernels** and **OpenAI**-compatible **servers**. Core objects:

- **`LLM`**: **offline** **batched** **inference**.
- **`AsyncLLMEngine`**: **async** **production** **serving** with **request** **queues**.
- ****Worker** processes** map to **GPU** **ranks**; **tensor parallel** **shards** **layers**.

!!! math-intuition "In Plain English"
    - **vLLM** is **both** a **library** and a **server**—choose **API** **surface** based on **integration** **needs**.

### TGI (Text Generation Inference)

**TGI** is Hugging Face’s **Rust**/**Python** **server** optimized for **Transformers** models with **flash-attention** integrations, **continuous** **batching**, and **Safetensors** **loading**. Strong **ecosystem** **fit** if you already **host** on **HF** **Hub**.

### Triton Inference Server

**NVIDIA Triton** is **model**-**agnostic**: serves **ONNX**, **TensorRT** **engines**, **Python** **backends**, **ensemble** **pipelines**. LLMs appear as **one** **backend** (e.g., **TensorRT-LLM** **or** **vLLM** **custom** **backend**) behind a **standard** **gRPC/HTTP** ** façade**.

!!! math-intuition "In Plain English"
    - Pick Triton when you **already** **standardize** **on** it for **vision** + **NLP** + **reco**—**unified** **observability** and **routing**.

### Ollama (Local Deployment)

**Ollama** packages **models** + **runners** for **macOS**/**Linux**/**Windows** with a **simple** **CLI** and **REST**—**GGUF** **weights**, **CPU**/**GPU** **hybrid**, **low** **ops** **overhead** for **developers**.

### TensorRT-LLM

**TensorRT-LLM** builds **optimized** **CUDA** **graphs** with **fused** **kernels**, **INT4/INT8** **weight** **formats**, **multi-GPU** **tensor** and **pipeline** **parallelism**, and **strong** **NVIDIA** **hardware** **integration**.

\[
\text{latency} \approx f(\text{engine},\ \text{KV layout},\ \text{batch},\ \text{precision})
\]

!!! math-intuition "In Plain English"
    - **Highest** **throughput** often requires **export**/**compile** **time**—**not** **dynamic** **Python** **eager** **loops**.

### Deployment Patterns

| Pattern | When | Caveats |
|---------|------|---------|
| **Single GPU** | Prototypes, small models | **OOM** at long context + batch |
| **Tensor parallel (TP)** | Large **dense** models | **NCCL** **latency** on **small** batches |
| **Pipeline parallel (PP)** | Extremely **deep** stacks | **Bubble** **overhead** |
| **Multi-node** | Frontier sizes | **Fault** **tolerance**, **network** |

!!! math-intuition "In Plain English"
    - **TP** splits **layers** **horizontally**; **PP** splits **vertically**—many **LLM** **servers** **default** to **TP** first.

### vLLM: Operational Knobs (Examples)

Common **CLI** flags (names may vary by version):

- **`--tensor-parallel-size`**: **shard** **attention**/**MLP** across **GPUs** in **one** **node**.
- **`--max-model-len`**: **hard** **cap** on **context**—drives **PagedAttention** **table** **size**.
- **`--gpu-memory-utilization`**: **fraction** of **device** **memory** **pre-reserved** for **KV** **pool**.
- **`--enable-prefix-caching`**: **hash** **shared** **prompt** **prefixes** to **reuse** **KV** **blocks**.

!!! math-intuition "In Plain English"
    - **Tuning** **`gpu-memory-utilization`** is how you **trade** **OOM** **risk** vs **empty** **reserved** **pool**.

### Ray + vLLM (Distributed)

**Ray** orchestrates **multi-node** **vLLM** **clusters**—**workers** **register** with **GCS**, **requests** **route** through **Ray Serve** or **custom** **gateways**. Useful when **single** **node** **TP** **exceeds** **PCIe**/**NVLink** **topology** **limits**.

### Multi-LoRA and Adapter Serving

Production stacks may **host** **one** **base** **model** with **many** **LoRA** **adapters** **selected** **per** **request**. **Memory** **savings** vs **N**× **full** **fine-tunes**—**routing** **layer** maps **tenant** → **adapter** **weights**.

### Triton: Ensemble Pipelines

**Ensemble** **models** **chain** **preprocessing** → **LLM** → **postprocessing** (e.g., **JSON** **schema** **validation**) inside **Triton** **graphs**. **Benefit**: **one** **network** **hop** **inside** **datacenter** **fabric** vs **scattered** **microservices**.

### Security and Multi-Tenancy

- **API** **keys** / **mTLS** for **edge** **ingress**
- **Per-tenant** **rate** **limits** (**token** **bucket**)
- **No** **raw** **prompt** **logging** without **PII** **redaction**
- **Isolation** of **adapter** **weights** and **KV** **caches** in **shared** **GPU** **processes**—**implementation**-dependent **risk**

### Cost Modeling (Sketch)

**$/1M tokens** \(\approx\) **GPU** **$/hour** × **hours** / **tokens** **processed**. **Quantization** and **batching** **raise** **tokens/hour** on **same** **hardware**—often **cheaper** than **adding** **replicas**.

\[
\text{cost per token} \propto \frac{1}{\text{tokens per second} \times \text{utilization}}.
\]

### Comparison Table (Qualitative — 2026 Snapshot)

| System | Throughput | Latency | Features | Notes |
|--------|------------|---------|----------|-------|
| **vLLM** | **High** | **Good** | PagedAttention, OpenAI API | Python/CUDA |
| **TGI** | **High** | **Good** | Hub integration, Rust core | HF ecosystem |
| **TensorRT-LLM** | **Very High** | **Excellent** | Fused kernels, quantization | NVIDIA-only |
| **Triton** | **Depends on backend** | **Depends** | Multi-model routing | Ops consolidation |
| **Ollama** | **Moderate** | **Good local** | Easy UX | GGUF focus |

**Caveat**: numbers **depend** on **model**, **GPU**, **precision**, **batch**, **sequence** lengths—treat table as **orientation**, not **benchmarks**.

### TensorRT-LLM Feature Highlights

- **Inflight batching** (continuous batching) with **KV** **cache** **management**
- **FP8/INT4/INT8** **weight** **formats** with **fused** **GEMMs**
- **Multi-GPU** **tensor**, **pipeline**, and **expert** **parallel** for **MoE**
- **CUDA graph** **capture** for **low** **kernel** **launch** **overhead**

!!! math-intuition "In Plain English"
    - **Compile** **time** can be **minutes**—**CI** must **cache** **engines** for **known** **model** **versions**.

### Kubernetes Patterns

- **Deployment** per **model** **SKU** with **HPA** on **GPU** **metrics**
- **Sidecar** **token** **counters** for **billing**
- **Pod** **topology** **spread** **constraints** for **NVLink** **locality**
- **Readiness** **probes** that **warm** **CUDA** **contexts** **without** **user** **traffic**

### Load Balancing Sticky Sessions

**Stateful** **LLM** **sessions** (**long** **chats**) **benefit** from **sticky** **routing** to **reuse** **prefix** **KV** **cache** on **same** **replica**—**classic** **L7** **LB** **feature**.

### SGLang / RadixAttention (Ecosystem)

**SGLang** focuses on **structured** **generation** **programs** and **radix** **attention** for **prefix** **sharing**—often **compared** alongside **vLLM** for **workloads** with **heavy** **prompt** **reuse**.

### Observability Stack

- **Prometheus** **metrics**: **time** **per** **forward** **stage**, **KV** **pool** **usage**, **queue** **depth**
- **OpenTelemetry** **traces** across **gateway** → **router** → **worker**
- **Structured** **logs** with **request** **ids**, **no** **PII** **payloads**

### OpenAI-Compatible Layers

Many stacks expose **`/v1/chat/completions`** with **streaming** **SSE**—**token** **deltas** reduce **time-to-first** **visible** **token** **latency** **perceived** by **users** even when **server** **TTFT** is **fixed**.

---

## Minimal Client Examples

### HTTP Streaming (Conceptual Python)

```python
"""
Minimal OpenAI-compatible streaming client (requests + SSE-style lines).
Replace BASE_URL with your vLLM/TGI/OpenAI endpoint.
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

import requests


def stream_chat(
    prompt: str,
    base_url: str = "http://127.0.0.1:8000/v1/chat/completions",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    api_key: Optional[str] = None,
) -> Iterator[str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.7,
    }

    with requests.post(base_url, headers=headers, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


if __name__ == "__main__":
    print("".join(stream_chat("Say hello in one sentence.")))
```

!!! math-intuition "In Plain English"
    - **Streaming** **does not** **reduce** **compute**—it **overlaps** **network** **I/O** with **generation**.

### Ollama REST (Illustrative)

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the KV cache linear in sequence length?",
  "stream": false
}'
```

---

## Autoscaling and Capacity Planning

Let **\(C\)** be **max** **sustainable** **tokens/s** per **replica** at **target** **latency**. **Replicas** **\(R\)** needed for **offered** **load** \(\lambda\) (tokens/s):

\[
R \approx \Bigl\lceil \frac{\lambda}{C} \cdot \text{safety factor} \Bigr\rceil.
\]

**HPA** on **GPU** **utilization** is **noisy**—prefer **custom** **metrics** (**queue** **depth**, **batch** **tokens**).

### Cold Start vs Warm GPU

**Cold** **start** includes **model** **weight** **load** from **disk**/**network**, **CUDA** **context** **init**, **kernel** **JIT**. **Warm** **pools** keep **replicas** **loaded**—**trade** **memory** **cost** for **latency** **SLOs**.

\[
T_{\text{cold}} \gg T_{\text{warm decode step}} \quad \text{(often).}
\]

### Batching vs Microbatching

**Microbatching** aggregates **several** **requests** **inside** **one** **forward** **iteration**—core of **continuous** **batching**. **Cross** **replica** **batching** (aggregation **proxy**) is **rarer** due to **state** **affinity**.

### Fault Tolerance

**GPU** **OOM**, **NCCL** **timeouts**, **driver** **resets** require **supervisor** **loops** to **restart** **workers** **without** **draining** **whole** **cluster**. **Checkpoint** **KV** **to** **CPU** is **expensive**—most **chat** **systems** **prefer** **client-side** **retry** **with** **truncated** **context**.

### Hardware Tiers

| GPU class | Typical sweet spot |
|-----------|-------------------|
| **Consumer** 24GB | 7B–13B **INT4** **chat** |
| **A100 80GB** | Many **concurrent** **users** **FP16** |
| **H100** | **FP8** **kernels**, **higher** **bandwidth** |

### Model Packaging Formats

- **Safetensors**: **safe** **mmap** **loading**, **no** **exec** **on** **pickle**
- **GGUF**: **CPU**/**edge** **friendly** **blocks**
- **TensorRT** **engines**: **compiled** **graphs** **locked** to **GPU** **arch**

### SLA Math: Tokens per Minute per User

If **product** **promises** **600** **tokens/minute** **per** **seat**, aggregate **arrival** **rate** \(\lambda\) scales with **seats** **×** **activity** **factor**—**dimension** **capacity** **plans**.

---

## Interview Takeaways

- **vLLM** = **PagedAttention** + **CUDA** **kernels** + **Python** **APIs**; **default** **choice** for **many** **GPU** **shops**.
- **TGI** = **Rust**/**Python** **server** with **strong** **HF** **Hub** **integration**.
- **TensorRT-LLM** = **NVIDIA**-**optimized** **compiled** **graphs**—**best** **throughput** when you can **pay** **compile** **cost**.
- **Triton** = **generic** **multi-model** **router**—**LLM** is **one** **backend** among **many**.
- **Ollama** = **developer**-**friendly** **local** **deployment** (**GGUF**).
- **TP** vs **PP** vs **single** **GPU** map to **model** **size** and **network** **environment**.
- **Streaming** **APIs** improve **perceived** **latency**; **SLOs** still need **TTFT** + **inter-token** **delay** **budgets**.

## References

- Kwon et al. (2023), *PagedAttention / vLLM* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Hugging Face, *Text Generation Inference* — [https://huggingface.co/docs/text-generation-inference](https://huggingface.co/docs/text-generation-inference)
- NVIDIA, *Triton Inference Server* — [https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)
- NVIDIA, *TensorRT-LLM* — [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- Ollama — [https://ollama.com](https://ollama.com)
