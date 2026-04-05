# Part 4 — Inference and Serving

Autoregressive LLMs are trained with next-token cross-entropy, but **production** behavior is governed by decoding policies, memory layout (KV cache), batching schedulers, quantization, and serving runtimes. Part 4 connects the **mathematics of sampling** to **systems engineering**: what you implement in PyTorch for research versus what vLLM or TensorRT-LLM does under load.

---

## Goals

After completing Part 4 you will be able to:

- Implement greedy, beam, top-\(k\), top-\(p\), and temperature sampling from first principles on logits
- Explain why naive autoregressive attention is \(O(T^2)\) per forward without a KV cache, and size cache memory for MHA/MQA/GQA
- Describe speculative decoding and acceptance sampling; sketch Medusa-style parallel heads
- Contrast static batching with continuous batching and PagedAttention-style KV virtual memory
- Compare post-training quantization families (GPTQ, AWQ, GGUF) and INT4/INT8 deployment paths
- Map major serving stacks (vLLM, TGI, Triton, Ollama, TensorRT-LLM) to throughput/latency trade-offs

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [Decoding Strategies](decoding_strategies.md) | Greedy, beam search, top-\(k\)/top-\(p\), temperature, repetition penalties |
| 2 | [KV Cache](kv_cache.md) | \(O(T^2)\) cost, cache tensors, MQA/GQA, eviction |
| 3 | [Speculative Decoding](speculative_decoding.md) | Draft/verify, acceptance rates, Medusa |
| 4 | [Continuous Batching & PagedAttention](continuous_batching.md) | Orca scheduling, vLLM paging, fragmentation |
| 5 | [Quantization for Inference](quantization_inference.md) | GPTQ, AWQ, GGUF, error bounds |
| 6 | [LLM Serving Systems](serving_systems.md) | vLLM, TGI, Triton, Ollama, TensorRT-LLM |

---

Each page pairs rigorous probability notation with runnable Python and pointers to canonical papers so you can defend design choices in **research** and **systems** interviews.
