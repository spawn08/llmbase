# Part 4 — Inference and Serving

The engineering that determines latency, throughput, and cost when deploying LLMs in production. From decoding algorithms to KV cache optimization, speculative decoding, continuous batching, and production serving systems.

---

## Goals

After completing Part 4 you will be able to:

- Implement greedy, beam search, top-k, top-p, and temperature decoding from scratch
- Calculate KV cache memory requirements for any model configuration
- Explain speculative decoding and prove it produces the same distribution as the target model
- Design a continuous batching system with PagedAttention for memory efficiency
- Choose the right serving framework for a given deployment scenario

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [Decoding Strategies](decoding_strategies.md) | Greedy, beam search, top-k, top-p, temperature, repetition penalty |
| 2 | [KV Cache](kv_cache.md) | Memory layout, GQA savings, PagedAttention, compression |
| 3 | [Speculative Decoding](speculative_decoding.md) | Draft-verify paradigm, acceptance probability, speedup analysis |
| 4 | [Continuous Batching](continuous_batching.md) | Iteration-level scheduling, PagedAttention, prefix caching |
| 5 | [Quantization for Inference](quantization_inference.md) | GPTQ, AWQ, GGUF, llama.cpp, quality benchmarks |
| 6 | [LLM Serving Systems](serving_systems.md) | vLLM, TGI, TensorRT-LLM, Ollama, deployment patterns |

---

Every page includes plain-English math walkthroughs, worked numerical examples, runnable Python code, and FAANG-level interview questions.
