# Part 4 — Inference & serving

Latency, throughput, and cost are decided here: **decoding algorithms**, **KV cache** layout, **speculative decoding**, **continuous batching**, and **quantized** runtimes (GGUF, AWQ, server stacks).

## Goals

- Implement or reason about **greedy, beam, top-k, top-p**, and **temperature** sampling.
- Explain **KV cache** memory growth with context length and batch size.
- Map **vLLM / TGI / Ollama**-style serving to production constraints.

## Planned topics

| Topic | What you will get |
| --- | --- |
| Decoding strategies | Algorithms + trade-offs |
| KV cache | Memory model, batching implications |
| Speculative decoding | Draft + verify intuition |
| Continuous batching | Paged attention idea |
| Quantization for inference | AWQ, GGUF, llama.cpp family |
| LLM serving systems | Triton, vLLM, TGI, Ollama comparison |

## Status

Section shell for **Phase 1**; serving deep dives in **Phase 5**.
