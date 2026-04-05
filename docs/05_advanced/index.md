# Part 5 — Advanced Topics

Retrieval-augmented generation, agents with tools, long-context modeling, multimodal systems, scaling laws and emergent behavior, evaluation and benchmarking, and hallucination mitigation with safety engineering. This part connects **foundation-model theory** to **systems you ship**: grounded answers, tool-using assistants, efficient attention at extreme sequence lengths, and measurable quality.

---

## Goals

After completing Part 5 you will be able to:

- Design an end-to-end RAG stack: chunking, embeddings, hybrid retrieval, and faithfulness-aware prompting
- Explain ReAct-style agent loops, tool schemas, and short- versus long-term memory patterns
- Compare FlashAttention, sliding-window attention, RoPE scaling, and sparse attention families for long sequences
- Describe CLIP-style alignment, vision encoders, and projector-based multimodal LLMs (e.g., LLaVA-class)
- State scaling-law relationships, emergent capabilities, and test-time compute trade-offs
- Select benchmarks (MMLU, HumanEval, MT-Bench) and interpret contamination risks
- Classify hallucinations, apply grounding and detection ideas, and reference standard safety benchmarks

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [Retrieval-Augmented Generation](rag.md) | Indexing, dense/sparse/hybrid retrieval, vector stores, evaluation |
| 2 | [Agents & Tool Use](agents.md) | ReAct, function calling, planning, memory, multi-agent patterns |
| 3 | [Long-Context Modeling](long_context.md) | FlashAttention, sliding windows, RoPE scaling, ring/sparse attention |
| 4 | [Multimodal LLMs](multimodal.md) | CLIP, ViT/SigLIP, LLaVA, cross-modal alignment |
| 5 | [Emergent Capabilities & Scaling Laws](emergent_capabilities.md) | Kaplan/Chinchilla, compute-optimal training, emergent skills, test-time scaling |
| 6 | [Evaluation & Benchmarking](evaluation.md) | Perplexity, knowledge/code/chat benchmarks, harness usage, gaming |
| 7 | [Hallucination & Safety](hallucination_safety.md) | Hallucination types, grounding, red-teaming, guardrails, TruthfulQA/BBQ |

---

Every page includes `!!! math-intuition` callouts, LaTeX where it clarifies scaling and algorithms, runnable Python where it helps, and **Interview Takeaways** plus primary **References** for deeper reading.
