# LLMBase

A comprehensive, engineer-focused guide to Large Language Models — built from the ground up for deep understanding and interview readiness.

---

## What is LLMBase?

LLMBase is an open knowledge base that takes you from the foundational mathematics of language modeling through modern Transformer architectures, training pipelines, and frontier research. Every concept includes plain-English explanations, step-by-step math walkthroughs, runnable code with full imports, and the specific questions that top tech companies ask in interviews.

This is not a surface-level overview. Each page is written to give you the depth needed to **explain concepts from first principles** — the standard expected at companies like Google, Meta, OpenAI, and Anthropic.

---

## Who is this for?

- **Software engineers** transitioning into AI/ML or applied LLM roles who need to build real understanding, not just API familiarity.
- **Interview candidates** preparing for ML fundamentals, LLM system design, and GenAI-focused rounds at top tech companies.
- **Practitioners** who want a single reference they can return to — with code they can run and math they can trace by hand.

---

## What you need before starting

- **Python 3.11+** — all code examples are self-contained and runnable.
- **Calculus and linear algebra** at an engineering level — gradients, matrix multiplication, eigenvalues.
- **Basic PyTorch** — tensors and `torch.nn` modules. Introduced gradually where needed.

---

## The learning path

LLMBase is organized into eight progressive sections. Start with Deep Learning Fundamentals if you need to build or refresh your foundations, then continue through each section in order — each builds on the previous one.

### [Deep Learning Fundamentals](00_prerequisites/index.md)

The building blocks that every neural network is made of. Perceptrons, activation functions, backpropagation, loss functions, CNNs, RNNs, and the encoder-decoder paradigm. Start here if you need to solidify your understanding of how neural networks learn before diving into language-specific concepts.

### [Foundations](01_foundations/index.md)

The math and theory that existed before the Transformer. Probability, embeddings, RNNs, sequence-to-sequence models, information theory, and the derivation of scaled dot-product attention. Every equation is walked through with real numbers.

### [Core Architectures](02_core_architectures/index.md)

How Transformers work at the tensor level. The full block (MHA, FFN, residual stream), positional encodings (sinusoidal, RoPE, ALiBi), and the three architecture families (GPT, BERT, T5). Plus Mixture of Experts and state-space models (Mamba).

### [Training and Alignment](03_training/index.md)

Pre-training pipelines, distributed training, quantization, instruction tuning (SFT), RLHF, DPO, and parameter-efficient fine-tuning (LoRA, QLoRA). How models go from raw text to following instructions.

### [Inference and Serving](04_inference/index.md)

Decoding strategies, KV cache mechanics, speculative decoding, continuous batching, and serving systems (vLLM, TGI, Ollama). The engineering that determines latency, throughput, and cost.

### [Advanced Topics](05_advanced/index.md)

RAG pipelines, agents and tool use, long-context modeling, multimodal LLMs, evaluation benchmarks, and hallucination/safety. Where LLMs meet production.

### [Landmark Research Papers](06_research_papers/index.md)

The 33 most influential papers in the LLM field — from "Attention Is All You Need" through DeepSeek-R1, GLM-5, and Kimi K2.5. Each with a concise summary explaining the problem it solved, the architecture it introduced, why it was needed, and the key ideas with step-by-step math, runnable code, and interview Q&A.

### [Recent Advances](07_recent_advances/index.md)

A rolling log of frontier research — reasoning models, test-time compute scaling, efficient inference, and new open-weight releases.

---

## How each page is structured

```
Why This Matters for LLMs
   → Why interviewers ask about this topic

Core Concepts
   → Plain-English explanation (no jargon)
   → The math (in a styled callout)
   → "In plain English, this says..." (after every equation)
   → Worked example with real numbers (step by step)

Deep Dive
   → Extended treatment for advanced understanding

Code
   → Complete, runnable Python with all imports

Interview Guide
   → FAANG-level questions with expected depth
   → Follow-up probes
   → Key phrases to use
```

---

---

## Interview Preparation

Every page in LLMBase includes a structured **Interview Guide** section with:

- Specific questions asked at FAANG-tier companies, with expected answer depth
- Common follow-up probes that test whether you truly understand or just memorized
- Key phrases that signal expertise to interviewers
- Red-flag answers that suggest shallow understanding

For a consolidated view, see the **[Interview Question Bank](interview_questions.md)** — 85+ questions organized by topic and difficulty.

---

*Built for engineers who want to understand LLMs deeply, not just use them.*
