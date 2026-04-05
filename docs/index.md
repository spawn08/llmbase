# LLMBase

> A living reference for engineers entering the world of Large Language Models. From first principles to cutting-edge research — with visualizations, code, and interview-ready depth.

Welcome to **LLMBase**. This guide is designed to take you from the fundamental mathematics of language modeling to building, serving, and understanding state-of-the-art Large Language Models.

**Live site:** [spawn08.github.io/llmbase](https://spawn08.github.io/llmbase/) (after GitHub Pages is enabled for the `gh-pages` branch).

## Who is this for?

- **Software engineers** moving into AI/ML or applied LLM roles.
- **Interview candidates** preparing for ML fundamentals, LLM internals, and GenAI system design.
- **Readers who want runnable code** alongside diagrams and math (formulas use MathJax, e.g. \(\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\bigl(\frac{QK^\top}{\sqrt{d_k}}\bigr)V\)).

## Prerequisites

- **Python:** Comfortable with Python 3.11+, basic OOP, and reading small PyTorch modules.
- **Math:** Derivatives, matrix multiplication, and probability at an engineering level.
- **PyTorch (later parts):** Tensors and `torch.nn` — introduced where needed.

## How to use this guide

LLMBase is split into **seven progressive parts**. Each part has an index with planned topics; individual concept pages will link diagrams (SVG), math, and self-contained Python snippets.

1. **[Foundations](01_foundations/index.md)** — Probability, embeddings, RNNs, seq2seq, information theory, attention math.
2. **[Core architectures](02_core_architectures/index.md)** — Transformer, GPT, BERT, T5, MoE, state-space models.
3. **[Training & alignment](03_training/index.md)** — Pre-training, distributed training, quantization, SFT, RLHF, PEFT.
4. **[Inference & serving](04_inference/index.md)** — Decoding, KV cache, speculative decoding, batching, quantization at serve time.
5. **[Advanced topics](05_advanced/index.md)** — RAG, agents, long context, multimodal, eval, safety.
6. **[Top 25 research papers](06_research_papers/index.md)** — Landmark papers with TL;DR, figures, and interview takeaways.
7. **[Recent advances](07_recent_advances/index.md)** — A rolling log of frontier work and links to code.

Use the tabs and search in the header to move between parts. Let's dive in.
