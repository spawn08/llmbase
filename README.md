# LLMBase

A ground-up guide to Large Language Models — from first principles to cutting-edge research. Built as a [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) site with math rendering, runnable code, and interview-ready depth.

**Live site:** [spawn08.github.io/llmbase](https://spawn08.github.io/llmbase/)

## Content

| Part | Topics | Status |
|------|--------|--------|
| **1. Foundations** | Language modeling, embeddings, RNNs/LSTMs, seq2seq, information theory, attention math | Complete |
| **2. Core Architectures** | Transformer, MHA/GQA, positional encoding, GPT, BERT, T5, MoE, Mamba/SSMs | Complete |
| **3. Training & Alignment** | Pre-training, distributed training, mixed precision, SFT, RLHF/DPO, Constitutional AI, LoRA/PEFT | Complete |
| **4. Inference & Serving** | Decoding strategies, KV cache, speculative decoding, continuous batching, quantization, serving systems | Complete |
| **5. Advanced Topics** | RAG, agents & tool use, long-context, multimodal LLMs, emergent capabilities, evaluation, safety | Complete |
| **6. Research Papers** | 25 landmark papers with TL;DR, key contributions, and code | Planned |
| **7. Recent Advances** | Rolling updates on frontier research | Planned |

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve        # http://127.0.0.1:8000
```

## Prerequisites

- Python 3.11+
- Engineering-level linear algebra and calculus
- Basic PyTorch (`nn.Module`, tensors)

## Tech Stack

| Component | Tool |
|-----------|------|
| Site generator | MkDocs + Material |
| Math rendering | MathJax 3 (Arithmatex) |
| Diagrams | draw.io → SVG |
| Code execution | Jupyter + Colab badges |
| Hosting | GitHub Pages |
| CI/CD | GitHub Actions |

## Project Structure

```
docs/
├── 01_foundations/          # 6 topics
├── 02_core_architectures/   # 8 topics
├── 03_training/             # 7 topics
├── 04_inference/            # 6 topics
├── 05_advanced/             # 7 topics
├── 06_research_papers/      # planned
├── 07_recent_advances/      # planned
└── interview_questions.md
```

## License

MIT
