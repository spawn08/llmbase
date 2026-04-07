# LLMBase

<p align="center">
  <strong>From First Principles to Frontier Research — Interview-Ready LLM Documentation</strong>
</p>

<p align="center">
  <a href="https://spawn08.github.io/llmbase/">
    <img src="https://img.shields.io/badge/📖-Live%20Documentation-blue?style=for-the-badge" alt="Live Documentation">
  </a>
  <a href="https://github.com/spawn08/llmbase/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/📜-MIT%20License-green?style=for-the-badge" alt="License">
  </a>
  <a href="https://github.com/spawn08/llmbase/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/spawn08/llmbase/ci.yml?style=for-the-badge&label=CI%2FCD" alt="CI/CD">
  </a>
  <img src="https://img.shields.io/badge/📊-35%20Papers-ff69b4?style=for-the-badge" alt="Research Papers">
</p>

<p align="center">
  A comprehensive, ground-up guide to Large Language Models. Built as a production-quality documentation site with mathematical rigor, runnable code examples, and interview-grade depth.
</p>

---

## 🎯 What is LLMBase?

LLMBase is a **curated knowledge base** for anyone working with or interviewing on Large Language Models — from ML engineers and researchers to students preparing for technical interviews at AI companies.

Unlike scattered blog posts or fragmented tutorials, LLMBase provides:

- **📐 Mathematical Rigor** — Every equation derived step-by-step, no black boxes
- **💻 Runnable Code** — Python implementations you can execute and modify
- **🎓 Interview Focus** — Q&A sections with common interview questions and expected depth
- **🔬 Research Coverage** — 35 landmark papers with deep dives, not just abstracts
- **🚀 Current Through April 2026** — Rolling updates on frontier research and techniques

**Live documentation:** [spawn08.github.io/llmbase](https://spawn08.github.io/llmbase/)

---

## 📚 Content Overview

| Section | Papers/Topics | What You'll Learn |
|---------|---------------|-------------------|
| **[Foundations](docs/00_prerequisites/)** | 6 topics | Language modeling, embeddings, RNNs/LSTMs, seq2seq, information theory, attention math |
| **[Core Architectures](docs/02_core_architectures/)** | 8 topics | Transformer, MHA/GQA, positional encoding (RoPE), GPT, BERT, T5, MoE, Mamba/SSMs |
| **[Training & Alignment](docs/03_training/)** | 7 topics | Pre-training, distributed training, mixed precision, SFT, RLHF/DPO, Constitutional AI, LoRA |
| **[Inference & Serving](docs/04_inference/)** | 6 topics | Decoding strategies, KV cache, speculative decoding, continuous batching, quantization |
| **[Advanced Topics](docs/05_advanced/)** | 7 topics | RAG, agents & tool use, long-context, multimodal LLMs, emergent capabilities, safety |
| **[Research Papers](docs/06_research_papers/)** | **35 papers** | Landmark papers with TL;DR, math, code, and interview Q&A |
| **[Recent Advances](docs/07_recent_advances/)** | **13 topics** | Rolling updates on frontier research (April 2026) |

**Total:** 99 documentation files, 78+ commits, continuously updated.

---

## 🌟 Key Features

### 📐 Mathematical Depth That Interviewers Expect

Every major concept includes **step-by-step math** with worked examples:

```
Attention:  Attention(Q, K, V) = softmax(QK^T / √d_k) V
           ↓
Step 1: Compute similarity scores S = QK^T
Step 2: Scale by √d_k to prevent vanishing gradients
Step 3: Apply softmax to get attention weights
Step 4: Weight values V by attention weights
```

Not hand-wavy explanations — **derivable formulas** you can whiteboard in an interview.

### 💻 Runnable Python Implementations

Every research paper includes **executable code** demonstrating core ideas:

```python
# FlashAttention-style tiled computation
def flash_attention(Q, K, V, block_size=256):
    """Exact attention without materializing full N×N matrix."""
    # Tiled computation in SRAM, online softmax
    # Produces identical output to standard attention
    ...
```

Code is **tested, documented, and ready to modify** for experiments or interviews.

### 🎓 Interview-Grade Q&A

Each paper and major topic includes **interview questions** with **expected depth**:

> **Q:** Why is attention often memory-bound, not compute-bound?  
> **Expected depth:** Discuss GPU memory hierarchy (HBM vs SRAM), IO complexity analysis, why standard attention materializes N×N matrix, and how FlashAttention reduces HBM traffic. Include specific bandwidth numbers (A100: 312 TFLOPS vs 2 TB/s).

Not just answers — **what interviewers are actually listening for**.

### 🔬 Research Paper Deep Dves

35 landmark papers covered with:

- **TL;DR** — One-paragraph summary
- **Key Contributions** — What this paper actually changed
- **Math Explained** — Every equation, step by step
- **Python Implementation** — Runnable code
- **Interview Q&A** — 5-10 questions with detailed answers
- **Connections** — How it relates to other papers

**Papers include:** Transformer, BERT, GPT-2/3, T5, RoPE, InstructGPT, LoRA, FlashAttention, LLaMA, Chinchilla, Mistral, Mixtral, Mamba, DeepSeek-V2/V3/R1, Constitutional AI, Chain-of-Thought, ReAct, Toolformer, **Attention Residuals (Kimi 2026)**, and 17 more.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Engineering-level linear algebra and calculus
- Basic PyTorch (`nn.Module`, tensors)

### Local Development

```bash
# Clone the repository
git clone https://github.com/spawn08/llmbase.git
cd llmbase

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start local development server
mkdocs serve  # → http://127.0.0.1:8000
```

The site hot-reloads on file changes. Open your browser and start reading!

### Build for Production

```bash
mkdocs build  # Output in site/ directory
mkdocs gh-deploy  # Deploy to GitHub Pages
```

---

## 🏗️ Architecture & Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Site Generator** | [MkDocs](https://www.mkdocs.org/) + [Material](https://squidfunk.github.io/mkdocs-material/) | Fast, clean documentation with excellent search |
| **Math Rendering** | MathJax 3 (Arithmatex) | Beautiful LaTeX equations in markdown |
| **Diagrams** | draw.io → SVG | Scalable, editable architecture diagrams |
| **Code Execution** | Jupyter + Colab badges | Runnable examples in browser |
| **Hosting** | GitHub Pages | Free, reliable, custom domain support |
| **CI/CD** | GitHub Actions | Automated builds on every push |

---

## 📁 Project Structure

```
llmbase/
├── docs/
│   ├── 00_prerequisites/          # Math foundations, loss functions, CNNs
│   ├── 01_foundations/            # NLMs, embeddings, RNNs, seq2seq, info theory
│   ├── 02_core_architectures/     # Transformer, GPT, BERT, T5, MoE, Mamba, RoPE
│   ├── 03_training/               # Pre-training, distributed training, RLHF, LoRA
│   ├── 04_inference/              # Decoding, KV cache, quantization, serving
│   ├── 05_advanced/               # RAG, agents, long-context, multimodal, safety
│   ├── 06_research_papers/        # 35 landmark papers with deep dives
│   ├── 07_recent_advances/        # 13 frontier topics (April 2026)
│   ├── assets/                    # Images and diagrams
│   └── interview_questions.md     # Consolidated interview Q&A
├── notebooks/                     # Jupyter notebooks for interactive exploration
├── diagrams/                      # draw.io source files
├── mkdocs.yml                     # Site configuration
├── requirements.txt               # Python dependencies
└── README.md                      # You are here
```

---

## 📊 Content Coverage by Interview Topic

| Interview Topic | Where to Find It |
|-----------------|------------------|
| **Transformer Architecture** | [Core Architectures → Transformer](docs/02_core_architectures/transformer.md) |
| **Attention Mechanisms** | [Core Architectures → Attention](docs/02_core_architectures/attention.md) |
| **Positional Encoding (RoPE)** | [Paper #34: RoPE](docs/06_research_papers/34_rope_roformer.md) |
| **Fine-tuning (LoRA/PEFT)** | [Training → LoRA](docs/03_training/lora.md), [Paper #13](docs/06_research_papers/13_lora.md) |
| **RLHF & Alignment** | [Training → RLHF](docs/03_training/rlhf.md), [Paper #9](docs/06_research_papers/09_instructgpt.md) |
| **KV Cache & Inference** | [Inference → KV Cache](docs/04_inference/kv_cache.md) |
| **Speculative Decoding** | [Inference → Speculative Decoding](docs/04_inference/speculative_decoding.md) |
| **FlashAttention** | [Paper #14](docs/06_research_papers/14_flash_attention.md) |
| **MoE (Mixtral, DeepSeek-V3)** | [Paper #16](docs/06_research_papers/16_mixtral.md), [Paper #27](docs/06_research_papers/27_deepseek_v3.md) |
| **Reasoning (CoT, DeepSeek-R1)** | [Paper #18](docs/06_research_papers/18_chain_of_thought.md), [Paper #28](docs/06_research_papers/28_deepseek_r1.md) |
| **Agents & Tools (ReAct)** | [Paper #19](docs/06_research_papers/19_react.md) |
| **Long Context** | [Advanced → Long Context](docs/05_advanced/long_context.md) |
| **Attention Residuals** | [Paper #35](docs/06_research_papers/35_attention_residuals.md) |

---

## 🗺️ Roadmap & Recent Updates

### Recently Added (April 2026)

- ✅ **Paper #34: RoPE/RoFormer** — Rotary Position Embedding, the industry standard for positional encoding
- ✅ **Paper #35: Attention Residuals** — Kimi's 2026 redesign of Transformer residual connections
- ✅ Updated paper index with interconnection diagrams and interview priority guide
- ✅ Enhanced RLHF and KV cache documentation with detailed interview questions

### Planned Updates

- [ ] Multi-token prediction (MTP) deep dive
- [ ] GRPO vs PPO vs DPO comparison paper
- [ ] Extended coverage of vision-language models (VLMs)
- [ ] Interactive Jupyter notebooks for each architecture
- [ ] Spanish translation of core sections

---

## 🤝 Contributing

Contributions are welcome! This is a living document that improves with community input.

### How to Contribute

1. **Report Issues** — Found a bug, typo, or unclear explanation? [Open an issue](https://github.com/spawn08/llmbase/issues)
2. **Improve Content** — Submit a PR with corrections, clarifications, or additional examples
3. **Add Papers** — Suggest new research papers or recent advances
4. **Interview Questions** — Share questions you've encountered or want answered

### Development Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/llmbase.git
cd llmbase

# Create feature branch
git checkout -b feature/improve-rpe-explanation

# Make changes, test locally
mkdocs serve

# Commit and push
git add .
git commit -m "Improve RoPE explanation with additional worked example"
git push origin feature/improve-rpe-explanation

# Open Pull Request
```

**PR Guidelines:**
- Keep technical content accurate and cite sources
- Include runnable Python code for new implementations
- Add interview Q&A for major additions
- Follow existing formatting and style conventions

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute.

```
MIT License — Copyright (c) 2024-2026 LLMBase Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## 🙏 Acknowledgments

LLMBase synthesizes knowledge from:
- **Original paper authors** — Vaswani, Devlin, Radford, Touvron, Dao, and hundreds of researchers
- **Open-source frameworks** — PyTorch, Hugging Face, vLLM, and the broader ML community
- **Interview candidates** — Whose questions and feedback shaped the interview-focused format

Built with ❤️ for the ML community. If this helps you land your dream AI role, consider contributing back!

---

## 📬 Stay Updated

- ⭐ **Star this repo** to follow updates and support the project
- 🐦 **Share with peers** preparing for ML interviews
- 📧 **Watch releases** for major content updates

---

<p align="center">
  <strong>Ready to dive in?</strong><br>
  <a href="https://spawn08.github.io/llmbase/">📖 Read the Documentation</a> •
  <a href="docs/06_research_papers/index.md">🔬 Browse Research Papers</a> •
  <a href="docs/interview_questions.md">🎓 Practice Interview Questions</a>
</p>
