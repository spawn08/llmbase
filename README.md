# LLMBase — A Ground-Up Guide to Large Language Models

> A living reference for engineers entering the world of Large Language Models.
> From first principles to cutting-edge research — with visualizations, code, and interview-ready depth.

---

## Vision

LLMBase is a self-contained, open knowledge base for engineers who want to deeply understand Large Language Models. It is structured as a GitHub Pages site that progresses from mathematical foundations through state-of-the-art architectures and research, with interactive diagrams, runnable code, and curated paper summaries.

The goal is a single resource you can return to for interviews, system design, or research exploration — without needing to chase scattered blog posts.

---

## Site Structure

```
llmbase/
├── README.md                        ← This file (project overview + roadmap)
├── requirements.txt                 ← Python dependencies for MkDocs
├── docs/                            ← GitHub Pages source (Jekyll or MkDocs)
│   ├── index.md                     ← Landing page
│   ├── 01_foundations/              ← Part 1: Foundations
│   ├── 02_core_architectures/       ← Part 2: Core Architectures
│   ├── 03_training/                 ← Part 3: Training & Alignment
│   ├── 04_inference/                ← Part 4: Inference & Serving
│   ├── 05_advanced/                 ← Part 5: Advanced Topics
│   ├── 06_research_papers/          ← Part 6: Top 25 Research Papers
│   ├── 07_recent_advances/          ← Part 7: Recent Advances
│   └── assets/
│       ├── diagrams/                ← draw.io XML + exported SVGs
│       ├── notebooks/               ← Jupyter notebooks (linked inline)
│       └── css/                     ← Custom theme overrides
├── notebooks/                       ← Source Jupyter notebooks
│   ├── foundations/
│   ├── architectures/
│   └── research/
├── diagrams/                        ← draw.io source files (.drawio)
└── mkdocs.yml                       ← Site config (or _config.yml for Jekyll)
```

---

## Prerequisites

To get the most out of LLMBase, you should have:

- **Python Proficiency**: Comfortable with Python 3.11+, OOP, and basic functional programming.
- **Basic Calculus & Linear Algebra**: Understanding of derivatives, matrix multiplication, and vectors.
- **Familiarity with PyTorch**: Basic tensor operations and neural network modules (`torch.nn`).

---

## Content Plan

### Part 1 — Foundations (`01_foundations/`)

| Topic | Description | Visualizations | Code |
|---|---|---|---|
| 1.1 Language Modeling Basics | N-grams, probability, perplexity | Probability tree diagram | Python n-gram LM |
| 1.2 Word Embeddings | Word2Vec, GloVe, FastText | 2D t-SNE embedding plot | `gensim` Word2Vec training |
| 1.3 Neural Language Models | FFNNs, RNNs, LSTMs for text | Unrolled RNN diagram | PyTorch LSTM LM |
| 1.4 Sequence-to-Sequence | Encoder-decoder, attention basics | Seq2Seq architecture diagram | Minimal seq2seq |
| 1.5 Information Theory | Entropy, cross-entropy, KL divergence | Visual intuitions | NumPy derivations |
| 1.6 Mathematics of Attention | Dot-product, scaled, softmax | Attention weight heatmap | From-scratch attention |

### Part 2 — Core Architectures (`02_core_architectures/`)

| Topic | Description | Visualizations | Code |
|---|---|---|---|
| 2.1 The Transformer | Full architecture walkthrough | Multi-layer Transformer diagram | Annotated PyTorch implementation |
| 2.2 Self-Attention & MHA | Query/Key/Value, multi-head | Head-by-head attention maps | `einops`-based MHA |
| 2.3 Positional Encoding | Sinusoidal, learned, RoPE, ALiBi | Frequency heatmaps | All four encodings implemented |
| 2.4 GPT — Decoder-Only | Causal masking, autoregressive generation | GPT block diagram | Minimal GPT (nanoGPT-style) |
| 2.5 BERT — Encoder-Only | Masked LM, NSP, fine-tuning | BERT pre-training diagram | HuggingFace fine-tune walkthrough |
| 2.6 T5 / Encoder-Decoder | Text-to-text framing | T5 architecture diagram | T5 fine-tuning example |
| 2.7 Mixture of Experts | Sparse routing, load balancing | MoE routing diagram | Simple MoE block |
| 2.8 State Space Models | Mamba, S4, selective state spaces | SSM state diagram | Minimal Mamba block |

### Part 3 — Training & Alignment (`03_training/`)

| Topic | Description | Visualizations | Code |
|---|---|---|---|
| 3.0 Data Preparation | Deduplication, MinHash, filtering, PII removal | Data pipeline diagram | `datatrove` pipeline |
| 3.1 Pre-training at Scale | Data pipelines, tokenization, BPE | BPE merge tree | `tokenizers` library walkthrough |
| 3.2 Distributed Training | Data/model/pipeline/tensor parallelism | Parallelism strategy diagram | `torch.distributed` overview |
| 3.3 Mixed Precision & Quantization | FP16, BF16, INT8, GPTQ | Precision format comparison | BitsAndBytes quantization |
| 3.4 Instruction Tuning | FLAN, InstructGPT, SFT | SFT training pipeline | HuggingFace SFT trainer |
| 3.5 RLHF | Reward model, PPO, DPO | RLHF loop diagram | TRL DPO example |
| 3.6 Constitutional AI | RLAIF, critique-revision | CAI feedback loop | Conceptual walkthrough |
| 3.7 Parameter-Efficient Fine-Tuning | LoRA, QLoRA, Adapters, Prefix Tuning | LoRA decomposition diagram | PEFT library examples |

### Part 4 — Inference & Serving (`04_inference/`)

| Topic | Description | Visualizations | Code |
|---|---|---|---|
| 4.1 Decoding Strategies | Greedy, beam, top-k, top-p, temperature | Decoding tree diagram | Implemented from scratch |
| 4.2 KV Cache | Memory layout, cache eviction | KV cache memory diagram | Manual KV cache in PyTorch |
| 4.3 Speculative Decoding | Draft model, verification | Speculative decoding flow | Conceptual code walkthrough |
| 4.4 Continuous Batching | vLLM paged attention, throughput | Paged attention diagram | vLLM API usage |
| 4.5 Quantization for Inference | AWQ, GGUF, llama.cpp | Quantization bit-width chart | `llama-cpp-python` example |
| 4.6 LLM Serving Systems | Triton, vLLM, TGI, Ollama | Serving architecture diagram | Deployment walkthrough |

### Part 5 — Advanced Topics (`05_advanced/`)

| Topic | Description | Visualizations | Code |
|---|---|---|---|
| 5.1 RAG — Retrieval-Augmented Generation | Indexing, retrieval, augmentation | RAG pipeline diagram | LangChain / LlamaIndex |
| 5.1.5 Vector Databases | FAISS, Chroma, Pinecone, HNSW | HNSW graph diagram | FAISS index from scratch |
| 5.2 Agents & Tool Use | ReAct, tool calling, planning | Agent loop diagram | OpenAI function calling |
| 5.3 Long-Context Modeling | FlashAttention, sliding window, RoPE scaling | Context window diagram | FlashAttention usage |
| 5.4 Multimodal LLMs | CLIP, vision encoders, LLaVA | Multimodal fusion diagram | LLaVA walkthrough |
| 5.5 Emergent Capabilities | In-context learning, chain-of-thought, scaling laws | Scaling law plots | Prompting experiments |
| 5.5.5 Prompt Engineering | Few-shot, ReAct, DSPy, optimization | Prompt flow diagram | DSPy optimization loop |
| 5.6 Evaluation & Benchmarking | MMLU, HellaSwag, HumanEval, ELO | Leaderboard comparison chart | `lm-evaluation-harness` |
| 5.7 Hallucination & Safety | Factuality, groundedness, red-teaming | Failure mode taxonomy | Detection heuristics |

### Part 6 — Top 25 Research Papers (`06_research_papers/`)

Each paper entry includes: **TL;DR summary**, **key contributions**, **architecture diagram or figure**, **reproduced code snippet**, **why it matters for interviews**.

| # | Paper | Year | Significance |
|---|---|---|---|
| 1 | Attention Is All You Need | 2017 | Introduced the Transformer |
| 2 | BERT | 2018 | Bidirectional pre-training |
| 3 | GPT-2 | 2019 | Scaling autoregressive LMs |
| 4 | GPT-3 | 2020 | Few-shot in-context learning |
| 5 | T5 | 2019 | Text-to-text unification |
| 6 | XLNet | 2019 | Permutation LM |
| 7 | RoBERTa | 2019 | Robust BERT pre-training |
| 8 | ELECTRA | 2020 | Replaced token detection |
| 9 | InstructGPT (RLHF) | 2022 | Alignment via human feedback |
| 10 | PaLM | 2022 | Pathways-scale LM |
| 11 | Chinchilla | 2022 | Compute-optimal scaling laws |
| 12 | LLaMA (1 & 2) | 2023 | Open-source competitive LM |
| 13 | LoRA | 2021 | Low-rank adaptation |
| 14 | FlashAttention (1 & 2) | 2022/23 | IO-aware exact attention |
| 15 | Mistral 7B | 2023 | Sliding window attention |
| 16 | Mixtral (MoE) | 2023 | Sparse MoE at scale |
| 17 | FLAN | 2021 | Instruction finetuning |
| 18 | Chain-of-Thought Prompting | 2022 | Reasoning via CoT |
| 19 | ReAct | 2022 | Reasoning + Acting agents |
| 20 | Toolformer | 2023 | Self-supervised tool use |
| 21 | Constitutional AI | 2022 | RLAIF alignment |
| 22 | CLIP | 2021 | Contrastive language-image |
| 23 | Codex | 2021 | Code generation |
| 24 | Mamba | 2023 | Selective state space models |
| 25 | Gemini | 2023 | Multimodal foundation model |

### Part 7 — Recent Advances (`07_recent_advances/`)

A regularly updated section tracking frontier research. Each entry includes a summary, key ideas, and code where available.

**Planned topics (as of April 2026):**

- GPT-4o and multimodal reasoning
- Claude 3 / Claude 4 architecture insights (publicly available)
- DeepSeek R1 and reasoning models
- Phi-3 / Phi-4 — small models, large knowledge
- Llama 3.1 and open-weight scaling
- Mixture-of-Depths
- Test-time compute scaling (o1 / o3 paradigm)
- Long-context retrieval without RAG
- Continuous learning / online fine-tuning
- LLM inference on edge devices

---

## Visualization Strategy

All architecture diagrams are authored in **draw.io** (`.drawio` XML format), exported to **SVG** for lossless rendering on the site, and embedded inline in Markdown.

Each concept page follows this layout:

```
## Concept Name

### Intuition
(plain English explanation — no jargon)

### Architecture / Mechanism
![Diagram](../assets/diagrams/concept-name.svg)

### Mathematical Formulation
(LaTeX equations rendered via MathJax/KaTeX)

### Code
(Python with all imports — runnable as-is)

### Interview Takeaways
(bullet points — what interviewers actually ask)

### References
(paper links, related topics)
```

---

## Code Standards

- All code is **self-contained** — every snippet includes its full imports.
- Primary libraries: `torch`, `transformers`, `tokenizers`, `datasets`, `peft`, `trl`, `einops`, `numpy`, `matplotlib`.
- Each major topic has a corresponding **Jupyter notebook** in `notebooks/` that can be run on Google Colab (badge included).
- Code targets Python 3.11+.

---

## Site Technology

| Concern | Choice | Rationale |
|---|---|---|
| Static site generator | MkDocs + Material theme | Best-in-class Markdown rendering, search, code highlighting, MathJax |
| Diagrams | draw.io → SVG | Version-controlled source, lossless web rendering |
| Math | MathJax | LaTeX support in Markdown |
| Code execution | Jupyter + Colab badges | Zero-friction for readers |
| Hosting | GitHub Pages | Free, versioned, CI-deployable |
| CI | GitHub Actions | Auto-deploy on push to `main` |

---

## Build & Deploy

```bash
# Install dependencies
pip install mkdocs-material mkdocs-jupyter

# Serve locally
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

GitHub Actions workflow (`.github/workflows/deploy.yml`) will auto-deploy on every push to `main`.

---

## Contribution & Maintenance

This repository follows a simple flow:

1. Each topic is a Markdown file in `docs/`.
2. Diagrams are `.drawio` files committed to `diagrams/` and exported SVGs committed to `docs/assets/diagrams/`.
3. Notebooks are committed to `notebooks/` and linked inline with a Colab badge.
4. The `07_recent_advances/` section is updated as new research lands.

---

## Interview Reference Quick Links

Once the site is live, these sections are most valuable for interview preparation:

- **Transformer internals** → `02_core_architectures/transformer.md`
- **Attention from scratch** → `01_foundations/attention-math.md`
- **RLHF / Alignment** → `03_training/rlhf.md`
- **RAG architecture** → `05_advanced/rag.md`
- **Scaling laws** → `05_advanced/emergent-capabilities.md`
- **Top papers TL;DR** → `06_research_papers/`
- **Decoding strategies** → `04_inference/decoding-strategies.md`
- **LoRA / PEFT** → `03_training/peft.md`

---

## Roadmap

- [ ] Phase 1 — Scaffold: MkDocs setup, CI/CD, site structure, landing page
- [ ] Phase 2 — Foundations (Part 1): All 6 topics with diagrams and code
- [ ] Phase 3 — Architectures (Part 2): Transformer deep-dive, GPT, BERT, T5
- [ ] Phase 4 — Training & Alignment (Part 3): Pre-training, SFT, RLHF, LoRA
- [ ] Phase 5 — Inference (Part 4): KV cache, decoding, vLLM
- [ ] Phase 6 — Advanced (Part 5): RAG, agents, multimodal, eval
- [ ] Phase 7 — Research Papers (Part 6): All 25 papers with summaries + code
- [ ] Phase 8 — Recent Advances (Part 7): Rolling updates
- [ ] Phase 9 — Polish: Search tuning, mobile layout, Colab badge verification

---

*Built for engineers, by engineers. Contributions welcome.*
