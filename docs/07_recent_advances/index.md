# Recent Advances in LLM Research

## How to Use This Section

A chronological log of frontier LLM developments from January 2025 onward. Each entry explains **what changed technically**, **why it matters**, and **what interviewers probe**. Use the timeline to orient yourself, then drill into individual entries for depth.

## Last Updated

April 2026

---

## Timeline at a Glance

| Date | Event | Significance |
|------|-------|-------------|
| Jan 2025 | DeepSeek-R1 & R1-Zero released | RL-only reasoning without SFT; GRPO goes mainstream |
| Jan 2025 | Kimi k1.5 (Moonshot AI) | Long-context RL scaling; matches o1 without MCTS |
| Feb 2025 | GPT-4.5 (OpenAI) | Last major GPT-4 generation; unsupervised learning scaling |
| Mar 2025 | Gemini 2.5 Pro preview (Google) | 1M context, Deep Think mode, configurable reasoning budget |
| Apr 2025 | Llama 4 Scout & Maverick (Meta) | First open-weight natively multimodal MoE; 10M ctx Scout |
| May 2025 | Gemini 2.5 Pro GA + I/O updates | Top LMArena ranking, grounded search |
| Jun 2025 | Gemini 2.5 Flash GA | 25% faster, 85% cheaper than Gemini 1.5 Pro |
| Aug 2025 | GPT-5 (OpenAI) | Unified fast+reasoning router; unified model replaces GPT-4o/o1 split |
| Sep 2025 | GLM-4.6 (Zhipu AI) | 357B MoE, 200K context, open-weight under MIT |
| Nov 2025 | Claude Opus 4.5 (Anthropic) | SotA on software engineering benchmarks |
| Feb 2026 | Gemini 3.1 Pro (Google) | 77.1% on ARC-AGI-2; cost-competitive with proprietary frontier |
| Mar 2026 | GPT-5.4 family (OpenAI) | 1M+ ctx, native computer use, mini/nano variants |
| Apr 2026 | Llama 4 Behemoth (Meta, in training) | 288B active / 16-expert MoE; distillation teacher |

---

## Individual Entries

| # | Entry | Date |
|---|-------|------|
| 1 | [DeepSeek-R1 and R1-Zero](01_deepseek_r1.md) | January 2025 |
| 2 | [Kimi k1.5](02_kimi_k1_5.md) | January 2025 |
| 3 | [GPT-4.5](03_gpt4_5.md) | February 2025 |
| 4 | [Gemini 2.5 Pro](04_gemini_2_5_pro.md) | March–June 2025 |
| 5 | [Llama 4](05_llama_4.md) | April 2025 |
| 6 | [GPT-5](06_gpt5.md) | August 2025 |
| 7 | [Claude Opus 4.5](07_claude_opus_4_5.md) | November 2025 |
| 8 | [Gemini 3.1 Pro](08_gemini_3_1_pro.md) | February 2026 |
| 9 | [GPT-5.4 Family](09_gpt5_4.md) | March 2026 |
| 10 | [GLM-4.6](10_glm4_6.md) | September 2025 |

Each entry covers: **What Changed**, **Key Technical Details** (with math and plain-English explanations), **Practical Implications**, **Interview Questions**, and **Code Examples** where applicable.

---

## Cross-Cutting Themes (2025–2026)

**Reasoning as a first-class axis**: every frontier lab now has a reasoning model (o3, R1, Gemini Deep Think, GLM-Z1, Claude extended thinking). RL with verifiable rewards is the standard recipe.

**Open-weight MoE at scale**: Llama 4, GLM-4.6, Qwen2.5-Plus — open MoE models are now competitive with closed frontiers. Self-hosting is viable for many production use cases.

**Context window beyond 128K**: 1M (Maverick, Gemini 2.5), 10M (Scout), 200K (GLM-4.6). Cost-per-token matters as much as quality at these lengths.

**Natively multimodal pretraining**: early fusion (Llama 4) is gaining over bolt-on vision encoders. Models learn joint text-image representations from day one.

**Model families via distillation**: GPT-5.4 (Pro/mini/nano), R1 (Distill variants), Scout (distilled from Behemoth). One training run, multiple deployment tiers.

**Cost as a frontier**: Gemini 3.1 at 1/3 the cost of GPT-5.4 Pro at comparable quality — efficiency is now a primary competitive dimension, not just a secondary consideration.

## Further Reading

- *Core Architectures* (MoE, MLA, iRoPE) — technical details on the attention and expert mechanisms underlying these models
- *Training and Alignment* (RLHF, DPO, GRPO) — the RL recipes powering reasoning models
- *Inference and Serving* (KV cache, speculative decoding, quantization) — how these large models run in production
- *Top 25 Papers* — the research lineage: DeepSeekMath → R1, Chinchilla → compute-optimal training, InstructGPT → RLHF

Verify **vendor docs** and **licenses** before production use — this page reflects the state as of April 2026.
